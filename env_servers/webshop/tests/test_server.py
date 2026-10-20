"""Engine-tier tests (py3.10 world).

Three layers, all against the fake engine factory (no pyserini/JVM/spaCy):
- WebshopEngineWorkerPool: spawns REAL worker processes; env behavior is
  scripted through the factory_module seam (fake_engine_factory) because
  monkeypatching cannot cross the spawn boundary. This is the concurrency lever
  under test — placement, isolation, respawn, stale-session invalidation,
  per-worker executor fairness, dead-slot parking/revival, shutdown.
- WebshopSharedBackend: in-process single engine + RLock; the factory functions
  are monkeypatched on the server module (no spawn, so monkeypatch is enough).
- create_app: the FastAPI protocol (410-on-unknown is the contract the py3.13
  adapter's _engine_client depends on; ValueError -> 400, other -> 500).
- Content addressing: reset by goal_id == reset by list position, an unknown
  goal_id is a 400 that registers nothing, /goals + /health.goals_digest are
  what export_goals.py freezes the dataset from (tested end to end over a real
  loopback port, since the CLI speaks stdlib urllib).

A gated TestRealEngine at the bottom runs the true vendored stack when
$WEBSHOP_VENDOR + $WEBSHOP_DATA_DIR point at a built catalog.
"""

import contextlib
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import fake_engine_factory as fake
import pytest
from _engine_worker import EngineWorkerDied, EngineWorkerError, WebshopEngineWorkerPool
from _webshop_engine import EngineConfig, UnknownGoal, is_reserved_search, step_env
from agent_utils.webshop_goals import goal_id, goals_set_digest
from fake_engine_factory import GOAL_BOOM, GOAL_DIE, GOAL_HANG, N_GOALS
from session_server import ResetResult, SessionBackend, StepResult, UnknownSession, base_arg_parser, create_app

FAKE_FACTORY = "fake_engine_factory"
# What the fake engine's goal list hashes to, computed independently of the
# server code under test (same shared module the dataset side uses).
FAKE_IDS = [goal_id(fake.fake_goal(i)) for i in range(N_GOALS)]
FAKE_DIGEST = goals_set_digest(FAKE_IDS)
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
UNKNOWN_ID = "0123456789abcdef"


def _pool_health(workers: int, dead_workers: int, respawns: int) -> dict:
    """The full /health backend block of a fake-backed pool."""
    return {
        "workers": workers,
        "dead_workers": dead_workers,
        "respawns": respawns,
        "goals": N_GOALS,
        "unique_goals": N_GOALS,
        "goals_digest": FAKE_DIGEST,
    }


def _cfg(**kw) -> EngineConfig:
    kw.setdefault("vendored_path", "/fake/vendor")
    kw.setdefault("data_dir", "/fake/data")
    kw.setdefault("factory_module", FAKE_FACTORY)
    return EngineConfig(**kw)


def _raises(fn) -> BaseException | None:
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 - surfaced to the test
        return e
    return None


def _run_threads(fns) -> list[BaseException]:
    """Run each callable on its own thread; return the exceptions they raised."""
    errors: list[BaseException] = []

    def wrap(fn):
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 - surfaced to the test
            errors.append(e)

    threads = [threading.Thread(target=wrap, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30.0)
    assert not any(t.is_alive() for t in threads), "a worker thread hung"
    return errors


# ────────────────────────────────────────────────────────── worker pool
class TestEngineWorkerPool:
    def test_reset_step_close_roundtrip(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            r = pool.reset("s1", 5, "train")
            assert r.obs == "fake obs for goal 5" and r.info["goal_idx"] == 5
            s = pool.step("s1", "search[mug]")
            assert (s.obs, s.reward, s.done) == ("echo:search[mug]:1", 0.0, False)
            s2 = pool.step("s1", "click[buy now]")
            assert s2.done is True and s2.reward == 1.0 and s2.info["won"] is True
            assert pool.live_sessions() == []  # terminal step forgets the session
            assert pool._counts == [0]
        finally:
            pool.shutdown()

    def test_many_sessions_one_worker_isolated(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("a", 1, "train")
            pool.reset("b", 2, "train")
            pool.step("a", "x")
            pool.step("a", "x")
            pool.step("b", "x")
            assert pool.step("a", "x").obs == "echo:x:3"  # independent counters
            assert pool.step("b", "x").obs == "echo:x:2"
        finally:
            pool.shutdown()

    def test_reset_spreads_to_least_loaded_worker(self):
        pool = WebshopEngineWorkerPool(_cfg(), 2, 30.0)
        try:
            for i in range(4):
                pool.reset(f"s{i}", i, "train")
            assert sorted(pool._counts) == [2, 2]
        finally:
            pool.shutdown()

    def test_factory_error_is_op_error_worker_survives(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            with pytest.raises(EngineWorkerError, match="scripted load failure"):
                pool.reset("s", GOAL_BOOM, "train")
            assert pool.respawns == 0
            r = pool.reset("s2", 1, "train")  # same worker, still healthy
            assert r.obs == "fake obs for goal 1"
            assert pool._counts == [1]  # failed reset rolled its count back
        finally:
            pool.shutdown()

    def test_step_error_is_op_error_session_survives(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("s", 1, "train")
            with pytest.raises(EngineWorkerError, match="scripted step failure"):
                pool.step("s", "explode")
            assert pool.step("s", "look").obs == "echo:look:1"  # env unharmed
        finally:
            pool.shutdown()

    def test_worker_death_respawns_and_invalidates_stale_sessions(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("survivor", 1, "train")
            with pytest.raises(EngineWorkerDied):
                pool.reset("victim", GOAL_DIE, "train")
            assert pool.respawns == 1
            with pytest.raises(UnknownSession):
                pool.step("survivor", "look")  # resident env died with the worker
            pool.close("survivor")  # stale close is a silent no-op
            r = pool.reset("fresh", 2, "train")  # respawned worker serves resets
            assert r.obs == "fake obs for goal 2"
            assert pool._counts == [1]
            assert pool.health() == _pool_health(1, 0, 1)
        finally:
            pool.shutdown()

    def test_wedged_worker_times_out_and_respawns(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 0.5)
        try:
            with pytest.raises(EngineWorkerDied, match="exceeded 0.5s"):
                pool.reset("s", GOAL_HANG, "train")
            assert pool.respawns == 1
            assert pool.reset("s2", 1, "train").obs == "fake obs for goal 1"
        finally:
            pool.shutdown()

    def test_wedged_worker_does_not_block_other_workers(self):
        # Failure isolation: while worker 0 is wedged inside a hanging reset,
        # ops pinned to worker 1 keep flowing (each worker's executor is its
        # own FIFO; no shared lock for the wedge to hold).
        pool = WebshopEngineWorkerPool(_cfg(), 2, 5.0)
        try:
            hang_errors: list[BaseException] = []

            def hang():
                try:
                    pool.reset("hang", GOAL_HANG, "train")  # reserves worker 0
                except EngineWorkerDied as e:
                    hang_errors.append(e)

            hang_thread = threading.Thread(target=hang)
            hang_thread.start()
            time.sleep(0.2)
            t0 = time.monotonic()
            pool.reset("s", 1, "train")  # least-loaded -> worker 1
            assert pool._sessions["s"][0] == 1
            for i in range(20):
                assert pool.step("s", "look").obs == f"echo:look:{i + 1}"
            assert time.monotonic() - t0 < 3.0  # finished while worker 0 still wedged
            hang_thread.join(15.0)
            assert hang_errors and "exceeded 5.0s" in str(hang_errors[0])
            assert pool.respawns == 1
            assert pool.live_sessions() == ["s"]  # the survivor was on the other worker
        finally:
            pool.shutdown()

    def test_concurrent_ops_through_executors(self):
        # 8 sessions x 20 steps across 2 workers from 8 threads: no errors, no
        # cross-talk between episodes, counts consistent afterwards.
        pool = WebshopEngineWorkerPool(_cfg(), 2, 30.0)
        try:
            for i in range(8):
                pool.reset(f"s{i}", i, "train")
            assert sorted(pool._counts) == [4, 4]

            def stepper(i):
                def run():
                    last = None
                    for _ in range(20):
                        last = pool.step(f"s{i}", "look")
                    assert last is not None and last.obs == "echo:look:20"

                return run

            assert _run_threads([stepper(i) for i in range(8)]) == []
            assert sorted(pool._counts) == [4, 4] and pool.respawns == 0
            for i in range(8):
                pool.close(f"s{i}")
            assert pool._counts == [0, 0] and pool.live_sessions() == []
        finally:
            pool.shutdown()

    def test_dead_slot_is_parked_and_revived(self, tmp_path):
        # A worker whose replacement cannot boot is parked (no respawn storm,
        # balancer routes around it) and comes back once the fault clears.
        marker = tmp_path / "BOOM"
        pool = WebshopEngineWorkerPool(_cfg(vendored_path=str(tmp_path)), 2, 30.0, revive_interval_s=0)
        try:
            marker.write_text("")  # every (re)build from now on fails
            with pytest.raises(EngineWorkerDied, match="failed to respawn"):
                pool.reset("victim", GOAL_DIE, "train")  # both idle -> worker 0 dies, reboot fails
            assert pool.health() == _pool_health(2, 1, 1)
            for i in range(3):
                assert pool.reset(f"s{i}", i, "train").obs == f"fake obs for goal {i}"
            assert {w for w, _, _ in pool._sessions.values()} == {1}  # parked slot skipped
            marker.unlink()
            assert pool._slots.revive_dead_slots() == 1
            assert pool.health() == _pool_health(2, 0, 2)
            pool.reset("back", 7, "train")
            assert pool._sessions["back"][0] == 0  # revived slot is now the least loaded
            assert pool.step("back", "look").obs == "echo:look:1"
        finally:
            pool.shutdown()

    def test_parked_slot_fails_fast_when_all_parked(self, tmp_path):
        marker = tmp_path / "BOOM"
        pool = WebshopEngineWorkerPool(_cfg(vendored_path=str(tmp_path)), 1, 30.0, revive_interval_s=0)
        try:
            marker.write_text("")
            with pytest.raises(EngineWorkerDied, match="failed to respawn"):
                pool.reset("victim", GOAL_DIE, "train")
            t0 = time.monotonic()
            with pytest.raises(EngineWorkerDied, match="parked"):
                pool.reset("s", 1, "train")  # no respawn attempt per op (that was the storm)
            assert time.monotonic() - t0 < 1.0
            assert pool.respawns == 1 and pool.live_sessions() == []
            marker.unlink()
            assert pool._slots.revive_dead_slots() == 1
            assert pool.reset("s", 1, "train").obs == "fake obs for goal 1"
        finally:
            pool.shutdown()

    def test_reset_racing_respawn_is_not_registered(self):
        # Another session on the same worker dies right after our reset
        # returned but before we registered: the env we built died with it, so
        # reset must fail (410 upstream) rather than hand back a live obs.
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            real_call = pool._call

            def racing_call(idx, gen, msg):
                result = real_call(idx, gen, msg)
                if msg[0] == "reset":
                    pool._slots._executors[idx].submit(pool._slots._respawn, idx, "test").result()
                return result

            pool._call = racing_call
            with pytest.raises(EngineWorkerDied, match="respawned mid-reset"):
                pool.reset("s", 1, "train")
            del pool._call  # back to the class method
            assert pool.live_sessions() == [] and pool._counts == [0] and pool.respawns == 1
            assert pool.reset("s2", 2, "train").obs == "fake obs for goal 2"
        finally:
            pool.shutdown()

    def test_same_session_id_concurrent_resets_leave_one_episode(self):
        pool = WebshopEngineWorkerPool(_cfg(), 2, 30.0)
        try:
            barrier = threading.Barrier(2)

            def racer(goal):
                def run():
                    barrier.wait(5.0)
                    pool.reset("dup", goal, "train")

                return run

            assert _run_threads([racer(1), racer(2)]) == []
            assert pool.live_sessions() == ["dup"]
            assert sum(pool._counts) == 1  # the displaced episode was closed, not leaked
            assert pool.step("dup", "look").obs == "echo:look:1"
        finally:
            pool.shutdown()

    def test_close_forgets_session(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("s", 1, "train")
            pool.close("s")
            assert pool.live_sessions() == []
            with pytest.raises(UnknownSession):
                pool.step("s", "look")
        finally:
            pool.shutdown()

    def test_reused_session_id_re_resets(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("s", 1, "train")
            pool.step("s", "look")
            pool.step("s", "look")
            pool.reset("s", 2, "train")  # re-reset drops the old episode
            assert pool.step("s", "look").obs == "echo:look:1"  # counter rewound
            assert sum(pool._counts) == 1  # count did not double
        finally:
            pool.shutdown()

    def test_step_after_worker_lost_episode_raises_unknown(self):
        # The close/TTL-vs-step race: the worker drops the env while the parent
        # still holds the session pointer. step must 410 (UnknownSession), not
        # surface a worker KeyError as a 500. Forced deterministically by
        # closing the episode inside the worker but leaving _sessions intact.
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("s", 1, "train")
            idx, gen, episode_id = pool._sessions["s"]
            pool._call(idx, gen, ("close", episode_id))  # worker drops the env
            with pytest.raises(UnknownSession):
                pool.step("s", "look")
            assert pool.live_sessions() == [] and pool._counts == [0]
        finally:
            pool.shutdown()

    def test_bad_task_id_is_value_error_without_touching_workers(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            with pytest.raises(ValueError, match="integer goal index"):
                pool.reset("s", "seven", "train")
            with pytest.raises(ValueError, match="integer goal index"):
                pool.reset("s", None, "train")
            with pytest.raises(ValueError, match="out of range"):
                pool.reset("s", N_GOALS, "train")
            assert pool._counts == [0] and pool.live_sessions() == []
        finally:
            pool.shutdown()

    def test_boot_failure_is_loud(self):
        with pytest.raises(RuntimeError, match="failed to boot") as ei:
            WebshopEngineWorkerPool(_cfg(vendored_path="/fake/BUILD-BOOM"), 1, 30.0)
        assert "Traceback (most recent call last)" in str(ei.value)  # child-side traceback travels with the failure

    def test_stale_gen_fails_fast_without_queueing_behind_busy_slot(self):
        # A stale generation is rejected at submit time: it must not wait behind
        # whatever occupies the slot's executor (a hung op, a booting replacement).
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        hung: list[BaseException] = []
        t = threading.Thread(target=lambda: hung.append(_raises(lambda: pool.reset("h", GOAL_HANG, "train"))))
        t.start()
        try:
            time.sleep(0.3)  # the hang is now in flight on slot 0's executor
            t0 = time.monotonic()
            with pytest.raises(EngineWorkerDied, match="was respawned"):
                pool._slots.call(0, pool._slots.gen(0) - 1, ("step", "x", "look"))
            assert time.monotonic() - t0 < 1.0
        finally:
            pool.shutdown()
            t.join(10.0)
        assert not t.is_alive() and isinstance(hung[0], EngineWorkerDied)

    def test_call_during_shutdown_raises_died(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool._slots.closing = True
            with pytest.raises(EngineWorkerDied, match="shutting down"):
                pool.reset("s", 1, "train")
            pool._slots.closing = False
        finally:
            pool.shutdown()

    def test_revive_only_touches_failed_slots(self):
        # A slot parked because a respawn is IN FLIGHT is not a revival target:
        # reviving it would kill the healthy replacement once it boots.
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0, revive_interval_s=0)
        try:
            pool._slots.dead_slots.add(0)  # simulate "respawn in flight" parking
            assert pool._slots.revive_dead_slots() == 0
            assert pool._slots._revive_one(0) is False
            pool._slots.dead_slots.discard(0)
            assert pool.respawns == 0
            assert pool.reset("s", 1, "train").obs == "fake obs for goal 1"
        finally:
            pool.shutdown()

    def test_start_failure_during_respawn_parks_slot(self):
        # Process.start() raising (EMFILE...) takes the parked-until-revival path
        # and surfaces as the died class, not a raw OSError.
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0, revive_interval_s=0)
        slots = pool._slots
        real_start = slots._start_proc

        def failing_start(idx: int) -> None:
            raise OSError(24, "Too many open files (scripted)")

        try:
            slots._start_proc = failing_start
            with pytest.raises(EngineWorkerDied, match="failed to respawn"):
                pool.reset("victim", GOAL_DIE, "train")
            assert pool.health() == _pool_health(1, 1, 1)
            with pytest.raises(EngineWorkerDied, match="parked"):
                pool.reset("s", 1, "train")
            slots._start_proc = real_start
            assert slots.revive_dead_slots() == 1
            assert pool.health() == _pool_health(1, 0, 2)
            assert pool.reset("s", 1, "train").obs == "fake obs for goal 1"
        finally:
            pool.shutdown()

    def test_shutdown_kills_hung_worker_promptly(self):
        pool = WebshopEngineWorkerPool(_cfg(), 2, 30.0)
        outcome: list[BaseException] = []

        def hang():
            try:
                pool.reset("hang", GOAL_HANG, "train")
            except EngineWorkerDied as e:
                outcome.append(e)

        hang_thread = threading.Thread(target=hang)
        hang_thread.start()
        time.sleep(0.2)
        t0 = time.monotonic()
        pool.shutdown(grace_s=1.0)
        assert time.monotonic() - t0 < 4.0
        assert all(not p.is_alive() for p in pool._slots._procs)
        hang_thread.join(5.0)
        assert not hang_thread.is_alive()
        assert outcome and pool.respawns == 0  # closing: the hung op fails, nothing respawns
        with pytest.raises(EngineWorkerDied, match="shutting down"):
            pool.reset("late", 1, "train")

    # ── content addressing (goal_id) ─────────────────────────────────
    def test_reset_by_goal_id_matches_reset_by_index(self):
        # The dataset's goal_id lands on the very episode the legacy position
        # would; task_id is ignored (may be absent) when goal_id is given.
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            by_idx = pool.reset("a", 5, "train")
            by_id = pool.reset("b", None, "train", goal_id=FAKE_IDS[5])
            assert by_id.obs == by_idx.obs == "fake obs for goal 5"
            assert by_id.info["goal_idx"] == by_idx.info["goal_idx"] == 5
            assert pool.step("b", "look").obs == "echo:look:1"
            assert sorted(pool.live_sessions()) == ["a", "b"] and pool._counts == [2]
        finally:
            pool.shutdown()

    def test_unknown_goal_id_is_value_error_and_registers_nothing(self):
        # The worker's UnknownGoal crosses the pipe as an op error and is
        # rebuilt as the ValueError the HTTP layer turns into 400 (never
        # retried), carrying the mismatch diagnosis; the reservation is rolled
        # back and the worker stays healthy.
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            with pytest.raises(UnknownGoal) as ei:
                pool.reset("s", None, "train", goal_id=UNKNOWN_ID)
            assert isinstance(ei.value, ValueError)
            assert str(ei.value) == (
                f"goal_id {UNKNOWN_ID!r} is not in the engine goal list (goals_digest {FAKE_DIGEST}): "
                "the dataset and the engine disagree on catalog / goal code"
            )
            assert pool.live_sessions() == [] and pool._counts == [0] and pool.respawns == 0
            assert pool.reset("s", 1, "train").obs == "fake obs for goal 1"
        finally:
            pool.shutdown()

    def test_health_digest_and_export_goals(self):
        pool = WebshopEngineWorkerPool(_cfg(), 2, 30.0)
        try:
            health = pool.health()
            assert health["goals"] == N_GOALS and health["goals_digest"] == FAKE_DIGEST
            assert _HEX16.match(health["goals_digest"])
            rows = pool.export_goals()
            assert [r["goal_idx"] for r in rows] == list(range(N_GOALS))
            assert [r["goal_id"] for r in rows] == FAKE_IDS
            assert all(goal_id(r) == r["goal_id"] for r in rows)  # rows are self-verifying content records
            assert goals_set_digest(r["goal_id"] for r in rows) == health["goals_digest"]
            assert rows[3]["instruction_text"] == "fake goal 3" and rows[3]["price_upper"] == 13.0
            assert "weight" not in rows[3]  # not an identity field, not exported
        finally:
            pool.shutdown()

    def test_export_goals_skips_parked_worker(self, tmp_path):
        marker = tmp_path / "BOOM"
        pool = WebshopEngineWorkerPool(_cfg(vendored_path=str(tmp_path)), 2, 30.0, revive_interval_s=0)
        try:
            marker.write_text("")
            with pytest.raises(EngineWorkerDied, match="failed to respawn"):
                pool.reset("victim", GOAL_DIE, "train")  # worker 0 dies, its reboot fails -> parked
            assert pool.health()["dead_workers"] == 1
            assert len(pool.export_goals()) == N_GOALS  # served by the live worker
        finally:
            pool.shutdown()

    def test_boot_fails_when_workers_disagree_on_goal_set(self, tmp_path):
        # A nondeterministic goal build would make /health's digest a lottery
        # and goal_id resolution worker-dependent: refuse to serve at all.
        (tmp_path / "SKEW").write_text("")  # every child builds a different goal list
        with pytest.raises(RuntimeError, match="disagree on the goal set"):
            WebshopEngineWorkerPool(_cfg(vendored_path=str(tmp_path)), 2, 30.0)

    def test_pool_over_http_goal_id_paths(self):
        # End to end through the HTTP layer: the cross-process UnknownGoal is a
        # 400 with the diagnosis, a known goal_id serves, /health and /goals
        # carry the content identity.
        from fastapi.testclient import TestClient

        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            client = TestClient(create_app(pool, session_ttl=1800.0, max_concurrency=4, sweep_interval=0))
            ok = client.post("/reset", json={"session_id": "a", "goal_id": FAKE_IDS[2]})
            assert ok.status_code == 200 and ok.json()["obs"] == "fake obs for goal 2"
            bad = client.post("/reset", json={"session_id": "b", "goal_id": UNKNOWN_ID})
            assert bad.status_code == 400 and "is not in the engine goal list" in bad.json()["detail"]
            assert pool.live_sessions() == ["a"]
            assert client.get("/health").json()["goals_digest"] == FAKE_DIGEST
            assert len(client.get("/goals").json()) == N_GOALS
        finally:
            pool.shutdown()


# ─────────────────────────────────────────────────────── shared backend
def _shared(monkeypatch) -> SessionBackend:
    import server as server_module

    monkeypatch.setattr(server_module, "build_engine", fake.build_engine)
    monkeypatch.setattr(server_module, "make_env", fake.make_env)
    monkeypatch.setattr(server_module, "step_env", fake.step_env)
    monkeypatch.setattr(server_module, "drop_engine_state", fake.drop_engine_state)
    return server_module.WebshopSharedBackend(_cfg())


class TestSharedBackend:
    def test_reset_step_close(self, monkeypatch):
        be = _shared(monkeypatch)
        assert be.reset("s", 3, "train").obs == "fake obs for goal 3"
        assert be.step("s", "search[x]").obs == "echo:search[x]:1"
        s2 = be.step("s", "click[buy now]")
        assert s2.done is True and s2.reward == 1.0
        assert be.live_sessions() == [] and be.worker_count() == 0

    def test_unknown_session_raises(self, monkeypatch):
        be = _shared(monkeypatch)
        with pytest.raises(UnknownSession):
            be.step("nope", "look")

    def test_session_key_isolation_and_cleanup(self, monkeypatch):
        # The GRPO-collision guard: concurrent episodes get per-session engine
        # keys, and close prunes only that session's keys.
        be = _shared(monkeypatch)
        be.reset("a", 1, "train")
        be.reset("b", 2, "train")
        assert any(k.startswith("a:") for k in be._engine.user_sessions)
        assert any(k.startswith("b:") for k in be._engine.user_sessions)
        be.close("a")
        assert not any(k.startswith("a:") for k in be._engine.user_sessions)
        assert any(k.startswith("b:") for k in be._engine.user_sessions)

    def test_reused_session_id_re_resets(self, monkeypatch):
        be = _shared(monkeypatch)
        be.reset("s", 1, "train")
        be.step("s", "look")
        be.reset("s", 2, "train")
        assert be.step("s", "look").obs == "echo:look:1"

    def test_health_and_bad_task_id(self, monkeypatch):
        be = _shared(monkeypatch)
        assert be.health() == {
            "workers": 0,
            "dead_workers": 0,
            "respawns": 0,
            "goals": N_GOALS,
            "unique_goals": N_GOALS,
            "goals_digest": FAKE_DIGEST,
        }
        with pytest.raises(ValueError):
            be.reset("s", "abc", "train")
        with pytest.raises(ValueError):
            be.reset("s", N_GOALS + 1, "train")
        assert be.live_sessions() == []

    def test_reset_by_goal_id_and_unknown_goal(self, monkeypatch):
        be = _shared(monkeypatch)
        by_id = be.reset("a", None, "train", goal_id=FAKE_IDS[7])
        by_idx = be.reset("b", 7, "train")
        assert by_id.obs == by_idx.obs == "fake obs for goal 7" and by_id.info["goal_idx"] == 7
        with pytest.raises(UnknownGoal, match=f"goals_digest {FAKE_DIGEST}"):
            be.reset("c", None, "train", goal_id=UNKNOWN_ID)
        assert sorted(be.live_sessions()) == ["a", "b"]

    def test_export_goals(self, monkeypatch):
        be = _shared(monkeypatch)
        rows = be.export_goals()
        assert [r["goal_id"] for r in rows] == FAKE_IDS and [r["goal_idx"] for r in rows] == list(range(N_GOALS))
        assert goals_set_digest(r["goal_id"] for r in rows) == be.health()["goals_digest"]


# ──────────────────────────────────────────── step_env semantics
class _FakeEng:
    def __init__(self):
        self.user_sessions = {}


class TestStepEnvTerminalAvail:
    """The vendored env auto-resets onto a new goal on done, so step_env must
    not leak that goal's admissible actions into the terminal observation."""

    class _FakeEnv:
        session_prefix = "s:"

        def __init__(self, done, avail):
            self._done = done
            self._avail = avail

        def step(self, action):
            return ("Thank you!" if self._done else "page", 1.0 if self._done else 0.0, self._done, {})

        def get_available_actions(self):
            return self._avail

    def test_done_step_blanks_admissible(self):
        env = self._FakeEnv(done=True, avail={"has_search_bar": True, "clickables": ["next-goal-item"]})
        obs, reward, done, info = step_env(_FakeEng(), env, "click[buy now]")
        assert done is True and reward == 1.0 and info["won"] is True
        assert info["available_actions"] == {"has_search_bar": False, "clickables": []}

    def test_nonterminal_step_keeps_admissible(self):
        env = self._FakeEnv(done=False, avail={"has_search_bar": False, "clickables": ["buy now"]})
        _, _, done, info = step_env(_FakeEng(), env, "click[item]")
        assert done is False and info["available_actions"]["clickables"] == ["buy now"]


class TestStepEnvReservedSearch:
    """Upstream's keyword dispatcher crashes on a query whose first token is a
    reserved marker; step_env must turn that into the ordinary no-op action
    (page unchanged, reward 0) and never raise."""

    class _RecordingEnv:
        session_prefix = "s:"

        def __init__(self):
            self.received: list[str] = []

        def step(self, action):
            self.received.append(action)
            return ("same page", 0.0, False, {})

        def get_available_actions(self):
            return {"has_search_bar": True, "clickables": ["b000000001"]}

    @pytest.mark.parametrize(
        "action",
        ["search[<a> red mug]", "search[<R>]", "search[<c> shoes]", "search[<q> some query]", "search[<a>  x]"],
    )
    def test_reserved_first_token_is_neutralized(self, action):
        env = self._RecordingEnv()
        obs, reward, done, info = step_env(_FakeEng(), env, action)
        assert env.received == ["invalid"]  # what the engine actually saw
        assert (obs, reward, done) == ("same page", 0.0, False)
        assert info["available_actions"]["has_search_bar"] is True and info["won"] is False

    @pytest.mark.parametrize(
        "action",
        [
            "search[red <a> mug]",  # marker not first
            "click[<a>]",  # not a search
            "search[ <a> x]",  # upstream splits on ' ': first token is '' -> plain Lucene search
            "Search[<a>]",  # upstream matches the action name case-sensitively
            "search[<abc>]",
            "search[shirt]",
        ],
    )
    def test_non_reserved_passes_through_unchanged(self, action):
        env = self._RecordingEnv()
        step_env(_FakeEng(), env, action)
        assert env.received == [action]

    def test_is_reserved_search_predicate(self):
        assert is_reserved_search("search[<a> x]") and is_reserved_search("search[<Q>]")
        assert not is_reserved_search("search[x <a>]") and not is_reserved_search("look")


# ─────────────────────────────────────────────────────────── HTTP layer
class _DictBackend(SessionBackend):
    def __init__(self):
        self.sessions: dict[str, int] = {}
        self.closed: list[str] = []  # every close() call, including failing ones
        self.resets: list[tuple] = []  # (task_id, goal_id) as received

    def reset(self, session_id, task_id, split, goal_id=None):
        self.resets.append((task_id, goal_id))
        if task_id == "bad":
            raise ValueError("goal index out of range")
        if task_id == "boom":
            raise RuntimeError("engine exploded")
        if goal_id == "unknown":
            raise UnknownGoal("goal_id 'unknown' is not in the engine goal list")
        self.sessions[session_id] = 0
        return ResetResult(obs=f"reset {goal_id or task_id} {split}", info={"goal_idx": task_id})

    def step(self, session_id, action):
        if session_id not in self.sessions:
            raise UnknownSession(session_id)
        self.sessions[session_id] += 1
        done = action == "buy"
        return StepResult(obs=f"step {action}", reward=1.0 if done else 0.0, done=done, info={})

    def close(self, session_id):
        self.closed.append(session_id)
        if session_id == "boom":
            raise RuntimeError("close exploded")
        self.sessions.pop(session_id, None)

    def live_sessions(self):
        return list(self.sessions)

    def worker_count(self):
        return 3

    def health(self):
        return {**super().health(), "goals": 42}


class TestHttpApp:
    def _client(self, backend=None, **kw):
        from fastapi.testclient import TestClient

        kw.setdefault("sweep_interval", 0)  # tests drive app.state.sweep_once directly
        backend = backend if backend is not None else _DictBackend()
        return TestClient(create_app(backend, session_ttl=1800.0, max_concurrency=4, **kw)), backend

    def test_reset_step_close_and_health(self):
        client, _ = self._client()
        r = client.post("/reset", json={"session_id": "s", "task_id": 7, "split": "eval"})
        assert r.status_code == 200 and r.json()["obs"] == "reset 7 eval"
        assert client.post("/step", json={"session_id": "s", "action": "look"}).json()["done"] is False
        buy = client.post("/step", json={"session_id": "s", "action": "buy"}).json()
        assert buy["done"] is True and buy["reward"] == 1.0
        assert client.get("/health").json() == {
            "status": "ok",
            "sessions": 1,
            "workers": 3,
            "dead_workers": 0,
            "respawns": 0,
            "goals": 42,
            "sweeper_alive": False,
        }
        assert client.post("/close", json={"session_id": "s"}).status_code == 200

    def test_unknown_session_returns_410(self):
        # The adapter treats 4xx as a fatal drop; 410 specifically means "your
        # episode is gone" (worker respawn / TTL evict), NOT a retriable error.
        client, _ = self._client()
        r = client.post("/step", json={"session_id": "ghost", "action": "look"})
        assert r.status_code == 410

    def test_value_error_on_reset_is_400(self):
        # Bad goal index / non-int task_id: the adapter must not retry it.
        client, backend = self._client()
        r = client.post("/reset", json={"session_id": "s", "task_id": "bad"})
        assert r.status_code == 400 and "out of range" in r.json()["detail"]
        assert backend.sessions == {} and client.get("/health").json()["sessions"] == 0

    def test_generic_backend_error_on_reset_is_500(self):
        client, _ = self._client()
        r = client.post("/reset", json={"session_id": "s", "task_id": "boom"})
        assert r.status_code == 500 and "engine exploded" in r.json()["detail"]

    def test_reset_passes_goal_id_only_when_present(self):
        client, backend = self._client()
        assert client.post("/reset", json={"session_id": "s", "task_id": 3}).status_code == 200
        r = client.post("/reset", json={"session_id": "t", "goal_id": "abc"})
        assert r.status_code == 200 and r.json()["obs"] == "reset abc train"
        assert backend.resets == [(3, None), (None, "abc")]

    def test_legacy_three_argument_backend_still_serves_legacy_resets(self):
        # Alfworld-style backends predating goal_id keep working as long as the
        # request carries no goal_id (the kwarg is only passed when present).
        class Legacy(_DictBackend):
            def reset(self, session_id, task_id, split):
                return super().reset(session_id, task_id, split)

        client, backend = self._client(Legacy())
        assert client.post("/reset", json={"session_id": "s", "task_id": 3}).status_code == 200
        assert backend.resets == [(3, None)]

    def test_unknown_goal_on_reset_is_400(self):
        client, backend = self._client()
        r = client.post("/reset", json={"session_id": "s", "goal_id": "unknown"})
        assert r.status_code == 400 and "is not in the engine goal list" in r.json()["detail"]
        assert backend.sessions == {} and client.get("/health").json()["sessions"] == 0

    def test_goals_is_501_without_a_goal_list(self):
        client, _ = self._client()
        r = client.get("/goals")
        assert r.status_code == 501 and "no goal list" in r.json()["detail"]

    def test_close_swallows_backend_errors(self):
        client, _ = self._client()
        r = client.post("/close", json={"session_id": "boom"})
        assert r.status_code == 200 and r.json() == {}

    def test_health_reports_live_sweeper(self):
        client, _ = self._client(sweep_interval=3600.0)
        assert client.get("/health").json()["sweeper_alive"] is True

    def test_sweep_once_evicts_only_stale_sessions(self):
        client, backend = self._client()
        state = client.app.state
        for sid in ("stale", "fresh"):
            assert client.post("/reset", json={"session_id": sid, "task_id": 1}).status_code == 200
        now = time.monotonic()
        state.last_seen["stale"] = now - 1800.0 - 1.0
        assert state.sweep_once(now) == ["stale"]
        assert backend.live_sessions() == ["fresh"]
        assert client.post("/step", json={"session_id": "stale", "action": "look"}).status_code == 410
        assert client.post("/step", json={"session_id": "fresh", "action": "look"}).status_code == 200
        assert state.sweep_once(now) == []  # idempotent; the touched session stays

    def test_sweep_once_survives_backend_close_error(self):
        client, _ = self._client()
        state = client.app.state
        assert client.post("/reset", json={"session_id": "boom", "task_id": 1}).status_code == 200
        now = time.monotonic()
        state.last_seen["boom"] = now - 1800.0 - 1.0
        assert state.sweep_once(now) == ["boom"]  # logged, not raised; TTL entry dropped
        assert "boom" not in state.last_seen

    def test_terminal_step_forgets_ttl_entry(self):
        # Backends drop a done episode themselves and the adapter sends no
        # /close after a terminal step, so the id must leave the TTL table at
        # once; otherwise the sweeper logs a spurious eviction (and closes a
        # ghost) session_ttl later, indistinguishable from a real leak.
        client, backend = self._client()
        state = client.app.state
        assert client.post("/reset", json={"session_id": "s", "task_id": 1}).status_code == 200
        assert client.post("/step", json={"session_id": "s", "action": "look"}).status_code == 200
        assert "s" in state.last_seen
        assert client.post("/step", json={"session_id": "s", "action": "buy"}).json()["done"] is True
        assert "s" not in state.last_seen
        assert state.sweep_once(time.monotonic() + 1800.0 + 1.0) == []
        assert backend.closed == []

    def test_410_forgets_ttl_entry(self):
        # Worker respawn / engine-side loss: the backend already lost the
        # episode, so the 410 must also drop the marker left by earlier steps.
        client, backend = self._client()
        state = client.app.state
        assert client.post("/reset", json={"session_id": "s", "task_id": 1}).status_code == 200
        assert client.post("/step", json={"session_id": "s", "action": "look"}).status_code == 200
        backend.sessions.pop("s")  # the engine lost the episode behind our back
        assert client.post("/step", json={"session_id": "s", "action": "look"}).status_code == 410
        assert "s" not in state.last_seen
        assert state.sweep_once(time.monotonic() + 1800.0 + 1.0) == []
        assert backend.closed == []


# ──────────────────────────────── HTTP over the fake engine + export CLI
def _fake_engine_app(monkeypatch):
    """The engine server as launch_engine.sh runs it (protocol block on /health),
    over the shared backend with the fake engine."""
    backend = _shared(monkeypatch)
    app = create_app(
        backend, session_ttl=1800.0, max_concurrency=4, sweep_interval=0, extra_health={"protocol": _cfg().protocol()}
    )
    return app, backend


@contextlib.contextmanager
def _serve_app(app):
    """Run ``app`` on a real loopback port in a daemon thread: export_goals.py
    speaks stdlib urllib, which TestClient cannot serve."""
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="test-uvicorn", daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(15.0)
        sock.close()


class TestHttpOverFakeEngine:
    def _client(self, monkeypatch):
        from fastapi.testclient import TestClient

        app, backend = _fake_engine_app(monkeypatch)
        return TestClient(app), backend

    def test_reset_by_goal_id_matches_index_and_unknown_is_400(self, monkeypatch):
        client, backend = self._client(monkeypatch)
        by_idx = client.post("/reset", json={"session_id": "a", "task_id": 5, "split": "train"}).json()
        by_id = client.post("/reset", json={"session_id": "b", "goal_id": FAKE_IDS[5], "split": "train"}).json()
        assert by_id == by_idx and by_id["info"]["goal_idx"] == 5
        r = client.post("/reset", json={"session_id": "c", "goal_id": UNKNOWN_ID})
        assert r.status_code == 400
        assert r.json()["detail"] == (
            f"goal_id {UNKNOWN_ID!r} is not in the engine goal list (goals_digest {FAKE_DIGEST}): "
            "the dataset and the engine disagree on catalog / goal code"
        )
        assert sorted(backend.live_sessions()) == ["a", "b"] and client.get("/health").json()["sessions"] == 2

    def test_goals_export_matches_health_digest(self, monkeypatch):
        client, _ = self._client(monkeypatch)
        health = client.get("/health").json()
        assert _HEX16.match(health["goals_digest"]) and health["goals"] == N_GOALS
        assert health["protocol"] == _cfg().protocol()  # the launch-time protocol block is still there
        rows = client.get("/goals").json()
        assert len(rows) == N_GOALS and [r["goal_idx"] for r in rows] == list(range(N_GOALS))
        assert all(goal_id(r) == r["goal_id"] for r in rows)
        assert goals_set_digest(r["goal_id"] for r in rows) == health["goals_digest"] == FAKE_DIGEST


class TestExportGoalsCli:
    def test_exports_jsonl_and_sidecar(self, monkeypatch, tmp_path, capsys):
        import export_goals

        app, _ = _fake_engine_app(monkeypatch)
        out = tmp_path / "data" / "webshop_goals.jsonl"
        with _serve_app(app) as url:
            export_goals.main(["--engine-url", url, "--output", str(out)])
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == N_GOALS and [r["goal_idx"] for r in rows] == list(range(N_GOALS))
        assert all(list(r) == list(export_goals.ROW_FIELDS) for r in rows)  # stable key order
        assert [r["goal_id"] for r in rows] == FAKE_IDS and all(goal_id(r) == r["goal_id"] for r in rows)
        assert rows[3]["instruction_text"] == "fake goal 3" and rows[3]["goal_options"] == {"size": "1"}
        meta = json.loads((tmp_path / "data" / "webshop_goals.jsonl.meta.json").read_text(encoding="utf-8"))
        assert meta["goals"] == N_GOALS and meta["goals_digest"] == FAKE_DIGEST
        assert meta["protocol"] == _cfg().protocol() and meta["engine_url"] == url
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", meta["exported_at"])
        assert f"exported {N_GOALS} goals (goals_digest {FAKE_DIGEST}" in capsys.readouterr().out

    def test_refuses_to_write_on_digest_mismatch(self, monkeypatch, tmp_path):
        # The engine changed under us (or /health lies): nothing may be written.
        import export_goals

        real_fetch = export_goals.fetch_json

        def skewed_fetch(url, timeout):
            payload = real_fetch(url, timeout)
            if url.endswith("/health"):
                payload["goals_digest"] = "0" * 16
            return payload

        monkeypatch.setattr(export_goals, "fetch_json", skewed_fetch)
        app, _ = _fake_engine_app(monkeypatch)
        out = tmp_path / "webshop_goals.jsonl"
        with _serve_app(app) as url, pytest.raises(SystemExit, match="refusing to write.*goals_digest"):
            export_goals.main(["--engine-url", url, "--output", str(out)])
        assert not out.exists() and not Path(f"{out}.meta.json").exists()

    def test_refuses_a_server_without_the_protocol(self, monkeypatch, tmp_path):
        # A /health without goals_digest is an old engine (or not one at all).
        import export_goals

        app = create_app(_DictBackend(), session_ttl=1800.0, max_concurrency=4, sweep_interval=0)
        out = tmp_path / "webshop_goals.jsonl"
        with _serve_app(app) as url, pytest.raises(SystemExit, match="carries no 'protocol'"):
            export_goals.main(["--engine-url", url, "--output", str(out)])
        assert not out.exists()


def _defaults_sh_values() -> dict[str, str]:
    """Source env_servers/webshop/defaults.sh in a clean environment (no MILES_AGENT_DATA, no
    WEBSHOP_*) and return the knobs server.py mirrors."""
    keys = (
        "WEBSHOP_SERVER_HOST",
        "WEBSHOP_SERVER_PORT",
        "WEBSHOP_ENGINE_WORKERS",
        "WEBSHOP_ENGINE_CALL_TIMEOUT",
        "WEBSHOP_ENGINE_SESSION_TTL",
        "WEBSHOP_VENDOR",
        "WEBSHOP_DATA_DIR",
        "WEBSHOP_GOAL_SEED",
        "WEBSHOP_OBS_MODE",
        "WEBSHOP_HUMAN_GOALS",
        "WEBSHOP_PROTOCOL",
    )
    script = f". '{Path(__file__).resolve().parents[1] / 'defaults.sh'}'; printf '%s\\n' " + " ".join(
        f'"${k}"' for k in keys
    )
    out = subprocess.run(["/bin/bash", "-c", script], env={}, check=True, capture_output=True, text=True).stdout
    return dict(zip(keys, out.splitlines(), strict=True))


def test_arg_parser_defaults_match_defaults_sh(monkeypatch):
    # Python-side defaults exist for manual runs and must equal defaults.sh
    # (the single source of truth) with WEBSHOP_HOME unset and no MILES_AGENT_DATA: a manual
    # `python server.py` must find the catalog setup.sh built, not the bare
    # committed tree. Cross-checked against the shell file itself so drift on
    # either side fails here.
    import server as server_module

    for var in ("WEBSHOP_VENDOR", "WEBSHOP_DATA_DIR", "MILES_AGENT_DATA"):
        monkeypatch.delenv(var, raising=False)
    args = server_module.build_arg_parser().parse_args([])
    assert (args.host, args.port) == ("127.0.0.1", 6200)
    assert (args.session_ttl, args.max_concurrency, args.sweep_interval) == (2400.0, 256, 60.0)
    assert (args.engine_workers, args.engine_call_timeout) == (8, 120.0)
    assert (args.vendored_path, args.data_dir) == ("/data/webshop/webshop", "/data/webshop/webshop/data")
    assert not args.full and not args.human_goals
    # TCOD protocol defaults: the original WebShop shuffle constant + text_rich pages.
    assert (args.goal_seed, args.observation_mode) == (233, "text_rich")
    assert base_arg_parser("x").parse_args([]).port == 6200  # the shared parser alone agrees too

    sh = _defaults_sh_values()
    assert args.goal_seed == int(sh["WEBSHOP_GOAL_SEED"]) and args.observation_mode == sh["WEBSHOP_OBS_MODE"]
    assert sh["WEBSHOP_HUMAN_GOALS"] == "0" and sh["WEBSHOP_PROTOCOL"] == "canonical"
    assert args.host == sh["WEBSHOP_SERVER_HOST"] and args.port == int(sh["WEBSHOP_SERVER_PORT"])
    assert args.engine_workers == int(sh["WEBSHOP_ENGINE_WORKERS"])
    assert args.engine_call_timeout == float(sh["WEBSHOP_ENGINE_CALL_TIMEOUT"])
    assert args.session_ttl == float(sh["WEBSHOP_ENGINE_SESSION_TTL"])
    assert (args.vendored_path, args.data_dir) == (sh["WEBSHOP_VENDOR"], sh["WEBSHOP_DATA_DIR"])

    # Manual runs may still point at another built runtime via the environment.
    monkeypatch.setenv("WEBSHOP_VENDOR", "/elsewhere/webshop")
    monkeypatch.setenv("WEBSHOP_DATA_DIR", "/elsewhere/webshop/data")
    args = server_module.build_arg_parser().parse_args([])
    assert (args.vendored_path, args.data_dir) == ("/elsewhere/webshop", "/elsewhere/webshop/data")


# ─────────────────────────────────────────────────── real vendored stack (gated)
def _real_cfg() -> EngineConfig | None:
    vendor = os.environ.get("WEBSHOP_VENDOR")
    data = os.environ.get("WEBSHOP_DATA_DIR")
    if vendor and data and (Path(vendor) / "web_agent_site").is_dir() and Path(data).is_dir():
        return EngineConfig(vendored_path=vendor, data_dir=data, small=True)
    return None


@pytest.mark.skipif(_real_cfg() is None, reason="$WEBSHOP_VENDOR + $WEBSHOP_DATA_DIR (built catalog) not available")
class TestRealEngine:
    def test_shared_backend_search_click_and_buy(self):
        """Real SimServer: reset goal 0, search, open the first product, buy.
        Pins the terminal contract: done, raw task_score in [0, 1], won iff a
        perfect purchase, blank admissible set, and no engine session left
        under this episode's prefix. Runs only where the catalog is built."""
        import server as server_module

        backend = server_module.WebshopSharedBackend(_real_cfg())
        r = backend.reset("real-1", 0, "eval")
        assert "Instruction:" in r.obs
        results = backend.step("real-1", "search[shirt]")
        assert isinstance(results.obs, str) and results.done is False
        clickables = results.info["available_actions"]["clickables"]
        asin = next(c for c in clickables if re.fullmatch(r"[A-Za-z0-9]{10}", c))  # product codes
        item = backend.step("real-1", f"click[{asin}]")
        assert any(c.lower() == "buy now" for c in item.info["available_actions"]["clickables"])
        buy = backend.step("real-1", "click[buy now]")
        assert buy.done is True
        assert 0.0 <= buy.info["task_score"] <= 1.0 and buy.reward == buy.info["task_score"]
        assert buy.info["won"] == (buy.info["task_score"] == 1.0)
        assert buy.info["available_actions"] == {"has_search_bar": False, "clickables": []}
        assert not any(k.startswith("real-1:") for k in backend._engine.user_sessions)
        assert backend.live_sessions() == []

    def test_shared_backend_reserved_search_is_a_noop(self):
        import server as server_module

        backend = server_module.WebshopSharedBackend(_real_cfg())
        r = backend.reset("real-2", 0, "eval")
        step = backend.step("real-2", "search[<a> shirt]")
        assert step.done is False and step.reward == 0.0
        assert step.info["available_actions"]["has_search_bar"] is True  # still the search page
        assert r.obs.split(" [SEP] ")[-1] in step.obs or "Instruction:" in step.obs
        backend.close("real-2")

    def test_goals_export_is_the_full_unique_goal_set(self):
        """/goals on the real small-catalog engine: 6910 rows = 6901 distinct
        questions (9 rows repeat an earlier row's content: a product listing an
        option value twice), the set digest recorded in
        the exporter's webshop_goals.meta.json sidecar, and a goal_id reset lands
        on the same goal as the legacy list position (a twin's id on the FIRST
        position)."""
        import server as server_module
        from fastapi.testclient import TestClient

        backend = server_module.WebshopSharedBackend(_real_cfg())
        client = TestClient(create_app(backend, session_ttl=1800.0, max_concurrency=4, sweep_interval=0))
        rows = client.get("/goals").json()
        ids = [r["goal_id"] for r in rows]
        health = client.get("/health").json()
        assert len(rows) == 6910 == health["goals"]
        assert len(set(ids)) == 6901 == health["unique_goals"]
        assert [r["goal_idx"] for r in rows] == list(range(6910))
        assert all(goal_id(r) == r["goal_id"] for r in rows)
        assert health["goals_digest"] == goals_set_digest(ids) == "455c9eaa96191605"
        assert ids[6766] == ids[98]  # the first twin pair of the export
        assert backend.reset("real-twin", None, "eval", goal_id=ids[6766]).info["goal_idx"] == 98
        backend.close("real-twin")
        by_id = backend.reset("real-3", None, "eval", goal_id=ids[5])
        by_idx = backend.reset("real-4", 5, "eval")
        assert by_id.info["goal_idx"] == by_idx.info["goal_idx"] == 5 and by_id.obs == by_idx.obs
        backend.close("real-3")
        backend.close("real-4")


# ── review follow-ups: twins, refusal paths, live-episode safety ─────────


def test_goal_table_collapses_repeated_content(caplog):
    # WebShop's synthetic generator emits the same goal at several positions;
    # the table keeps one id per question, serves it from the FIRST position,
    # exports every row (the file is the faithful engine list) and hashes the SET.
    from _webshop_engine import GoalTable

    with caplog.at_level("WARNING", logger="webshop_engine"):
        table = GoalTable([fake.fake_goal(0), fake.fake_goal(1), fake.fake_goal(1)])
    assert len(table) == 3 and table.unique == 2
    assert table.ids[1] == table.ids[2] and table.resolve(table.ids[1]) == 1
    assert table.export()[2] == {"goal_id": table.ids[1], "goal_idx": 2, **table.export()[1]} | {"goal_idx": 2}
    assert table.digest == goals_set_digest(table.ids[:2]) == goals_set_digest(table.ids)
    assert "1 of 3 goals duplicate another goal's content" in caplog.text


class TestUnknownGoalKeepsLiveEpisode:
    """A re-reset with an unknown goal_id is refused BEFORE the caller's live
    episode under that session_id is touched (like a bad legacy index)."""

    def test_pool(self):
        pool = WebshopEngineWorkerPool(_cfg(), 1, 30.0)
        try:
            pool.reset("s", 1, "train")
            assert pool.step("s", "look").obs == "echo:look:1"
            with pytest.raises(UnknownGoal, match=f"goals_digest {FAKE_DIGEST}"):
                pool.reset("s", None, "train", goal_id=UNKNOWN_ID)
            assert pool.live_sessions() == ["s"] and pool._counts == [1] and pool.respawns == 0
            assert pool.step("s", "look").obs == "echo:look:2"
        finally:
            pool.shutdown()

    def test_shared_backend(self, monkeypatch):
        be = _shared(monkeypatch)
        be.reset("s", 1, "train")
        assert be.step("s", "look").obs == "echo:look:1"
        with pytest.raises(UnknownGoal, match=f"goals_digest {FAKE_DIGEST}"):
            be.reset("s", None, "train", goal_id=UNKNOWN_ID)
        assert be.live_sessions() == ["s"] and be.step("s", "look").obs == "echo:look:2"


class TestExportGoalsRefusals:
    """Every integrity check of export_goals.export() refuses to write."""

    @staticmethod
    def _rows_and_health():
        from agent_utils.webshop_goals import goal_record

        rows = [{"goal_id": FAKE_IDS[i], "goal_idx": i, **goal_record(fake.fake_goal(i))} for i in range(N_GOALS)]
        health = {"protocol": _cfg().protocol(), "goals_digest": FAKE_DIGEST, "goals": N_GOALS}
        return rows, health

    @pytest.mark.parametrize(
        "skew, match",
        [
            (
                lambda rows, health: rows[3].__setitem__("price_upper", 999.0),
                r"goal_idx 3 carries goal_id .* but its content",
            ),
            (lambda rows, health: rows.pop(3), "not the contiguous range"),
            (lambda rows, health: health.__setitem__("goals", N_GOALS + 1), "/health reports goals="),
        ],
        ids=["tampered-row", "gap-in-goal_idx", "health-count"],
    )
    def test_refuses(self, monkeypatch, tmp_path, skew, match):
        import export_goals

        rows, health = self._rows_and_health()
        skew(rows, health)
        fetch = lambda url, timeout: health if url.endswith("/health") else rows  # noqa: E731
        monkeypatch.setattr(export_goals, "fetch_json", fetch)
        out = tmp_path / "webshop_goals.jsonl"
        with pytest.raises(SystemExit, match=match):
            export_goals.export("http://engine", out)
        assert not out.exists() and not Path(f"{out}.meta.json").exists()

    def test_refuses_a_protocol_server_without_a_digest(self, monkeypatch, tmp_path):
        # The realistic "engine predates content-addressed goals" case: /health
        # carries the protocol block but no goals_digest.
        import export_goals

        extra = {"protocol": _cfg().protocol()}
        app = create_app(_DictBackend(), session_ttl=1800.0, max_concurrency=4, sweep_interval=0, extra_health=extra)
        out = tmp_path / "webshop_goals.jsonl"
        with _serve_app(app) as url, pytest.raises(SystemExit, match="carries no 'goals_digest'"):
            export_goals.main(["--engine-url", url, "--output", str(out)])
        assert not out.exists()
