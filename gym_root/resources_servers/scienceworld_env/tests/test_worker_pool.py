"""Process backend tests (_worker.WorkerPool + server with env_workers > 0).

These spawn REAL worker processes; env behavior is scripted through the
env_factory config seam (fake_sciworld) because monkeypatching cannot cross
the spawn boundary. TestRealJVM at the bottom is gated on a working
scienceworld + java install.
"""

import asyncio
import os
import shutil
import time
from unittest.mock import MagicMock

import fake_sciworld
import pytest
from fastapi import HTTPException
from scienceworld_env._worker import EnvWorkerDied, EnvWorkerError, WorkerPool
from scienceworld_env.app import ScienceworldResourcesServer, ScienceworldResourcesServerConfig
from test_app import ROW, _act_response

FAKE_FACTORY = "fake_sciworld:make_env"
SID = "session-1"


def _make_server(**overrides) -> ScienceworldResourcesServer:
    overrides.setdefault("sweep_interval_s", 0.0)
    overrides.setdefault("env_workers", 2)
    overrides.setdefault("env_call_timeout_s", 30.0)
    overrides.setdefault("env_factory", FAKE_FACTORY)
    overrides.setdefault("slot_wait_timeout_s", 2.0)
    config = ScienceworldResourcesServerConfig(
        host="0.0.0.0", port=8100, entrypoint="", name="scienceworld_env", **overrides
    )
    from nemo_gym.server_utils import ServerClient

    return ScienceworldResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _pool(n=1, timeout=30.0, **kw) -> WorkerPool:
    return WorkerPool(n, timeout, FAKE_FACTORY, **kw)


# ──────────────────────────────────────────────────────────
# WorkerPool directly (no server, no fastapi)
# ──────────────────────────────────────────────────────────


class TestWorkerPool:
    async def test_load_step_release_roundtrip(self):
        pool = _pool()
        idx, gen = await pool.acquire(1.0)
        ep, obs, info = await pool.load(idx, gen, "boil", 0)
        assert obs == fake_sciworld.INITIAL_OBS
        assert info["task_desc"] == "Your task is to do boil variation 0."
        assert info["templates"] == fake_sciworld.TEMPLATES and info["objects"] == fake_sciworld.OBJECTS
        obs, sinfo = await ep.step("half")
        assert (sinfo["best_score"], sinfo["best_delta"]) == (50.0, 50.0)
        assert (sinfo["completed"], sinfo["done"]) == (False, False)
        obs, sinfo = await ep.step("win")
        assert (sinfo["best_delta"], sinfo["completed"], sinfo["done"]) == (50.0, True, True)
        assert pool.free_slots == 0
        await ep.close()
        assert pool.free_slots == 1
        await ep.close()  # idempotent

    async def test_jvm_is_reused_across_episodes(self):
        pool = _pool()
        idx, gen = await pool.acquire(1.0)
        ep, _, _ = await pool.load(idx, gen, "boil", 0)
        await ep.step("half")
        await ep.close()
        idx, gen = await pool.acquire(1.0)
        ep2, _, info = await pool.load(idx, gen, "melt", 1)
        assert info["task_desc"] == "Your task is to do melt variation 1."
        _, sinfo = await ep2.step("look around")
        assert sinfo["best_score"] == 0.0  # fresh episode, not the previous score

    async def test_admission_waits_then_times_out(self):
        pool = _pool(1)
        idx, gen = await pool.acquire(1.0)
        ep, _, _ = await pool.load(idx, gen, "boil", 0)
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="no free ScienceWorld worker"):
            await pool.acquire(0.5)
        assert 0.4 < time.monotonic() - t0 < 5.0
        waiter = asyncio.create_task(pool.acquire(5.0))
        await asyncio.sleep(0.1)
        await ep.close()
        idx2, _ = await waiter  # released slot wakes the waiter
        assert idx2 == idx

    async def test_factory_error_is_op_error_worker_survives_slot_returns(self):
        pool = _pool()
        idx, gen = await pool.acquire(1.0)
        with pytest.raises(EnvWorkerError, match="scripted load failure"):
            await pool.load(idx, gen, "boil", fake_sciworld.BOOM_VAR)
        assert pool.respawns == 0 and pool.free_slots == 1
        idx, gen = await pool.acquire(1.0)
        ep, _, _ = await pool.load(idx, gen, "boil", 0)  # same worker, still healthy
        assert (await ep.step("look around"))[0] == "echo:look around:1"

    async def test_step_error_is_op_error(self):
        pool = _pool()
        idx, gen = await pool.acquire(1.0)
        ep, _, _ = await pool.load(idx, gen, "boil", 0)
        with pytest.raises(EnvWorkerError, match="scripted step failure"):
            await ep.step("explode")
        await ep.close()
        assert pool.free_slots == 1

    async def test_worker_death_respawns_and_invalidates_stale_handle(self):
        pool = _pool(2)
        idx, gen = await pool.acquire(1.0)
        survivor, _, _ = await pool.load(idx, gen, "boil", 0)
        idx2, gen2 = await pool.acquire(1.0)
        with pytest.raises(EnvWorkerDied):
            await pool.load(idx2, gen2, "boil", fake_sciworld.DIE_VAR)
        assert pool.respawns == 1 and pool.free_slots == 1
        assert (await survivor.step("look around"))[0] == "echo:look around:1"  # other worker untouched
        pool._slots._procs[survivor.worker_idx].kill()
        with pytest.raises(EnvWorkerDied):
            await survivor.step("look around")
        await survivor.close()  # stale gen: no wire op, slot still returns
        assert pool.free_slots == 2 and pool.respawns == 2

    async def test_wedged_worker_times_out_and_respawns(self):
        pool = _pool(1, 0.5)
        idx, gen = await pool.acquire(1.0)
        with pytest.raises(EnvWorkerDied, match="exceeded 0.5s"):
            await pool.load(idx, gen, "boil", fake_sciworld.HANG_VAR)
        assert pool.respawns == 1 and pool.free_slots == 1
        idx, gen = await pool.acquire(1.0)
        ep, _, _ = await pool.load(idx, gen, "boil", 0)
        assert (await ep.step("look around"))[0] == "echo:look around:1"

    async def test_wedged_worker_does_not_block_other_workers(self):
        pool = _pool(2, 5.0)
        idx, gen = await pool.acquire(1.0)
        hang_task = asyncio.create_task(pool.load(idx, gen, "boil", fake_sciworld.HANG_VAR))
        await asyncio.sleep(0.1)
        t0 = time.monotonic()
        idx2, gen2 = await pool.acquire(1.0)
        ep, _, _ = await pool.load(idx2, gen2, "boil", 0)
        for _ in range(3):
            await ep.step("look around")
        assert time.monotonic() - t0 < 3.0 and idx2 != idx
        with pytest.raises(EnvWorkerDied):
            await hang_task
        assert pool.respawns == 1


# ──────────────────────────────────────────────────────────
# Server with env_workers > 0 (fake factory in real subprocesses)
# ──────────────────────────────────────────────────────────


class TestServerProcessBackend:
    async def test_reset_step_win_through_workers(self):
        server = _make_server()
        obs, info = await server.reset(dict(ROW), SID)
        assert obs.startswith("Your ScienceWorld task is: Your task is to do boil variation 3.")
        assert info == {"task_name": "boil", "var_num": "3", "env_worker_respawns": 0, "env_free_workers": 1}
        obs, reward, terminated, truncated, sinfo = await server.step(_act_response("half"), {}, SID)
        assert obs.startswith("You focus on the thermometer.")
        assert (reward, terminated, truncated, sinfo["task_score"]) == (0.5, False, False, 0.5)
        obs, reward, terminated, truncated, sinfo = await server.step(_act_response("win"), {}, SID)
        assert (reward, terminated, truncated, sinfo["won"]) == (0.5, True, False, True)
        assert obs.endswith("Task completed.")
        await server.close_session(SID)
        assert server._procpool.free_slots == 2

    async def test_all_workers_busy_is_503(self):
        server = _make_server(env_workers=1, slot_wait_timeout_s=0.3)
        await server.reset(dict(ROW), "a")
        with pytest.raises(HTTPException) as exc:
            await server.reset(dict(ROW), "b")
        assert exc.value.status_code == 503
        await server.close_session("a")
        await server.reset(dict(ROW), "b")  # slot freed by the close

    async def test_load_on_dead_slot_retries_after_respawn(self):
        server = _make_server(env_workers=1)
        await server.reset(dict(ROW), "warm")
        await server.close_session("warm")
        # Kill the only worker; the load on it dies (respawn) and the retry
        # re-acquires the respawned slot instead of failing the /reset.
        server._procpool._slots._procs[0].kill()
        obs, info = await server.reset(dict(ROW), SID)
        assert info["env_worker_respawns"] == 1
        assert obs.startswith("Your ScienceWorld task is")
        assert server._procpool.free_slots == 0

    async def test_session_on_dead_worker_500s_on_step_and_frees_slot(self):
        server = _make_server(env_workers=1)
        await server.reset(dict(ROW), SID)
        server._procpool._slots._procs[0].kill()
        with pytest.raises(EnvWorkerDied):
            await server.step(_act_response("look around"), {}, SID)
        assert SID not in server.session_state
        assert server._procpool.free_slots == 1
        await server.reset(dict(ROW), SID)  # respawned worker serves again

    async def test_step_failure_frees_slot(self):
        server = _make_server(env_workers=1)
        await server.reset(dict(ROW), SID)
        with pytest.raises(EnvWorkerError):
            await server.step(_act_response("explode"), {}, SID)
        assert server._procpool.free_slots == 1


# ──────────────────────────────────────────────────────────
# Real simulator (gated): scienceworld + py4j installed and java on PATH
# ──────────────────────────────────────────────────────────


def _real_available() -> bool:
    if os.environ.get("SCIENCEWORLD_REAL_TESTS", "") != "1":
        return False
    try:
        import scienceworld  # noqa: F401
    except ImportError:
        return False
    return shutil.which("java") is not None


@pytest.mark.skipif(not _real_available(), reason="set SCIENCEWORLD_REAL_TESTS=1 with scienceworld + java installed")
class TestRealJVM:
    async def test_boil_episode_roundtrip(self):
        server = _make_server(
            env_workers=1,
            env_factory="scienceworld_env._worker:make_env",
            env_call_timeout_s=120.0,
            jar_path=os.environ.get("SCIENCEWORLD_JAR", ""),  # "" = the wheel's bundled jar
        )
        obs, info = await server.reset({"task_name": "boil", "var_num": 0, "split": "train"}, SID)
        assert obs.startswith("Your ScienceWorld task is: Your task is to boil ")
        assert "Available action commands: ['" in obs and "'help'" not in obs
        obs, reward, terminated, truncated, sinfo = await server.step(_act_response("look around"), {}, SID)
        assert "This room is called" in obs and terminated is False
        obs, *_, sinfo = await server.step(_act_response("frobnicate the widget"), {}, SID)
        assert sinfo["invalid_actions"] == 1
        await server.close_session(SID)
        # the JVM stays warm: a second episode on the same worker is fast
        t0 = time.monotonic()
        await server.reset({"task_name": "melt", "var_num": 0, "split": "train"}, SID)
        assert time.monotonic() - t0 < 5.0
