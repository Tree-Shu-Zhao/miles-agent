"""Engine-worker process pool for the WebShop engine server (the concurrency
lever; see docs/webshop_integration.md).

Why processes: the vendored WebShop stack is single-threaded by construction
(global Flask app contexts, module-level ``random`` reseeded per env, one
shared ``user_sessions`` dict), so ONE engine can only be touched serially and
a single global lock caps throughput under an RL wave. Each worker process
owns its OWN full ``SimServer`` engine and hosts MANY lightweight per-episode
envs; K workers run truly in parallel (separate interpreters, separate JVMs,
no shared lock), giving K x the Lucene-search throughput. Sync facade (the
engine server dispatches backend calls on a thread pool), many envs per worker
(the engine is the only heavy resource, and it is shared within a worker).

Built on ``agent_utils.procslots.ProcSlots``, shared with the alfworld env
worker pool: a dedicated single-thread executor per worker serializes its pipe
(FIFO; a wedged worker blocks only the callers pinned to it), a generation
counter per worker invalidates stale pins after a respawn, and a worker whose
replacement fails to boot is parked (``/health`` ``dead_workers``) and retried
by procslots' revive thread every 60 s. This module adds only what is
WebShop-specific: the eager engine build in the child (ready payload = goal
count + goal-set digest, surfaced on ``/health`` for the inner script's digest
gate), the session_id -> (worker, gen, episode) pin table, the
``UnknownSession`` (HTTP 410) mapping, and the ``UnknownGoal`` (HTTP 400)
round trip: a goal_id is resolved INSIDE the worker that owns the goal list,
so its ValueError comes back over the pipe as an op error and is rebuilt here
(the one op error that is the client's fault, not the engine's).

Failure model: an op exceeding call_timeout_s, or a dead pipe, SIGKILLs and
respawns the worker. Every episode resident in that worker is gone; its pins
are dropped in the same critical section as the gen bump, so their next op
raises UnknownSession (the HTTP layer returns 410, the adapter drops the
trajectory as a transport failure) instead of touching a stranger's env after
the respawn. Boot handshake: each child builds its engine BEFORE sending
ready, so a data/JVM breakage fails pool construction loudly with the real
cause instead of a silent per-op respawn storm.

Same-session_id races: two concurrent /reset for one session_id used to leak
the loser's episode until its worker respawned. Registration is now atomic
under the slot lock and the displaced episode is closed, so exactly one live
episode remains; independently, the adapter regenerates session_id per
attempt (gym_root/resources_servers/webshop_env/_engine_client.py), so its
retries never collide in the first place.
"""

import importlib
import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from _webshop_engine import EngineConfig, GoalTable, UnknownGoal, goal_index
from session_server import ResetResult, SessionBackend, StepResult, UnknownSession

try:
    from agent_utils.procslots import UNKNOWN, ProcSlots, SlotDied, SlotOpError, serve_loop
except ImportError:
    # The py3.10 engine env cannot pip-install gym_root/agent_utils (it requires
    # py>=3.12 + nemo-gym); procslots itself is stdlib-only, so reach it by path:
    # env_servers/webshop/ and gym_root/ are siblings under the checkout root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gym_root"))
    from agent_utils.procslots import UNKNOWN, ProcSlots, SlotDied, SlotOpError, serve_loop

logger = logging.getLogger("webshop_engine.pool")

# Per-worker ready-handshake budget, measured from call time while all workers
# build in PARALLEL. This — not the inner-script health gate — is the effective
# boot ceiling. 300s is generous for the default --small catalog (~tens of s per
# engine); a --full-catalog build with large K may need it raised. A build that
# *raises* (missing data/JVM) reports on the ready channel immediately and does
# not consume this budget; the timeout only fires on a genuine hang.
SPAWN_READY_TIMEOUT_S = 300.0
# worker->parent marker: the episode was closed/evicted between the parent
# pinning it and this op, so step must 410 (UnknownSession), not 500.
_UNKNOWN_EPISODE = UNKNOWN
# serve_loop flattens a child-side exception to "ExcType: msg"; this prefix
# identifies the UnknownGoal raised by the worker's goal_id lookup.
_UNKNOWN_GOAL_PREFIX = f"{UnknownGoal.__name__}: "


class _EngineHost:
    """Child side: one engine plus this worker's resident episodes (episode_id -> env).
    The factory module is resolved here, inside the child, so tests can swap in
    a fake stack by name (monkeypatching cannot cross the spawn boundary)."""

    def __init__(self, cfg: EngineConfig):
        self._cfg = cfg
        self._envs: dict[str, Any] = {}

    def build(self) -> dict:
        factory = importlib.import_module(self._cfg.factory_module)
        self._engine, self._env_cls, self._file_paths = factory.build_engine(self._cfg)
        self._make_env = factory.make_env
        self._step_env = factory.step_env
        self._drop = factory.drop_engine_state
        # Content identity of this worker's goal list; every worker builds the
        # same list from the same catalog + seed, which the pool verifies via
        # the digest in this ready payload.
        self._goals = GoalTable(self._engine.goals)
        return {
            "goals": len(self._goals),
            "unique_goals": self._goals.unique,
            "goals_digest": self._goals.digest,
            # The parent pre-validates goal_ids against this set so an unknown id
            # is refused before any episode is touched (110 KB per worker).
            "goal_ids": self._goals.ids,
        }

    def reset(self, episode_id: str, goal_ref: Any, prefix: str):
        # goal_ref: str goal_id (content address, resolved here because the goal
        # list lives in this process) or a legacy int position. UnknownGoal /
        # ValueError travel back to the parent as an op error (-> HTTP 400).
        goal_idx = self._goals.resolve(goal_ref)
        old = self._envs.pop(episode_id, None)
        if old is not None:
            self._drop(self._engine, getattr(old, "session_prefix", None))
        env, obs, info = self._make_env(
            self._engine,
            self._env_cls,
            self._file_paths,
            goal_idx,
            prefix,
            observation_mode=self._cfg.observation_mode,
        )
        self._envs[episode_id] = env
        return (obs, info)

    def step(self, episode_id: str, action: str):
        env = self._envs.get(episode_id)
        if env is None:
            # Closed/evicted (concurrent /close or TTL sweep) between the parent
            # pinning this session and the op reaching us. Tell the parent to
            # 410 rather than raising KeyError (which would 500).
            return serve_loop.UNKNOWN
        obs, reward, done, info = self._step_env(self._engine, env, action)
        if done:
            self._envs.pop(episode_id, None)
        return (obs, reward, done, info)

    def close(self, episode_id: str) -> None:
        env = self._envs.pop(episode_id, None)
        if env is not None:
            self._drop(self._engine, getattr(env, "session_prefix", None))
        return None

    def export_goals(self) -> list[dict]:
        return self._goals.export()


def engine_worker_main(conn, cfg: EngineConfig) -> None:
    """Body of one engine-hosting worker process (spawn target): build the
    engine eagerly, then serve reset/step/close/export_goals until EOF or
    shutdown."""
    host = _EngineHost(cfg)
    handlers = {"reset": host.reset, "step": host.step, "close": host.close, "export_goals": host.export_goals}
    serve_loop(conn, handlers, build=host.build)


class EngineWorkerError(SlotOpError):
    """The op failed inside the worker; the worker itself is still healthy."""


class EngineWorkerDied(SlotDied):
    """The worker was killed (op timeout / crash); its resident envs are gone."""


def _unknown_goal_from(e: EngineWorkerError) -> UnknownGoal | None:
    """Rebuild the worker's UnknownGoal from its flattened op error, or None
    for any other worker-side failure (which stays a 500-class error)."""
    msg = str(e)
    if msg.startswith(_UNKNOWN_GOAL_PREFIX):
        return UnknownGoal(msg[len(_UNKNOWN_GOAL_PREFIX) :])
    return None


class WebshopEngineWorkerPool(SessionBackend):
    """N spawn'd engine-hosting processes behind ``ProcSlots``. Sessions are
    placed least-loaded and pinned to a worker for the episode's life.

    ``_sessions`` (session_id -> (worker idx, gen, episode_id)) is guarded by
    ``self._slots.lock`` so pin/unpin and the slot count stay consistent with
    respawns (which drop a worker's pins in the same critical section).
    """

    def __init__(
        self, cfg: EngineConfig, num_workers: int, call_timeout_s: float = 120.0, revive_interval_s: float = 60.0
    ):
        assert num_workers > 0
        self._cfg = cfg
        self._n = num_workers
        self._sessions: dict[str, tuple[int, int, str]] = {}
        # The goal set every worker must have built, fixed by the first worker
        # to complete the boot handshake and enforced on every later handshake
        # (including respawns) by _check_goal_set.
        self._goal_set_lock = threading.Lock()
        self._ref_digest: str | None = None
        self._goal_ids: frozenset[str] = frozenset()
        self._slots = ProcSlots(
            num_workers,
            call_timeout_s,
            engine_worker_main,
            lambda _idx: (cfg,),
            name="webshop-engine",
            spawn_ready_timeout_s=SPAWN_READY_TIMEOUT_S,
            revive_interval_s=revive_interval_s,
            op_error_cls=EngineWorkerError,
            died_cls=EngineWorkerDied,
            on_respawn=self._drop_pins_of,
            validate_ready=self._check_goal_set,
        )
        # Belt and braces: _check_goal_set already failed any disagreeing boot.
        digests = sorted({p["goals_digest"] for p in self._ready_info() if "goals_digest" in p})
        if len(digests) > 1:
            self._slots.shutdown()
            raise RuntimeError(f"engine workers disagree on the goal set (goals_digest {digests}); refusing to serve")

    def _check_goal_set(self, idx: int, ready: Any) -> None:
        """Boot-handshake hook (first boot and every respawn): a worker whose
        goal set differs from the first worker's never serves. A disagreement
        means the goal build is nondeterministic or the on-disk catalog / seed /
        vendored code changed under a running server; a respawned worker would
        otherwise silently hand out different questions for the same goal_id.
        """
        digest, ids = (ready or {}).get("goals_digest"), (ready or {}).get("goal_ids")
        if not digest or ids is None:
            raise RuntimeError(f"engine worker {idx} reported no goal set in its ready payload: {ready!r}")
        with self._goal_set_lock:
            if self._ref_digest is None:
                self._ref_digest, self._goal_ids = digest, frozenset(ids)
            elif digest != self._ref_digest:
                raise RuntimeError(
                    f"engine worker {idx} built a different goal set (goals_digest {digest} != {self._ref_digest}); "
                    "workers disagree on the goal set"
                )

    # ── slot views (tests read these) ────────────────────────────────
    @property
    def _counts(self) -> list[int]:
        return self._slots._counts

    @property
    def respawns(self) -> int:
        return self._slots.respawns

    def _ready_info(self) -> list[dict]:
        return [p for p in self._slots.ready_payloads if isinstance(p, dict)]

    @property
    def goals(self) -> int:
        """Goal count known to the engines (min over workers; identical data ->
        identical counts). 0 only if no worker ever reported one."""
        counts = [p["goals"] for p in self._ready_info() if "goals" in p]
        return min(counts) if counts else 0

    @property
    def unique_goals(self) -> int:
        """Distinct questions across the workers' (identical) goal lists."""
        counts = [p["unique_goals"] for p in self._ready_info() if "unique_goals" in p]
        return min(counts) if counts else 0

    @property
    def goals_digest(self) -> str | None:
        """Order-free digest of the engines' goal set (the inner script's gate
        against the dataset). Checked equal across workers at boot; a respawned
        worker that disagrees is logged here and the first worker's wins."""
        digests = [p["goals_digest"] for p in self._ready_info() if "goals_digest" in p]
        if len(set(digests)) > 1:
            logger.error("engine workers disagree on the goal set: goals_digest %s", sorted(set(digests)))
        return digests[0] if digests else None

    def _call(self, idx: int, gen: int, msg: tuple) -> Any:
        return self._slots.call(idx, gen, msg)

    # ── pin table ────────────────────────────────────────────────────
    def _drop_pins_of(self, idx: int, _new_gen: int) -> None:
        # ProcSlots calls this under its lock right after bumping worker idx's
        # gen: every episode resident there died with it (the respawn already
        # discarded the slot's counts), so their pins go in the same step.
        for sid in [s for s, (w, _, _) in self._sessions.items() if w == idx]:
            del self._sessions[sid]

    def _unpin(self, session_id: str, expect: tuple | None = None) -> tuple | None:
        """Drop the session's pin (only if it still equals ``expect`` when
        given, so a step racing a re-reset never unpins the new episode) and
        its slot count. Returns the dropped pin or None."""
        with self._slots.lock:
            entry = self._sessions.get(session_id)
            if entry is None or (expect is not None and entry != expect):
                return None
            del self._sessions[session_id]
            self._slots.release(entry[0], entry[1])
            return entry

    def _close_episode(self, idx: int, gen: int, episode_id: str) -> None:
        try:
            self._call(idx, gen, ("close", episode_id))
        except (EngineWorkerDied, EngineWorkerError):
            pass  # worker gone (the episode with it) or episode already dropped

    # ── SessionBackend API ───────────────────────────────────────────
    def reset(self, session_id: str, task_id: Any, split: str, goal_id: str | None = None) -> ResetResult:
        # A goal_id (content address) is resolved by the worker, which owns the
        # goal list; a legacy int position is range-checked here so a bad index
        # never touches a worker. Both errors are ValueErrors -> HTTP 400.
        if goal_id is not None and goal_id not in self._goal_ids:
            # Refused here, before the caller's live episode (if any) is touched:
            # a re-reset with a bad id must leave the old episode intact, like a
            # bad legacy index does. Same message as GoalTable.resolve.
            raise UnknownGoal(
                f"goal_id {goal_id!r} is not in the engine goal list (goals_digest {self._ref_digest}): "
                "the dataset and the engine disagree on catalog / goal code"
            )
        goal_ref: Any = goal_id if goal_id is not None else goal_index(task_id, self.goals)
        # A reused session_id re-resets: drop the old episode first.
        self.close(session_id)
        idx, gen = self._slots.reserve()
        episode_id = uuid.uuid4().hex
        try:
            obs, info = self._call(idx, gen, ("reset", episode_id, goal_ref, f"{session_id}:"))
        except BaseException as e:
            self._slots.rollback(idx, gen)
            unknown = _unknown_goal_from(e) if isinstance(e, EngineWorkerError) else None
            if unknown is not None:
                raise unknown from e
            raise
        with self._slots.lock:
            if self._slots.gen(idx) != gen:
                # Worker respawned during our reset (a different session on this
                # worker timed out/died): the env we just built died with it, and
                # the respawn already discarded this slot's counts — nothing to
                # roll back. Don't register a doomed session behind a live obs.
                raise EngineWorkerDied(f"worker {idx} respawned mid-reset of {session_id}")
            displaced = self._sessions.get(session_id)
            self._sessions[session_id] = (idx, gen, episode_id)
            if displaced is not None:
                self._slots.release(displaced[0], displaced[1])
        if displaced is not None:
            # A concurrent reset on the same session_id registered first; close
            # its episode so exactly one stays live (last writer wins the pin).
            self._close_episode(*displaced)
        return ResetResult(obs=obs, info=info)

    def step(self, session_id: str, action: str) -> StepResult:
        with self._slots.lock:
            entry = self._sessions.get(session_id)
        if entry is None:
            raise UnknownSession(session_id)
        idx, gen, episode_id = entry
        try:
            obs, reward, done, info = self._call(idx, gen, ("step", episode_id, action))
        except EngineWorkerDied as e:
            self._unpin(session_id, expect=entry)
            raise UnknownSession(session_id) from e
        except EngineWorkerError as e:
            if str(e) == _UNKNOWN_EPISODE:
                # Episode vanished under us (concurrent close/TTL evict): the
                # worker is healthy but the env is gone -> 410, not 500.
                self._unpin(session_id, expect=entry)
                raise UnknownSession(session_id) from e
            raise
        if done:
            self._unpin(session_id, expect=entry)  # the worker dropped a done episode itself
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def close(self, session_id: str) -> None:
        entry = self._unpin(session_id)
        if entry is not None:
            self._close_episode(*entry)

    def live_sessions(self) -> list[str]:
        with self._slots.lock:
            return list(self._sessions)

    def worker_count(self) -> int:
        return self._n

    def health(self) -> dict[str, Any]:
        return {
            **self._slots.health(),
            "goals": self.goals,
            "unique_goals": self.unique_goals,
            "goals_digest": self.goals_digest,
        }

    def export_goals(self) -> list[dict]:
        """The goal list (goal_id + goal_idx + content record per goal) from the
        first live worker; all workers hold the same list (checked at boot)."""
        last: BaseException | None = None
        for idx in range(self._n):
            try:
                return self._call(idx, self._slots.gen(idx), ("export_goals",))
            except EngineWorkerDied as e:
                last = e  # parked / respawning: try the next worker
        raise EngineWorkerDied(f"no live engine worker could export the goal list: {last}")

    def shutdown(self, grace_s: float = 5.0) -> None:
        self._slots.shutdown(grace_s)
