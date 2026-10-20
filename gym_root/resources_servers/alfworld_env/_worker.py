"""Process backend for TextWorld envs (the ProcessPoolBackend pattern of the author's earlier environment code).

Why processes: fast_downward's pure-python PDDL->SAS translate runs on every
env load/reset, is GIL-bound (~seconds each) and full of module globals, so
in-process loads must serialize on TW_LOAD_LOCK — a synchronized wave of N
resets drains in N * load_time and blows past the trainer's /run timeout
(livelock evidence: 2026-08-31, 512-session waves). Worker
processes each own an interpreter: loads parallelize num_workers-ways with no
lock at all.

Two halves, one module, both thin wrappers over ``agent_utils.procslots``
(the process-slot machinery shared with the WebShop engine tier):

- Child side: ``env_worker_main`` (spawn target) — ``procslots.serve_loop``
  with load/reset/step/close handlers, hosting many envs (env_id -> env).
  Single-threaded means strictly sequential env ops, so the thread-unsafety
  that forces TW_LOAD_LOCK in-process needs no locking here. ``make_env``
  (the real TextWorld factory) lives here too. NOTE: because the production
  entrypoint is ``python app.py``, multiprocessing's __mp_main__ fixup
  re-executes app.py's module body in every child (~4 s / ~100 MB for the
  fastapi/nemo_gym/ray imports, once per worker) — harmless but not free;
  the factory's own textworld/alfworld imports stay lazy until the first load.

- Parent side: ``WorkerPool`` / ``RemoteEnv``, an ASYNC facade over
  ``ProcSlots``. Every op on worker i executes on worker i's dedicated
  single-thread executor (which is also what serializes its pipe — no
  per-worker lock), and callers await the future without holding any shared
  thread. A wedged worker therefore stalls only its own ops; steps/loads on
  the other workers keep flowing, and the event loop never blocks. (v1 of
  this module dispatched blocking calls onto the server's shared executor;
  one wedged worker then pinned every thread for env_call_timeout_s,
  stalling all sessions server-wide.)

Failure model (see procslots for the mechanics): an op exceeding
call_timeout_s, or a dead pipe, kills and respawns the worker (SIGKILL — it
may be wedged in C code). Every env resident in that worker is gone; the
generation counter makes stale RemoteEnv handles raise EnvWorkerDied instead
of touching a stranger's env after the respawn. Sessions discover the loss on
their next op: 500 -> the agent reports env_metrics.run_transport_failed and
the recipe's dynamic-sampling filter
(miles_agent.dapo_filter.check_no_aborted_no_transport_failure) drops the
group so it is re-sampled, never trained as reward 0.
Parked pool handles die lazily via the broken-env fallback in app.py's
reset(). Ops that fail while the worker stays alive (bad game file, env
exception) raise EnvWorkerError and cost nothing else. A worker whose
replacement fails to boot is parked and retried periodically by procslots'
revive thread; a load that hits a parked worker fails fast (app.py's reset
retries once, landing elsewhere).

Boot handshake: each child sends one ready message before serving; spawn
waits for it, so a deploy/venv breakage that kills children during bootstrap
fails the pool construction loudly with the real cause instead of degrading
into a silent per-op respawn storm.
"""

import importlib
import uuid
from typing import Any

from agent_utils.procslots import ProcSlots, SlotDied, SlotOpError, serve_loop

from alfworld_env._tw_compat import apply_textworld_patches

# Child bootstrap = interpreter + __mp_main__ re-import of app.py (~4 s local);
# generous headroom for cold cluster filesystems.
SPAWN_READY_TIMEOUT_S = 120.0


def make_env(game_file: str, want_admissible: bool):
    """Load + reset one game. Blocking. Returns (env, initial_obs, info).

    step() semantics of the returned env are pinned by
    tests/test_app.py::test_real_game_step_semantics:
    (obs: str, score: int, done: bool, info: dict) — success is info["won"],
    never a tuple slot (tale-suite's positional unpacking mislabels them).
    """
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    from textworld.envs.wrappers import Filter

    apply_textworld_patches()

    infos = textworld.EnvInfos(
        score=True,
        max_score=True,
        won=True,
        lost=True,
        feedback=True,
        moves=True,
        admissible_commands=want_admissible,
    )
    env = textworld.start(game_file, infos, wrappers=[Filter, AlfredDemangler()])
    obs, info = env.reset()
    return env, obs, info


def _resolve_factory(spec: str):
    module_name, _, func_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), func_name)


def env_worker_main(conn, factory_spec: str) -> None:
    """Body of one env-hosting worker process (spawn target).

    Sends one ready message, then replies exactly once per received message:
    (True, payload) or (False, "ExcType: message"). Exits on EOF/broken pipe
    (parent died or respawned us) or an explicit shutdown op. The factory is
    resolved lazily on the first load so the child boots without importing
    textworld/alfworld.
    """
    envs: dict[str, Any] = {}
    factory = None

    def load(env_id: str, game_file: str, want_admissible: bool):
        nonlocal factory
        if factory is None:
            factory = _resolve_factory(factory_spec)
        env, obs, info = factory(game_file, want_admissible)
        envs[env_id] = env
        return (obs, info)

    def reset(env_id: str):
        return envs[env_id].reset()

    def step(env_id: str, command: str):
        return envs[env_id].step(command)

    def close(env_id: str) -> None:
        env = envs.pop(env_id, None)
        if env is not None:
            env.close()
        return None

    serve_loop(conn, {"load": load, "reset": reset, "step": step, "close": close})


class EnvWorkerError(SlotOpError):
    """The op failed inside the worker; the worker itself is still healthy."""


class EnvWorkerDied(SlotDied):
    """The worker was killed (op timeout / crash); its resident envs are gone."""


class RemoteEnv:
    """Handle to one env resident in a worker process; all methods are
    coroutines awaited on the event loop (app.py branches on procpool mode,
    so these never need to duck-type the sync in-process env API)."""

    __slots__ = ("_pool", "worker_idx", "gen", "env_id")

    def __init__(self, pool: "WorkerPool", worker_idx: int, gen: int, env_id: str):
        self._pool = pool
        self.worker_idx = worker_idx
        self.gen = gen
        self.env_id = env_id

    async def reset(self):
        return await self._pool.op(self, "reset")

    async def step(self, command: str):
        return await self._pool.op(self, "step", command)

    async def close(self) -> None:
        await self._pool.release(self)


class WorkerPool:
    """num_workers spawn'd env-hosting processes behind ``ProcSlots``.

    Placement: load() reserves the least-loaded worker before the (slow) IPC
    round trip so concurrent loads spread instead of stampeding one worker;
    the reservation is rolled back generation-aware if the load fails.
    ``_counts``/``_gens``/``_procs``/``respawns`` are views onto the slots
    (tests and app.py's reset_info read them).
    """

    def __init__(self, num_workers: int, call_timeout_s: float, factory_spec: str):
        assert num_workers > 0
        self._factory_spec = factory_spec
        self._slots = ProcSlots(
            num_workers,
            call_timeout_s,
            env_worker_main,
            lambda _idx: (factory_spec,),
            name="alfworld-env-worker",
            spawn_ready_timeout_s=SPAWN_READY_TIMEOUT_S,
            op_error_cls=EnvWorkerError,
            died_cls=EnvWorkerDied,
        )

    @property
    def respawns(self) -> int:
        return self._slots.respawns

    @property
    def _counts(self) -> list[int]:
        return self._slots._counts

    @property
    def _gens(self) -> list[int]:
        return self._slots._gens

    @property
    def _procs(self) -> list[Any]:
        return self._slots._procs

    # ── async facade ──────────────────────────────────────────────────

    async def load(self, game_file: str, want_admissible: bool) -> tuple[RemoteEnv, str, dict]:
        idx, gen = self._slots.reserve()  # reserve the slot before the (slow) IPC call
        env_id = uuid.uuid4().hex
        try:
            obs, info = await self._slots.acall(idx, gen, ("load", env_id, game_file, want_admissible))
        except BaseException:
            self._slots.rollback(idx, gen)  # only our own reservation, and only on our gen
            raise
        return RemoteEnv(self, idx, gen, env_id), obs, info

    async def op(self, handle: RemoteEnv, op: str, *args):
        return await self._slots.acall(handle.worker_idx, handle.gen, (op, handle.env_id, *args))

    async def release(self, handle: RemoteEnv) -> None:
        """close() for RemoteEnv: drop the count, best-effort close in the worker."""
        if not self._slots.release(handle.worker_idx, handle.gen):
            return  # worker died since; the env is already gone
        try:
            await self.op(handle, "close")
        except (EnvWorkerDied, EnvWorkerError):
            pass

    def shutdown(self) -> None:
        self._slots.shutdown()
