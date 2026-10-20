"""Process backend for ScienceWorld: one persistent JVM per worker process,
one episode per worker at a time (the earlier environment code's env_servers/scienceworld
pattern, rebuilt on ``agent_utils.procslots`` like alfworld_env/_worker.py).

Why processes: ScienceWorld is a Scala simulator behind a py4j gateway; each
``ScienceWorldEnv`` owns one JVM child (~350 MB RSS, 2-4 s to start) and one
simulation, so a JVM can host exactly one episode. Paying the JVM start per
episode would dominate short episodes; instead every worker keeps ITS env warm
across episodes and ``env.load(task, variation)`` fully resets the simulation
(TCOD's constructor ``ScienceWorldEnv("", jar, envStepLimit=100)`` +
``load(task, var, "easy", generateGoldPath=False)``).

Two halves, one module, both thin wrappers over ``agent_utils.procslots``:

- Child side: ``env_worker_main`` (spawn target) — ``procslots.serve_loop``
  with load/step/release handlers around the one resident env. The factory
  (``"module:function"``, resolved INSIDE the child so tests can substitute a
  fake without crossing the spawn boundary) builds the env lazily on the first
  load, so a child boots without a JVM and a broken JDK surfaces on the first
  /reset as a 500, not as a pool boot failure. A py4j/JVM failure discards the
  env so the next load builds a fresh gateway.

- Parent side: ``WorkerPool`` / ``RemoteEpisode``, an ASYNC facade over
  ``ProcSlots`` plus the one thing alfworld's pool does not need: SLOT
  ADMISSION. A slot hosts one episode, so /reset must wait for a free slot
  (``acquire``: asyncio.Condition, ``slot_wait_timeout_s``); loaded slots go
  back to the free list on ``release`` (terminal step, /close, sweeper, failed
  step). The trainer's concurrency (rollout batch x samples, eval waves) must
  stay under ``env_workers`` or waves serialize on the wait — the wait is
  bounded and surfaces as a 503, never a silent stall.

Failure model (see procslots): an op exceeding ``call_timeout_s``, or a dead
pipe, SIGKILLs and respawns the worker; its episode is gone and the session's
next op raises ``EnvWorkerDied`` (500 -> ``run_transport_failed`` -> the
recipe's dynamic-sampling filter drops the group). The stale generation makes
the owning session's ``release`` a no-op on the wire (the slot still returns
to the free list). A slot whose replacement fails to boot is parked by
procslots and retried by its revive thread; ``acquire`` prefers live slots and
a load that lands on a parked one fails fast (app.py retries once).
"""

import asyncio
import importlib
import os
import time

from agent_utils.procslots import ProcSlots, SlotDied, SlotOpError, serve_loop

# Child bootstrap = interpreter + __mp_main__ re-import of app.py (~4 s local);
# generous headroom for cold cluster filesystems. No JVM here (lazy factory).
SPAWN_READY_TIMEOUT_S = 180.0

_NO_ACTION_MATCH = "No known action matches that input"


def make_env(jar_path: str, env_step_limit: int):
    """The real factory: one ScienceWorldEnv = one JVM child (py4j gateway).
    Blocking (JVM start 2-4 s). ``jar_path`` "" -> the wheel's bundled jar.
    """
    from scienceworld import ScienceWorldEnv

    return ScienceWorldEnv("", jar_path or None, envStepLimit=env_step_limit)


def _resolve_factory(spec: str):
    module_name, _, func_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), func_name)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(v) for v in value]


def _get_completed(env) -> bool:
    """Raw simulator completion (TCOD's _get_scienceworld_completed): the
    success flag, NOT the wrapper's done (which also fires on the step limit)."""
    completed = env.server.getCompleted()
    if isinstance(completed, (list, tuple)) and len(completed) == 1:
        completed = completed[0]
    if isinstance(completed, str):
        value = completed.strip().lower()
        if value in {"true", "1"}:
            return True
        if value in {"false", "0"}:
            return False
        raise RuntimeError(f"unexpected ScienceWorld completion value: {completed!r}")
    return bool(completed)


class EpisodeHost:
    """The child-side episode logic: one resident env (built lazily by the
    factory), load/step/release. Also used in-process by app.py when
    ``env_workers=0`` (unit tests, ops fallback), so the two paths cannot
    drift."""

    def __init__(self, factory_spec: str, jar_path: str, simplification: str, env_step_limit: int):
        self._factory_spec = factory_spec
        self._factory = None
        self._jar_path = jar_path
        self._simplification = simplification
        self._env_step_limit = env_step_limit
        self.env = None
        self._best = 0.0
        self._templates: list[str] = []

    def _ensure_env(self):
        if self.env is None:
            if self._factory is None:
                self._factory = _resolve_factory(self._factory_spec)
            self.env = self._factory(self._jar_path, self._env_step_limit)
        return self.env

    def discard_env(self) -> None:
        env = self.env
        self.env = None
        if env is not None:
            try:
                env.close()
                env.close = lambda: None  # ScienceWorldEnv.__del__ double-close guard
            except Exception:
                pass

    def load(self, task_name: str, var_num: int) -> tuple[str, dict]:
        """Start an episode: returns (obs, info) with the compact action
        presentation TCOD uses (get_possible_actions ~23 templates +
        get_possible_objects ~20 objects; full valid-combination lists blow
        the token budget: >1100 actions)."""
        try:
            env = self._ensure_env()
            env.load(task_name, int(var_num), self._simplification, generateGoldPath=False)
            obs, info = env.reset()
            task_desc = str(env.get_task_description()).strip()
            templates = _as_list(env.get_possible_actions())
            objects = _as_list(env.get_possible_objects())
        except Exception:
            self.discard_env()  # a broken JVM must not poison the next episode
            raise
        if not task_desc:
            raise RuntimeError(f"empty task description for {task_name}/{var_num}")
        score = float((info or {}).get("score", 0.0) or 0.0)
        self._best = max(0.0, score)
        self._templates = templates
        return str(obs), {
            "task_desc": task_desc,
            "score": score,
            "best_score": self._best,
            "completed": False,
            "templates": templates,
            "objects": objects,
        }

    def step(self, command: str) -> tuple[str, dict]:
        env = self.env
        if env is None:
            raise RuntimeError("step before load")
        try:
            obs, _reward, done, info = env.step(str(command))
            completed = _get_completed(env)
            objects = _as_list(env.get_possible_objects())
        except Exception:
            self.discard_env()
            raise
        score = float((info or {}).get("score", 0.0) or 0.0)
        prev_best = self._best
        self._best = max(prev_best, score)
        return str(obs), {
            "score": score,
            "best_score": self._best,
            "best_delta": self._best - prev_best,
            "completed": completed,
            "done": bool(done) or completed,
            "templates": self._templates,
            "objects": objects,
            "invalid": str(obs).strip().startswith(_NO_ACTION_MATCH),
        }

    def release(self) -> None:
        # keep the JVM warm; the next load() fully resets the simulation
        return None


def env_worker_main(
    conn,
    factory_spec: str,
    jar_path: str,
    simplification: str,
    env_step_limit: int,
    java_home: str,
    java_opts: str,
) -> None:
    """Body of one env-hosting worker process (spawn target).

    py4j execs the first ``java`` on PATH: ``java_home`` (if set) is put in
    front of PATH here, in the child, so the server's own environment does not
    matter. ``java_opts`` goes to JAVA_TOOL_OPTIONS (bounded heap: the JVM
    default heap is 1/4 of physical RAM, unbudgetable for 100+ workers).
    """
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    if java_opts:
        os.environ["JAVA_TOOL_OPTIONS"] = java_opts
    host = EpisodeHost(factory_spec, jar_path, simplification, env_step_limit)
    serve_loop(conn, {"load": host.load, "step": host.step, "release": host.release})


class EnvWorkerError(SlotOpError):
    """The op failed inside the worker; the worker (and its JVM) is still healthy."""


class EnvWorkerDied(SlotDied):
    """The worker was killed (op timeout / crash); its episode is gone."""


class RemoteEpisode:
    """Handle to the episode resident in one worker; coroutines only."""

    __slots__ = ("_pool", "worker_idx", "gen", "released")

    def __init__(self, pool: "WorkerPool", worker_idx: int, gen: int):
        self._pool = pool
        self.worker_idx = worker_idx
        self.gen = gen
        self.released = False

    async def step(self, command: str):
        return await self._pool.step(self, command)

    async def close(self) -> None:
        await self._pool.release(self)


class WorkerPool:
    """``num_workers`` spawn'd env-hosting processes behind ``ProcSlots``, with
    one-episode-per-slot admission on top."""

    def __init__(
        self,
        num_workers: int,
        call_timeout_s: float,
        factory_spec: str,
        *,
        jar_path: str = "",
        simplification: str = "easy",
        env_step_limit: int = 100,
        java_home: str = "",
        java_opts: str = "",
        spawn_ready_timeout_s: float = SPAWN_READY_TIMEOUT_S,
    ):
        assert num_workers > 0
        self._slots = ProcSlots(
            num_workers,
            call_timeout_s,
            env_worker_main,
            lambda _idx: (factory_spec, jar_path, simplification, env_step_limit, java_home, java_opts),
            name="scienceworld-env-worker",
            spawn_ready_timeout_s=spawn_ready_timeout_s,
            op_error_cls=EnvWorkerError,
            died_cls=EnvWorkerDied,
        )
        self._n = num_workers
        self._free: list[int] = list(range(num_workers))
        # Created lazily on the serving loop (the pool itself is built off-loop).
        self._cond: asyncio.Condition | None = None

    # ── views (tests / app.py reset_info) ──────────────────────────
    @property
    def respawns(self) -> int:
        return self._slots.respawns

    @property
    def num_workers(self) -> int:
        return self._n

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def health(self) -> dict[str, int]:
        return self._slots.health() | {"free_workers": len(self._free)}

    # ── admission ─────────────────────────────────────────────────
    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def _pick_free(self) -> int | None:
        """Least-recently-freed live slot first; parked slots only when
        nothing else is free (they fail fast at load, app.py retries once)."""
        if not self._free:
            return None
        dead = self._slots.dead_slots
        for pos, idx in enumerate(self._free):
            if idx not in dead:
                return self._free.pop(pos)
        return self._free.pop(0)

    async def acquire(self, timeout_s: float) -> tuple[int, int]:
        """Wait for a free slot; returns ``(idx, gen)``. Raises TimeoutError
        when every slot stays busy for ``timeout_s`` (the trainer's
        concurrency exceeds env_workers — a 503 upstream)."""
        cond = self._condition()
        deadline = time.monotonic() + timeout_s
        async with cond:
            while True:
                idx = self._pick_free()
                if idx is not None:
                    return idx, self._slots.gen(idx)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no free ScienceWorld worker within {timeout_s:.0f}s ({self._n} slots busy)")
                try:
                    await asyncio.wait_for(cond.wait(), remaining)
                except TimeoutError:
                    raise TimeoutError(
                        f"no free ScienceWorld worker within {timeout_s:.0f}s ({self._n} slots busy)"
                    ) from None

    async def _put_back(self, idx: int) -> None:
        cond = self._condition()
        async with cond:
            if idx not in self._free:
                self._free.append(idx)
            cond.notify()

    # ── ops ───────────────────────────────────────────────────────
    async def load(self, idx: int, gen: int, task_name: str, var_num: int) -> tuple[RemoteEpisode, str, dict]:
        """Start an episode on an acquired slot. On failure the slot goes back
        to the free list (a respawned or parked slot included) and the error
        propagates."""
        try:
            obs, info = await self._slots.acall(idx, gen, ("load", task_name, int(var_num)))
        except BaseException:
            await self._put_back(idx)
            raise
        return RemoteEpisode(self, idx, gen), obs, info

    async def step(self, ep: RemoteEpisode, command: str) -> tuple[str, dict]:
        return await self._slots.acall(ep.worker_idx, ep.gen, ("step", command))

    async def release(self, ep: RemoteEpisode) -> None:
        """Return the slot. The wire op is skipped when the worker was
        respawned meanwhile (stale gen: nothing to release there)."""
        if ep.released:
            return
        ep.released = True
        try:
            if self._slots.gen(ep.worker_idx) == ep.gen:
                await self._slots.acall(ep.worker_idx, ep.gen, ("release",))
        except (SlotDied, SlotOpError):
            pass
        finally:
            await self._put_back(ep.worker_idx)

    def shutdown(self, grace_s: float = 5.0) -> None:
        self._slots.shutdown(grace_s)
