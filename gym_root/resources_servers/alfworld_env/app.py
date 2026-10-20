"""ALFWorld resources server — Gymnasium protocol (/reset + /step), driven by env_loop_agent.

One episode = one signed-cookie session (upstream GymnasiumServer machinery,
shared session/step/sweeper template in agent_utils.session_gym):
/reset loads the TextWorld game named by ``game_file`` in the run body
(relative paths resolve under ``alfworld_data`` / $ALFWORLD_DATA), /step
plays the FIRST ``act`` tool call in the model turn (structured
``{"command": ...}`` arguments — the dataset row declares the tool and the
trainer's tool-call parser produces it; no text parsing here). Strict
Gymnasium endings: terminated on the game's ``done``, truncated at
``max_episode_steps``; reward is 1.0 only at a terminal ``info["won"]``
(binary success, the ALFWorld convention).

Sessions whose episode is killed upstream (policy 400s, trainer aborts, /run
timeouts) never see a terminal /step; a TTL sweeper evicts them, and
env_loop_agent POSTs /close best-effort on non-terminal exits.

Loaded envs are recycled through a bounded per-game_file pool (EnvPool):
close paths park the env instead of closing it, and /reset re-resets a
parked env of the same game (~0.15 s) instead of a fresh load (~0.8 s warm,
seconds at the tail). Only pool evictions and load-failure fallbacks run the
real env.close().

With env_workers > 0 (production default) envs live in worker PROCESSES
(_worker.py — the ProcessPoolBackend pattern of the earlier environment code): loads/resets
parallelize env_workers-ways instead of serializing on TW_LOAD_LOCK, which
is what let a 512-session reset wave livelock the trainer (2026-08-31).
Session state and EnvPool then hold RemoteEnv handles with async
reset/step/close (awaited directly — a per-worker single-thread executor
inside WorkerPool does the blocking pipe IO, so remote ops never occupy
self._executor and a wedged worker stalls only its own envs). The env
dispatch sites branch on self._procpool; everything else is shared.
env_workers=0 keeps the legacy in-process envs (unit tests, ops fallback).
"""

import asyncio
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_utils.session_gym import (
    BaseSessionState,
    SessionGymConfig,
    SessionGymnasiumServer,
    format_fault_hint,
    truncate_obs,
)
from agent_utils.session_gym import extract_command as extract_command  # re-export (tests import it here)
from fastapi import HTTPException

from alfworld_env._tw_compat import TW_LOAD_LOCK
from alfworld_env._worker import EnvWorkerDied, WorkerPool

# TextWorld's per-step feedback for any command the game cannot parse/apply.
_NOTHING_HAPPENS = "Nothing happens."


class AlfworldResourcesServerConfig(SessionGymConfig):
    # "" -> read $ALFWORLD_DATA at reset time (exported by the training entry;
    # local dev points it at an `alfworld-download` tree).
    alfworld_data: str = ""
    # Env-step budget per episode; dataset-row metadata overrides per game.
    max_episode_steps: int = 30
    # Name of the action tool the dataset rows declare (prepare_data.py).
    action_tool_name: str = "act"
    expose_admissible_commands: bool = True
    max_admissible_commands: int = 60
    # Context safety: episodes must stay inside the model window; ALFWorld
    # observations are short but pathological ones exist.
    max_obs_chars: int = 2048
    # Executor threads for blocking env.step/env.close calls — env_workers=0
    # mode only (remote ops run on WorkerPool's per-worker threads instead).
    # In that mode env loads get their own single-thread executor: they
    # serialize on TW_LOAD_LOCK anyway (see _tw_compat), and sharing this
    # pool would let a reset wave occupy every thread and starve in-flight
    # steps. CAVEAT (env_workers=0 only): serial loads mean a synchronized
    # wave of N resets drains in N * load_time and the tail /run can exceed
    # NEMO_GYM_RUN_TIMEOUT for N in the hundreds — the exact livelock that
    # motivated env_workers (2026-08-31, 512-session waves).
    max_concurrency: int = 32
    # Env-hosting worker processes (see _worker.py). Loads parallelize
    # env_workers-ways: a 512-reset wave drains in ~512/16 * load_time ≈
    # tens of seconds instead of ~30 min. Ops queue per worker, so one slow
    # or wedged worker delays only its own envs. 0 = legacy in-process envs.
    env_workers: int = 16
    # Per-IPC-op budget (load/reset/step/close). A worker exceeding it is
    # SIGKILLed and respawned: its resident envs are lost, affected sessions
    # 500 on their next op (-> run_transport_failed; the recipe's
    # miles_agent.dapo_filter.check_no_aborted_no_transport_failure drops the
    # group so it is re-sampled, never trained as reward 0), parked pool
    # handles die lazily via the broken-env fallback in reset(). Loads are ~seconds;
    # this only fires on genuine wedges.
    env_call_timeout_s: float = 300.0
    # Worker-side env factory "module:function", resolved INSIDE the worker
    # process (monkeypatching app._make_env cannot cross the spawn boundary;
    # tests point this at a fake factory).
    env_factory: str = "alfworld_env._worker:make_env"
    # Max idle loaded envs parked for reuse, keyed by game_file (~5 MB each;
    # 128 ≈ 640 MB). 0 disables pooling: close paths run the real env.close().
    env_pool_size: int = 128
    # Must exceed NEMO_GYM_RUN_TIMEOUT (1200 s in recipes/alfworld) so a live
    # episode is never evicted mid-run.
    session_ttl_s: float = 1800.0
    # <= 0 disables the sweeper (unit tests).
    sweep_interval_s: float = 60.0


@dataclass
class AlfworldSessionState(BaseSessionState):
    env: Any
    # Resolved absolute game path; the EnvPool key the env is parked under.
    game_file: str


def format_observation(feedback: str, admissible: list[str] | None, config: AlfworldResourcesServerConfig) -> str:
    obs = truncate_obs((feedback or "").strip(), config.max_obs_chars)
    if config.expose_admissible_commands and admissible:
        commands = list(admissible)[: config.max_admissible_commands]
        obs += "\nYour admissible actions are: [" + ", ".join(commands) + "]"
    return obs


def _make_env(game_file: str, want_admissible: bool):
    """Load + reset one game (env_workers=0 path). Blocking; runs in the load
    executor under TW_LOAD_LOCK. Returns (env, initial_obs, info).

    The body lives in _worker.make_env so worker processes import it without
    this module's server stack; this wrapper stays module-level as the
    monkeypatch seam that lets unit tests run without alfworld installed.
    step() semantics are pinned by test_real_game_step_semantics (see there).
    """
    from alfworld_env._worker import make_env

    return make_env(game_file, want_admissible)


def _locked_make_env(game_file: str, want_admissible: bool):
    with TW_LOAD_LOCK:
        return _make_env(game_file, want_admissible)


def _locked_reset_env(env):
    """Re-reset a pooled env to its initial state; returns (obs, info).

    Blocking; runs in the LOAD executor under TW_LOAD_LOCK: env.reset()
    rebuilds PddlState via fast_downward's pure-python translate — the same
    global-state-unsafe pass a fresh load runs (see _tw_compat). Running it
    locked on the step pool instead would let a reset wave park on the lock
    across every step thread and starve in-flight steps.
    Reuse semantics (byte-identical initial obs/admissible_commands, winnable
    again, ~0.15 s, from finished or mid-episode state) validated against
    real json_2.1.1 games, 2026-08-31.
    """
    with TW_LOAD_LOCK:
        return env.reset()


class EnvPool:
    """Bounded pool of idle loaded envs, keyed by game_file.

    take() removes the env it returns, so an env is owned by the pool XOR by
    exactly one session — never shared. take() prefers the most recently
    parked env of a game; eviction drops the env idle the longest, and the
    CALLER must run the real env.close() on everything put() returns. All
    methods are synchronous dict ops, only called from the event loop thread,
    so they need no locking.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._items: OrderedDict[int, tuple[str, Any]] = OrderedDict()  # id -> (game_file, env)
        self._ids_by_game: dict[str, list[int]] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._items)

    def take(self, game_file: str) -> Any | None:
        ids = self._ids_by_game.get(game_file)
        if not ids:
            return None
        item_id = ids.pop()
        if not ids:
            del self._ids_by_game[game_file]
        _, env = self._items.pop(item_id)
        return env

    def put(self, game_file: str, env: Any) -> list[Any]:
        """Park an idle env; returns the envs the caller must close."""
        if self.max_size <= 0:
            return [env]
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = (game_file, env)
        self._ids_by_game.setdefault(game_file, []).append(item_id)
        evicted = []
        while len(self._items) > self.max_size:
            old_id, (old_game, old_env) = self._items.popitem(last=False)
            old_ids = self._ids_by_game[old_game]
            old_ids.remove(old_id)
            if not old_ids:
                del self._ids_by_game[old_game]
            evicted.append(old_env)
        return evicted


class AlfworldResourcesServer(SessionGymnasiumServer):
    config: AlfworldResourcesServerConfig

    def _post_init(self) -> None:
        # tatsu's PEG recursion overflows default worker-thread stacks (512 KB
        # on macOS); must be set before the executors spawn their threads.
        threading.stack_size(64 * 1024 * 1024)
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrency, thread_name_prefix="alfworld")
        # Loads serialize on TW_LOAD_LOCK; a dedicated single thread keeps a
        # reset wave from occupying the step/close pool (see max_concurrency).
        # Both executors are only used with env_workers=0 — remote ops run on
        # WorkerPool's own per-worker threads.
        self._load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alfworld-load")
        self._env_pool = EnvPool(self.config.env_pool_size)
        # Created lazily on the first /reset: eager spawn here would also fire
        # in config dry-runs and venv-builder probes (nemo-gym instantiates
        # the server class before the dry_run early-return) and in any test
        # constructing a default-config server.
        self._procpool: WorkerPool | None = None
        self._procpool_init_lock = asyncio.Lock()

    def _resolve_game_file(self, metadata: dict) -> Path:
        """Row validation (three 400s). Runs BEFORE the base frees a re-reset's
        previous episode (``_validate_row``) and again, cheaply, in ``_do_reset``
        before acquiring an env: a failure after acquisition but before session
        registration would leak the env (worker-resident envs have no GC
        fallback, unlike the in-process path)."""
        game_file = metadata.get("game_file")
        if not game_file:
            raise HTTPException(status_code=400, detail="Missing game_file in the run body.")
        path = Path(game_file)
        if not path.is_absolute():
            data_root = self.config.alfworld_data or os.environ.get("ALFWORLD_DATA", "")
            if not data_root:
                raise HTTPException(
                    status_code=400,
                    detail="Relative game_file requires the alfworld_data config or $ALFWORLD_DATA.",
                )
            path = Path(data_root) / game_file
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"No such game file: {path}")
        return path

    def _validate_row(self, metadata: dict) -> None:
        # Pre-refactor ordering: a bad row 400s WITHOUT touching the previous
        # episode of a reused cookie.
        self._resolve_game_file(metadata)

    async def _do_reset(
        self, metadata: dict, session_id: str, max_episode_steps: int
    ) -> tuple[AlfworldSessionState, str, dict]:
        path = self._resolve_game_file(metadata)
        await self._ensure_workers()

        loop = asyncio.get_running_loop()
        want_admissible = self.config.expose_admissible_commands
        game_file = str(path)
        load_retries = 0
        pool_hit = 0
        env = self._env_pool.take(game_file)
        if env is not None:
            try:
                if self._procpool is not None:
                    # Async IPC; the worker's own single-threadedness
                    # replaces TW_LOAD_LOCK.
                    obs, info = await env.reset()
                else:
                    obs, info = await loop.run_in_executor(self._load_executor, _locked_reset_env, env)
                pool_hit = 1
            except Exception:
                # A parked env can come back broken (e.g. its last episode
                # 500'd mid-step, or its worker process was respawned); drop
                # it and fall through to a fresh load.
                try:
                    if self._procpool is not None:
                        await env.close()
                    else:
                        await loop.run_in_executor(self._executor, env.close)
                except Exception:
                    pass
                env = None
        if env is None:
            try:
                env, obs, info = await self._load_env(loop, game_file, want_admissible)
            except EnvWorkerDied as e:
                if isinstance(e.__cause__, TimeoutError):
                    # A load that blew the whole per-op budget signals a
                    # systemic stall (wedged translate, dead $ALFWORLD_DATA
                    # mount); retrying would wedge the fresh worker for
                    # another full budget. Fail fast instead.
                    raise
                load_retries = 1
                env, obs, info = await self._load_env(loop, game_file, want_admissible)
            except Exception:
                # Transient load flakes exist even with the _tw_compat hardening;
                # a second failure propagates (500 -> run_transport_failed, see
                # env_call_timeout_s for what that means to training).
                load_retries = 1
                env, obs, info = await self._load_env(loop, game_file, want_admissible)

        state = AlfworldSessionState(env=env, game_file=game_file, max_episode_steps=max_episode_steps)
        observation = format_observation(obs, info.get("admissible_commands"), self.config)
        # Only counters worth forwarding to wandb env/* belong here; the agent
        # merges numeric info values into its /run response.
        reset_info = {"env_load_retries": load_retries, "env_pool_hit": pool_hit}
        if self._procpool is not None:
            # Server-lifetime gauge (mean over samples in wandb): nonzero
            # means wedged/crashed env workers — investigate.
            reset_info["env_worker_respawns"] = self._procpool.respawns
        return state, observation, reset_info

    async def _load_env(self, loop, game_file: str, want_admissible: bool):
        """Fresh env load -> (env, initial_obs, info); blocking work off-loop."""
        if self._procpool is not None:
            return await self._procpool.load(game_file, want_admissible)
        return await loop.run_in_executor(self._load_executor, _locked_make_env, game_file, want_admissible)

    async def _ensure_workers(self) -> None:
        # Built off-loop: WorkerPool.__init__ blocks on the children's ready
        # handshakes (seconds — each child re-imports the app stack). A boot
        # failure propagates (500) and is retried by the next /reset.
        if self.config.env_workers <= 0 or self._procpool is not None:
            return
        async with self._procpool_init_lock:
            if self._procpool is None:
                self._procpool = await asyncio.get_running_loop().run_in_executor(
                    None,
                    WorkerPool,
                    self.config.env_workers,
                    self.config.env_call_timeout_s,
                    self.config.env_factory,
                )

    async def _do_step(self, state: AlfworldSessionState, command: str) -> tuple[str, float, bool, dict]:
        if self._procpool is not None:
            # Async IPC on the env's own worker thread; a wedged or busy
            # worker delays only its own envs' ops.
            raw_obs, _score, done, info = await state.env.step(command)
        else:
            raw_obs, _score, done, info = await asyncio.get_running_loop().run_in_executor(
                self._executor, state.env.step, command
            )
        if (raw_obs or "").strip() == _NOTHING_HAPPENS:
            state.invalid_actions += 1
        done = bool(done)
        state.won = bool(info.get("won"))
        obs = format_observation(raw_obs, info.get("admissible_commands"), self.config)
        if done:
            obs += "\nTask completed." if state.won else "\nTask failed."
        reward = 1.0 if (done and state.won) else 0.0
        # `lost` is a per-step signal (not carried on the state): the base's
        # early returns report False and this overrides it on a real step.
        return obs, reward, done, {"lost": bool(info.get("lost"))}

    def _extra_info(self, state: AlfworldSessionState) -> dict:
        return {"lost": False}

    def _format_fault_observation(self, state: AlfworldSessionState) -> str:
        return f"{_NOTHING_HAPPENS} " + format_fault_hint(self.config.action_tool_name, "<one admissible command>")

    async def _release_state(self, state: AlfworldSessionState) -> None:
        # Park the env for reuse; textworld/libdownward is not safe against
        # step racing close/reset on one env, so the base calls this only
        # under the session lock, after the state left session_state. Pool
        # evictions (and pooling disabled) run the real close here.
        for env in self._env_pool.put(state.game_file, state.env):
            try:
                if self._procpool is not None:
                    await env.close()
                else:
                    await asyncio.get_running_loop().run_in_executor(self._executor, env.close)
            except Exception:
                pass


if __name__ == "__main__":
    AlfworldResourcesServer.run_webserver()
