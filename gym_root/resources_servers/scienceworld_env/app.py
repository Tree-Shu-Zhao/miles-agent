"""ScienceWorld resources server — Gymnasium protocol (/reset + /step), driven by env_loop_agent.

Third env on the shared session-Gymnasium template (agent_utils.session_gym):
same cookie-keyed session, same FIRST-``act``-tool-call action transport
(structured ``{"command": ...}``; no text parsing), same TTL sweeper; this
module owns only the ScienceWorld mechanics and presentation.

- /reset: the run body names the task by ``task_name`` + ``var_num`` (plus an
  optional ``split`` that is validated for membership, see task_splits.py).
  The episode is loaded into one worker process holding one warm JVM
  (_worker.py: ``env.load(task, var, "easy")`` on a persistent
  ``ScienceWorldEnv``); the opening observation is TCOD's wording — the task
  description, the current observation, the compact action templates and the
  interactable objects (arXiv 2604.24005, SCIWORLD_TEMPLATE_NO_HIS).
- /step: the command goes to the simulator UNCHANGED (no admissibility
  check; TCOD has none). The observation is the simulator feedback followed
  by the (unchanged) action templates and the CURRENT objects — TCOD's
  per-step template re-lists both. Strict Gymnasium endings: terminated on
  the simulator's done OR raw completion, truncated at ``max_episode_steps``.
- Reward is dense and neutral: the running-max score delta / 100, so the
  episode sum is TCOD's episode reward best_score/100 in [0, 1] (webshop-style
  task_score); ``won`` is the raw simulator completion flag
  (``env.server.getCompleted()``), never the wrapper's done. recipes own any
  remap (the OPD arms zero the train reward anyway).
- "No known action matches that input." counts as ``invalid_actions``
  (no penalty, no special feedback — TCOD parity); a turn with no ``act``
  call costs a step and returns the retry hint (base template).

Failure semantics (frozen, see session_gym): env feedback is never reshaped;
a JVM/worker failure raises (500) -> the base evicts the session and
env_loop_agent reports ``run_transport_failed`` so the trainer drops the
group. A bad row (unknown task, variation out of range, split mismatch) is a
400 before any slot is taken. All slots busy for ``slot_wait_timeout_s`` is
a 503 (trainer concurrency above ``env_workers``).

``env_workers=0`` hosts the env in-process (one JVM per live session,
blocking calls on an executor) — unit tests with a fake factory and an ops
fallback, not the production shape.
"""

import asyncio
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utils.session_gym import (
    BaseSessionState,
    SessionGymConfig,
    SessionGymnasiumServer,
    format_fault_hint,
    truncate_obs,
)
from fastapi import HTTPException

from scienceworld_env._worker import EnvWorkerDied, EpisodeHost, WorkerPool
from scienceworld_env.task_splits import validate_task

logger = logging.getLogger("scienceworld_env")

# Wording the model sees on a turn without an act tool call.
_FAULT_EXAMPLE = "<one action, e.g. look around or open door to kitchen>"


class ScienceworldResourcesServerConfig(SessionGymConfig):
    # Pinned scienceworld.jar; "" -> the jar bundled in the scienceworld wheel.
    # The server logs the sha256 of the jar it serves at the first /reset.
    jar_path: str = ""
    # JDK the worker processes put on PATH (py4j execs `java`); "" -> whatever
    # `java` the server process already sees. The training entry points this
    # at $SCIW_JAVA_HOME (a JDK 17).
    java_home: str = ""
    # JAVA_TOOL_OPTIONS for the worker JVMs. Bounded heap: the default heap is
    # 1/4 of physical RAM, which makes 128 JVMs unbudgetable; 2g is ~6x the
    # observed per-JVM RSS (~350 MB).
    java_opts: str = "-Xmx2g -Xms256m -Xss2m"
    # TCOD protocol knobs: simplification string and the simulator-internal
    # step cap (the episode is capped by max_episode_steps, below it).
    simplification: str = "easy"
    env_step_limit: int = 100
    # Env-step budget per episode (TCOD: 30); dataset-row metadata overrides.
    max_episode_steps: int = 30
    # Worker processes = max concurrent episodes (one warm JVM each).
    env_workers: int = 128
    # Per IPC op (load/step/release). A worker exceeding it is SIGKILLed and
    # respawned (its JVM with it); the affected episode 500s on its next op.
    env_call_timeout_s: float = 300.0
    # Child boot handshake budget (no JVM at boot: the env is built lazily).
    env_spawn_timeout_s: float = 180.0
    # Worker-side env factory "module:function" (tests point it at a fake).
    env_factory: str = "scienceworld_env._worker:make_env"
    # /reset waits this long for a free worker before answering 503.
    slot_wait_timeout_s: float = 600.0
    # ScienceWorld feedback is longer than ALFWorld's (room descriptions list
    # every object); the lists below are capped by max_admissible_commands.
    max_obs_chars: int = 4096
    max_admissible_commands: int = 60
    # Must exceed NEMO_GYM_RUN_TIMEOUT (1200 s in the recipes).
    session_ttl_s: float = 1800.0
    sweep_interval_s: float = 60.0
    # In-process mode only (env_workers=0): executor threads for blocking calls.
    max_concurrency: int = 32


@dataclass
class ScienceworldSessionState(BaseSessionState):
    episode: Any  # RemoteEpisode (workers) or EpisodeHost (in-process)
    task_name: str
    var_num: int
    task_desc: str
    templates: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    best_score: float = 0.0  # 0-100, running max of the simulator score
    score: float = 0.0  # last raw simulator score


def format_actions(templates: list[str], limit: int) -> str:
    """Compact TCOD rendering: single-quoted, ', '-joined, 'help' dropped."""
    items = [t for t in templates if t != "help"][:limit]
    return ", ".join(f"'{t}'" for t in items)


def format_objects(objects: list[str], limit: int) -> str:
    return ", ".join(f"'{o}'" for o in list(objects)[:limit])


def format_reset_observation(task_desc: str, obs: str, templates: list[str], objects: list[str], config) -> str:
    """Opening observation (TCOD SCIWORLD_TEMPLATE_NO_HIS body, minus the
    per-turn protocol lines the system prompt already states)."""
    return (
        f"Your ScienceWorld task is: {task_desc}\n"
        f"Your current observation is: {truncate_obs(obs.strip(), config.max_obs_chars)}\n"
        f"Available action commands: [{format_actions(templates, config.max_admissible_commands)}]\n"
        f"Available objects you can interact with: [{format_objects(objects, config.max_admissible_commands)}]"
    )


def format_step_observation(obs: str, templates: list[str], objects: list[str], config) -> str:
    """Per-step feedback: simulator text + the re-listed templates/objects
    (TCOD's per-step template re-lists both; objects change as the agent moves)."""
    return (
        f"{truncate_obs(obs.strip(), config.max_obs_chars)}\n"
        f"Available action commands: [{format_actions(templates, config.max_admissible_commands)}]\n"
        f"Available objects you can interact with: [{format_objects(objects, config.max_admissible_commands)}]"
    )


def _task_ref(metadata: dict) -> tuple[str, int, str | None]:
    """(task_name, var_num, split) from the run body; HTTP 400 on a bad row."""
    task_name = metadata.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        raise HTTPException(status_code=400, detail="Missing task_name in the run body.")
    var_num = metadata.get("var_num")
    if isinstance(var_num, bool) or not isinstance(var_num, int):
        try:
            var_num = int(var_num)  # jsonl rows may carry it as a numeric string
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Bad var_num: {metadata.get('var_num')!r}") from None
    split = metadata.get("split")
    if split is not None and not isinstance(split, str):
        raise HTTPException(status_code=400, detail=f"Bad split: {split!r}")
    error = validate_task(task_name, var_num, split)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return task_name, var_num, split


class ScienceworldResourcesServer(SessionGymnasiumServer):
    config: ScienceworldResourcesServerConfig

    def _post_init(self) -> None:
        # Created lazily on the first /reset: eager spawn here would also fire
        # in config dry-runs and venv-builder probes.
        self._procpool: WorkerPool | None = None
        self._procpool_init_lock = asyncio.Lock()
        # In-process mode only.
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrency, thread_name_prefix="sciworld")
        self._jar_logged = False

    # ------------------------------------------------------------------ reset
    def _validate_row(self, metadata: dict) -> None:
        # Before a reused cookie's previous episode is released (base contract).
        _task_ref(metadata)

    def _log_jar_once(self) -> None:
        if self._jar_logged:
            return
        self._jar_logged = True
        jar = self.config.jar_path
        if not jar:
            try:
                from scienceworld.constants import JAR_PATH  # type: ignore[import-not-found]

                jar = str(JAR_PATH)
            except Exception:
                logger.info("scienceworld jar: wheel default (path unavailable)")
                return
        try:
            digest = hashlib.sha256(Path(jar).read_bytes()).hexdigest()
        except OSError as e:
            logger.warning("scienceworld jar %s unreadable: %s", jar, e)
            return
        java_home = self.config.java_home or os.environ.get("JAVA_HOME", "")
        logger.info("scienceworld jar %s sha256=%s java_home=%s", jar, digest, java_home)

    async def _ensure_workers(self) -> None:
        # Built off-loop: WorkerPool.__init__ blocks on the children's ready
        # handshakes. A boot failure propagates (500) and is retried next /reset.
        if self.config.env_workers <= 0 or self._procpool is not None:
            return
        async with self._procpool_init_lock:
            if self._procpool is None:
                cfg = self.config
                self._procpool = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: WorkerPool(
                        cfg.env_workers,
                        cfg.env_call_timeout_s,
                        cfg.env_factory,
                        jar_path=cfg.jar_path,
                        simplification=cfg.simplification,
                        env_step_limit=cfg.env_step_limit,
                        java_home=cfg.java_home,
                        java_opts=cfg.java_opts,
                        spawn_ready_timeout_s=cfg.env_spawn_timeout_s,
                    ),
                )

    async def _load_remote(self, task_name: str, var_num: int):
        pool = self._procpool
        assert pool is not None
        try:
            idx, gen = await pool.acquire(self.config.slot_wait_timeout_s)
        except TimeoutError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        try:
            return await pool.load(idx, gen, task_name, var_num)
        except EnvWorkerDied as e:
            if isinstance(e.__cause__, TimeoutError):
                # A load that blew the whole per-op budget is a systemic stall
                # (wedged JVM start, dead jar mount); fail fast.
                raise
            # Respawned/parked slot: one retry lands on another slot.
            idx, gen = await pool.acquire(self.config.slot_wait_timeout_s)
            return await pool.load(idx, gen, task_name, var_num)

    def _load_inproc(self, task_name: str, var_num: int):
        cfg = self.config
        host = EpisodeHost(cfg.env_factory, cfg.jar_path, cfg.simplification, cfg.env_step_limit)
        obs, info = host.load(task_name, var_num)
        return host, obs, info

    async def _do_reset(
        self, metadata: dict, session_id: str, max_episode_steps: int
    ) -> tuple[ScienceworldSessionState, str, dict]:
        task_name, var_num, _split = _task_ref(metadata)
        self._log_jar_once()
        await self._ensure_workers()
        if self._procpool is not None:
            episode, obs, info = await self._load_remote(task_name, var_num)
        else:
            episode, obs, info = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._load_inproc, task_name, var_num
            )
        state = ScienceworldSessionState(
            episode=episode,
            task_name=task_name,
            var_num=var_num,
            task_desc=str(info.get("task_desc", "")),
            templates=list(info.get("templates") or []),
            objects=list(info.get("objects") or []),
            best_score=float(info.get("best_score", 0.0) or 0.0),
            score=float(info.get("score", 0.0) or 0.0),
            max_episode_steps=max_episode_steps,
        )
        observation = format_reset_observation(state.task_desc, obs, state.templates, state.objects, self.config)
        # Numeric info only (the agent forwards numerics into wandb env/*);
        # task identity rides as strings so it is visible on the wire but
        # never averaged.
        reset_info: dict[str, Any] = {"task_name": task_name, "var_num": str(var_num)}
        if self._procpool is not None:
            reset_info["env_worker_respawns"] = self._procpool.respawns
            reset_info["env_free_workers"] = self._procpool.free_slots
        return state, observation, reset_info

    # ------------------------------------------------------------------- step
    async def _do_step(self, state: ScienceworldSessionState, command: str) -> tuple[str, float, bool, dict]:
        if self._procpool is not None:
            obs, info = await state.episode.step(command)
        else:
            obs, info = await asyncio.get_running_loop().run_in_executor(self._executor, state.episode.step, command)
        state.score = float(info.get("score", 0.0) or 0.0)
        state.best_score = float(info.get("best_score", state.best_score) or 0.0)
        if info.get("objects") is not None:
            state.objects = list(info["objects"])
        if info.get("templates"):
            state.templates = list(info["templates"])
        if info.get("invalid"):
            state.invalid_actions += 1
        completed = bool(info.get("completed", False))
        state.won = completed
        terminated = bool(info.get("done", False)) or completed
        reward = float(info.get("best_delta", 0.0) or 0.0) / 100.0
        observation = format_step_observation(obs, state.templates, state.objects, self.config)
        if terminated:
            observation += "\nTask completed." if completed else "\nTask finished."
        return observation, reward, terminated, {}

    def _extra_info(self, state: ScienceworldSessionState) -> dict:
        # task_score = TCOD's episode reward (best score / 100); won = raw completion.
        return {"task_score": state.best_score / 100.0}

    def _format_fault_observation(self, state: ScienceworldSessionState) -> str:
        return format_fault_hint(self.config.action_tool_name, _FAULT_EXAMPLE)

    async def _release_state(self, state: ScienceworldSessionState) -> None:
        if self._procpool is not None:
            await state.episode.close()
        else:
            await asyncio.get_running_loop().run_in_executor(self._executor, state.episode.discard_env)


if __name__ == "__main__":
    ScienceworldResourcesServer.run_webserver()
