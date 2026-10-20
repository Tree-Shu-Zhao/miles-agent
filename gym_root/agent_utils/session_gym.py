"""Shared session-Gymnasium base for the env resources servers (alfworld_env, webshop_env, scienceworld_env).

One episode = one signed-cookie session (upstream GymnasiumServer machinery):
/reset creates the episode, /step plays the FIRST ``act`` tool call of the
model turn (structured ``{"command": ...}`` arguments; no text parsing), and
the base /step endpoint auto-closes the session on a terminal or truncated
step. This module owns everything that is env-agnostic about that protocol so
the two env servers only implement the env mechanics:

- ``SessionGymConfig``: the config fields every session env has.
- ``BaseSessionState``: counters, the per-session lock and ``last_access``.
- ``extract_command`` / ``truncate_obs`` / ``format_fault_hint``: the pure
  helpers both envs used to carry as copies.
- ``SessionGymnasiumServer``: the /reset and /step templates (session-id
  handling, ``max_episode_steps`` parsing, re-reset frees the old episode,
  per-session lock with liveness re-check, done-replay guard, format-fault
  hint, budget/truncation, failed-step eviction), the /close and /finalize
  endpoints and the TTL sweeper. Envs fill in ``_do_reset`` / ``_do_step`` /
  ``_release_state`` (and ``_settle_unfinished`` to score an agent-side stop).

Failure semantics (user decision, frozen): env feedback — invalid actions,
no-op clicks, bad purchases, low scores — is never filtered or reshaped here.
Only infrastructure failures raise: an exception out of ``_do_step`` evicts
the session and releases its resources immediately (instead of leaving a
half-played episode to the TTL sweeper) and propagates as a 5xx, which the
agent turns into ``run_transport_failed`` so the trainer drops the group.
"""

import asyncio
import json
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_utils.disconnect import RunToCompletionOnDisconnect
from fastapi import FastAPI, HTTPException, Request
from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY
from pydantic import BaseModel
from resources_servers.gymnasium import GymnasiumServer

logger = logging.getLogger("agent_utils.session_gym")


class FinalizeRequest(BaseModel):
    submission: str | None = None


class SessionGymConfig(BaseResourcesServerConfig):
    # Env-step budget per episode; dataset-row metadata overrides per row.
    # Envs give the default (alfworld 30, webshop 15).
    max_episode_steps: int
    # Name of the action tool the dataset rows declare (prepare_data.py).
    action_tool_name: str = "act"
    # Context safety: episodes must stay inside the model window.
    max_obs_chars: int = 2048
    max_admissible_commands: int = 60
    # Must exceed NEMO_GYM_RUN_TIMEOUT (1200 s in the recipes) so a live
    # episode is never evicted mid-run.
    session_ttl_s: float = 1800.0
    # <= 0 disables the sweeper (unit tests).
    sweep_interval_s: float = 60.0


@dataclass(kw_only=True)
class BaseSessionState:
    """Per-episode bookkeeping; envs subclass it with their handle fields
    (alfworld: env/game_file; webshop: engine/task/avail/task_score). The base
    fields are keyword-only so subclasses can add positional fields freely."""

    max_episode_steps: int
    step_count: int = 0
    done: bool = False
    won: bool = False
    invalid_actions: int = 0
    action_format_errors: int = 0
    last_access: float = field(default_factory=time.monotonic)
    # Serializes ops on one session: the env step (step() holds it), releasing
    # the env resources (_release_session takes it after popping, before
    # parking/closing), and duplicate/late requests. step() must re-check
    # liveness after acquiring it — the session may have been popped
    # (auto-close, sweeper, /close) while it waited, and by then the resources
    # may already belong to another session.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def extract_command(response: NeMoGymResponse, tool_name: str) -> str | None:
    """Command from the FIRST function_call item; None on any format fault
    (no call, wrong tool name, bad JSON args, missing/empty command).

    First-call-wins mirrors env_loop_agent, which replies the observation to
    the first tool_call_id and answers extras with an "ignored" notice.
    """
    calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
    if not calls:
        return None
    call = calls[0]
    if call.name != tool_name:
        return None
    try:
        args = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    command = args.get("command") if isinstance(args, dict) else None
    if not isinstance(command, str) or not command.strip():
        return None
    return command.strip()


def truncate_obs(text: str, max_chars: int) -> str:
    """Hard cap on a model-visible observation, with an explicit marker."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n[observation truncated]"
    return text


def format_fault_hint(tool_name: str, example_arg: str) -> str:
    """Model-visible retry hint for a turn with no usable tool call.

    Names the tool and what its command argument takes, never a markup: the
    server does not know the model, whose call format is its own (its chat
    template; the example call in the system prompt). A markup here is one
    family's, and models copy the hint literally (live-probed on Qwen2.5,
    2026-08-31), so another family would be taught a reply its parser cannot
    read. Each env keeps its wording by choosing the prefix and
    ``example_arg`` in ``_format_fault_observation``.
    """
    return (
        f"Invalid: no {tool_name} tool call found. Reply with exactly one call to the {tool_name} tool, "
        f"its command argument set to {example_arg}, in the tool-call format of your instructions."
    )


class SessionGymnasiumServer(RunToCompletionOnDisconnect, GymnasiumServer):
    """Gymnasium server with in-process per-session state and the shared
    /reset, /step, /close and TTL-sweeper machinery. Subclasses implement
    ``_do_reset``, ``_do_step`` and ``_release_state`` (see each docstring).

    Per-episode state objects live in the base class's ``session_state`` dict
    (typed Dict[str, Any] there precisely because state objects — these hold
    an asyncio.Lock — are not pydantic-schemable).
    """

    config: SessionGymConfig

    def model_post_init(self, context) -> None:
        # Session state is an in-process dict: >1 uvicorn worker would shard
        # /reset and /step of one episode onto different processes.
        num_workers = getattr(self.config, "num_workers", None)
        assert num_workers in (None, 1), (
            f"{type(self).__name__} keeps per-session state in-process and requires num_workers=1, got {num_workers}"
        )
        # Created lazily on the first /reset (see _ensure_sweeper).
        self._sweeper_task: asyncio.Task | None = None
        self._post_init()

    def _post_init(self) -> None:
        """Env hook: build executors/pools/semaphores. No running loop yet."""

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()  # /reset, /step, /aggregate_metrics + session middleware
        app.post("/close")(self._close_endpoint)
        app.post("/finalize")(self._finalize_endpoint)
        return app

    # ------------------------------------------------------------------ hooks
    def _validate_row(self, metadata: dict) -> None:
        """Optional cheap row validation (raise HTTPException 400) that runs
        BEFORE a reused cookie's previous episode is released, so a bad re-reset
        leaves the live episode untouched. ``_do_reset`` must still validate
        (or re-validate) before acquiring resources."""

    @abstractmethod
    async def _do_reset(
        self, metadata: dict, session_id: str, max_episode_steps: int
    ) -> tuple[BaseSessionState, str | None, dict]:
        """Create the episode: validate the row (raise HTTPException 400 for a
        bad row, 5xx for an infrastructure failure), acquire the env resources
        and return ``(state, observation, info)``. The base registers ``state``
        under ``session_id``. Only counters worth forwarding to wandb env/*
        belong in ``info`` (the agent merges NUMERIC info values into its /run
        response; non-numerics are ignored there)."""

    @abstractmethod
    async def _do_step(self, state: Any, command: str) -> tuple[str | None, float, bool, dict]:
        """Apply ``command`` and return ``(observation, reward, terminated,
        info_extra)``; ``info_extra`` is merged over ``_session_info``. Runs
        under the session lock. Update env counters (``invalid_actions``,
        ``won``, ...) on ``state``; the base handles ``done``, ``step_count``,
        truncation and ``last_access``. Raise for infrastructure failures: the
        base evicts the session, releases it and re-raises."""

    @abstractmethod
    async def _release_state(self, state: Any) -> None:
        """Free the episode's resources (park/close the env, close the engine
        session). Called under the session lock, at most once per state, after
        it left ``session_state``; exceptions are logged and swallowed."""

    def _extra_info(self, state: Any) -> dict:
        """Env-specific keys merged into every step info (alfworld: lost;
        webshop: task_score)."""
        return {}

    async def _settle_unfinished(self, state: Any, submission: str | None) -> tuple[float, dict] | None:
        """The agent stopped on its own (/finalize, e.g. a black-box harness
        that exited). An env that scores such a stop returns ``(reward,
        info_extra)``; None keeps reward 0. ``submission`` is the agent's final
        answer, if it gave one. Runs under the session lock; raise for
        infrastructure failures (the base evicts the session)."""
        return None

    def _format_fault_observation(self, state: Any) -> str:
        """Observation returned for a turn with no usable action tool call."""
        return format_fault_hint(self.config.action_tool_name, "<one admissible command>")

    # ------------------------------------------------------------------ reset
    async def reset(self, metadata: dict, session_id: str | None = None) -> tuple[str | None, dict]:
        self._ensure_sweeper()
        if session_id is None:
            raise HTTPException(status_code=400, detail="Missing session id.")
        # Validate BEFORE acquiring an env: a failure after acquisition but
        # before session registration would leak the env (worker-resident
        # envs have no GC fallback, unlike an in-process one).
        try:
            max_episode_steps = int(metadata.get("max_episode_steps") or self.config.max_episode_steps)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail=f"Bad max_episode_steps: {metadata.get('max_episode_steps')!r}"
            ) from None
        self._validate_row(metadata)
        # A reused cookie re-resets: free the previous episode first (never
        # leak its env; a pooled env may be taken right back by this reset).
        await self._release_session(session_id)
        state, observation, info = await self._do_reset(metadata, session_id, max_episode_steps)
        self.session_state[session_id] = state
        return observation, info

    # ------------------------------------------------------------------- step
    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: str | None = None
    ) -> tuple[str | None, float, bool, bool, dict]:
        sid = session_id or ""
        state = self.session_state.get(sid)
        if state is None:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")

        async with state.lock:
            if self.session_state.get(sid) is not state:
                # Closed while we waited for the lock (terminal/truncated
                # auto-close, sweeper, /close, re-reset): the resources may
                # already be freed or owned by another session — never step.
                return None, 0.0, True, False, self._session_info(state)
            state.last_access = time.monotonic()
            if state.done:
                # Defensive replay of a finished episode (the base endpoint
                # already closed the session on the terminal step).
                return None, 0.0, True, False, self._session_info(state)

            command = extract_command(action, self.config.action_tool_name)
            state.step_count += 1
            if command is None:
                # Format fault: a model-visible retry hint plus a counter,
                # never a crash and no env step (there is no command); the
                # step still spends episode budget.
                state.action_format_errors += 1
                obs = self._format_fault_observation(state)
                truncated = state.step_count >= state.max_episode_steps
                return obs, 0.0, False, truncated, self._session_info(state)

            try:
                obs, reward, terminated, info_extra = await self._do_step(state, command)
            except Exception:
                if self._evict(sid, state):
                    await self._release_state_safely(state)
                raise
            state.done = bool(terminated)
            state.last_access = time.monotonic()
            truncated = (not state.done) and state.step_count >= state.max_episode_steps
            return obs, reward, state.done, truncated, self._session_info(state) | info_extra

    def _evict(self, sid: str, state: Any) -> bool:
        """Infrastructure failure mid-episode: drop the session now instead of
        letting the half-played episode linger to the TTL sweeper. True = we
        popped it and own its release; a concurrent close that popped it first
        (waiting on the session lock) releases it once we return."""
        if self.session_state.get(sid) is not state:
            return False
        del self.session_state[sid]
        return True

    def _session_info(self, state: Any) -> dict:
        return {
            "won": state.won,
            "step_count": state.step_count,
            "invalid_actions": state.invalid_actions,
            "action_format_errors": state.action_format_errors,
        } | self._extra_info(state)

    # ------------------------------------------------------------------ close
    async def _close_endpoint(self, request: Request) -> dict:
        session_id = request.session.get(SESSION_ID_KEY)
        existed = (session_id or "") in self.session_state
        await self.close_session(session_id)
        return {"closed": existed}

    async def _finalize_endpoint(self, request: Request, body: FinalizeRequest | None = None) -> dict:
        submission = body.submission if body is not None else None
        return await self.finalize_session(request.session.get(SESSION_ID_KEY), submission)

    async def finalize_session(self, session_id: str | None, submission: str | None = None) -> dict:
        """Agent-side stop: settle the live episode (``_settle_unfinished``),
        then close it. Returns the reward the stop earns and its info."""
        sid = session_id or ""
        reward, info = 0.0, {}
        state = self.session_state.get(sid)
        if state is not None:
            async with state.lock:
                if self.session_state.get(sid) is state and not state.done:
                    state.done = True
                    try:
                        settled = await self._settle_unfinished(state, submission)
                    except Exception:
                        if self._evict(sid, state):
                            await self._release_state_safely(state)
                        raise
                    if settled is not None:
                        reward, extra = settled
                        info = self._session_info(state) | extra
                    else:
                        info = self._session_info(state)
        await self.close_session(session_id)
        return {"reward": reward, "info": info}

    async def close_session(self, session_id: str | None) -> None:
        await self._release_session(session_id)
        await super().close_session(session_id)

    async def _release_session(self, session_id: str | None) -> None:
        state = self.session_state.pop(session_id or "", None)
        if state is None:
            return
        # Pop first (new resets on the same cookie get fresh state), then wait
        # for any in-flight step before releasing: env backends are not safe
        # against step racing close/reset on one env, and a pooled env may be
        # handed to another session the moment it is parked.
        async with state.lock:
            await self._release_state_safely(state)

    async def _release_state_safely(self, state: Any) -> None:
        try:
            await self._release_state(state)
        except Exception:
            # Best effort: a failed release must never mask the caller's
            # outcome (the env/engine TTL-evicts orphans on its side).
            logger.warning("releasing session resources failed", exc_info=True)

    # ---------------------------------------------------------------- sweeper
    def _ensure_sweeper(self) -> None:
        # Started lazily from the serving loop on the first /reset: newer
        # starlette is lifespan-only (no add_event_handler), and
        # model_post_init has no running loop yet.
        if self.config.sweep_interval_s <= 0:
            return
        task = self._sweeper_task
        if task is None or task.done():
            self._sweeper_task = asyncio.get_running_loop().create_task(self._sweep_loop())

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.sweep_interval_s)
            try:
                evicted = await self.sweep_stale_sessions()
            except Exception:
                logger.exception("session sweep failed; retrying next interval")
                continue
            if evicted:
                logger.info("swept %d stale session(s) (ttl=%.0fs)", evicted, self.config.session_ttl_s)

    async def sweep_stale_sessions(self) -> int:
        """Close every session idle longer than session_ttl_s; returns how many."""
        cutoff = time.monotonic() - self.config.session_ttl_s
        stale = [sid for sid, state in self.session_state.items() if state.last_access < cutoff]
        for sid in stale:
            try:
                await self.close_session(sid)
            except Exception:
                # One bad session must not stop the sweep for the others.
                logger.exception("closing stale session %s failed", sid)
        return len(stale)
