"""WebShop resources server — Gymnasium protocol (/reset + /step), driven by env_loop_agent.

Thin py3.13 nemo-gym adapter over the py3.10 WebShop engine server (which owns
the catalog + Lucene/JVM search + spaCy reward; see env_servers/webshop/ and
docs/webshop_integration.md for why the two are separate processes). The
session/step/sweeper template is the shared agent_utils.session_gym; this
adapter owns training-semantics presentation only:

- /reset: proxy the engine, extract the goal instruction, render the initial
  page + admissible actions as the opening observation. Per-episode identity
  rides the /run body into here: ``goal_id`` (the goal's CONTENT id, see
  agent_utils/webshop_goals.py — what the frozen datasets carry) or the legacy
  positional ``task_id`` (engine list index), plus ``split``; the signed
  session cookie is the episode handle thereafter. The engine resolves
  ``goal_id`` itself and 400s an id it does not have: that is a dataset/engine
  goal-set mismatch (a row error, surfaced as 400 here), never infra.
- /step: read the FIRST ``act`` tool call in the model turn (structured
  ``{"command": ...}``; no text parsing). An admissible command is proxied
  UNCHANGED to the engine and the next page + admissible actions rendered.
  Strict Gymnasium endings: terminated on the engine's ``done``, truncated at
  ``max_episode_steps``. Reward is the raw WebShop score (``info.task_score``)
  at the terminal step, 0 otherwise — the neutral env truth;
  recipes/webshop/reward.py owns the binary/dense remap.
- A non-admissible command (``invalid_action_feedback``): ``error`` (TCOD
  protocol, default) spends the turn WITHOUT stepping the engine and shows
  TCOD's error sentence as the observation (admissible list unchanged);
  ``noop`` forwards it and WebShop's unchanged page is the feedback
  (wstrain1/2). Either way it is counted in ``invalid_actions``.
- A turn with no ``act`` call costs a step (format error counter) and returns
  a retry hint; the engine is not stepped (there is no command).

Failure semantics (user decision): env feedback — non-admissible or no-op
actions, bad purchases, low scores — is never filtered or reshaped; it only
drives counters (``invalid_actions``). Infrastructure failures (engine
error/410/timeout, malformed engine payload) raise HTTP 502; the base evicts
the session on the spot, env_loop_agent reports ``run_transport_failed`` and
the trainer's dynamic-sampling filter drops the whole group for re-sampling.
Row errors (no goal reference, an engine 400 for an unknown ``goal_id`` / bad
``task_id``) raise HTTP 400 instead. The distinction is adapter-local: the
engine client never retries a 4xx and the adapter's status/log line carries
the engine's detail (which goal_id, which goals_digest), but downstream
env_loop_agent raises on any non-2xx /reset, nemo-gym turns that into a /run
500 and gym_bridge reports ``run_transport_failed`` = 1 exactly as for a
wedged engine, so the group is dropped by the same filter and the wandb metric
does not tell the two apart; the boot-time goals_digest gate is what keeps
row errors from happening at all.
Reserved search tokens (``<a>``/``<c>``/``<q>``/``<r>`` first in the query)
crash the upstream engine code; they are never admissible here (under
``noop`` the engine glue neutralizes them into the unchanged-page no-op).

Page rendering (``observation_mode``) must match the engine's
--observation-mode: ``text_rich`` (TCOD / original baselines, default) or
``text`` (verl-agent ``[SEP]`` form); see _format.py.

Sessions whose episode is killed upstream (policy 400s, trainer aborts, /run
timeouts) never see a terminal /step; a TTL sweeper closes them on the engine,
and env_loop_agent POSTs /close best-effort on non-terminal exits.

Unlike alfworld_env (heavy env resident in-process) this server holds no env
state beyond a per-episode EngineSession handle, so it stays trivially
horizontally-scalable; the engine tier owns concurrency (engine_workers).
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from agent_utils.session_gym import (
    BaseSessionState,
    SessionGymConfig,
    SessionGymnasiumServer,
    format_fault_hint,
)
from fastapi import HTTPException

from webshop_env._engine_client import EngineError, EngineSession
from webshop_env._format import (
    OBSERVATION_MODES,
    WebshopFormatError,
    extract_task,
    invalid_action_message,
    is_reserved_search,
    render_observation,
    render_page,
)

INVALID_FEEDBACK_MODES = ("error", "noop")


class WebshopResourcesServerConfig(SessionGymConfig):
    # Base URL of the py3.10 engine server (its internal worker pool fans out
    # across engines; this adapter proxies to the one URL, like wiki_search
    # proxies its retriever). Loopback + a port below a 9000+ ephemeral
    # range (a common ip_local_port_range on big nodes). Shell-side
    # source of truth: env_servers/webshop/defaults.sh; the training entry
    # overrides this via +webshop_env.resources_servers.webshop_env.engine_base_url.
    engine_base_url: str = "http://127.0.0.1:6200"
    # Env-step budget per episode; dataset-row metadata overrides per goal.
    max_episode_steps: int = 15
    # Page rendering the engine was started with (env_servers/webshop
    # --observation-mode; shell knob WEBSHOP_OBS_MODE feeds both tiers):
    # text_rich (TCOD / original baselines) or text (verl-agent [SEP] form).
    observation_mode: str = "text_rich"
    # Reply to a non-admissible command: error (TCOD protocol — turn spent, no
    # engine step, TCOD's error sentence as the page) or noop (forward it; the
    # unchanged page is the feedback). Shell knob WEBSHOP_INVALID_FEEDBACK.
    invalid_action_feedback: str = "error"
    # Bounds in-flight engine calls (protects the engine tier); the engine's
    # own worker pool is the real parallelism, this only caps the adapter.
    max_concurrency: int = 256
    # Per engine HTTP call. Must exceed the engine's --engine-call-timeout
    # (120 s) plus its worker-respawn margin (kill/join <= 5 s + the
    # replacement engine's boot, tens of seconds with --small), so a wedged
    # engine worker normally surfaces here as the engine's definitive 410/500
    # rather than a client-side ReadTimeout on the non-idempotent /step. A
    # slower replacement boot (--full catalog) exhausts this first; the
    # outcome is the same 502 -> group dropped, only the log differs. Raise
    # this or lower --engine-call-timeout to restore the margin.
    engine_timeout_s: float = 240.0
    # Must exceed NEMO_GYM_RUN_TIMEOUT (1200 s in recipes/webshop/run.py) so a
    # live episode is never evicted mid-run, and stay below the engine's
    # --session-ttl (2400 s) so the engine never evicts a session this adapter
    # still considers live.
    session_ttl_s: float = 1800.0
    # <= 0 disables the sweeper (unit tests).
    sweep_interval_s: float = 60.0


@dataclass
class WebshopSessionState(BaseSessionState):
    engine: EngineSession
    task: str  # goal instruction, extracted at reset (stable across the episode)
    avail: dict[str, Any]  # current page's available_actions dict
    task_score: float = 0.0


def _is_valid_command(command: str, avail: dict[str, Any]) -> bool:
    """Whether the command is admissible on the current page. WebShop silently
    no-ops non-admissible actions; this only drives the invalid-action counter
    (the penalty lives in recipes/webshop/reward.py)."""
    cmd = command.lower().strip()
    if not cmd.endswith("]"):
        return False
    if cmd.startswith("search[") and avail.get("has_search_bar"):
        return True
    if cmd.startswith("click["):
        target = cmd[len("click[") : -1]
        # Upstream step() no-ops click[search] unconditionally (the search-bar
        # submit button is listed among the clickables but never acts), so it
        # is not admissible even though the raw clickables contain it.
        return target != "search" and target in {c.lower() for c in avail.get("clickables", [])}
    return False


# What the retry hint says the command argument takes on a turn without an act
# tool call (format_fault_hint; models copy the hint literally, keep it stable).
_FAULT_EXAMPLE = "<one admissible action, e.g. search[query] or click[button]>"
# Anything the engine can hand back that the renderer cannot interpret is a
# protocol fault (infra), never a model mistake.
_PAYLOAD_ERRORS = (WebshopFormatError, AttributeError, TypeError, ValueError)


class WebshopResourcesServer(SessionGymnasiumServer):
    config: WebshopResourcesServerConfig

    def _post_init(self) -> None:
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        if self.config.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {OBSERVATION_MODES}, got {self.config.observation_mode!r}"
            )
        if self.config.invalid_action_feedback not in INVALID_FEEDBACK_MODES:
            raise ValueError(
                f"invalid_action_feedback must be one of {INVALID_FEEDBACK_MODES}, "
                f"got {self.config.invalid_action_feedback!r}"
            )

    # ------------------------------------------------------------------ reset
    @staticmethod
    def _goal_ref(metadata: dict) -> tuple[str | None, Any]:
        """The row's goal reference as ``(goal_id, task_id)`` — exactly one is
        set. ``goal_id`` (non-empty str) wins: it is the content address the
        frozen datasets carry; ``task_id`` (engine list index) is the legacy
        positional form. Raises 400 when neither is usable."""
        goal_id = metadata.get("goal_id")
        if goal_id is not None and goal_id != "":
            if not isinstance(goal_id, str):
                # A non-string id is a corrupt row, not a legacy one: refusing
                # beats silently falling back to a task_id that may be stale.
                raise HTTPException(
                    status_code=400, detail=f"goal_id must be a string, got {type(goal_id).__name__}: {goal_id!r}"
                )
            return goal_id, None
        if "task_id" not in metadata:
            raise HTTPException(status_code=400, detail="Missing goal_id (or legacy task_id) in the run body.")
        return None, metadata["task_id"]

    def _validate_row(self, metadata: dict) -> None:
        # Runs before a reused cookie's live episode is released, so a bad
        # re-reset leaves it untouched (base contract).
        self._goal_ref(metadata)

    async def _do_reset(
        self, metadata: dict, session_id: str, max_episode_steps: int
    ) -> tuple[WebshopSessionState, str, dict]:
        goal_id, task_id = self._goal_ref(metadata)
        split = metadata.get("split", "train")

        engine = EngineSession(self.config.engine_base_url, self.config.engine_timeout_s)
        try:
            async with self._semaphore:
                obs, info = await engine.reset(task_id, split, goal_id=goal_id)
        except EngineError as e:
            if e.status == 400:
                # The engine's definitive verdict on the ROW (unknown goal_id =
                # the dataset and the engine disagree on catalog / goal code;
                # bad legacy index): a client error the engine never retries
                # and a retry could not fix. 400 here only stops the engine
                # client retrying and labels the adapter's response/log with the
                # engine's detail; downstream it is dropped like any failed
                # /reset (see the module docstring).
                raise HTTPException(status_code=400, detail=str(e)) from e
            # Dead/wedged engine, engine 5xx after retries, malformed body:
            # never an episode. 502 -> the agent reports run_transport_failed
            # and the group is re-sampled.
            raise HTTPException(status_code=502, detail=str(e)) from e
        try:
            task = extract_task(obs)
            avail = self._require_avail(info)
            # The opening observation surfaces the goal explicitly (the env
            # speaks first; the system prompt is task-agnostic) — TCOD's
            # "Your task is to: ..." line, whatever the page rendering does
            # with the instruction.
            observation = f"Your task is to: {task}.\n" + self._render(obs, task, avail, next_step=1)
        except _PAYLOAD_ERRORS as e:
            await engine.close()
            raise HTTPException(status_code=502, detail=f"malformed engine reset payload: {e}") from e

        state = WebshopSessionState(engine=engine, task=task, avail=avail, max_episode_steps=max_episode_steps)
        # Both ids are labels, not metrics: env_loop_agent forwards NUMERIC
        # info values into wandb env/*, so they stay strings (visible on the
        # wire for debugging, never averaged). goal_id is the row's content id
        # (or whatever the engine reports for a legacy row, usually nothing);
        # goal_idx is the engine's list index for this boot.
        echo_goal_id = goal_id if goal_id is not None else info.get("goal_id")
        echo_goal_idx = info.get("goal_idx", task_id)
        return (
            state,
            observation,
            {
                "goal_id": "" if echo_goal_id is None else str(echo_goal_id),
                "goal_idx": "" if echo_goal_idx is None else str(echo_goal_idx),
            },
        )

    # ------------------------------------------------------------------- step
    async def _do_step(self, state: WebshopSessionState, command: str) -> tuple[str, float, bool, dict]:
        # The base already counted this turn (step_count); the model's NEXT
        # action is step_count + 1.
        next_step = state.step_count + 1
        # Reserved search tokens are the one syntactically valid search upstream
        # crashes on; they are never admissible (under noop the engine glue
        # turns them into the unchanged-page no-op).
        if not _is_valid_command(command, state.avail) or is_reserved_search(command):
            state.invalid_actions += 1
            if self.config.invalid_action_feedback == "error":
                # TCOD protocol: the env is not stepped; the turn is spent and
                # the model reads TCOD's error sentence as its observation,
                # with the (unchanged) admissible list.
                observation = render_page(
                    invalid_action_message(command, state.avail),
                    state.avail,
                    next_step,
                    self.config.max_admissible_commands,
                )
                return observation, 0.0, False, {}
            # noop: forward UNCHANGED — WebShop itself no-ops what it cannot
            # apply and that unchanged page is the feedback the model sees.

        try:
            async with self._semaphore:
                raw_obs, _engine_reward, done, info = await state.engine.step(command)
        except EngineError as e:
            if e.status == 410:
                # The engine no longer has this episode (TTL, worker respawn):
                # a /close round trip would only 410 again.
                state.engine.forget()
            # Infra failure mid-episode: 502; the base evicts the session and
            # releases it (engine.close unless forgotten). A half-played
            # episode with a made-up reward is worse than a dropped one.
            raise HTTPException(status_code=502, detail=str(e)) from e

        try:
            avail = self._require_avail(info)
            task_score = float(info.get("task_score", 0.0))
            won = bool(info.get("won", False))
            observation = self._render(raw_obs, state.task, avail, next_step=next_step)
        except _PAYLOAD_ERRORS as e:
            raise HTTPException(status_code=502, detail=f"malformed engine step payload: {e}") from e
        state.avail = avail
        state.task_score = task_score
        state.won = won

        if done:
            # Terminal page precedes the vendored env's auto-reset; render it,
            # then the base endpoint closes the session (dropping the
            # auto-reset junk). Reward is the raw score at the buy step. The
            # engine already dropped the finished episode, so mark the client
            # session closed without a redundant /close round trip.
            observation += "\nTask completed." if won else "\nTask finished."
            state.engine.forget()
            return observation, task_score, True, {}
        return observation, 0.0, False, {}

    def _extra_info(self, state: WebshopSessionState) -> dict:
        # Success is ``won`` (score == 1.0) only. WebShop's ``done`` flag (buy
        # now reached, any score) is deliberately NOT exported: it is not a
        # success measure (user decision, 2026-09-04).
        return {"task_score": state.task_score}

    def _format_fault_observation(self, state: WebshopSessionState) -> str:
        return format_fault_hint(self.config.action_tool_name, _FAULT_EXAMPLE)

    async def _release_state(self, state: WebshopSessionState) -> None:
        # No-op after forget() (terminal step / 410); otherwise one /close so
        # the engine frees the episode now instead of at its TTL.
        await state.engine.close()

    # -------------------------------------------------------------- helpers
    def _render(self, raw_obs: str, task: str, avail: dict, *, next_step: int) -> str:
        return render_observation(
            raw_obs,
            task,
            avail,
            self.config.max_obs_chars,
            self.config.max_admissible_commands,
            observation_mode=self.config.observation_mode,
            next_step=next_step,
        )

    @staticmethod
    def _require_avail(info: dict) -> dict:
        avail = info.get("available_actions")
        if not isinstance(avail, dict):
            raise WebshopFormatError("engine response missing info['available_actions']")
        return avail


if __name__ == "__main__":
    WebshopResourcesServer.run_webserver()
