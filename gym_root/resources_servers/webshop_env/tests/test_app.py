"""webshop_env adapter behavior against a fake engine session (no pyserini/JVM).

The adapter's only outbound dependency is EngineSession (the HTTP client to the
py3.10 engine); we monkeypatch it with a scripted fake so these run with just
nemo-gym. The engine's own logic has its own tests (env_servers/webshop/tests);
the real wire contract is covered by tests/test_contract.py.
"""

import asyncio
import json
import time

import pytest

pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")

from unittest.mock import MagicMock  # noqa: E402

import webshop_env.app as app_module  # noqa: E402
from agent_utils.webshop_goals import goal_id as content_goal_id  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponse  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from webshop_env._engine_client import EngineError  # noqa: E402
from webshop_env._format import (  # noqa: E402
    INVALID_CLICK_PREFIX,
    INVALID_GENERIC,
    INVALID_SEARCH_EMPTY,
    INVALID_SEARCH_NO_BAR,
    available_commands,
    extract_task,
    format_obs,
    invalid_action_message,
    is_reserved_search,
    render_observation,
)
from webshop_env.app import WebshopResourcesServer, WebshopResourcesServerConfig, _is_valid_command  # noqa: E402

SID = "sess-1"
GOAL = "i need a red ceramic mug, and price lower than 30.00 dollars"
# The fake engine's goal list, addressed by CONTENT like the real one: ids come
# from the shared module so a test row's goal_id has the production shape.
FAKE_GOALS = [
    {
        "asin": f"B{i:05d}",
        "name": f"product {i}",
        "category": "kitchen",
        "query": "ceramic mug",
        "product_category": "Home > Kitchen",
        "instruction_text": f"fake goal {i}",
        "attributes": [f"attr{i % 3}"],
        "goal_options": {"size": str(i % 2)},
        "price_upper": float(10 + i),
    }
    for i in range(3)
]
FAKE_GOAL_INDEX = {content_goal_id(g): i for i, g in enumerate(FAKE_GOALS)}
GID = content_goal_id(FAKE_GOALS[1])  # -> goal_idx 1
UNKNOWN_GID = "0" * 16
INITIAL_OBS = f"Amazon Shopping Game [SEP] Instruction: [SEP] {GOAL} [SEP] Search"
# Upstream's real initial page: the search bar plus its "search" submit button.
INITIAL_AVAIL = {"has_search_bar": True, "clickables": ["search"]}
RESULTS_OBS = f"Instruction: [SEP] {GOAL} [SEP] Back to Search [SEP] Page 1 (Total results: 5) [SEP] B0001 [SEP] $10.9"
RESULTS_AVAIL = {"has_search_bar": False, "clickables": ["back to search", "b0001"]}
ITEM_OBS = f"Instruction: [SEP] {GOAL} [SEP] red ceramic mug [SEP] Price: $10.99 [SEP] Buy Now"
ITEM_AVAIL = {"has_search_bar": False, "clickables": ["buy now"]}
DONE_OBS = f"Instruction: [SEP] {GOAL} [SEP] Thank you for shopping with us!"
POST_RESET_AVAIL = {"has_search_bar": True, "clickables": []}
# The same pages in the vendored env's text_rich rendering (TCOD / original
# WebShop baselines): one visible text per line, buttons marked up.
RICH_INITIAL_OBS = f"WebShop\nInstruction:\n{GOAL}\n[button] Search [button_]\n"
RICH_RESULTS_OBS = (
    f"Instruction:\n{GOAL}\n[button] Back to Search [button_]\nPage 1 (Total results: 5)\n"
    "[button] Next > [button_]\n\n[button] B0001 [button_]\nred ceramic mug\n$10.99\n"
)
RICH_RESULTS_STEP = (
    RICH_RESULTS_OBS,
    0.0,
    False,
    {"available_actions": RESULTS_AVAIL, "task_score": 0.0, "won": False},
)

RESULTS_STEP = (RESULTS_OBS, 0.0, False, {"available_actions": RESULTS_AVAIL, "task_score": 0.0, "won": False})
# A search that lands back on a search page (keeps the search bar admissible).
SEARCH_PAGE_STEP = (INITIAL_OBS, 0.0, False, {"available_actions": INITIAL_AVAIL, "task_score": 0.0, "won": False})
# The wire reward here is DELIBERATELY not the task_score. The real engine puts the raw
# score in both slots (pinned in tests/test_contract.py), so only a divergent fake can
# prove the adapter reads info["task_score"] and ignores the reward slot.
WIN_STEP = (DONE_OBS, 10.0, True, {"available_actions": POST_RESET_AVAIL, "task_score": 1.0, "won": True})

# Exact model-visible retry hint (models copy it literally; pinned). It names
# no tool-call markup: that is the model's own (its chat template).
EXPECTED_FAULT_HINT = (
    "Invalid: no act tool call found. Reply with exactly one call to the act tool, its command argument "
    "set to <one admissible action, e.g. search[query] or click[button]>, in the tool-call format of "
    "your instructions."
)


class FakeEngine:
    """Scripted EngineSession: reset -> initial page; step consumes a queue of
    (obs, reward, done, info) tuples (an exception instance is raised instead).
    Mirrors the real client's close/forget bookkeeping: ``closed`` means an
    HTTP /close was sent, ``forgotten`` means forget() marked it closed without
    one, and close() after either is a no-op. A ``goal_id`` is resolved against
    FAKE_GOAL_INDEX; an unknown one raises what the real client raises for the
    engine's 400 (EngineError, status=400)."""

    def __init__(self, step_script=(), *, reset_exc=None, reset_info=None, reset_obs=None, close_exc=None):
        self._steps = list(step_script)
        self.reset_exc = reset_exc
        self.reset_info = reset_info
        self.reset_obs = INITIAL_OBS if reset_obs is None else reset_obs
        self.close_exc = close_exc
        self.actions: list[str] = []
        self.reset_calls: list[dict] = []
        self.closed = False
        self.forgotten = False
        self.close_calls = 0
        self._closed = False
        self.step_gate: asyncio.Event | None = None  # when set, step() blocks until it fires
        self.order: list[str] = []

    async def reset(self, task_id=None, split="train", goal_id=None):
        self.reset_calls.append({"task_id": task_id, "split": split, "goal_id": goal_id})
        if self.reset_exc is not None:
            raise self.reset_exc
        if goal_id is not None:
            if goal_id not in FAKE_GOAL_INDEX:
                raise EngineError(
                    f"engine http://fake/reset failed (status=400): goal_id {goal_id!r} is not in the engine goal "
                    "list (goals_digest 0123456789abcdef): the dataset and the engine disagree on catalog / goal code",
                    status=400,
                )
            goal_idx = FAKE_GOAL_INDEX[goal_id]
        else:
            goal_idx = task_id
        info = {"available_actions": INITIAL_AVAIL, "task_score": 0.0, "won": False, "goal_idx": goal_idx}
        if self.reset_info is not None:
            info = self.reset_info
        return self.reset_obs, info

    async def step(self, action):
        self.actions.append(action)
        self.order.append("step_enter")
        if self.step_gate is not None:
            await self.step_gate.wait()
        self.order.append("step_exit")
        item = self._steps.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def forget(self):
        self.forgotten = True
        self._closed = True

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1
        self.order.append("close")
        if self.close_exc is not None:
            raise self.close_exc
        self.closed = True


def _config(**overrides) -> WebshopResourcesServerConfig:
    overrides.setdefault("sweep_interval_s", 0.0)  # no background task per test loop
    # The fixtures above are the verl-agent [SEP] rendering and most tests pin
    # the forward-everything (wstrain1/2) feedback; the TCOD defaults
    # (text_rich + error feedback) have their own section below.
    overrides.setdefault("observation_mode", "text")
    overrides.setdefault("invalid_action_feedback", "noop")
    return WebshopResourcesServerConfig(host="0.0.0.0", port=9021, entrypoint="", name="webshop_env", **overrides)


def _server(
    monkeypatch, step_script=(), *, engine_kwargs=None, **overrides
) -> tuple[WebshopResourcesServer, FakeEngine]:
    """Server whose every reset hands out the SAME fake engine (one episode per test)."""
    engine = FakeEngine(step_script, **(engine_kwargs or {}))
    monkeypatch.setattr(app_module, "EngineSession", lambda *a, **k: engine)
    server = WebshopResourcesServer(config=_config(**overrides), server_client=MagicMock(spec=ServerClient))
    return server, engine


def _server_per_reset(monkeypatch, step_script=(), **overrides) -> tuple[WebshopResourcesServer, list[FakeEngine]]:
    """Server whose every reset creates a fresh fake engine (multi-episode tests)."""
    engines: list[FakeEngine] = []

    def factory(*a, **k):
        engines.append(FakeEngine(step_script))
        return engines[-1]

    monkeypatch.setattr(app_module, "EngineSession", factory)
    server = WebshopResourcesServer(config=_config(**overrides), server_client=MagicMock(spec=ServerClient))
    return server, engines


def _response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "t",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": output,
        }
    )


def _act(command: str, *, name: str = "act") -> NeMoGymResponse:
    return _response(
        [
            {
                "type": "function_call",
                "id": "fc_0",
                "call_id": "c1",
                "name": name,
                "arguments": json.dumps({"command": command}),
                "status": "completed",
            }
        ]
    )


def _text(text: str) -> NeMoGymResponse:
    return _response(
        [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "msg_0",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )


def _status_410(msg: str = "session gone") -> EngineError:
    return EngineError(msg, status=410)  # the engine's definitive "episode already dropped"


# ──────────────────────────────────────────────────────────
# reset
# ──────────────────────────────────────────────────────────


async def test_reset_surfaces_goal_and_admissible(monkeypatch):
    server, _ = _server(monkeypatch)
    obs, info = await server.reset({"task_id": 7, "split": "train"}, SID)
    assert f"Your task is to: {GOAL}." in obs
    assert "\nsearch[<query>]\n" in obs  # admissible actions rendered (TCOD form: one per line)
    assert info["goal_idx"] == "7"
    assert SID in server.session_state


async def test_reset_by_goal_id_sends_field_and_needs_no_task_id(monkeypatch):
    # Content-addressed row (what the frozen datasets carry): goal_id goes to
    # the engine as its own field, no task_id required, and the engine's
    # resolved list index comes back as goal_idx.
    server, engine = _server(monkeypatch)
    obs, info = await server.reset({"goal_id": GID, "split": "eval"}, SID)
    assert engine.reset_calls == [{"task_id": None, "split": "eval", "goal_id": GID}]
    assert f"Your task is to: {GOAL}." in obs
    assert info == {"goal_id": GID, "goal_idx": "1"}
    assert SID in server.session_state


async def test_reset_legacy_task_id_path_unchanged(monkeypatch):
    # A positional row still resets by index; goal_id is not put on the wire
    # (the engine client omits it) and echoes empty.
    server, engine = _server(monkeypatch)
    _, info = await server.reset({"task_id": 7}, SID)
    assert engine.reset_calls == [{"task_id": 7, "split": "train", "goal_id": None}]
    assert info == {"goal_id": "", "goal_idx": "7"}


async def test_reset_goal_id_wins_over_task_id(monkeypatch):
    # A row carrying both: the content id is authoritative, the stale position
    # is not forwarded (it may point at another question on this engine).
    server, engine = _server(monkeypatch)
    _, info = await server.reset({"goal_id": GID, "task_id": 7}, SID)
    assert engine.reset_calls == [{"task_id": None, "split": "train", "goal_id": GID}]
    assert info == {"goal_id": GID, "goal_idx": "1"}


@pytest.mark.parametrize("row", [{"task_id": 7}, {"goal_id": GID}])
async def test_reset_info_ids_are_strings_never_numeric(monkeypatch, row):
    # env_loop_agent forwards every numeric info value into wandb env/*; ids
    # averaged over samples are noise, so both stay strings.
    server, _ = _server(monkeypatch)
    _, info = await server.reset(row, SID)
    assert set(info) == {"goal_id", "goal_idx"}
    assert all(isinstance(v, str) for v in info.values())


async def test_reset_unknown_goal_id_is_400_not_502_no_session(monkeypatch):
    # The engine's 400 for an id it does not have is a dataset/engine goal-set
    # mismatch: a row error (400), never a transport failure (502).
    server, engine = _server(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        await server.reset({"goal_id": UNKNOWN_GID}, SID)
    assert ei.value.status_code == 400
    assert "is not in the engine goal list" in ei.value.detail
    assert isinstance(ei.value.__cause__, EngineError) and ei.value.__cause__.status == 400
    # exactly one engine call from _do_reset (no adapter-level retry); the
    # client's own 4xx no-retry is pinned in test_contract.py
    assert len(engine.reset_calls) == 1
    assert SID not in server.session_state
    assert engine.closed is False  # nothing to close: the engine opened no episode


async def test_engine_5xx_on_reset_stays_502(monkeypatch):
    # Only the engine's definitive 400 is a row error; a 5xx after retries is
    # still infra (group re-sampled).
    server, _ = _server(monkeypatch, engine_kwargs={"reset_exc": EngineError("engine 503 after retries", status=503)})
    with pytest.raises(HTTPException) as ei:
        await server.reset({"goal_id": GID}, SID)
    assert ei.value.status_code == 502
    assert SID not in server.session_state


@pytest.mark.parametrize("row", [{"split": "train"}, {"goal_id": ""}, {"goal_id": None}])
async def test_reset_requires_goal_id_or_task_id(monkeypatch, row):
    server, engine = _server(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        await server.reset(row, SID)
    assert ei.value.status_code == 400
    assert ei.value.detail == "Missing goal_id (or legacy task_id) in the run body."
    assert engine.reset_calls == [] and SID not in server.session_state


async def test_reset_non_string_goal_id_is_400(monkeypatch):
    # A numeric goal_id is a corrupt row, not a legacy one: refuse rather than
    # silently fall back to a possibly stale task_id.
    server, engine = _server(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        await server.reset({"goal_id": 123, "task_id": 7}, SID)
    assert ei.value.status_code == 400 and "goal_id must be a string" in ei.value.detail
    assert engine.reset_calls == []


async def test_bad_re_reset_leaves_live_episode_untouched(monkeypatch):
    # Row validation runs before the reused cookie's episode is released, so a
    # malformed re-reset cannot kill a live episode.
    server, engines = _server_per_reset(monkeypatch)
    await server.reset({"goal_id": GID}, SID)
    with pytest.raises(HTTPException) as ei:
        await server.reset({"split": "train"}, SID)
    assert ei.value.status_code == 400
    assert len(engines) == 1 and engines[0].closed is False
    assert server.session_state[SID].engine is engines[0]


async def test_initial_admissible_has_no_click_search(monkeypatch):
    # The search-bar submit button is a no-op upstream: not an action.
    server, _ = _server(monkeypatch)
    obs, _ = await server.reset({"task_id": 7}, SID)
    assert "click[search]" not in obs
    assert "\nsearch[<query>]\n" in obs


async def test_reset_requires_session_id(monkeypatch):
    server, _ = _server(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        await server.reset({"task_id": 7}, None)
    assert ei.value.status_code == 400


async def test_reset_bad_max_episode_steps_400(monkeypatch):
    server, engine = _server(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        await server.reset({"task_id": 7, "max_episode_steps": "lots"}, SID)
    assert ei.value.status_code == 400
    assert engine.actions == [] and SID not in server.session_state


async def test_engine_error_on_reset_is_502_and_no_session(monkeypatch):
    server, engine = _server(monkeypatch, engine_kwargs={"reset_exc": EngineError("engine down")})
    with pytest.raises(HTTPException) as ei:
        await server.reset({"task_id": 7}, SID)
    assert ei.value.status_code == 502
    assert SID not in server.session_state
    assert engine.closed is False  # nothing to close: the engine never opened an episode


async def test_malformed_reset_payload_is_502_and_engine_closed(monkeypatch):
    server, engine = _server(monkeypatch, engine_kwargs={"reset_info": {"task_score": 0.0}})  # no available_actions
    with pytest.raises(HTTPException) as ei:
        await server.reset({"task_id": 7}, SID)
    assert ei.value.status_code == 502
    assert SID not in server.session_state
    assert engine.closed is True  # the engine DID open an episode; free it now


async def test_reused_cookie_re_resets_and_closes_old_engine(monkeypatch):
    server, engines = _server_per_reset(monkeypatch)
    await server.reset({"task_id": 7}, SID)
    await server.reset({"task_id": 8}, SID)
    assert len(engines) == 2
    assert engines[0].closed is True and engines[1].closed is False
    assert server.session_state[SID].engine is engines[1]


# ──────────────────────────────────────────────────────────
# step
# ──────────────────────────────────────────────────────────


async def test_step_without_reset_400(monkeypatch):
    server, _ = _server(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        await server.step(_act("search[mug]"), {}, SID)
    assert ei.value.status_code == 400


async def test_step_proxies_command_and_formats(monkeypatch):
    server, engine = _server(monkeypatch, [RESULTS_STEP])
    await server.reset({"task_id": 7}, SID)
    obs, reward, term, trunc, info = await server.step(_act("search[red ceramic mug]"), {}, SID)
    assert engine.actions == ["search[red ceramic mug]"]
    assert "\nclick[b0001]\n" in obs
    assert obs.startswith("You are now at step 2 and your current observation is: ")
    assert (reward, term, trunc) == (0.0, False, False)
    assert info["step_count"] == 1 and info["invalid_actions"] == 0
    assert info["task_score"] == 0.0 and info["won"] is False


async def test_buy_now_terminal_reward_is_task_score(monkeypatch):
    server, engine = _server(monkeypatch, [WIN_STEP])
    await server.reset({"task_id": 7}, SID)
    # put the session on the item page so buy now is admissible
    server.session_state[SID].avail = ITEM_AVAIL
    obs, reward, term, trunc, info = await server.step(_act("click[buy now]"), {}, SID)
    assert reward == 1.0  # info["task_score"], not the deliberately divergent wire reward slot
    assert term is True and trunc is False
    assert info["won"] is True and info["task_score"] == 1.0
    assert obs.endswith("Task completed.")
    # The engine dropped the finished episode itself: no /close round trip.
    assert engine.forgotten is True and engine.closed is False


async def test_imperfect_purchase_scores_partial(monkeypatch):
    step = (DONE_OBS, 0.0, True, {"available_actions": POST_RESET_AVAIL, "task_score": 0.5, "won": False})
    server, _ = _server(monkeypatch, [step])
    await server.reset({"task_id": 7}, SID)
    server.session_state[SID].avail = ITEM_AVAIL
    obs, reward, term, _, info = await server.step(_act("click[buy now]"), {}, SID)
    assert reward == 0.5 and term is True and info["won"] is False
    assert obs.endswith("Task finished.")


async def test_format_fault_hint_no_engine_step(monkeypatch):
    server, engine = _server(monkeypatch)  # engine must NOT be stepped
    await server.reset({"task_id": 7}, SID)
    obs, reward, term, trunc, info = await server.step(_text("I will search for a mug."), {}, SID)
    assert engine.actions == []  # no command -> no engine step
    assert obs == EXPECTED_FAULT_HINT
    assert info["action_format_errors"] == 1 and info["step_count"] == 1
    assert (reward, term, trunc) == (0.0, False, False)


async def test_wrong_tool_name_is_format_fault(monkeypatch):
    server, engine = _server(monkeypatch)
    await server.reset({"task_id": 7}, SID)
    _, _, _, _, info = await server.step(_act("search[mug]", name="search"), {}, SID)
    assert engine.actions == [] and info["action_format_errors"] == 1


async def test_invalid_command_counted(monkeypatch):
    # a command that is not admissible on the results page still steps the
    # engine (WebShop no-ops it) but is counted invalid
    server, engine = _server(monkeypatch, [RESULTS_STEP])
    await server.reset({"task_id": 7}, SID)  # initial page: only search admissible
    _, _, _, _, info = await server.step(_act("click[nonexistent]"), {}, SID)
    assert info["invalid_actions"] == 1
    assert engine.actions == ["click[nonexistent]"]  # env feedback stays the env's job


async def test_click_search_counted_invalid_and_forwarded(monkeypatch):
    # upstream lists the search-bar button among the clickables yet no-ops
    # click[search]; the adapter neither advertises it nor counts it valid,
    # but still forwards it (env feedback stays the env's job)
    server, engine = _server(monkeypatch, [SEARCH_PAGE_STEP])
    obs0, _ = await server.reset({"task_id": 7}, SID)
    assert "click[search]" not in obs0
    _, _, _, _, info = await server.step(_act("click[search]"), {}, SID)
    assert info["invalid_actions"] == 1
    assert engine.actions == ["click[search]"]


async def test_reserved_token_search_counted_invalid_and_forwarded(monkeypatch):
    # `search[<a> ...]` crashes upstream; the engine glue neutralizes it into a
    # no-op, the adapter only counts it and forwards the command unchanged.
    server, engine = _server(monkeypatch, [SEARCH_PAGE_STEP, RESULTS_STEP])
    await server.reset({"task_id": 7}, SID)
    _, _, _, _, info = await server.step(_act("search[<a> red mug]"), {}, SID)
    assert info["invalid_actions"] == 1
    assert engine.actions == ["search[<a> red mug]"]
    _, _, _, _, info = await server.step(_act("search[red <a> mug]"), {}, SID)  # only the FIRST token is reserved
    assert info["invalid_actions"] == 1
    assert engine.actions[-1] == "search[red <a> mug]"


async def test_truncation_at_max_steps(monkeypatch):
    server, _ = _server(monkeypatch, [RESULTS_STEP])
    await server.reset({"task_id": 7, "max_episode_steps": 1}, SID)
    _, _, term, trunc, _ = await server.step(_act("search[mug]"), {}, SID)
    assert term is False and trunc is True


async def test_config_step_cap_when_no_metadata(monkeypatch):
    server, _ = _server(monkeypatch, [RESULTS_STEP], max_episode_steps=1)
    await server.reset({"task_id": 7}, SID)
    _, _, _, trunc, _ = await server.step(_act("search[mug]"), {}, SID)
    assert trunc is True


async def test_format_fault_spends_budget_and_truncates(monkeypatch):
    server, _ = _server(monkeypatch, max_episode_steps=1)
    await server.reset({"task_id": 7}, SID)
    _, _, term, trunc, _ = await server.step(_text("hmm"), {}, SID)
    assert (term, trunc) == (False, True)


async def test_done_replay_defense(monkeypatch):
    server, _ = _server(monkeypatch, [WIN_STEP])
    await server.reset({"task_id": 7}, SID)
    await server.step(_act("click[buy now]"), {}, SID)  # direct call: no endpoint auto-close
    obs, reward, term, trunc, _ = await server.step(_act("search[again]"), {}, SID)
    assert (obs, reward, term, trunc) == (None, 0.0, True, False)


async def test_engine_error_on_step_is_502_evicts_and_closes(monkeypatch):
    server, engine = _server(monkeypatch, [EngineError("engine 500 after retries")])
    await server.reset({"task_id": 7}, SID)
    with pytest.raises(HTTPException) as ei:
        await server.step(_act("search[mug]"), {}, SID)
    assert ei.value.status_code == 502
    assert SID not in server.session_state  # evicted now, not at TTL
    assert engine.closed is True and engine.close_calls == 1  # engine episode freed
    with pytest.raises(HTTPException) as ei:  # the dead cookie 400s from here on
        await server.step(_act("search[mug]"), {}, SID)
    assert ei.value.status_code == 400


async def test_engine_410_on_step_is_502_evicts_and_forgets(monkeypatch):
    server, engine = _server(monkeypatch, [_status_410()])
    await server.reset({"task_id": 7}, SID)
    with pytest.raises(HTTPException) as ei:
        await server.step(_act("search[mug]"), {}, SID)
    assert ei.value.status_code == 502
    assert SID not in server.session_state
    assert engine.forgotten is True and engine.closed is False  # no /close to an engine that already 410'd


async def test_malformed_step_payload_is_502_and_evicts(monkeypatch):
    bad = (RESULTS_OBS, 0.0, False, {"task_score": 0.0, "won": False})  # no available_actions
    server, engine = _server(monkeypatch, [bad])
    await server.reset({"task_id": 7}, SID)
    with pytest.raises(HTTPException) as ei:
        await server.step(_act("search[mug]"), {}, SID)
    assert ei.value.status_code == 502
    assert SID not in server.session_state and engine.closed is True


async def test_step_liveness_recheck_after_eviction(monkeypatch):
    # A /step that fetched the state but lost the lock race to a close
    # (auto-close, sweeper, /close) must NOT step the engine.
    server, engine = _server(monkeypatch, [RESULTS_STEP])
    await server.reset({"task_id": 7}, SID)
    state = server.session_state[SID]
    async with state.lock:
        task = asyncio.create_task(server.step(_act("search[mug]"), {}, SID))
        await asyncio.sleep(0)  # step fetched the state ref, now blocked on the lock
        del server.session_state[SID]  # the close path popped it
    obs, reward, term, trunc, _ = await task
    assert (obs, reward, term, trunc) == (None, 0.0, True, False)
    assert engine.actions == []


async def test_close_waits_for_inflight_step(monkeypatch):
    # engine.close must not run concurrently with engine.step on one session.
    server, engine = _server(monkeypatch, [RESULTS_STEP])
    engine.step_gate = asyncio.Event()
    await server.reset({"task_id": 7}, SID)
    step_task = asyncio.create_task(server.step(_act("search[mug]"), {}, SID))
    while "step_enter" not in engine.order:
        await asyncio.sleep(0.01)
    close_task = asyncio.create_task(server.close_session(SID))
    await asyncio.sleep(0.05)
    assert not engine.closed  # close is parked on the session lock
    assert SID not in server.session_state  # but the cookie is already dead
    engine.step_gate.set()
    await asyncio.gather(step_task, close_task)
    assert engine.order == ["step_enter", "step_exit", "close"]


async def test_sessions_are_isolated(monkeypatch):
    server, engines = _server_per_reset(monkeypatch, [RESULTS_STEP, RESULTS_STEP])
    await server.reset({"task_id": 1}, "a")
    await server.reset({"task_id": 2}, "b")
    await server.step(_act("search[a1]"), {}, "a")
    await server.step(_act("search[b1]"), {}, "b")
    await server.step(_act("click[b0001]"), {}, "a")
    assert engines[0].actions == ["search[a1]", "click[b0001]"]
    assert engines[1].actions == ["search[b1]"]


# ──────────────────────────────────────────────────────────
# session lifecycle
# ──────────────────────────────────────────────────────────


async def test_close_frees_engine(monkeypatch):
    server, engine = _server(monkeypatch)
    await server.reset({"task_id": 7}, SID)
    await server.close_session(SID)
    assert engine.closed is True and SID not in server.session_state
    await server.close_session(SID)  # idempotent
    assert engine.close_calls == 1


async def test_sweeper_evicts_stale_keeps_fresh(monkeypatch):
    server, engines = _server_per_reset(monkeypatch, session_ttl_s=1000.0)
    await server.reset({"task_id": 1}, "stale")
    await server.reset({"task_id": 2}, "fresh")
    server.session_state["stale"].last_access = time.monotonic() - 2000.0
    evicted = await server.sweep_stale_sessions()
    assert evicted == 1
    assert engines[0].closed is True and engines[1].closed is False
    assert set(server.session_state) == {"fresh"}


async def test_sweeper_survives_engine_close_failure(monkeypatch):
    server, engine = _server(monkeypatch, engine_kwargs={"close_exc": RuntimeError("engine unreachable")})
    await server.reset({"task_id": 1}, SID)
    server.session_state[SID].last_access = time.monotonic() - 1e6
    assert await server.sweep_stale_sessions() == 1
    assert server.session_state == {}  # evicted locally even though the engine close failed


async def test_sweeper_task_starts_lazily(monkeypatch):
    server, _ = _server(monkeypatch, sweep_interval_s=30.0)
    assert server._sweeper_task is None
    await server.reset({"task_id": 7}, SID)
    assert server._sweeper_task is not None and not server._sweeper_task.done()
    server._sweeper_task.cancel()


def test_num_workers_guard(monkeypatch):
    # pydantic wraps model_post_init assertion failures into ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="num_workers"):
        _server(monkeypatch, num_workers=4)


# ──────────────────────────────────────────────────────────
# HTTP cookie flow (full FastAPI app, signed-cookie sessions)
# ──────────────────────────────────────────────────────────


class TestHttpFlow:
    def _client(self, server):
        from fastapi.testclient import TestClient

        # No context manager: startup handlers stay off (sweeper is lazy anyway).
        return TestClient(server.setup_webserver())

    @staticmethod
    def _reset_body(task_id: int = 7) -> dict:
        return {"responses_create_params": {"input": []}, "task_id": task_id, "split": "train"}

    @staticmethod
    def _reset_body_goal_id(goal_id: str = GID) -> dict:
        # The shape prepare_data.py emits today: goal_id, no task_id.
        return {"responses_create_params": {"input": []}, "goal_id": goal_id, "split": "train"}

    @staticmethod
    def _step_body(command: str) -> dict:
        return {"responses_create_params": {"input": []}, "response": _act(command).model_dump(mode="json")}

    def test_reset_step_terminal_autoclose(self, monkeypatch):
        server, engine = _server(monkeypatch, [WIN_STEP])
        client = self._client(server)

        r = client.post("/reset", json=self._reset_body())
        assert r.status_code == 200
        assert r.json()["observation"].startswith(f"Your task is to: {GOAL}.")
        assert r.json()["info"] == {"goal_id": "", "goal_idx": "7"}
        assert len(server.session_state) == 1
        next(iter(server.session_state.values())).avail = ITEM_AVAIL

        r = client.post("/step", json=self._step_body("click[buy now]"))
        assert r.status_code == 200
        body = r.json()
        assert body["terminated"] is True and body["truncated"] is False
        assert body["reward"] == 1.0
        assert body["info"]["won"] is True and body["info"]["task_score"] == 1.0
        # Terminal step: the base endpoint auto-closed the session; the engine
        # already dropped the done episode, so no /close HTTP was sent.
        assert server.session_state == {}
        assert engine.forgotten is True and engine.closed is False and engine.close_calls == 0

    def test_truncated_step_autocloses_with_engine_close(self, monkeypatch):
        server, engine = _server(monkeypatch, [RESULTS_STEP], max_episode_steps=1)
        client = self._client(server)
        client.post("/reset", json=self._reset_body())
        r = client.post("/step", json=self._step_body("search[mug]"))
        assert r.status_code == 200 and r.json()["truncated"] is True
        assert server.session_state == {}
        assert engine.closed is True  # non-terminal exit: the engine still holds the episode

    def test_reset_by_goal_id_echoes_both_ids_as_strings(self, monkeypatch):
        server, engine = _server(monkeypatch, [RESULTS_STEP])
        client = self._client(server)
        r = client.post("/reset", json=self._reset_body_goal_id())
        assert r.status_code == 200
        assert r.json()["info"] == {"goal_id": GID, "goal_idx": "1"}
        assert engine.reset_calls == [{"task_id": None, "split": "train", "goal_id": GID}]
        assert client.post("/step", json=self._step_body("search[mug]")).status_code == 200

    def test_reset_unknown_goal_id_is_400(self, monkeypatch):
        server, engine = _server(monkeypatch)
        client = self._client(server)
        r = client.post("/reset", json=self._reset_body_goal_id(UNKNOWN_GID))
        assert r.status_code == 400
        assert "is not in the engine goal list" in r.json()["detail"]
        assert server.session_state == {} and engine.closed is False

    def test_reset_without_any_goal_ref_is_400(self, monkeypatch):
        server, _ = _server(monkeypatch)
        client = self._client(server)
        r = client.post("/reset", json={"responses_create_params": {"input": []}, "split": "train"})
        assert r.status_code == 400
        assert r.json()["detail"] == "Missing goal_id (or legacy task_id) in the run body."

    def test_step_without_reset_is_400(self, monkeypatch):
        server, _ = _server(monkeypatch)
        client = self._client(server)
        assert client.post("/step", json=self._step_body("search[mug]")).status_code == 400

    def test_engine_error_on_step_is_502(self, monkeypatch):
        server, engine = _server(monkeypatch, [EngineError("engine down")])
        client = self._client(server)
        client.post("/reset", json=self._reset_body())
        r = client.post("/step", json=self._step_body("search[mug]"))
        assert r.status_code == 502
        assert server.session_state == {} and engine.closed is True

    def test_close_endpoint(self, monkeypatch):
        server, engine = _server(monkeypatch)
        client = self._client(server)
        assert client.post("/close", json={}).json() == {"closed": False}
        client.post("/reset", json=self._reset_body())
        assert client.post("/close", json={}).json() == {"closed": True}
        assert engine.closed is True and server.session_state == {}
        assert client.post("/close", json={}).json() == {"closed": False}

    def test_two_clients_isolated(self, monkeypatch):
        server, engines = _server_per_reset(monkeypatch, [RESULTS_STEP])
        client_a, client_b = self._client(server), self._client(server)
        client_a.post("/reset", json=self._reset_body(1))
        client_b.post("/reset", json=self._reset_body(2))
        client_a.post("/step", json=self._step_body("search[a]"))
        client_b.post("/step", json=self._step_body("search[b]"))
        assert engines[0].actions == ["search[a]"]
        assert engines[1].actions == ["search[b]"]


# ──────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────


def test_is_valid_command():
    avail = {"has_search_bar": True, "clickables": ["b0001", "buy now"]}
    assert _is_valid_command("search[red mug]", avail)
    assert _is_valid_command("click[b0001]", avail)
    assert _is_valid_command("click[BUY NOW]", avail)  # case-insensitive
    assert not _is_valid_command("click[missing]", avail)
    assert not _is_valid_command("search[x]", {"has_search_bar": False, "clickables": []})
    assert not _is_valid_command("go north", avail)
    # the search-bar submit button is listed among upstream's clickables but
    # step() no-ops it unconditionally -> not admissible
    assert not _is_valid_command("click[search]", {"has_search_bar": True, "clickables": ["search", "b0001"]})


def test_is_reserved_search():
    for tok in ("<a>", "<c>", "<q>", "<r>"):
        assert is_reserved_search(f"search[{tok} red mug]")
        assert is_reserved_search(f"search[{tok.upper()}]")  # upstream lowercases the argument
    assert is_reserved_search("Search[<a> mug]")  # name matched case-insensitively
    assert not is_reserved_search("search[red <a> mug]")  # only the first token
    assert not is_reserved_search("search[ <a> mug]")  # upstream splits on ' ': keywords[0] == ''
    assert not is_reserved_search("search[<b> mug]")
    assert not is_reserved_search("click[<a>]")
    assert not is_reserved_search("search[]")
    assert not is_reserved_search("<a> mug")


def test_available_commands_drops_search_button_only_with_search_bar():
    assert available_commands({"has_search_bar": True, "clickables": ["search"]}) == ["search[<query>]"]
    assert available_commands({"has_search_bar": True, "clickables": ["Search", "b0001"]}) == [
        "search[<query>]",
        "click[b0001]",
    ]
    # Without a search bar a "search" clickable would be a real button: kept.
    assert available_commands({"has_search_bar": False, "clickables": ["search", "b0001"]}) == [
        "click[search]",
        "click[b0001]",
    ]


# ──────────────────────────────────────────────────────────
# TCOD protocol defaults: text_rich pages + error feedback
# ──────────────────────────────────────────────────────────


def test_tcod_defaults_are_the_config_defaults():
    cfg = WebshopResourcesServerConfig(host="0.0.0.0", port=9021, entrypoint="", name="webshop_env")
    assert (cfg.observation_mode, cfg.invalid_action_feedback) == ("text_rich", "error")


def test_bad_protocol_knobs_fail_at_construction(monkeypatch):
    with pytest.raises(ValueError, match="observation_mode"):
        _server(monkeypatch, observation_mode="html")
    with pytest.raises(ValueError, match="invalid_action_feedback"):
        _server(monkeypatch, invalid_action_feedback="penalize")


def test_extract_task_and_format_obs_text_rich():
    assert extract_task(RICH_INITIAL_OBS) == GOAL
    assert extract_task(INITIAL_OBS) == GOAL  # the [SEP] rendering still parses
    # Multi-line instruction: joined until the first [button] line.
    assert (
        extract_task("Instruction:\nbuy a mug\nunder 30 dollars\n[button] Search [button_]")
        == "buy a mug under 30 dollars"
    )
    with pytest.raises(app_module.WebshopFormatError):
        extract_task("WebShop\n[button] Search [button_]\n")
    # text_rich pages pass through verbatim (TCOD's format_observation), instruction included.
    assert format_obs(RICH_RESULTS_OBS, GOAL, "text_rich") == RICH_RESULTS_OBS.strip()
    assert (
        format_obs(RESULTS_OBS, GOAL, "text")
        == "'Back to Search' [SEP] 'Page 1 (Total results: 5)' [SEP] 'B0001' [SEP] '$10.9'"
    )
    with pytest.raises(ValueError):
        format_obs(RESULTS_OBS, GOAL, "html")


def test_render_observation_tcod_wording():
    first = render_observation(
        RICH_INITIAL_OBS, GOAL, INITIAL_AVAIL, 2048, 60, observation_mode="text_rich", next_step=1
    )
    assert first == (
        f"Your current observation is: WebShop\nInstruction:\n{GOAL}\n[button] Search [button_].\n"
        "Your admissible actions of the current situation are:\n[\nsearch[<query>]\n]."
    )
    later = render_observation(
        RICH_RESULTS_OBS, GOAL, RESULTS_AVAIL, 2048, 60, observation_mode="text_rich", next_step=3
    )
    assert later.startswith("You are now at step 3 and your current observation is: Instruction:\n")
    assert later.endswith(
        "\nYour admissible actions of the current situation are:\n[\nclick[back to search]\nclick[b0001]\n]."
    )


async def test_text_rich_reset_and_step(monkeypatch):
    server, engine = _server(
        monkeypatch, [RICH_RESULTS_STEP], engine_kwargs={"reset_obs": RICH_INITIAL_OBS}, observation_mode="text_rich"
    )
    obs, _ = await server.reset({"task_id": 7}, SID)
    assert obs == (
        f"Your task is to: {GOAL}.\nYour current observation is: WebShop\nInstruction:\n{GOAL}\n"
        "[button] Search [button_].\nYour admissible actions of the current situation are:\n[\nsearch[<query>]\n]."
    )
    assert server.session_state[SID].task == GOAL
    obs, _, _, _, info = await server.step(_act("search[red ceramic mug]"), {}, SID)
    assert engine.actions == ["search[red ceramic mug]"]
    assert obs.startswith("You are now at step 2 and your current observation is: Instruction:\n")
    assert "[button] B0001 [button_]" in obs and "\nclick[b0001]\n" in obs
    assert info["invalid_actions"] == 0


def test_invalid_action_message_matches_tcod_branches():
    page = {"has_search_bar": False, "clickables": ["back to search", "b0001"]}
    assert invalid_action_message("search", page) == INVALID_SEARCH_EMPTY  # bare name, no argument
    assert invalid_action_message("search[red mug]", page) == INVALID_SEARCH_NO_BAR
    assert invalid_action_message("click[missing]", page) == INVALID_CLICK_PREFIX + "['back to search', 'b0001']"
    assert (
        invalid_action_message("click[search]", INITIAL_AVAIL) == INVALID_CLICK_PREFIX + "[]"
    )  # search button hidden
    assert invalid_action_message("search[<a> red mug]", INITIAL_AVAIL) == INVALID_GENERIC  # reserved first token
    assert invalid_action_message("search[]", page) == INVALID_GENERIC  # TCOD: no regex match -> generic
    assert invalid_action_message("go north", page) == INVALID_GENERIC


async def test_error_feedback_spends_turn_without_engine_step(monkeypatch):
    # TCOD protocol: a non-admissible command never reaches the engine; the model
    # reads TCOD's error sentence with the unchanged admissible list.
    server, engine = _server(monkeypatch, [RESULTS_STEP], invalid_action_feedback="error")
    await server.reset({"task_id": 7}, SID)  # initial page: only search admissible
    obs, reward, term, trunc, info = await server.step(_act("click[nonexistent]"), {}, SID)
    assert engine.actions == []
    assert obs == (
        "You are now at step 2 and your current observation is: " + INVALID_CLICK_PREFIX + "[].\n"
        "Your admissible actions of the current situation are:\n[\nsearch[<query>]\n]."
    )
    assert (reward, term, trunc) == (0.0, False, False)
    assert info["invalid_actions"] == 1 and info["step_count"] == 1
    # reserved-token searches are never admissible either
    _, _, _, _, info = await server.step(_act("search[<a> red mug]"), {}, SID)
    assert engine.actions == [] and info["invalid_actions"] == 2
    # an admissible command is still forwarded verbatim
    obs, _, _, _, info = await server.step(_act("search[red ceramic mug]"), {}, SID)
    assert engine.actions == ["search[red ceramic mug]"] and info["invalid_actions"] == 2
    assert "\nclick[b0001]\n" in obs
    # ... and on the results page a search has no bar
    _, _, _, _, info = await server.step(_act("search[again]"), {}, SID)
    assert engine.actions == ["search[red ceramic mug]"] and info["invalid_actions"] == 3


async def test_error_feedback_still_truncates_at_max_steps(monkeypatch):
    server, engine = _server(monkeypatch, invalid_action_feedback="error")
    await server.reset({"task_id": 7, "max_episode_steps": 2}, SID)
    _, _, term, trunc, _ = await server.step(_act("click[x]"), {}, SID)
    assert (term, trunc) == (False, False)
    _, _, term, trunc, info = await server.step(_act("click[y]"), {}, SID)
    assert (term, trunc) == (False, True) and info["invalid_actions"] == 2
    assert engine.actions == []
