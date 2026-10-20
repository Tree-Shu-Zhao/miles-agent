"""Adapter <-> engine wire contract, end to end and in-process.

test_app.py fakes EngineSession; test_engine_client.py fakes the engine's HTTP
replies. Here both fakes are removed at once: the REAL webshop_env adapter
drives the REAL engine HTTP layer (env_servers/webshop/session_server.create_app)
over httpx.ASGITransport, with only the engine *backend* scripted. A drift in
field names, status codes, or the 410 / terminal-close semantics between the
py3.13 and py3.10 tiers shows up here and nowhere else.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")

# The engine tier is a set of top-level modules under env_servers/webshop
# (repo root is 4 levels above tests/).
_ENGINE_DIR = Path(__file__).resolve().parents[4] / "env_servers" / "webshop"
sys.path.insert(0, str(_ENGINE_DIR))

import httpx  # noqa: E402
from _webshop_engine import UnknownGoal, step_env  # noqa: E402
from agent_utils.webshop_goals import goal_id, goals_set_digest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponse  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from session_server import ResetResult, SessionBackend, StepResult, UnknownSession, create_app  # noqa: E402
from webshop_env._engine_client import EngineError, EngineSession  # noqa: E402
from webshop_env.app import WebshopResourcesServer, WebshopResourcesServerConfig  # noqa: E402

ENGINE_URL = "http://engine"
SID = "cookie-1"
GOAL = "i need a red ceramic mug, and price lower than 30.00 dollars"
# The scripted engine's goal list, content-addressed with the shared module
# exactly like the real _EngineHost (goal_id -> list index, order-free digest).
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
GOAL_IDS = [goal_id(g) for g in FAKE_GOALS]
GOAL_INDEX = {gid: i for i, gid in enumerate(GOAL_IDS)}
GOALS_DIGEST = goals_set_digest(GOAL_IDS)
UNKNOWN_GID = "0" * 16
INITIAL_OBS = f"Amazon Shopping Game [SEP] Instruction: [SEP] {GOAL} [SEP] Search"
INITIAL_AVAIL = {"has_search_bar": True, "clickables": []}
RESULTS_OBS = (
    f"Instruction: [SEP] {GOAL} [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] B0001 [SEP] $10.99"
)
RESULTS_AVAIL = {"has_search_bar": False, "clickables": ["back to search", "b0001"]}
ITEM_AVAIL = {"has_search_bar": False, "clickables": ["buy now"]}
DONE_OBS = f"Instruction: [SEP] {GOAL} [SEP] Thank you for shopping with us!"
TERMINAL_AVAIL = {"has_search_bar": False, "clickables": []}


def _engine_step(obs: str, *, score: float, done: bool, avail: dict) -> StepResult:
    """Shape of the real ``step_env`` result: the reward slot carries the RAW
    [0, 1] score, identical to ``info["task_score"]``. The engine never remaps
    rewards (recipes/webshop/reward.py owns the training reward), so a fake that
    put anything else on the wire would pin a value production never produces."""
    info = {"available_actions": avail, "task_score": score, "won": done and score == 1.0}
    return StepResult(obs=obs, reward=score, done=done, info=info)


class ScriptedBackend(SessionBackend):
    """Stand-in for WebshopEngineWorkerPool with the real engine's payload shapes:
    reset -> initial page; any non-buy action -> results page; click[buy now] ->
    terminal (raw task_score 1.0 in BOTH the reward slot and info, episode
    forgotten). A ``goal_id`` is resolved through GOAL_INDEX; an unknown one
    raises the engine's real UnknownGoal (a ValueError, so session_server maps
    it, like a bad legacy index, to 400 — the adapter never retries either)."""

    def __init__(self):
        self.sessions: dict[str, int] = {}
        self.calls: list[tuple] = []

    def reset(self, session_id, task_id, split, goal_id=None):
        self.calls.append(("reset", session_id, task_id, split, goal_id))
        if goal_id is not None:
            if goal_id not in GOAL_INDEX:
                raise UnknownGoal(
                    f"goal_id {goal_id!r} is not in the engine goal list (goals_digest {GOALS_DIGEST}): "
                    "the dataset and the engine disagree on catalog / goal code"
                )
            goal_idx = GOAL_INDEX[goal_id]
        else:
            if not isinstance(task_id, int) or task_id < 0:
                raise ValueError(f"goal index out of range: {task_id!r}")
            goal_idx = task_id
        self.sessions[session_id] = 0
        info = {"available_actions": INITIAL_AVAIL, "task_score": 0.0, "won": False, "goal_idx": goal_idx}
        return ResetResult(obs=INITIAL_OBS, info=info)

    def step(self, session_id, action):
        self.calls.append(("step", session_id, action))
        if session_id not in self.sessions:
            raise UnknownSession(session_id)
        self.sessions[session_id] += 1
        if action == "click[buy now]":
            self.sessions.pop(session_id)  # the real pool forgets a done episode
            return _engine_step(DONE_OBS, score=1.0, done=True, avail=TERMINAL_AVAIL)
        return _engine_step(RESULTS_OBS, score=0.0, done=False, avail=RESULTS_AVAIL)

    def close(self, session_id):
        self.calls.append(("close", session_id))
        self.sessions.pop(session_id, None)

    def live_sessions(self):
        return list(self.sessions)

    def worker_count(self):
        return 1

    def health(self):
        return {"workers": 1, "dead_workers": 0, "respawns": 0, "goals": len(GOAL_IDS), "goals_digest": GOALS_DIGEST}

    def ops(self, kind: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]


class _RecordingASGI:
    """ASGI wrapper that keeps the decoded JSON body of every request, so a
    test can pin the exact wire shape the adapter sends (independent of how
    the engine's pydantic model happens to parse it)."""

    def __init__(self, app):
        self.app = app
        self.bodies: list[tuple[str, dict]] = []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks: list[bytes] = []

        async def recv():
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body"):
                    self.bodies.append((scope["path"], json.loads(b"".join(chunks) or b"null")))
            return message

        return await self.app(scope, recv, send)

    def sent(self, path: str) -> list[dict]:
        return [body for p, body in self.bodies if p == path]


@pytest.fixture
async def rig(monkeypatch):
    backend = ScriptedBackend()
    wire = _RecordingASGI(create_app(backend, session_ttl=1800.0, max_concurrency=4, sweep_interval=0))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=wire), base_url=ENGINE_URL)
    monkeypatch.setattr(EngineSession, "_clients", {ENGINE_URL: client})
    # The scripted backend speaks the verl-agent [SEP] rendering; the TCOD
    # text_rich default is covered in test_app.py.
    config = WebshopResourcesServerConfig(
        host="0.0.0.0",
        port=9021,
        entrypoint="",
        name="webshop_env",
        engine_base_url=ENGINE_URL,
        sweep_interval_s=0.0,
        observation_mode="text",
    )
    server = WebshopResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    yield SimpleNamespace(backend=backend, server=server, client=client, wire=wire)
    await client.aclose()


def _act(command: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "t",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_0",
                    "call_id": "c1",
                    "name": "act",
                    "arguments": json.dumps({"command": command}),
                    "status": "completed",
                }
            ],
        }
    )


def _engine_error(exc: HTTPException) -> EngineError:
    cause = exc.__cause__
    assert isinstance(cause, EngineError), f"adapter {exc.status_code} must chain from EngineError, got {cause!r}"
    return cause


# ────────────────────────────────────────── EngineSession alone vs the engine app
async def test_engine_session_raw_wire(rig):
    """The client-side view of the protocol, before the adapter reshapes anything."""
    s = EngineSession(ENGINE_URL, step_timeout=5.0)
    obs, info = await s.reset(3, "eval")
    assert rig.backend.calls == [("reset", s.session_id, 3, "eval", None)]
    # Legacy body, byte-for-byte: no goal_id key (not even null) when none is given.
    assert rig.wire.sent("/reset") == [{"session_id": s.session_id, "task_id": 3, "split": "eval"}]
    assert obs == INITIAL_OBS and info["goal_idx"] == 3 and info["available_actions"] == INITIAL_AVAIL

    obs, reward, done, info = await s.step("search[mug]")
    assert (obs, done) == (RESULTS_OBS, False) and info["available_actions"] == RESULTS_AVAIL
    assert reward == info["task_score"] == 0.0

    obs, reward, done, info = await s.step("click[buy now]")
    # The wire reward IS the raw score: nothing is remapped engine-side (see step_env).
    assert (obs, done) == (DONE_OBS, True) and reward == info["task_score"] == 1.0
    assert info["won"] is True and info["available_actions"] == TERMINAL_AVAIL
    assert rig.backend.live_sessions() == []  # the engine dropped the done episode

    with pytest.raises(EngineError) as ei:
        await s.step("look")  # stepping a dropped episode
    assert ei.value.status == 410

    await s.close()
    await s.close()
    assert rig.backend.ops("close") == [("close", s.session_id)]  # idempotent client side


async def test_engine_session_bad_goal_is_400_not_retried(rig):
    s = EngineSession(ENGINE_URL, step_timeout=5.0)
    with pytest.raises(EngineError) as ei:
        await s.reset(-1)
    assert ei.value.status == 400
    assert len(rig.backend.ops("reset")) == 1  # 4xx is never retried
    assert rig.backend.live_sessions() == []


async def test_engine_session_reset_by_goal_id_wire(rig):
    """Content-addressed reset: ``goal_id`` is its own body field (task_id rides
    along as null), the engine resolves it to the list index it reports back."""
    s = EngineSession(ENGINE_URL, step_timeout=5.0)
    obs, info = await s.reset(split="eval", goal_id=GOAL_IDS[2])
    assert rig.wire.sent("/reset") == [
        {"session_id": s.session_id, "task_id": None, "split": "eval", "goal_id": GOAL_IDS[2]}
    ]
    assert rig.backend.calls == [("reset", s.session_id, None, "eval", GOAL_IDS[2])]
    assert obs == INITIAL_OBS and info["goal_idx"] == 2


async def test_engine_session_unknown_goal_id_is_400_with_mismatch_message(rig):
    s = EngineSession(ENGINE_URL, step_timeout=5.0)
    with pytest.raises(EngineError) as ei:
        await s.reset(goal_id=UNKNOWN_GID)
    assert ei.value.status == 400
    assert f"goal_id {UNKNOWN_GID!r} is not in the engine goal list (goals_digest {GOALS_DIGEST})" in str(ei.value)
    assert len(rig.backend.ops("reset")) == 1  # never retried: the engine's verdict on the row is final
    assert rig.backend.live_sessions() == []


async def test_engine_health_carries_goals_digest(rig):
    """The training entry's digest gate reads /health.goals_digest; the HTTP layer
    must merge the backend's health fields verbatim."""
    health = (await rig.client.get("/health")).json()
    assert health["goals"] == len(GOAL_IDS) and health["goals_digest"] == GOALS_DIGEST
    assert len(health["goals_digest"]) == 16 and int(health["goals_digest"], 16) >= 0


# ──────────────────────────────── the scripted stand-in vs the real engine glue
class _StubEngine:
    def __init__(self):
        self.user_sessions = {"sid:0": object()}


class _StubEnv:
    """The WebAgentTextEnv surface step_env touches, with a scripted step result."""

    session_prefix = "sid:"

    def __init__(self, score: float, done: bool):
        self._ret = (DONE_OBS if done else RESULTS_OBS, score, done, {})

    def step(self, action):
        return self._ret

    def get_available_actions(self):
        return RESULTS_AVAIL


@pytest.mark.parametrize("score", [0.0, 0.75, 1.0])
def test_real_step_env_reward_slot_is_raw_task_score(score):
    """Anchor for ScriptedBackend: the real engine glue returns the RAW score in
    the reward slot (== info.task_score); there is no engine-side remap."""
    engine = _StubEngine()
    obs, reward, done, info = step_env(engine, _StubEnv(score, done=True), "click[buy now]")
    assert (obs, done) == (DONE_OBS, True)
    assert reward == info["task_score"] == score
    assert info["won"] is (score == 1.0)
    assert engine.user_sessions == {}  # done episode pruned under its prefix


def test_scripted_backend_mirrors_step_env_reward_contract():
    """The stand-in must not drift from step_env: reward slot == task_score on every step."""
    backend = ScriptedBackend()
    backend.reset("e1", 0, "train")
    for action, score, done in (("search[mug]", 0.0, False), ("click[buy now]", 1.0, True)):
        r = backend.step("e1", action)
        assert r.done is done and r.reward == r.info["task_score"] == score


def _backend_idx(gid: str) -> int:
    return ScriptedBackend().reset("probe", None, "train", goal_id=gid).info["goal_idx"]


def test_scripted_backend_goal_ids_are_content_ids():
    """The stand-in's ids must be what the shared module computes from the goal
    content, and its digest must be the order-free set digest of those ids."""
    for i, g in enumerate(FAKE_GOALS):
        assert _backend_idx(GOAL_IDS[i]) == i and goal_id(g) == GOAL_IDS[i]
    assert len(set(GOAL_IDS)) == len(GOAL_IDS)
    assert goals_set_digest(reversed(GOAL_IDS)) == GOALS_DIGEST


# ───────────────────────────────────────────── adapter -> engine, full episode
async def test_adapter_reset_step_terminal_contract(rig):
    server, backend = rig.server, rig.backend

    obs, info = await server.reset({"task_id": 7, "split": "train"}, SID)
    state = server.session_state[SID]
    engine_sid = state.engine.session_id
    assert backend.calls == [("reset", engine_sid, 7, "train", None)]
    assert backend.live_sessions() == [engine_sid]
    assert obs.startswith(f"Your task is to: {GOAL}.\n") and "\nsearch[<query>]\n" in obs
    assert info == {"goal_id": "", "goal_idx": "7"}  # legacy row: index echoed, no content id

    obs, reward, term, trunc, info = await server.step(_act("search[red ceramic mug]"), {}, SID)
    assert backend.ops("step") == [("step", engine_sid, "search[red ceramic mug]")]  # command forwarded verbatim
    assert (reward, term, trunc) == (0.0, False, False)
    assert "\nclick[b0001]\n" in obs
    assert info["step_count"] == 1 and info["invalid_actions"] == 0

    state.avail = ITEM_AVAIL  # put the episode on the item page so buy now is admissible
    obs, reward, term, trunc, info = await server.step(_act("click[buy now]"), {}, SID)
    assert backend.ops("step")[-1] == ("step", engine_sid, "click[buy now]")
    assert reward == info["task_score"] == 1.0  # the adapter's reward is the raw task_score
    assert (term, trunc) == (True, False)
    assert info["won"] is True and info["task_score"] == 1.0 and info["step_count"] == 2
    assert obs.endswith("Task completed.")

    # F33: the engine already dropped the done episode, so the adapter marks the
    # client closed (forget) and the release path spends no /close round trip.
    assert state.engine.closed is True
    await server.close_session(SID)
    assert SID not in server.session_state
    assert backend.ops("close") == []


async def test_adapter_step_on_engine_dropped_session_is_410_then_502(rig):
    server, backend = rig.server, rig.backend
    await server.reset({"task_id": 7}, SID)
    state = server.session_state[SID]
    engine_sid = state.engine.session_id

    backend.close(engine_sid)  # engine-side loss: worker respawn / TTL eviction
    backend.calls.clear()

    with pytest.raises(HTTPException) as ei:
        await server.step(_act("search[mug]"), {}, SID)
    assert ei.value.status_code == 502
    assert _engine_error(ei.value).status == 410
    assert backend.ops("step") == [("step", engine_sid, "search[mug]")]  # exactly one attempt

    # The failed step evicts the adapter session (no lingering to TTL), and a
    # 410 means the engine has nothing left to close: no /close round trip.
    assert SID not in server.session_state
    assert state.engine.closed is True
    assert backend.ops("close") == []


async def test_adapter_reset_by_goal_id_contract(rig):
    """The frozen-dataset row shape (goal_id, no task_id) end to end: the
    adapter puts goal_id on the wire, the engine resolves it, and both ids come
    back as STRING labels."""
    server, backend = rig.server, rig.backend
    obs, info = await server.reset({"goal_id": GOAL_IDS[1], "split": "eval"}, SID)
    state = server.session_state[SID]
    engine_sid = state.engine.session_id
    assert rig.wire.sent("/reset") == [
        {"session_id": engine_sid, "task_id": None, "split": "eval", "goal_id": GOAL_IDS[1]}
    ]
    assert backend.calls == [("reset", engine_sid, None, "eval", GOAL_IDS[1])]
    assert backend.live_sessions() == [engine_sid]
    assert obs.startswith(f"Your task is to: {GOAL}.\n")
    assert info == {"goal_id": GOAL_IDS[1], "goal_idx": "1"}

    _, reward, term, trunc, step_info = await server.step(_act("search[red ceramic mug]"), {}, SID)
    assert backend.ops("step") == [("step", engine_sid, "search[red ceramic mug]")]
    assert (reward, term, trunc) == (0.0, False, False) and step_info["step_count"] == 1


@pytest.fixture
def no_backoff(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("webshop_env._engine_client.asyncio.sleep", fake_sleep)
    return sleeps


async def test_adapter_bad_goal_index_is_400_not_retried_no_session(rig, no_backoff):
    """Engine 400 (bad legacy index) -> adapter 400: a row error, not a transport failure."""
    server, backend = rig.server, rig.backend
    with pytest.raises(HTTPException) as ei:
        await server.reset({"task_id": -1}, SID)
    assert ei.value.status_code == 400
    assert _engine_error(ei.value).status == 400
    assert len(backend.ops("reset")) == 1 and no_backoff == []  # 4xx: no retry, no back-off
    assert SID not in server.session_state and backend.live_sessions() == []


async def test_adapter_unknown_goal_id_is_400_not_retried_no_session(rig, no_backoff):
    """Engine 400 for an id it does not have (dataset/engine goal-set mismatch)
    -> adapter 400 carrying the engine's message, exactly one attempt, no
    session on either side. Never 502: it is not infra and must not read as a
    wedged engine."""
    server, backend = rig.server, rig.backend
    with pytest.raises(HTTPException) as ei:
        await server.reset({"goal_id": UNKNOWN_GID}, SID)
    assert ei.value.status_code == 400
    assert _engine_error(ei.value).status == 400
    assert f"goal_id {UNKNOWN_GID!r} is not in the engine goal list (goals_digest {GOALS_DIGEST})" in ei.value.detail
    assert len(backend.ops("reset")) == 1 and no_backoff == []
    assert SID not in server.session_state and backend.live_sessions() == []


async def test_adapter_engine_5xx_on_reset_is_502_after_retries(rig, no_backoff, monkeypatch):
    """Contrast with the 400 path: an engine failure that is NOT a row verdict
    stays a transport failure (502) and is retried with fresh session ids."""
    server, backend = rig.server, rig.backend

    def boom(*a, **k):
        raise RuntimeError("worker respawning")

    monkeypatch.setattr(backend, "reset", boom)
    with pytest.raises(HTTPException) as ei:
        await server.reset({"goal_id": GOAL_IDS[0]}, SID)
    assert ei.value.status_code == 502
    assert _engine_error(ei.value).status == 500
    assert len(rig.wire.sent("/reset")) == 3 and no_backoff == [2.0, 4.0]  # first try + 2 retries
    assert len({b["session_id"] for b in rig.wire.sent("/reset")}) == 3
    assert SID not in server.session_state


async def test_adapter_reused_cookie_re_resets_and_closes_old_engine_session(rig):
    server, backend = rig.server, rig.backend
    await server.reset({"task_id": 1}, SID)
    old_sid = server.session_state[SID].engine.session_id
    await server.reset({"task_id": 2}, SID)
    new_sid = server.session_state[SID].engine.session_id
    assert new_sid != old_sid
    assert backend.ops("close") == [("close", old_sid)]  # the previous episode was released on the engine
    assert backend.live_sessions() == [new_sid]
    assert backend.ops("reset")[-1] == ("reset", new_sid, 2, "train", None)


async def test_adapter_close_endpoint_path_releases_engine_session(rig):
    server, backend = rig.server, rig.backend
    await server.reset({"task_id": 1}, SID)
    engine_sid = server.session_state[SID].engine.session_id
    await server.close_session(SID)
    assert SID not in server.session_state
    assert backend.ops("close") == [("close", engine_sid)]
    assert backend.live_sessions() == []
    await server.close_session(SID)  # idempotent
    assert backend.ops("close") == [("close", engine_sid)]
