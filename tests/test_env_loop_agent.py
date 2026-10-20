"""Behavior tests for env_loop_agent's episode loop (policy + env both faked).

Actions travel as calls to the dataset-declared tool ("act" here); the agent
forwards them to /step as function_call items and replies observations as
tool messages. The agent module is loaded by file path (it lives in gym_root,
not a root package); nemo_gym must be importable (uv sync --group gym).
"""

import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")

from nemo_gym.server_utils import ServerClient  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
POLICY_URL = "http://trainer:30000/sessions/abc/v1"

ACT_TOOL = {
    "type": "function",
    "name": "act",
    "description": "Execute one admissible command.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _load_module():
    path = REPO / "gym_root" / "responses_api_agents" / "env_loop_agent" / "app.py"
    spec = importlib.util.spec_from_file_location("env_loop_agent_app", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeResponse:
    """Duck-types the aiohttp ClientResponse surface the agent touches."""

    def __init__(self, payload: dict, status: int = 200, cookies: dict | None = None):
        self._payload = payload
        self.status = status
        self.ok = status < 400
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    @property
    def content(self):
        return self  # raise_for_status reads .content.read() on errors

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status}")


class FakeEnvServer:
    """Scripted resources server behind server_client.post."""

    def __init__(self, step_payloads: list[dict], reset_status: int = 200):
        self.step_payloads = list(step_payloads)
        self.reset_status = reset_status
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self._cookie_serial = 0

    async def post(self, server_name: str, url_path: str, json: dict | None = None, cookies: dict | None = None):
        self.calls.append((server_name, url_path, json, cookies))
        self._cookie_serial += 1
        fresh = {"session": f"c{self._cookie_serial}"}
        if url_path == "/reset":
            return FakeResponse(
                {"observation": "obs0", "info": {"env_load_retries": 0}},
                status=self.reset_status,
                cookies=fresh,
            )
        if url_path == "/step":
            return FakeResponse(self.step_payloads.pop(0), cookies=fresh)
        if url_path == "/close":
            return FakeResponse({"closed": True}, cookies=fresh)
        raise AssertionError(f"unexpected url_path {url_path!r}")

    def paths(self) -> list[str]:
        return [c[1] for c in self.calls]


def _make_agent(mod, env: FakeEnvServer, max_turns: int = 30):
    config = mod.EnvLoopAgentConfig(
        host="0.0.0.0",
        port=8300,
        entrypoint="",
        name="env_loop_agent",
        resources_server={"type": "resources_servers", "name": "alfworld_env"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        max_turns=max_turns,
    )
    client = MagicMock(spec=ServerClient)
    client.post = env.post
    return mod.EnvLoopAgent(config=config, server_client=client)


def _make_body(mod, **overrides):
    payload = {
        "responses_create_params": {
            "input": [{"role": "system", "content": "sys prompt"}],
            "tools": [ACT_TOOL],
            "temperature": 1.0,
            "max_output_tokens": 128,
        },
        "policy_base_url": POLICY_URL,
        "policy_api_key": "k",
        "game_file": "json_2.1.1/train/x/game.tw-pddl",
        "max_episode_steps": 5,
        **overrides,
    }
    return mod.EnvLoopAgentRunRequest.model_validate(payload)


def _act_msg(*commands: str, content: str = "", start: int = 1) -> dict:
    """Assistant chat message calling act once per command."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": f"tc{start + i}",
                "type": "function",
                "function": {"name": "act", "arguments": json.dumps({"command": c})},
            }
            for i, c in enumerate(commands)
        ],
    }


def _fake_policy(monkeypatch, mod, replies: list):
    """replies: assistant message dict per call, or an int status code."""
    calls: list[dict] = []

    async def fake_http_request(method, url, **kwargs):
        assert method == "POST"
        assert url == f"{POLICY_URL}/chat/completions"
        # Deep-copy: the agent keeps appending to the same messages list.
        calls.append(copy.deepcopy(kwargs))
        reply = replies[len(calls) - 1]
        if isinstance(reply, int):
            return FakeResponse({}, status=reply)
        return FakeResponse({"choices": [{"message": reply}]})

    monkeypatch.setattr(mod, "http_request", fake_http_request)
    return calls


STEP_MID = {
    "observation": "obs1",
    "reward": 0.0,
    "terminated": False,
    "truncated": False,
    "info": {"won": False, "lost": False, "step_count": 1, "invalid_actions": 0, "action_format_errors": 0},
}
STEP_WON = {
    "observation": "obs2\nTask completed.",
    "reward": 1.0,
    "terminated": True,
    "truncated": False,
    "info": {"won": True, "lost": False, "step_count": 2, "invalid_actions": 0, "action_format_errors": 0},
}
STEP_ENV_TRUNCATED = {
    "observation": "obs1",
    "reward": 0.0,
    "terminated": False,
    "truncated": True,
    "info": {"won": False, "lost": False, "step_count": 1, "invalid_actions": 0, "action_format_errors": 0},
}
STEP_FORMAT_ERROR = {
    "observation": "Nothing happens. Call the act tool with exactly one admissible command.",
    "reward": 0.0,
    "terminated": False,
    "truncated": False,
    "info": {"won": False, "lost": False, "step_count": 1, "invalid_actions": 0, "action_format_errors": 1},
}


async def test_happy_path_terminated(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_MID, STEP_WON])
    policy_calls = _fake_policy(
        monkeypatch,
        mod,
        [_act_msg("go north", content="Going north.", start=1), _act_msg("take mug", content="Taking it.", start=2)],
    )
    agent = _make_agent(mod, env)

    result = await agent.run(request=None, body=_make_body(mod))

    assert result["reward"] == 1.0
    assert result["turns"] == 2.0
    assert result["won"] == 1.0
    assert result["step_count"] == 2.0
    assert result["env_truncated"] == 0.0
    assert result["policy_400_truncated"] == 0
    assert result["extra_tool_calls"] == 0.0
    assert result["env_load_retries"] == 0.0
    assert result["agent_metrics"]["total_tool_time"] >= 0.0
    # Terminal step: the env already closed the session -> no /close.
    assert env.paths() == ["/reset", "/step", "/step"]

    # The tool is declared on every policy call.
    assert policy_calls[0]["json"]["tools"][0]["function"]["name"] == "act"
    assert policy_calls[0]["json"]["max_tokens"] == 128
    assert policy_calls[0]["headers"]["Authorization"] == "Bearer k"

    # Message accumulation: system, reset obs, assistant, tool reply (obs1).
    second_call_messages = policy_calls[1]["json"]["messages"]
    assert [m["role"] for m in second_call_messages] == ["system", "user", "assistant", "tool"]
    assert second_call_messages[1]["content"] == "obs0"
    assert second_call_messages[3] == {"role": "tool", "tool_call_id": "tc1", "content": "obs1"}
    # Assistant message forwarded verbatim (tool_calls intact).
    assert second_call_messages[2]["tool_calls"][0]["function"]["name"] == "act"

    # The /run response transcript carries both assistant content turns.
    assert len(result["response"]["output"]) == 2


async def test_env_body_excludes_policy_fields_and_threads_cookies(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_MID, STEP_WON])
    _fake_policy(monkeypatch, mod, [_act_msg("go north"), _act_msg("take mug", start=2)])
    agent = _make_agent(mod, env)

    await agent.run(request=None, body=_make_body(mod))

    (_, _, reset_json, reset_cookies) = env.calls[0]
    assert reset_cookies is None
    assert reset_json["game_file"] == "json_2.1.1/train/x/game.tw-pddl"
    assert reset_json["max_episode_steps"] == 5
    assert "policy_base_url" not in reset_json and "policy_api_key" not in reset_json

    (_, _, step1_json, step1_cookies) = env.calls[1]
    (_, _, _, step2_cookies) = env.calls[2]
    # Cookie chain: reset's cookies to step 1, step 1's cookies to step 2.
    assert step1_cookies == {"session": "c1"}
    assert step2_cookies == {"session": "c2"}
    # The step body is the run body plus the single-turn response object,
    # action as a structured function_call item.
    assert step1_json["game_file"] == "json_2.1.1/train/x/game.tw-pddl"
    calls = [item for item in step1_json["response"]["output"] if item["type"] == "function_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "act"
    assert json.loads(calls[0]["arguments"]) == {"command": "go north"}
    assert calls[0]["call_id"] == "tc1"


async def test_extra_tool_calls_answered_ignored(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_MID, STEP_WON])
    policy_calls = _fake_policy(
        monkeypatch, mod, [_act_msg("go north", "look", start=1), _act_msg("take mug", start=3)]
    )
    agent = _make_agent(mod, env)

    result = await agent.run(request=None, body=_make_body(mod))

    assert result["extra_tool_calls"] == 1.0
    second_call_messages = policy_calls[1]["json"]["messages"]
    # Both tool_call_ids answered: observation to the first, notice to the second.
    assert [m["role"] for m in second_call_messages] == ["system", "user", "assistant", "tool", "tool"]
    assert second_call_messages[3] == {"role": "tool", "tool_call_id": "tc1", "content": "obs1"}
    assert second_call_messages[4]["tool_call_id"] == "tc2"
    assert "Ignored" in second_call_messages[4]["content"]


async def test_no_tool_call_turn_still_steps_env(monkeypatch):
    mod = _load_module()
    # Counters are cumulative per session, so the terminal step's info still
    # carries the earlier format error (the last-merged info wins in metrics).
    step_won = {**STEP_WON, "info": {**STEP_WON["info"], "action_format_errors": 1}}
    env = FakeEnvServer([STEP_FORMAT_ERROR, step_won])
    policy_calls = _fake_policy(
        monkeypatch,
        mod,
        [{"role": "assistant", "content": "I go north."}, _act_msg("go north")],
    )
    agent = _make_agent(mod, env)

    result = await agent.run(request=None, body=_make_body(mod))

    # The text-only turn was still an action attempt: /step was called with a
    # text-only response object (the env counts the format error).
    (_, _, step1_json, _) = env.calls[1]
    assert [item["type"] for item in step1_json["response"]["output"]] == ["message"]
    assert result["action_format_errors"] == 1.0
    # Its retry hint rode back as a user message (no call id to answer).
    second_call_messages = policy_calls[1]["json"]["messages"]
    assert [m["role"] for m in second_call_messages] == ["system", "user", "assistant", "user"]
    assert second_call_messages[3]["content"].startswith("Nothing happens.")


async def test_max_turns_exhaustion_truncates_and_closes(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_MID])
    _fake_policy(monkeypatch, mod, [_act_msg("go north")])
    agent = _make_agent(mod, env, max_turns=1)

    result = await agent.run(request=None, body=_make_body(mod))

    assert env.paths() == ["/reset", "/step", "/close"]
    # /close rides the last step's cookies.
    assert env.calls[2][3] == {"session": "c2"}
    assert result["env_truncated"] == 1.0
    assert result["reward"] == 0.0
    assert result["turns"] == 1.0


async def test_env_declared_truncation_no_close(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_ENV_TRUNCATED])
    _fake_policy(monkeypatch, mod, [_act_msg("go north")])
    agent = _make_agent(mod, env)

    result = await agent.run(request=None, body=_make_body(mod))

    # Env-declared truncation is a finished episode: the base server already
    # closed the session on that step.
    assert env.paths() == ["/reset", "/step"]
    assert result["env_truncated"] == 1.0
    assert result["turns"] == 1.0


async def test_policy_400_truncates_and_closes(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([])
    _fake_policy(monkeypatch, mod, [400])
    agent = _make_agent(mod, env)

    result = await agent.run(request=None, body=_make_body(mod))

    assert env.paths() == ["/reset", "/close"]
    assert result["policy_400_truncated"] == 1
    assert result["env_truncated"] == 1.0
    assert result["reward"] == 0.0
    assert result["turns"] == 0.0
    assert result["response"]["output"] == []


async def test_reset_failure_propagates(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([], reset_status=500)
    _fake_policy(monkeypatch, mod, [])
    agent = _make_agent(mod, env)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await agent.run(request=None, body=_make_body(mod))


async def test_missing_policy_base_url_rejected(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([])
    _fake_policy(monkeypatch, mod, [])
    agent = _make_agent(mod, env)

    with pytest.raises(AssertionError, match="policy_base_url"):
        await agent.run(request=None, body=_make_body(mod, policy_base_url=None))


async def test_last_observation_is_the_observation_after_the_final_action(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_MID, STEP_WON])
    _fake_policy(monkeypatch, mod, [_act_msg("go north"), _act_msg("take mug", start=2)])

    result = await _make_agent(mod, env).run(request=None, body=_make_body(mod))

    # The terminal step's observation, which the policy never saw; a string,
    # so gym_bridge's numeric env_metrics filter keeps it out of wandb env/*.
    assert result["last_observation"] == STEP_WON["observation"]
    # An episode the policy never stepped (400 on its first call) reports the
    # reset observation.
    env = FakeEnvServer([])
    _fake_policy(monkeypatch, mod, [400])
    result = await _make_agent(mod, env).run(request=None, body=_make_body(mod))
    assert result["last_observation"] == "obs0"


# ---- scripted prefix: recorded assistant turns replayed through /step before the first policy call ----

STEP_INVALID = {
    "observation": "Nothing happens.",
    "reward": 0.0,
    "terminated": False,
    "truncated": False,
    "info": {"won": False, "lost": False, "step_count": 1, "invalid_actions": 1, "action_format_errors": 0},
}
STEP_FORMAT_ERROR_AFTER_INVALID = {
    "observation": "Nothing happens. Call the act tool with exactly one admissible command.",
    "reward": 0.0,
    "terminated": False,
    "truncated": False,
    "info": {"won": False, "lost": False, "step_count": 2, "invalid_actions": 1, "action_format_errors": 1},
}


async def test_scripted_prefix_replays_before_the_first_policy_call(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_MID, STEP_MID, STEP_WON])
    policy_calls = _fake_policy(monkeypatch, mod, [_act_msg("take mug", content="Taking it.", start=9)])
    agent = _make_agent(mod, env)
    prefix = [_act_msg("go north", start=1), _act_msg("open box", start=2)]

    result = await agent.run(request=None, body=_make_body(mod, scripted_prefix=prefix))

    # two replayed steps, then one policy step; the env body never sees the prefix
    assert env.paths() == ["/reset", "/step", "/step", "/step"]
    assert all("scripted_prefix" not in (call[2] or {}) for call in env.calls)
    assert env.calls[1][2]["response"]["output"][0]["arguments"] == json.dumps({"command": "go north"})
    # the policy's first (and only) call sees the replayed turns and their observations
    msgs = policy_calls[0]["json"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert msgs[2] == prefix[0] and msgs[4] == prefix[1]
    assert msgs[3] == {"role": "tool", "tool_call_id": "tc1", "content": "obs1"}
    assert len(policy_calls) == 1
    assert result["scripted_steps"] == 2.0 and result["scripted_terminated"] == 0.0
    assert result["scripted_invalid_actions"] == 0.0 and result["scripted_format_errors"] == 0.0
    # replayed turns are not policy turns: transcript and turn count cover the policy only
    assert result["turns"] == 1.0 and len(result["response"]["output"]) == 1
    assert result["reward"] == 1.0 and result["won"] == 1.0 and result["step_count"] == 2.0
    # no prefix: the metrics still exist and are zero
    env = FakeEnvServer([STEP_WON])
    _fake_policy(monkeypatch, mod, [_act_msg("take mug")])
    result = await _make_agent(mod, env).run(request=None, body=_make_body(mod))
    assert result["scripted_steps"] == 0.0 and result["scripted_terminated"] == 0.0
    assert result["scripted_invalid_actions"] == 0.0 and result["scripted_format_errors"] == 0.0


async def test_scripted_prefix_that_ends_the_episode_leaves_the_policy_no_turn(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_WON])
    policy_calls = _fake_policy(monkeypatch, mod, [])
    agent = _make_agent(mod, env)

    result = await agent.run(
        request=None, body=_make_body(mod, scripted_prefix=[_act_msg("a"), _act_msg("b", start=2)])
    )

    assert env.paths() == ["/reset", "/step"] and policy_calls == []  # terminal: env closed it, no /close
    assert result["scripted_steps"] == 1.0 and result["scripted_terminated"] == 1.0
    assert result["turns"] == 0.0 and result["env_truncated"] == 0.0 and result["reward"] == 1.0
    # an env-declared truncation during the replay behaves the same way
    env = FakeEnvServer([STEP_ENV_TRUNCATED])
    result = await _make_agent(mod, env).run(request=None, body=_make_body(mod, scripted_prefix=[_act_msg("a")]))
    assert (
        env.paths() == ["/reset", "/step"] and result["scripted_terminated"] == 1.0 and result["env_truncated"] == 1.0
    )


async def test_scripted_prefix_reports_replay_faults(monkeypatch):
    mod = _load_module()
    env = FakeEnvServer([STEP_INVALID, STEP_FORMAT_ERROR_AFTER_INVALID, STEP_WON])
    policy_calls = _fake_policy(monkeypatch, mod, [_act_msg("take mug", start=9)])
    agent = _make_agent(mod, env)
    prefix = [_act_msg("fly", start=1), {"role": "assistant", "content": "no call this turn"}]

    result = await agent.run(request=None, body=_make_body(mod, scripted_prefix=prefix))

    assert result["scripted_steps"] == 2.0 and result["scripted_invalid_actions"] == 1.0
    assert result["scripted_format_errors"] == 1.0 and result["scripted_terminated"] == 0.0
    # the no-call replay turn got the env's hint as a user message, like a live format fault
    msgs = policy_calls[0]["json"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert msgs[5]["content"].startswith("Nothing happens.")


async def test_run_response_carries_the_structured_transcript_and_terminated(monkeypatch):
    """The /run response returns the chat transcript as the agent built it - the prompt messages, the reset
    observation, the replayed scripted turns and the policy's assistant messages verbatim (tool calls as the engine's
    parser returned them), each followed by its observation - and the Gymnasium ``terminated`` flag. gym_bridge
    forwards the transcript for the R2S pilot dump under MILES_OPD_DUMP_DIR; the takeover stage reads both."""
    mod = _load_module()
    env = FakeEnvServer([STEP_MID, STEP_MID, STEP_WON])
    prefix = [_act_msg("go north", content="North first.", start=1)]
    policy = [_act_msg("open box", start=2), _act_msg("take mug", content="Taking it.", start=3)]
    _fake_policy(monkeypatch, mod, policy)

    result = await _make_agent(mod, env).run(request=None, body=_make_body(mod, scripted_prefix=prefix))

    assert result["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "obs0"},
        prefix[0],
        {"role": "tool", "tool_call_id": "tc1", "content": "obs1"},
        policy[0],
        {"role": "tool", "tool_call_id": "tc2", "content": "obs1"},
        policy[1],
        {"role": "tool", "tool_call_id": "tc3", "content": STEP_WON["observation"]},
    ]
    assert result["terminated"] == 1.0 and result["won"] == 1.0 and result["env_truncated"] == 0.0
    json.dumps(result["messages"])  # plain JSON: it rides the /run response and the dump row
    # a locally truncated episode (max_turns) is not terminated; a format fault's hint is a user message
    env = FakeEnvServer([STEP_FORMAT_ERROR])
    _fake_policy(monkeypatch, mod, [{"role": "assistant", "content": "I go north."}])
    result = await _make_agent(mod, env, max_turns=1).run(request=None, body=_make_body(mod))
    assert result["terminated"] == 0.0 and result["env_truncated"] == 1.0
    assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant", "user"]
    assert result["messages"][3]["content"].startswith("Nothing happens.")
