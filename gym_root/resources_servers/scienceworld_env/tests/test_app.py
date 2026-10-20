"""scienceworld_env server tests (in-process mode, env_workers=0, fake factory).

The env factory is a config seam ("module:function"); pointing it at
fake_sciworld.make_env exercises the whole reset/step/close template without a
JVM. The process backend has its own tests (test_worker_pool.py).
"""

import json
import time
from unittest.mock import MagicMock

import fake_sciworld
import pytest
from fastapi import HTTPException
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from scienceworld_env.app import (
    ScienceworldResourcesServer,
    ScienceworldResourcesServerConfig,
    format_actions,
    format_reset_observation,
)
from scienceworld_env.task_splits import (
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    TASK_VARIATIONS,
    TRAIN_TASK_NAMES,
    build_train,
    build_train_heldout,
    validate_task,
)

SID = "session-1"
FAKE_FACTORY = "fake_sciworld:make_env"
ROW = {"task_name": "boil", "var_num": 3, "split": "train"}


def _make_config(**overrides) -> ScienceworldResourcesServerConfig:
    overrides.setdefault("sweep_interval_s", 0.0)
    overrides.setdefault("env_workers", 0)
    overrides.setdefault("env_factory", FAKE_FACTORY)
    return ScienceworldResourcesServerConfig(
        host="0.0.0.0", port=8100, entrypoint="", name="scienceworld_env", **overrides
    )


def _make_server(**overrides) -> ScienceworldResourcesServer:
    return ScienceworldResourcesServer(config=_make_config(**overrides), server_client=MagicMock(spec=ServerClient))


def _response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "test",
            "created_at": 0.0,
            "model": "test_model",
            "object": "response",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _text_item(text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": "msg_0",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _act_item(command: str, *, name: str = "act", call_id: str = "call_0") -> dict:
    return {
        "type": "function_call",
        "id": "fc_0",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps({"command": command}),
        "status": "completed",
    }


def _act_response(command: str, **kwargs) -> NeMoGymResponse:
    return _response([_act_item(command, **kwargs)])


# ──────────────────────────────────────────────────────────
# task_splits
# ──────────────────────────────────────────────────────────


class TestTaskSplits:
    def test_sizes_match_tcod(self):
        assert len(build_train()) == EXPECTED_TRAIN
        assert EXPECTED_TEST == 1308
        assert sum(TASK_VARIATIONS[n] - int(TASK_VARIATIONS[n] * 0.5) for n in TRAIN_TASK_NAMES) == len(
            build_train_heldout()
        )

    def test_train_and_heldout_disjoint(self):
        assert not set(build_train()) & set(build_train_heldout())

    def test_validate_task(self):
        assert validate_task("boil", 0, "train") is None
        assert validate_task("boil", 15, "train_heldout") is None
        assert validate_task("boil", 15, "train") is not None
        assert validate_task("boil", 0, "train_heldout") is not None
        assert validate_task("boil", 30, None) is not None  # out of range
        assert validate_task("grow-fruit", 0, "test") is None
        assert validate_task("boil", 0, "test") is not None  # train task type
        assert validate_task("nope", 0, None) is not None
        assert validate_task("boil", 0, "eval") is not None
        assert validate_task("boil", True, None) is not None


# ──────────────────────────────────────────────────────────
# formatting
# ──────────────────────────────────────────────────────────


class TestFormatting:
    def test_actions_drop_help_and_quote(self):
        assert format_actions(["look around", "help", "focus on OBJ"], 60) == "'look around', 'focus on OBJ'"

    def test_reset_observation_is_tcod_wording(self):
        cfg = _make_config()
        text = format_reset_observation(
            "Your task is to boil water.", " You see a stove. ", ["look around"], ["stove"], cfg
        )
        assert text == (
            "Your ScienceWorld task is: Your task is to boil water.\n"
            "Your current observation is: You see a stove.\n"
            "Available action commands: ['look around']\n"
            "Available objects you can interact with: ['stove']"
        )

    def test_truncation_marker(self):
        cfg = _make_config(max_obs_chars=10)
        text = format_reset_observation("t", "x" * 50, [], [], cfg)
        assert "[observation truncated]" in text


# ──────────────────────────────────────────────────────────
# reset
# ──────────────────────────────────────────────────────────


class TestReset:
    async def test_returns_formatted_obs_and_info(self):
        server = _make_server()
        obs, info = await server.reset(dict(ROW), SID)
        assert obs.startswith("Your ScienceWorld task is: Your task is to do boil variation 3.\n")
        assert "Your current observation is: This room is called the hallway." in obs
        assert "Available action commands: ['look around', 'focus on OBJ', 'open OBJ', 'go to OBJ']" in obs
        assert obs.endswith("Available objects you can interact with: ['air', 'door to kitchen', 'hallway']")
        assert info == {"task_name": "boil", "var_num": "3"}
        state = server.session_state[SID]
        assert state.episode.env.task == ("boil", 3, "easy")
        assert state.max_episode_steps == 30

    async def test_metadata_overrides_step_cap(self):
        server = _make_server()
        await server.reset(dict(ROW, max_episode_steps=5), SID)
        assert server.session_state[SID].max_episode_steps == 5

    async def test_var_num_as_string_is_coerced(self):
        server = _make_server()
        await server.reset({"task_name": "boil", "var_num": "3"}, SID)
        assert server.session_state[SID].var_num == 3

    @pytest.mark.parametrize(
        "row, needle",
        [
            ({"var_num": 0}, "task_name"),
            ({"task_name": "boil"}, "var_num"),
            ({"task_name": "boil", "var_num": "x"}, "var_num"),
            ({"task_name": "nope", "var_num": 0}, "unknown"),
            ({"task_name": "boil", "var_num": 99}, "out of range"),
            ({"task_name": "boil", "var_num": 20, "split": "train"}, "not in the train split"),
            ({"task_name": "boil", "var_num": 0, "split": "test"}, "train task type"),
            ({"task_name": "boil", "var_num": 0, "split": "eval"}, "unknown split"),
        ],
    )
    async def test_bad_rows_400_before_any_env(self, row, needle):
        server = _make_server()
        with pytest.raises(HTTPException) as exc:
            await server.reset(row, SID)
        assert exc.value.status_code == 400 and needle in exc.value.detail
        assert SID not in server.session_state

    async def test_bad_re_reset_leaves_prior_episode_intact(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        state = server.session_state[SID]
        with pytest.raises(HTTPException):
            await server.reset({"task_name": "nope", "var_num": 0}, SID)
        assert server.session_state[SID] is state
        assert state.episode.env.closed is False

    async def test_reused_session_releases_prior_env(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        first = server.session_state[SID].episode.env
        await server.reset({"task_name": "melt", "var_num": 1}, SID)
        assert first.closed is True
        assert server.session_state[SID].episode.env.task[0] == "melt"

    async def test_load_failure_propagates_and_registers_nothing(self):
        server = _make_server()
        with pytest.raises(RuntimeError, match="scripted load failure"):
            await server.reset({"task_name": "boil", "var_num": fake_sciworld.BOOM_VAR}, SID)
        assert SID not in server.session_state


# ──────────────────────────────────────────────────────────
# step
# ──────────────────────────────────────────────────────────


class TestStep:
    async def test_without_reset_400(self):
        server = _make_server()
        with pytest.raises(HTTPException) as exc:
            await server.step(_act_response("look around"), {}, SID)
        assert exc.value.status_code == 400

    async def test_happy_step_relists_actions_and_current_objects(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        obs, reward, terminated, truncated, info = await server.step(_act_response("look around"), {}, SID)
        assert obs == (
            "echo:look around:1\n"
            "Available action commands: ['look around', 'focus on OBJ', 'open OBJ', 'go to OBJ']\n"
            "Available objects you can interact with: ['air', 'door to kitchen', 'hallway', 'kitchen']"
        )
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert info == {
            "won": False,
            "step_count": 1,
            "invalid_actions": 0,
            "action_format_errors": 0,
            "task_score": 0.0,
        }

    async def test_dense_reward_is_running_max_delta_over_100(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        _, r1, *_ = await server.step(_act_response("half"), {}, SID)
        _, r2, _, _, info2 = await server.step(_act_response("drop"), {}, SID)  # score falls back to 0
        _, r3, term, trunc, info3 = await server.step(_act_response("win"), {}, SID)
        assert (r1, r2, r3) == (0.5, 0.0, 0.5)
        assert info2["task_score"] == 0.5 and info2["won"] is False
        assert (term, trunc) == (True, False)
        assert info3["won"] is True and info3["task_score"] == 1.0
        assert server.session_state[SID].done is True  # the base /step endpoint closes it

    async def test_terminal_completed_marks_done_and_text(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        obs, reward, terminated, _, info = await server.step(_act_response("win"), {}, SID)
        assert terminated is True and obs.endswith("\nTask completed.")
        assert server.session_state[SID].done is True
        # defensive replay of a finished episode
        obs2, r2, t2, _, _ = await server.step(_act_response("win"), {}, SID)
        assert (obs2, r2, t2) == (None, 0.0, True)

    async def test_no_match_counts_invalid_no_penalty(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        obs, reward, terminated, _, info = await server.step(_act_response("bogus"), {}, SID)
        assert obs.startswith("No known action matches that input.")
        assert (reward, terminated, info["invalid_actions"]) == (0.0, False, 1)

    async def test_format_error_is_observation_not_env_step(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        env = server.session_state[SID].episode.env
        obs, reward, terminated, truncated, info = await server.step(_response([_text_item("I think")]), {}, SID)
        assert obs.startswith("Invalid: no act tool call found.")
        assert "look around" in obs and "<tool_call>" not in obs  # the call format is the model's own
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert info["action_format_errors"] == 1 and info["step_count"] == 1
        assert env.steps == 0

    async def test_wrong_tool_name_is_format_error(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        *_, info = await server.step(_act_response("look around", name="look around"), {}, SID)
        assert info["action_format_errors"] == 1

    async def test_truncated_at_step_cap(self):
        server = _make_server()
        await server.reset(dict(ROW, max_episode_steps=2), SID)
        _, _, t1, tr1, _ = await server.step(_act_response("look around"), {}, SID)
        _, _, t2, tr2, _ = await server.step(_act_response("look around"), {}, SID)
        assert (t1, tr1, t2, tr2) == (False, False, False, True)

    async def test_simulator_done_without_completion_is_not_won(self):
        server = _make_server(env_step_limit=1)
        await server.reset(dict(ROW), SID)
        obs, reward, terminated, _, info = await server.step(_act_response("look around"), {}, SID)
        assert terminated is True and info["won"] is False and obs.endswith("\nTask finished.")

    async def test_step_failure_evicts_and_releases(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        env = server.session_state[SID].episode.env
        with pytest.raises(RuntimeError, match="scripted step failure"):
            await server.step(_act_response("explode"), {}, SID)
        assert SID not in server.session_state
        assert env.closed is True

    async def test_sessions_are_isolated(self):
        server = _make_server()
        await server.reset(dict(ROW), "a")
        await server.reset({"task_name": "melt", "var_num": 2}, "b")
        await server.step(_act_response("half"), {}, "a")
        *_, info_b = await server.step(_act_response("look around"), {}, "b")
        assert info_b["task_score"] == 0.0
        assert server.session_state["a"].best_score == 50.0


# ──────────────────────────────────────────────────────────
# close / sweeper / guards
# ──────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_close_session_releases_env(self):
        server = _make_server()
        await server.reset(dict(ROW), SID)
        env = server.session_state[SID].episode.env
        await server.close_session(SID)
        assert SID not in server.session_state and env.closed is True
        await server.close_session(SID)  # idempotent

    async def test_sweeper_evicts_stale_keeps_fresh(self):
        server = _make_server(session_ttl_s=1.0)
        await server.reset(dict(ROW), "old")
        await server.reset(dict(ROW), "new")
        server.session_state["old"].last_access = time.monotonic() - 5.0
        assert await server.sweep_stale_sessions() == 1
        assert set(server.session_state) == {"new"}

    def test_num_workers_guard(self):
        # pydantic wraps model_post_init assertion failures into ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="num_workers"):
            _make_server(num_workers=2)
