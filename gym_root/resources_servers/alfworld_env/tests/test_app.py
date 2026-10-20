"""alfworld_env server tests.

Everything except TestRealGame runs without alfworld installed: the blocking
loader `_make_env` is module-level exactly so these tests can monkeypatch it
with a scripted fake env (nemo-gym is still required for the base classes).
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import alfworld_env.app as app_module
import pytest
from alfworld_env.app import (
    AlfworldResourcesServer,
    AlfworldResourcesServerConfig,
    EnvPool,
    extract_command,
    format_observation,
)
from fastapi import HTTPException
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient

SID = "session-1"


def _make_config(data_root: str, **overrides) -> AlfworldResourcesServerConfig:
    overrides.setdefault("sweep_interval_s", 0.0)  # no background task per test loop
    # In-process envs so the fake-env monkeypatch works (spawn'd workers
    # would not see it); the process backend has its own tests below.
    overrides.setdefault("env_workers", 0)
    return AlfworldResourcesServerConfig(
        host="0.0.0.0",
        port=8100,
        entrypoint="",
        name="alfworld_env",
        alfworld_data=data_root,
        **overrides,
    )


def _make_server(data_root: str, **overrides) -> AlfworldResourcesServer:
    return AlfworldResourcesServer(
        config=_make_config(data_root, **overrides), server_client=MagicMock(spec=ServerClient)
    )


@pytest.fixture
def game_dir(tmp_path):
    game = tmp_path / "json_2.1.1" / "train" / "t1" / "game.tw-pddl"
    game.parent.mkdir(parents=True)
    game.write_text("{}")
    return tmp_path


GAME_REL = "json_2.1.1/train/t1/game.tw-pddl"


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


def _act_item(
    command: str | None = None, *, name: str = "act", arguments: str | None = None, call_id: str = "call_0"
) -> dict:
    return {
        "type": "function_call",
        "id": "fc_0",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps({"command": command}) if arguments is None else arguments,
        "status": "completed",
    }


def _act_response(command: str, **kwargs) -> NeMoGymResponse:
    return _response([_act_item(command, **kwargs)])


FAKE_INITIAL_OBS = "You are in the kitchen."
FAKE_INITIAL_INFO = {"admissible_commands": ["look", "go north"]}


class FakeEnv:
    """Scripted (obs, score, done, info) tuples; falls back to a neutral step."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.commands: list[str] = []
        self.closed = False
        self.resets = 0

    def step(self, command: str):
        self.commands.append(command)
        if self.script:
            return self.script.pop(0)
        return ("You move.", 0, False, {"won": False, "lost": False, "admissible_commands": ["look", "go north"]})

    def reset(self):
        # Pool-reuse path: same (obs, info) as a fresh load of the same game.
        self.resets += 1
        return FAKE_INITIAL_OBS, dict(FAKE_INITIAL_INFO)

    def close(self):
        self.closed = True


def install_fake_env(monkeypatch, script=None, fail_first=False):
    """Replace the blocking loader; returns the list of envs it created."""
    created: list[FakeEnv] = []
    calls = {"n": 0}

    def fake_make_env(game_file: str, want_admissible: bool):
        calls["n"] += 1
        if fail_first and calls["n"] == 1:
            raise RuntimeError("transient tatsu flake")
        env = FakeEnv(script)
        created.append(env)
        return env, FAKE_INITIAL_OBS, dict(FAKE_INITIAL_INFO)

    monkeypatch.setattr(app_module, "_make_env", fake_make_env)
    return created


# ──────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────


class TestExtractCommand:
    def test_text_only_is_none(self):
        assert extract_command(_response([_text_item("I go north.")]), "act") is None

    def test_empty_output_is_none(self):
        assert extract_command(_response([]), "act") is None

    def test_valid_call(self):
        assert extract_command(_act_response(" go north "), "act") == "go north"

    def test_first_call_wins(self):
        resp = _response([_act_item("look"), _act_item("go north", call_id="call_1")])
        assert extract_command(resp, "act") == "look"

    def test_text_before_call_ok(self):
        resp = _response([_text_item("Let me look around."), _act_item("look")])
        assert extract_command(resp, "act") == "look"

    def test_wrong_tool_name_is_none(self):
        assert extract_command(_act_response("look", name="search"), "act") is None

    def test_bad_json_args_is_none(self):
        assert extract_command(_act_response("x", arguments="{not json"), "act") is None

    def test_non_dict_args_is_none(self):
        assert extract_command(_act_response("x", arguments='"look"'), "act") is None

    def test_missing_command_key_is_none(self):
        assert extract_command(_act_response("x", arguments='{"cmd": "look"}'), "act") is None

    def test_non_string_command_is_none(self):
        assert extract_command(_act_response("x", arguments='{"command": 3}'), "act") is None

    def test_empty_command_is_none(self):
        assert extract_command(_act_response("  "), "act") is None


class TestFormatObservation:
    def _config(self, **overrides):
        return _make_config("/data", **overrides)

    def test_appends_admissible(self):
        obs = format_observation("You see a table.", ["look", "go north"], self._config())
        assert obs == "You see a table.\nYour admissible actions are: [look, go north]"

    def test_flag_off(self):
        obs = format_observation("You see a table.", ["look"], self._config(expose_admissible_commands=False))
        assert obs == "You see a table."

    def test_admissible_capped(self):
        config = self._config(max_admissible_commands=2)
        obs = format_observation("x", ["a", "b", "c", "d"], config)
        assert obs.endswith("[a, b]")

    def test_truncation(self):
        config = self._config(max_obs_chars=10)
        obs = format_observation("y" * 50, None, config)
        assert obs.startswith("y" * 10 + "\n[observation truncated]")

    def test_no_admissible(self):
        assert format_observation("plain", None, self._config()) == "plain"


# ──────────────────────────────────────────────────────────
# reset
# ──────────────────────────────────────────────────────────


class TestReset:
    async def test_returns_formatted_obs_and_info(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        obs, info = await server.reset({"game_file": GAME_REL}, SID)
        assert obs == "You are in the kitchen.\nYour admissible actions are: [look, go north]"
        assert info == {"env_load_retries": 0, "env_pool_hit": 0}

    async def test_missing_game_file_400(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        with pytest.raises(HTTPException) as e:
            await server.reset({}, SID)
        assert e.value.status_code == 400

    async def test_nonexistent_game_400(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        with pytest.raises(HTTPException) as e:
            await server.reset({"game_file": "json_2.1.1/train/nope/game.tw-pddl"}, SID)
        assert e.value.status_code == 400

    async def test_relative_path_without_data_root_400(self, monkeypatch):
        install_fake_env(monkeypatch)
        monkeypatch.delenv("ALFWORLD_DATA", raising=False)
        server = _make_server("")
        with pytest.raises(HTTPException) as e:
            await server.reset({"game_file": GAME_REL}, SID)
        assert e.value.status_code == 400

    async def test_env_var_fallback(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        monkeypatch.setenv("ALFWORLD_DATA", str(game_dir))
        server = _make_server("")
        obs, _ = await server.reset({"game_file": GAME_REL}, SID)
        assert obs.startswith("You are in the kitchen.")

    async def test_load_retry_once(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch, fail_first=True)
        server = _make_server(str(game_dir))
        _, info = await server.reset({"game_file": GAME_REL}, SID)
        assert info == {"env_load_retries": 1, "env_pool_hit": 0}

    async def test_reused_session_recycles_prior_env(self, monkeypatch, game_dir):
        # A reused cookie re-resets: the prior env is parked in the pool and
        # (same game) immediately taken back — one load, one reuse-reset.
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)
        _, info = await server.reset({"game_file": GAME_REL}, SID)
        assert len(created) == 1
        assert created[0].resets == 1 and not created[0].closed
        assert info == {"env_load_retries": 0, "env_pool_hit": 1}

    async def test_bad_re_reset_leaves_prior_episode_intact(self, monkeypatch, game_dir):
        # Row validation runs BEFORE the reused cookie's previous episode is
        # released: a bad row 400s and the live episode is untouched.
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)
        state = server.session_state[SID]
        bad_rows = (
            {"game_file": "json_2.1.1/train/nope/game.tw-pddl"},
            {},
            {"game_file": GAME_REL, "max_episode_steps": "abc"},
        )
        for bad in bad_rows:
            with pytest.raises(HTTPException) as ei:
                await server.reset(bad, SID)
            assert ei.value.status_code == 400
        assert server.session_state[SID] is state and len(created) == 1 and not created[0].closed
        obs, *_ = await server.step(_act_response("go north"), {}, SID)
        assert obs

    async def test_reused_session_closes_prior_env_pool_disabled(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir), env_pool_size=0)
        await server.reset({"game_file": GAME_REL}, SID)
        await server.reset({"game_file": GAME_REL}, SID)
        assert len(created) == 2
        assert created[0].closed and not created[1].closed


# ──────────────────────────────────────────────────────────
# step
# ──────────────────────────────────────────────────────────


class TestStep:
    async def _reset(self, server, metadata=None):
        return await server.reset({"game_file": GAME_REL, **(metadata or {})}, SID)

    async def test_without_reset_400(self, game_dir):
        server = _make_server(str(game_dir))
        with pytest.raises(HTTPException) as e:
            await server.step(_act_response("look"), {}, SID)
        assert e.value.status_code == 400

    async def test_happy_step(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await self._reset(server)
        obs, reward, terminated, truncated, info = await server.step(_act_response("go north"), {}, SID)
        assert created[0].commands == ["go north"]
        assert obs == "You move.\nYour admissible actions are: [look, go north]"
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert info["step_count"] == 1
        assert info["won"] is False

    async def test_format_error_is_observation_not_env_step(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await self._reset(server)
        obs, reward, terminated, truncated, info = await server.step(_response([_text_item("I go north.")]), {}, SID)
        assert created[0].commands == []  # env never stepped
        assert "no act tool call found" in obs and "<one admissible command>" in obs
        assert "<tool_call>" not in obs  # the call format is the model's own, never the env's
        assert info["action_format_errors"] == 1
        assert info["step_count"] == 1  # still spends budget
        assert (reward, terminated, truncated) == (0.0, False, False)

    async def test_nothing_happens_counts_invalid(self, monkeypatch, game_dir):
        script = [("Nothing happens.", 0, False, {"won": False, "lost": False})]
        install_fake_env(monkeypatch, script=script)
        server = _make_server(str(game_dir))
        await self._reset(server)
        _, _, _, _, info = await server.step(_act_response("eat table"), {}, SID)
        assert info["invalid_actions"] == 1

    async def test_terminal_won(self, monkeypatch, game_dir):
        script = [("You put the mug on the desk.", 1, True, {"won": True, "lost": False})]
        install_fake_env(monkeypatch, script=script)
        server = _make_server(str(game_dir))
        await self._reset(server)
        obs, reward, terminated, truncated, info = await server.step(_act_response("put mug on desk"), {}, SID)
        assert (reward, terminated, truncated) == (1.0, True, False)
        assert obs.endswith("Task completed.")
        assert info["won"] is True

    async def test_terminal_lost(self, monkeypatch, game_dir):
        script = [("You broke it.", 0, True, {"won": False, "lost": True})]
        install_fake_env(monkeypatch, script=script)
        server = _make_server(str(game_dir))
        await self._reset(server)
        obs, reward, terminated, truncated, info = await server.step(_act_response("smash mug"), {}, SID)
        assert (reward, terminated, truncated) == (0.0, True, False)
        assert obs.endswith("Task failed.")
        assert info["lost"] is True

    async def test_truncated_at_step_cap(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await self._reset(server, {"max_episode_steps": 2})
        _, _, terminated, truncated, _ = await server.step(_act_response("look"), {}, SID)
        assert (terminated, truncated) == (False, False)
        _, reward, terminated, truncated, info = await server.step(_act_response("look"), {}, SID)
        assert (terminated, truncated) == (False, True)
        assert reward == 0.0
        assert info["step_count"] == 2

    async def test_config_step_cap_when_no_metadata(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir), max_episode_steps=1)
        await self._reset(server)
        _, _, _, truncated, _ = await server.step(_act_response("look"), {}, SID)
        assert truncated is True

    async def test_done_replay_defense(self, monkeypatch, game_dir):
        script = [("Done.", 1, True, {"won": True, "lost": False})]
        install_fake_env(monkeypatch, script=script)
        server = _make_server(str(game_dir))
        await self._reset(server)
        await server.step(_act_response("finish"), {}, SID)
        obs, reward, terminated, truncated, _ = await server.step(_act_response("look"), {}, SID)
        assert (obs, reward, terminated, truncated) == (None, 0.0, True, False)


# ──────────────────────────────────────────────────────────
# session lifecycle
# ──────────────────────────────────────────────────────────


class TestSessions:
    async def test_close_session_parks_env_in_pool(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)
        await server.close_session(SID)
        assert not created[0].closed
        assert len(server._env_pool) == 1
        assert server.session_state == {}

    async def test_close_session_pool_disabled_closes_env(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir), env_pool_size=0)
        await server.reset({"game_file": GAME_REL}, SID)
        await server.close_session(SID)
        assert created[0].closed
        assert len(server._env_pool) == 0
        assert server.session_state == {}

    async def test_sweeper_evicts_stale_keeps_fresh(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir), session_ttl_s=1000.0)
        await server.reset({"game_file": GAME_REL}, "stale")
        await server.reset({"game_file": GAME_REL}, "fresh")
        server.session_state["stale"].last_access = time.monotonic() - 2000.0
        evicted = await server.sweep_stale_sessions()
        assert evicted == 1
        assert not created[0].closed and len(server._env_pool) == 1  # parked, not closed
        assert set(server.session_state) == {"fresh"}

    async def test_sessions_are_isolated(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, "a")
        await server.reset({"game_file": GAME_REL}, "b")
        await server.step(_act_response("go north"), {}, "a")
        await server.step(_act_response("look"), {}, "b")
        await server.step(_act_response("take mug"), {}, "a")
        assert created[0].commands == ["go north", "take mug"]
        assert created[1].commands == ["look"]

    async def test_step_liveness_recheck_after_eviction(self, monkeypatch, game_dir):
        # A /step that fetched the state but lost the lock race to a close
        # (truncated auto-close, sweeper, /close) must NOT step the env.
        import asyncio

        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)
        state = server.session_state[SID]
        async with state.lock:
            task = asyncio.create_task(server.step(_act_response("look"), {}, SID))
            await asyncio.sleep(0)  # step fetched the state ref, now blocked on the lock
            del server.session_state[SID]  # the close path popped it
        obs, reward, terminated, truncated, _ = await task
        assert (obs, reward, terminated, truncated) == (None, 0.0, True, False)
        assert created[0].commands == []  # env never stepped

    @staticmethod
    def _install_blocking_env(monkeypatch):
        # An env whose step blocks until the gate opens, recording event order.
        import threading as _threading

        gate = _threading.Event()
        order: list[str] = []
        created: list[FakeEnv] = []

        class BlockingEnv(FakeEnv):
            def step(self, command):
                order.append("step_enter")
                gate.wait(5)
                order.append("step_exit")
                return super().step(command)

            def close(self):
                order.append("close")
                super().close()

        def fake_make_env(game_file, want_admissible):
            env = BlockingEnv()
            created.append(env)
            return env, "obs", {"admissible_commands": ["look"]}

        monkeypatch.setattr(app_module, "_make_env", fake_make_env)
        return gate, order, created

    async def test_close_waits_for_inflight_step(self, monkeypatch, game_dir):
        # env.close must not run concurrently with env.step on the same env.
        import asyncio

        gate, order, created = self._install_blocking_env(monkeypatch)
        server = _make_server(str(game_dir), env_pool_size=0)  # close is the real close
        await server.reset({"game_file": GAME_REL}, SID)

        step_task = asyncio.create_task(server.step(_act_response("look"), {}, SID))
        while "step_enter" not in order:  # step is inside env.step, holding the lock
            await asyncio.sleep(0.01)
        close_task = asyncio.create_task(server.close_session(SID))
        await asyncio.sleep(0.05)
        assert not created[0].closed  # close is parked on the session lock
        gate.set()
        await asyncio.gather(step_task, close_task)
        assert order == ["step_enter", "step_exit", "close"]

    async def test_pool_park_waits_for_inflight_step(self, monkeypatch, game_dir):
        # Same discipline with pooling on: the env must not become takeable
        # by another session while a step is still inside it.
        import asyncio

        gate, order, created = self._install_blocking_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)

        step_task = asyncio.create_task(server.step(_act_response("look"), {}, SID))
        while "step_enter" not in order:
            await asyncio.sleep(0.01)
        close_task = asyncio.create_task(server.close_session(SID))
        await asyncio.sleep(0.05)
        assert len(server._env_pool) == 0  # park is waiting on the session lock
        gate.set()
        await asyncio.gather(step_task, close_task)
        assert order == ["step_enter", "step_exit"]
        assert len(server._env_pool) == 1 and not created[0].closed

    def test_num_workers_guard(self, game_dir):
        # pydantic wraps model_post_init assertion failures into ValidationError
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="num_workers"):
            _make_server(str(game_dir), num_workers=4)

    async def test_sweeper_task_starts_lazily(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir), sweep_interval_s=30.0)
        assert getattr(server, "_sweeper_task", None) is None
        await server.reset({"game_file": GAME_REL}, SID)
        assert server._sweeper_task is not None and not server._sweeper_task.done()
        server._sweeper_task.cancel()


# ──────────────────────────────────────────────────────────
# EnvPool + reuse through the server
# ──────────────────────────────────────────────────────────


class TestEnvPool:
    def test_take_from_empty_is_none(self):
        assert EnvPool(4).take("g1") is None

    def test_put_take_roundtrip_removes(self):
        pool, env = EnvPool(4), FakeEnv()
        assert pool.put("g1", env) == []
        assert len(pool) == 1
        assert pool.take("g1") is env
        assert len(pool) == 0
        assert pool.take("g1") is None  # ownership transferred, not shared

    def test_take_wrong_game_is_none(self):
        pool = EnvPool(4)
        pool.put("g1", FakeEnv())
        assert pool.take("g2") is None
        assert len(pool) == 1

    def test_take_prefers_most_recently_parked(self):
        pool, e1, e2 = EnvPool(4), FakeEnv(), FakeEnv()
        pool.put("g1", e1)
        pool.put("g1", e2)
        assert pool.take("g1") is e2
        assert pool.take("g1") is e1

    def test_eviction_drops_longest_idle_first(self):
        pool, e1, e2, e3 = EnvPool(2), FakeEnv(), FakeEnv(), FakeEnv()
        assert pool.put("g1", e1) == []
        assert pool.put("g2", e2) == []
        assert pool.put("g3", e3) == [e1]  # caller closes e1
        assert len(pool) == 2
        assert pool.take("g1") is None
        assert pool.take("g2") is e2 and pool.take("g3") is e3

    def test_reparked_env_refreshes_idle_age(self):
        pool, e1, e2, e3 = EnvPool(2), FakeEnv(), FakeEnv(), FakeEnv()
        pool.put("g1", e1)
        pool.put("g2", e2)
        assert pool.take("g1") is e1
        pool.put("g1", e1)  # e1 parked again -> e2 is now the longest idle
        assert pool.put("g3", e3) == [e2]

    def test_zero_size_pool_returns_env_for_closing(self):
        pool, env = EnvPool(0), FakeEnv()
        assert pool.put("g1", env) == [env]
        assert len(pool) == 0


class TestEnvReuse:
    async def test_reset_reuses_parked_env_with_identical_obs(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        obs1, info1 = await server.reset({"game_file": GAME_REL}, "ep-1")
        env = server.session_state["ep-1"].env
        await server.close_session("ep-1")
        obs2, info2 = await server.reset({"game_file": GAME_REL}, "ep-2")
        assert len(created) == 1  # no second load
        assert server.session_state["ep-2"].env is env
        assert env.resets == 1
        assert obs2 == obs1
        assert (info1["env_pool_hit"], info2["env_pool_hit"]) == (0, 1)
        assert len(server._env_pool) == 0  # taken back out

    async def test_pool_miss_on_other_game_loads_fresh(self, monkeypatch, game_dir):
        other = game_dir / "json_2.1.1" / "train" / "t2" / "game.tw-pddl"
        other.parent.mkdir(parents=True)
        other.write_text("{}")
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)
        await server.close_session(SID)
        _, info = await server.reset({"game_file": "json_2.1.1/train/t2/game.tw-pddl"}, SID)
        assert len(created) == 2
        assert info["env_pool_hit"] == 0
        assert len(server._env_pool) == 1  # the t1 env stays parked

    async def test_pool_bound_eviction_closes_env(self, monkeypatch, game_dir):
        games = [GAME_REL]
        for name in ("t2", "t3"):
            g = game_dir / "json_2.1.1" / "train" / name / "game.tw-pddl"
            g.parent.mkdir(parents=True)
            g.write_text("{}")
            games.append(f"json_2.1.1/train/{name}/game.tw-pddl")
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir), env_pool_size=2)
        for i, game in enumerate(games):
            await server.reset({"game_file": game}, f"s{i}")
            await server.close_session(f"s{i}")
        assert len(server._env_pool) == 2
        assert created[0].closed  # evicted: longest idle
        assert not created[1].closed and not created[2].closed

    async def test_no_cross_session_sharing(self, monkeypatch, game_dir):
        # One parked env, two new sessions on the same game: exactly one gets
        # the parked env, the other a fresh load, and steps stay routed.
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, "old")
        await server.close_session("old")
        await server.reset({"game_file": GAME_REL}, "a")
        await server.reset({"game_file": GAME_REL}, "b")
        env_a, env_b = server.session_state["a"].env, server.session_state["b"].env
        assert env_a is not env_b
        assert len(created) == 2  # one reuse, one fresh load
        await server.step(_act_response("go north"), {}, "a")
        await server.step(_act_response("look"), {}, "b")
        assert env_a.commands == ["go north"]
        assert env_b.commands == ["look"]

    async def test_broken_parked_env_falls_back_to_fresh_load(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        await server.reset({"game_file": GAME_REL}, SID)
        await server.close_session(SID)
        created[0].reset = MagicMock(side_effect=RuntimeError("downward state trashed"))
        obs, info = await server.reset({"game_file": GAME_REL}, SID)
        assert created[0].closed  # broken env dropped for real
        assert len(created) == 2
        assert obs.startswith(FAKE_INITIAL_OBS)
        assert info == {"env_load_retries": 0, "env_pool_hit": 0}


# ──────────────────────────────────────────────────────────
# HTTP cookie flow (full FastAPI app, signed-cookie sessions)
# ──────────────────────────────────────────────────────────


class TestHttpFlow:
    def _client(self, server):
        from fastapi.testclient import TestClient

        # No context manager: startup handlers (the TTL sweeper) stay off.
        return TestClient(server.setup_webserver())

    def _step_body(self, command: str) -> dict:
        return {
            "responses_create_params": {"input": []},
            "game_file": GAME_REL,
            "response": _act_response(command).model_dump(mode="json"),
        }

    def test_reset_step_terminal_autoclose(self, monkeypatch, game_dir):
        script = [("Done.", 1, True, {"won": True, "lost": False})]
        created = install_fake_env(monkeypatch, script=script)
        server = _make_server(str(game_dir))
        client = self._client(server)

        r = client.post("/reset", json={"responses_create_params": {"input": []}, "game_file": GAME_REL})
        assert r.status_code == 200
        assert r.json()["observation"].startswith("You are in the kitchen.")
        assert len(server.session_state) == 1

        r = client.post("/step", json=self._step_body("finish"))
        assert r.status_code == 200
        body = r.json()
        assert body["terminated"] is True
        assert body["reward"] == 1.0
        # Terminal step: the base endpoint auto-closed the session, which
        # parks the env in the pool for the next reset of this game.
        assert server.session_state == {}
        assert not created[0].closed and len(server._env_pool) == 1

    def test_step_without_reset_is_400(self, monkeypatch, game_dir):
        install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        client = self._client(server)
        r = client.post("/step", json=self._step_body("look"))
        assert r.status_code == 400

    def test_close_endpoint(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        client = self._client(server)
        assert client.post("/close", json={}).json() == {"closed": False}
        client.post("/reset", json={"responses_create_params": {"input": []}, "game_file": GAME_REL})
        assert client.post("/close", json={}).json() == {"closed": True}
        assert not created[0].closed and len(server._env_pool) == 1  # parked for reuse

    def test_two_clients_isolated(self, monkeypatch, game_dir):
        created = install_fake_env(monkeypatch)
        server = _make_server(str(game_dir))
        client_a, client_b = self._client(server), self._client(server)
        client_a.post("/reset", json={"responses_create_params": {"input": []}, "game_file": GAME_REL})
        client_b.post("/reset", json={"responses_create_params": {"input": []}, "game_file": GAME_REL})
        client_a.post("/step", json=self._step_body("go north"))
        client_b.post("/step", json=self._step_body("look"))
        assert created[0].commands == ["go north"]
        assert created[1].commands == ["look"]


# ──────────────────────────────────────────────────────────
# Real game (needs alfworld + $ALFWORLD_DATA; pins upstream tuple semantics)
# ──────────────────────────────────────────────────────────


def _real_data_root() -> Path | None:
    root = os.environ.get("ALFWORLD_DATA")
    if root and (Path(root) / "json_2.1.1").is_dir():
        return Path(root)
    return None


@pytest.mark.skipif(_real_data_root() is None, reason="$ALFWORLD_DATA with json_2.1.1 not available")
class TestRealGame:
    def test_real_game_step_semantics(self):
        """Pin the (obs, score, done, info) step-tuple order and info["won"]
        as the success signal — the whole reward path assumes them.
        (tale-suite unpacks the same tuple as `obs, done, reward, info`,
        which mislabels the slots; never copy that.)"""
        pytest.importorskip("alfworld", reason="alfworld not installed in this venv")

        games = sorted(_real_data_root().glob("json_2.1.1/valid_seen/**/game.tw-pddl"))
        assert games, "no valid_seen games under $ALFWORLD_DATA"
        env, obs, info = app_module._locked_make_env(str(games[0]), want_admissible=True)
        try:
            assert isinstance(obs, str) and obs
            admissible = info["admissible_commands"]
            assert isinstance(admissible, list) and admissible
            step_obs, score, done, step_info = env.step(admissible[0])
            assert isinstance(step_obs, str)
            assert isinstance(score, (int, float)) and not isinstance(score, bool)
            assert isinstance(done, bool)
            assert isinstance(step_info, dict)
            assert isinstance(step_info["won"], bool)
        finally:
            env.close()

    async def test_real_game_full_reset_step(self):
        """One /reset + /step through the server against a real game."""
        pytest.importorskip("alfworld", reason="alfworld not installed in this venv")

        server = _make_server(str(_real_data_root()), max_concurrency=2)
        games = sorted(_real_data_root().glob("json_2.1.1/valid_seen/**/game.tw-pddl"))
        rel = str(games[0].relative_to(_real_data_root()))
        obs, reset_info = await server.reset({"game_file": rel}, SID)
        assert "Your admissible actions are: [" in obs
        assert reset_info["env_load_retries"] in (0, 1)
        obs2, reward, terminated, truncated, info = await server.step(_act_response("look"), {}, SID)
        assert isinstance(obs2, str) and obs2
        assert reward == 0.0
        assert terminated is False and truncated is False
        assert info["step_count"] == 1
        await server.close_session(SID)  # parks the env
        server._env_pool.take(str(games[0])).close()

    async def test_real_game_walkthrough_twice_through_pooled_env(self):
        """Reuse guarantee end-to-end: episode 2 takes the parked env, starts
        from a byte-identical initial observation, and the walkthrough wins
        again (reward 1.0 at the terminal step)."""
        pytest.importorskip("alfworld", reason="alfworld not installed in this venv")

        root = _real_data_root()
        game = next(
            g
            for g in sorted(root.glob("json_2.1.1/valid_seen/**/game.tw-pddl"))
            if json.loads(g.read_text()).get("walkthrough")
        )
        walkthrough = json.loads(game.read_text())["walkthrough"]
        server = _make_server(str(root), max_concurrency=2)
        metadata = {
            "game_file": str(game.relative_to(root)),
            "max_episode_steps": len(walkthrough) + 5,
        }

        async def play(sid):
            obs0, reset_info = await server.reset(metadata, sid)
            env = server.session_state[sid].env
            reward = terminated = None
            for cmd in walkthrough:
                _, reward, terminated, truncated, _ = await server.step(_act_response(cmd), {}, sid)
                if terminated or truncated:
                    break
            assert terminated is True and reward == 1.0
            # Direct method calls bypass the base /step endpoint's auto-close.
            await server.close_session(sid)
            return obs0, reset_info, env

        obs1, info1, env1 = await play("ep-1")
        obs2, info2, env2 = await play("ep-2")
        assert (info1["env_pool_hit"], info2["env_pool_hit"]) == (0, 1)
        assert env2 is env1
        assert obs2 == obs1
        server._env_pool.take(str(game)).close()
