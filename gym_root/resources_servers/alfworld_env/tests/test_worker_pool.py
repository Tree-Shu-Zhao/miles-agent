"""Process backend tests (_worker.WorkerPool + server with env_workers > 0).

These spawn REAL worker processes; env behavior is scripted through the
env_factory config seam (fake_worker_factory) because monkeypatching cannot
cross the spawn boundary. The gated TestRealGameProcessBackend at the bottom
runs the true alfworld factory end to end.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from alfworld_env._worker import EnvWorkerDied, EnvWorkerError, WorkerPool
from alfworld_env.app import AlfworldResourcesServer, AlfworldResourcesServerConfig
from test_app import GAME_REL, _act_response

FAKE_FACTORY = "fake_worker_factory:make_env"
SID = "session-1"


def _make_server(data_root: str, **overrides) -> AlfworldResourcesServer:
    overrides.setdefault("sweep_interval_s", 0.0)
    overrides.setdefault("env_workers", 2)
    overrides.setdefault("env_call_timeout_s", 30.0)
    overrides.setdefault("env_factory", FAKE_FACTORY)  # real one gated below
    config = AlfworldResourcesServerConfig(
        host="0.0.0.0",
        port=8100,
        entrypoint="",
        name="alfworld_env",
        alfworld_data=data_root,
        **overrides,
    )
    from nemo_gym.server_utils import ServerClient

    return AlfworldResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def game_dir(tmp_path):
    game = tmp_path / "json_2.1.1" / "train" / "t1" / "game.tw-pddl"
    game.parent.mkdir(parents=True)
    game.write_text("{}")
    return tmp_path


# ──────────────────────────────────────────────────────────
# WorkerPool directly (no server, no fastapi)
# ──────────────────────────────────────────────────────────


class TestWorkerPool:
    async def test_load_step_close_roundtrip(self):
        pool = WorkerPool(1, 30.0, FAKE_FACTORY)
        env, obs, info = await pool.load("g1", True)
        assert obs == "fake obs for g1"
        assert info["admissible_commands"] == ["look", "win"]
        step_obs, score, done, step_info = await env.step("look")
        assert (step_obs, score, done) == ("echo:look:1", 0, False)
        assert step_info["won"] is False
        step_obs, score, done, step_info = await env.step("win")
        assert (done, step_info["won"]) == (True, True)
        await env.close()
        assert pool._counts == [0]

    async def test_many_envs_one_worker_stay_isolated(self):
        pool = WorkerPool(1, 30.0, FAKE_FACTORY)
        env_a, _, _ = await pool.load("gA", True)
        env_b, _, _ = await pool.load("gB", True)
        await env_a.step("look")
        await env_a.step("look")
        await env_b.step("look")
        assert (await env_a.step("look"))[0] == "echo:look:3"
        assert (await env_b.step("look"))[0] == "echo:look:2"

    async def test_remote_reset_restores_initial_state(self):
        pool = WorkerPool(1, 30.0, FAKE_FACTORY)
        env, obs, info = await pool.load("g1", True)
        await env.step("look")
        obs2, info2 = await env.reset()
        assert obs2 == obs and info2 == info
        assert (await env.step("look"))[0] == "echo:look:1"  # counter rewound

    async def test_load_spreads_to_least_loaded_worker(self):
        pool = WorkerPool(2, 30.0, FAKE_FACTORY)
        envs = [(await pool.load(f"g{i}", True))[0] for i in range(4)]
        assert sorted(pool._counts) == [2, 2]
        assert {e.worker_idx for e in envs} == {0, 1}

    async def test_factory_error_is_op_error_worker_survives(self):
        pool = WorkerPool(1, 30.0, FAKE_FACTORY)
        with pytest.raises(EnvWorkerError, match="scripted load failure"):
            await pool.load("GAME-BOOM", True)
        assert pool.respawns == 0
        env, _, _ = await pool.load("g1", True)  # same worker, still healthy
        assert (await env.step("look"))[0] == "echo:look:1"

    async def test_step_error_is_op_error_env_survives(self):
        pool = WorkerPool(1, 30.0, FAKE_FACTORY)
        env, _, _ = await pool.load("g1", True)
        with pytest.raises(EnvWorkerError, match="scripted step failure"):
            await env.step("explode")
        assert (await env.step("look"))[0] == "echo:look:1"

    async def test_worker_death_respawns_and_invalidates_stale_handles(self):
        pool = WorkerPool(1, 30.0, FAKE_FACTORY)
        survivor, _, _ = await pool.load("g1", True)
        with pytest.raises(EnvWorkerDied):
            await pool.load("GAME-DIE", True)
        assert pool.respawns == 1
        with pytest.raises(EnvWorkerDied):
            await survivor.step("look")  # resident env died with the worker
        await survivor.close()  # stale close is a silent no-op
        env, obs, _ = await pool.load("g2", True)  # respawned worker serves loads
        assert obs == "fake obs for g2"
        assert pool._counts == [1]

    async def test_wedged_worker_times_out_and_respawns(self):
        pool = WorkerPool(1, 0.5, FAKE_FACTORY)
        with pytest.raises(EnvWorkerDied, match="exceeded 0.5s"):
            await pool.load("GAME-HANG", True)
        assert pool.respawns == 1
        env, _, _ = await pool.load("g1", True)
        assert (await env.step("look"))[0] == "echo:look:1"

    async def test_wedged_worker_does_not_block_other_workers(self):
        # Failure isolation: while worker 0 is wedged inside a hanging load,
        # ops on worker 1 keep flowing (they queue per worker, and awaiting
        # callers hold no shared executor thread).
        import asyncio
        import time

        pool = WorkerPool(2, 5.0, FAKE_FACTORY)
        hang_task = asyncio.create_task(pool.load("GAME-HANG", True))  # reserves worker 0
        await asyncio.sleep(0.1)
        t0 = time.monotonic()
        env, _, _ = await pool.load("g1", True)  # placed on worker 1
        for _ in range(3):
            await env.step("look")
        assert time.monotonic() - t0 < 3.0  # finished while worker 0 still wedged
        assert env.worker_idx != 0
        with pytest.raises(EnvWorkerDied):
            await hang_task
        assert pool.respawns == 1


# ──────────────────────────────────────────────────────────
# Server with env_workers > 0 (fake factory in real subprocesses)
# ──────────────────────────────────────────────────────────


class TestServerProcessBackend:
    async def test_reset_step_win_through_workers(self, game_dir):
        server = _make_server(str(game_dir))
        obs, info = await server.reset({"game_file": GAME_REL}, SID)
        assert obs.startswith("fake obs for ")
        assert "Your admissible actions are: [look, win]" in obs
        assert info == {"env_load_retries": 0, "env_pool_hit": 0, "env_worker_respawns": 0}
        obs, reward, terminated, truncated, sinfo = await server.step(_act_response("look"), {}, SID)
        assert obs.startswith("echo:look:1")
        assert (reward, terminated, truncated) == (0.0, False, False)
        obs, reward, terminated, truncated, sinfo = await server.step(_act_response("win"), {}, SID)
        assert (reward, terminated, truncated) == (1.0, True, False)
        assert obs.endswith("Task completed.")
        assert sinfo["won"] is True

    async def test_pool_reuse_round_trips_the_worker(self, game_dir):
        server = _make_server(str(game_dir))
        obs1, info1 = await server.reset({"game_file": GAME_REL}, "ep-1")
        handle = server.session_state["ep-1"].env
        await server.step(_act_response("look"), {}, "ep-1")
        await server.close_session("ep-1")
        assert len(server._env_pool) == 1
        obs2, info2 = await server.reset({"game_file": GAME_REL}, "ep-2")
        assert server.session_state["ep-2"].env is handle  # same parked handle
        assert (info1["env_pool_hit"], info2["env_pool_hit"]) == (0, 1)
        assert obs2 == obs1  # reset rewound the env inside the worker
        step_obs, *_ = await server.step(_act_response("look"), {}, "ep-2")
        assert step_obs.startswith("echo:look:1")

    async def test_parked_handle_of_dead_worker_falls_back_to_fresh_load(self, game_dir):
        server = _make_server(str(game_dir))
        obs1, _ = await server.reset({"game_file": GAME_REL}, SID)
        handle = server.session_state[SID].env
        await server.close_session(SID)  # parked
        server._procpool._procs[handle.worker_idx].kill()
        obs2, info2 = await server.reset({"game_file": GAME_REL}, SID)  # broken-env fallback
        assert obs2 == obs1
        assert info2["env_pool_hit"] == 0
        assert info2["env_worker_respawns"] == 1
        assert server.session_state[SID].env is not handle
        step_obs, *_ = await server.step(_act_response("look"), {}, SID)
        assert step_obs.startswith("echo:look:1")

    async def test_session_on_dead_worker_500s_on_step(self, game_dir):
        server = _make_server(str(game_dir), env_workers=1)
        await server.reset({"game_file": GAME_REL}, SID)
        server._procpool._procs[0].kill()
        with pytest.raises(EnvWorkerDied):
            await server.step(_act_response("look"), {}, SID)  # -> HTTP 500, trajectory dropped

    async def test_concurrent_resets_spread_across_workers(self, game_dir):
        import asyncio

        server = _make_server(str(game_dir), env_workers=2, env_pool_size=0)
        await asyncio.gather(*(server.reset({"game_file": GAME_REL}, f"s{i}") for i in range(4)))
        idxs = {state.env.worker_idx for state in server.session_state.values()}
        assert idxs == {0, 1}
        results = await asyncio.gather(*(server.step(_act_response("look"), {}, f"s{i}") for i in range(4)))
        assert all(obs.startswith("echo:look:1") for obs, *_ in results)


# ──────────────────────────────────────────────────────────
# Real alfworld games through worker processes (gated)
# ──────────────────────────────────────────────────────────


def _real_data_root() -> Path | None:
    root = os.environ.get("ALFWORLD_DATA")
    if root and (Path(root) / "json_2.1.1").is_dir():
        return Path(root)
    return None


@pytest.mark.skipif(_real_data_root() is None, reason="$ALFWORLD_DATA with json_2.1.1 not available")
class TestRealGameProcessBackend:
    async def test_real_walkthrough_win_and_pooled_reuse(self):
        """Full episode via the DEFAULT (real alfworld) factory inside worker
        processes: walkthrough wins with reward 1.0, and a second episode
        reuses the parked remote env with a byte-identical initial obs."""
        pytest.importorskip("alfworld", reason="alfworld not installed in this venv")

        root = _real_data_root()
        game = next(
            g
            for g in sorted(root.glob("json_2.1.1/valid_seen/**/game.tw-pddl"))
            if json.loads(g.read_text()).get("walkthrough")
        )
        walkthrough = json.loads(game.read_text())["walkthrough"]
        server = _make_server(
            str(root),
            env_workers=2,
            max_concurrency=4,
            env_call_timeout_s=300.0,
            env_factory="alfworld_env._worker:make_env",
        )
        metadata = {
            "game_file": str(game.relative_to(root)),
            "max_episode_steps": len(walkthrough) + 5,
        }

        async def play(sid):
            obs0, reset_info = await server.reset(metadata, sid)
            assert "Your admissible actions are: [" in obs0
            reward = terminated = None
            for cmd in walkthrough:
                _, reward, terminated, truncated, _ = await server.step(_act_response(cmd), {}, sid)
                if terminated or truncated:
                    break
            assert terminated is True and reward == 1.0
            await server.close_session(sid)
            return obs0, reset_info

        obs1, info1 = await play("ep-1")
        obs2, info2 = await play("ep-2")
        assert (info1["env_pool_hit"], info2["env_pool_hit"]) == (0, 1)
        assert obs2 == obs1
        assert info2["env_worker_respawns"] == 0
        await server._env_pool.take(str(game)).close()
