"""miles_harbor.trial_server: the /run answers gym_bridge reads, from a stand-in for Harbor's trial (Trial.create
patched: it writes mini-swe-agent's trajectory where Harbor's mounted logs put it, emits its start, end or cancel to
the hooks, and returns a result shaped as Harbor's TrialResult); the harness's rollout kwargs and outcome on their
own."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

pytest.importorskip("harbor")  # the Harbor package is in gym_root/miles_harbor's venv, not in every checkout

import _fake_backend  # noqa: E402  (the tests' sandbox backend; this directory is on sys.path as pytest collects)
import httpx  # noqa: E402
import miles_harbor.trial_server as trial_server  # noqa: E402
import yaml  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from harbor.models.task.config import TaskConfig as TaskSpec  # noqa: E402
from harbor.trial.hooks import TrialEvent  # noqa: E402
from miles_harbor.harnesses import MiniSweAgent, Outcome, VanilluxAgent  # noqa: E402
from miles_harbor.trial_server import (  # noqa: E402
    ServerConfig,
    TrialServer,
    episode_metrics,
    make_app,
    missing_tasks,
    resolve_dataset,
    trial_config,
    trial_reward,
)

T0 = datetime(2026, 10, 14, 20, 0, 0)


def timing(seconds):
    return SimpleNamespace(started_at=T0, finished_at=T0 + timedelta(seconds=seconds))


def harbor_result(reward=1.0, exception=None, rewards=None, agent_s=100.0):
    """The parts of Harbor's TrialResult the server reads."""
    verifier = None if reward is None and rewards is None else SimpleNamespace(rewards=rewards or {"reward": reward})
    return SimpleNamespace(
        verifier_result=verifier,
        exception_info=SimpleNamespace(exception_type=exception, exception_message="it broke " * 3)
        if exception
        else None,
        agent_result=SimpleNamespace(n_input_tokens=1000, n_output_tokens=50),
        environment_setup=timing(3),
        agent_setup=timing(1),
        agent_execution=timing(agent_s),
        verifier=timing(20),
    )


class FakeTrial:
    """Harbor's Trial as the server drives it: create(config), add_hook(event, hook) -> run() -> a TrialResult."""

    created: list = []
    # how the next trials run: an awaitable to wait on, the trajectory's info (vanillux's status), the reward files
    # the agent writes to /logs/verifier, the result
    gate: asyncio.Event | None = None
    info: dict | None = {"exit_status": "Submitted", "model_stats": {"api_calls": 12}}
    status: dict | None = None
    planted: dict[str, str] = {}
    result = harbor_result()
    cancelled = 0
    # the verifier dir's files as the tests start
    seen_by_tests: list[str] | None = None

    def __init__(self, config):
        self.config = config
        # the task's config as Trial.create loads it (task.toml's: no network policy of its own)
        self.task = SimpleNamespace(config=TaskSpec())
        self.hooks = {event: [] for event in TrialEvent}

    def add_hook(self, event, hook):
        self.hooks[event].append(hook)

    async def emit(self, event):
        for hook in self.hooks[event]:
            await hook(SimpleNamespace(event=event, config=self.config))

    @classmethod
    async def create(cls, config):
        trial = cls(config)
        cls.created.append(trial)
        return trial

    async def run(self):
        await self.emit(TrialEvent.START)
        agent_dir = self.config.trials_dir / self.config.trial_name / "agent"
        agent_dir.mkdir(parents=True)
        try:
            if FakeTrial.gate is not None:
                await FakeTrial.gate.wait()
        except asyncio.CancelledError:
            FakeTrial.cancelled += 1
            await self.emit(TrialEvent.CANCEL)
            raise
        if FakeTrial.info is not None:
            (agent_dir / "mini-swe-agent.trajectory.json").write_text(
                json.dumps({"info": FakeTrial.info, "messages": []})
            )
        if FakeTrial.status is not None:
            (agent_dir / "vanillux.json").write_text(json.dumps(FakeTrial.status))
        verifier_dir = agent_dir.parent / "verifier"  # mounted at /logs/verifier
        verifier_dir.mkdir()
        for name, content in FakeTrial.planted.items():
            (verifier_dir / name).write_text(content)
        await self.emit(TrialEvent.AGENT_END)
        result = FakeTrial.result
        if self.config.verifier.disable:  # a hook skipped the verifier
            result = SimpleNamespace(**{**vars(result), "verifier_result": None, "verifier": None})
        else:
            await self.emit(TrialEvent.VERIFICATION_START)
            FakeTrial.seen_by_tests = sorted(path.name for path in verifier_dir.iterdir())
        await self.emit(TrialEvent.END)
        return result


@pytest.fixture(autouse=True)
def fake_trial(monkeypatch):
    FakeTrial.created, FakeTrial.gate, FakeTrial.cancelled = [], None, 0
    FakeTrial.info = {"exit_status": "Submitted", "model_stats": {"api_calls": 12}}
    FakeTrial.status, FakeTrial.planted, FakeTrial.seen_by_tests = None, {}, None
    FakeTrial.result = harbor_result()
    _fake_backend.CALLS.clear()
    monkeypatch.setattr(trial_server, "Trial", FakeTrial)


@pytest.fixture
def mini_config(tmp_path):
    path = tmp_path / "mini.yaml"
    path.write_text(
        yaml.safe_dump({"agent": {"step_limit": 250, "system_template": "s"}, "environment": {"env": {"A": "1"}}})
    )
    return path


@pytest.fixture
def config(tmp_path, mini_config):
    return ServerConfig(
        dataset="swebench-verified@1.0",
        download_dir=tmp_path / "tasks",
        harness="mini-swe-agent",
        agent_kwargs={"config_file": str(mini_config)},
        environment_import_path="_fake_backend:FakeEnvironment",
        environment_kwargs={"token": "t"},
        trials_dir=tmp_path / "trials",
        records=tmp_path / "records" / "trials.jsonl",
        max_concurrent=4,
        agent_timeout_s=600,
    )


@pytest.fixture
def server(config, tmp_path):
    return TrialServer(config, {"django__django-15098": tmp_path / "tasks" / "django__django-15098"})


def body(**extra):
    return {
        "instance_id": "django__django-15098",
        "max_episode_steps": 30,
        "policy_base_url": "http://10.0.0.1:9000/sessions/abc/v1/",
        "responses_create_params": {"input": [], "max_output_tokens": 1536},
        **extra,
    }


def records(config):
    return [json.loads(line) for line in config.records.read_text().splitlines()]


async def until(predicate, timeout=5.0):
    """Wait for ``predicate()``; fail instead of hanging."""

    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


# ---- the pieces ----


def test_the_dataset_is_spelled_as_harbor_run_spells_it(tmp_path):
    registry = resolve_dataset("swebench-verified@1.0", tmp_path, ["a", "b"])
    assert (registry.name, registry.version, registry.ref, registry.task_names) == (
        "swebench-verified",
        "1.0",
        None,
        ["a", "b"],
    )
    hub = resolve_dataset("harbor/terminal-bench@2.0", tmp_path, None)
    assert (hub.name, hub.ref, hub.version, hub.download_dir) == ("harbor/terminal-bench", "2.0", None, tmp_path)
    local = resolve_dataset("/data/harbor/tmax", tmp_path, ["t1"])
    assert (local.path, local.name, local.task_names, local.is_local()) == (
        Path("/data/harbor/tmax"),
        None,
        ["t1"],
        True,
    )


def test_every_plain_task_name_must_be_a_task():
    known = {"django__django-15098", "org/task-a", "task-a"}
    assert missing_tasks(None, known) == [] and missing_tasks(["django__*", "task-?", "org/task-a"], known) == []
    assert missing_tasks(["django__django-15098", "sympy__sympy-1", "astropy__astropy-2"], known) == [
        "astropy__astropy-2",
        "sympy__sympy-1",
    ]


def write_task(root, *, docker_image=None, dockerfile=None):
    """A task dir as Harbor's Task loads it: a prebuilt ``docker_image``, else ``environment/Dockerfile``."""
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (root / "instruction.md").write_text("Do the thing.\n")
    (root / "task.toml").write_text("[environment]\n" + (f'docker_image = "{docker_image}"\n' if docker_image else ""))
    if dockerfile is not None:
        (root / "environment" / "Dockerfile").write_text(dockerfile)
    return root


def test_a_local_datasets_named_tasks_are_read_from_their_own_dirs(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    for name in ("t1", "t2", "t3"):
        write_task(root / name, docker_image="org/img")
    harbors = asyncio.run(trial_server.load_tasks(str(root), tmp_path / "dl", ["t?"]))
    assert harbors == {"t1": root / "t1", "t2": root / "t2", "t3": root / "t3"}
    # plain names that are its dirs: those dirs alone, as Harbor's resolution (a scan of every dir) gives them
    monkeypatch.setattr(trial_server, "resolve_dataset", lambda *a: pytest.fail("scanned the dataset"))
    assert asyncio.run(trial_server.load_tasks(str(root), tmp_path / "dl", ["t2", "t1", "t2"])) == {
        "t2": root / "t2",
        "t1": root / "t1",
    }
    # a glob, a name no dir of it has, none, or a dataset not local: Harbor's resolution decides
    for dataset, names in ((root, ["t*"]), (root, ["t1", "t9"]), (root, None), ("org/tasks@1", ["t1"])):
        assert trial_server.local_named_tasks(str(dataset), names) is None


def test_a_trial_is_harbors_with_the_rollouts_session_and_limits(config, tmp_path):
    trial = trial_config(
        config, tmp_path / "t", "http://10.0.0.1:9000/sessions/abc/v1/", step_limit=30, max_tokens=1536
    )
    assert trial.task.path == tmp_path / "t" and trial.trials_dir == config.trials_dir
    agent = trial.agent
    assert agent.import_path == "miles_harbor.harnesses:MiniSweAgent"
    assert agent.model_name == "openai/model" and agent.override_timeout_sec == 600
    assert (
        agent.env["OPENAI_BASE_URL"] == "http://10.0.0.1:9000/sessions/abc/v1"
        and agent.env["MSWEA_API_KEY"] == "EMPTY"
    )
    assert agent.kwargs["max_tokens"] == 1536 and "config_file" not in agent.kwargs
    assert agent.kwargs["config"]["agent"] == {"step_limit": 30, "system_template": "s"}
    assert trial.environment.import_path == "_fake_backend:FakeEnvironment" and trial.environment.delete
    # agent_network=endpoint: the backend is asked for the endpoint's kwargs (the fake adds none) after the config's
    assert trial.environment.kwargs == config.environment_kwargs
    assert [name for name, _ in _fake_backend.CALLS] == ["environment_kwargs", "network_kwargs"]
    assert _fake_backend.CALLS[1] == ("network_kwargs", ("http://10.0.0.1:9000/sessions/abc/v1/",))


def test_the_tasks_own_network_forwards_nothing(config, tmp_path):
    config = config.model_copy(update={"agent_network": "task"})
    trial = trial_config(config, tmp_path / "t", "http://node7:9000/v1", step_limit=None, max_tokens=None)
    assert trial.environment.kwargs == config.environment_kwargs
    assert [name for name, _ in _fake_backend.CALLS] == ["environment_kwargs"]  # no endpoint kwargs asked


def test_mini_keeps_its_config_file_without_a_step_limit(mini_config):
    kwargs = MiniSweAgent.rollout_kwargs({"config_file": str(mini_config)}, step_limit=None, max_tokens=None)
    assert kwargs == {"config_file": str(mini_config)}


def test_mini_sets_the_step_limit_over_an_inline_config_without_touching_the_callers():
    inline = {"agent": {"step_limit": 250}, "model": {"x": 1}}
    kwargs = MiniSweAgent.rollout_kwargs({"config": inline}, step_limit=100, max_tokens=None)
    assert kwargs["config"] == {"agent": {"step_limit": 100}, "model": {"x": 1}}
    assert inline["agent"]["step_limit"] == 250


def test_mini_without_a_config_gets_one_with_the_step_limit_alone():
    # Harbor puts mini's own mini.yaml first (-c mini), this over it
    assert MiniSweAgent.rollout_kwargs({}, step_limit=40, max_tokens=None) == {"config": {"agent": {"step_limit": 40}}}


@pytest.mark.parametrize(
    "content, outcome",
    [
        ({"info": {"exit_status": "Submitted", "model_stats": {"api_calls": 86}}}, Outcome("Submitted", True, 86)),
        (
            {"info": {"exit_status": "LimitsExceeded", "model_stats": {"api_calls": 250}}},
            Outcome("LimitsExceeded", False, 250),
        ),
        ({"info": {"exit_status": "ContextWindowExceededError"}}, Outcome("ContextWindowExceededError", False, None)),
        ({"messages": []}, Outcome(None, False, None)),
        ("not json", Outcome(None, False, None)),
        ([1, 2], Outcome(None, False, None)),
        (None, Outcome(None, False, None)),
    ],
)
def test_mini_tells_how_it_ended_from_its_trajectory(tmp_path, content, outcome):
    if content is not None:
        text = content if isinstance(content, str) else json.dumps(content)
        (tmp_path / "mini-swe-agent.trajectory.json").write_text(text)
    assert MiniSweAgent.outcome(tmp_path) == outcome


def test_the_reward_is_the_verifiers():
    assert trial_reward(harbor_result(reward=0.0)) == 0.0
    assert trial_reward(harbor_result(rewards={"reward": 1, "extra": 5})) == 1.0
    assert trial_reward(harbor_result(rewards={"pass": 1})) == 1.0
    assert trial_reward(harbor_result(rewards={"a": 1, "b": 0})) is None
    assert trial_reward(harbor_result(reward=None)) is None


@pytest.mark.parametrize(
    "exception, outcome, ended",
    [
        (None, Outcome("Submitted", True, 5), "submitted"),
        (None, Outcome("LimitsExceeded", False, 30), "limit"),
        (None, Outcome("step_limit", False, 64), "limit"),
        (None, Outcome("context", False, 20), "context"),
        (None, Outcome("no_tool_call", False, 2), "no_tool_call"),
        ("NonZeroAgentExitCodeError", Outcome("BadRequestError", False, 3), "error"),
        (None, Outcome(None, False, None), "error"),
        # Harbor stopped the agent: whatever its trajectory last said
        ("AgentTimeoutError", Outcome("Submitted", True, 7), "timeout"),
    ],
)
def test_how_the_agent_ended_is_one_exit_and_its_error_flag(exception, outcome, ended):
    metrics = episode_metrics(harbor_result(exception=exception), outcome)
    exits = {key.removeprefix("exit_"): value for key, value in metrics.items() if key.startswith("exit_")}
    assert exits == {
        name: float(name == ended) for name in ("submitted", "limit", "context", "no_tool_call", "timeout", "error")
    }
    assert metrics["agent_error"] == float(ended != "submitted")
    assert metrics["agent_s"] == 100.0 and metrics["env_setup_s"] == 3.0 and metrics["verifier_s"] == 20.0
    assert metrics["agent_input_tokens"] == 1000.0
    assert ("agent_steps" in metrics) == (outcome.steps is not None)
    # output tokens a step, when the trial took any
    per_step = metrics.get("agent_output_tokens_per_step")
    assert per_step == (50.0 / outcome.steps if outcome.steps else None)


# ---- the server ----


async def test_a_graded_trial_answers_its_reward_and_how_it_ended(server, config):
    answer = await server.run(body())
    assert answer["reward"] == 1.0 and answer["aborted"] == 0.0 and answer["agent_error"] == 0.0
    assert answer["exit_submitted"] == 1.0 and answer["agent_steps"] == 12.0
    assert answer["agent_metrics"]["total_tool_time"] >= 0.0
    # every top-level number but reward is a gym_bridge env metric
    assert all(isinstance(value, float) for key, value in answer.items() if key != "agent_metrics")
    (trial,) = FakeTrial.created
    assert (
        trial.config.agent.kwargs["config"]["agent"]["step_limit"] == 30
        and trial.config.agent.kwargs["max_tokens"] == 1536
    )
    # agent_network=endpoint: the backend was asked for the endpoint's kwargs as the trial was built, and to limit the
    # agent's run to the session's address as it began (the fake records the calls; what they do is a backend's)
    url = body()["policy_base_url"]
    assert ("network_kwargs", (url,)) in _fake_backend.CALLS
    assert ("limit_agent_network", (trial.task.config, url)) in _fake_backend.CALLS
    assert trial.config.environment.kwargs == config.environment_kwargs
    (record,) = records(config)
    assert (
        record["task"] == "django__django-15098" and record["reward"] == 1.0 and record["exit_status"] == "Submitted"
    )
    assert not (config.trials_dir / trial.config.trial_name).exists()


def trial_lines(caplog):
    """The server's TRIAL lines: [trial, event] each."""
    return [r.getMessage().split()[1:3] for r in caplog.records if r.getMessage().startswith("TRIAL ")]


async def test_each_trial_event_is_a_log_line_with_its_seconds(server, caplog):
    caplog.set_level(logging.INFO, logger="miles_harbor.trial_server")
    await server.run(body())
    name = FakeTrial.created[0].config.trial_name
    events = ["start", "agent-end", "verification-start", "end"]
    assert trial_lines(caplog) == [[name, event] for event in events]
    assert all(float(r.getMessage().split()[3]) >= 0.0 for r in caplog.records if r.getMessage().startswith("TRIAL "))


async def test_agent_network_task_leaves_the_tasks_policy(server, config):
    server.config = config.model_copy(update={"agent_network": "task"})
    await server.run(body(policy_base_url="http://node7:9000/v1"))
    (trial,) = FakeTrial.created
    assert trial.task.config.agent.network_mode is None and "allowlist_ports" not in trial.config.environment.kwargs
    # none of the network hooks is called: the URL is not checked, no endpoint kwargs, no limit
    assert [name for name, _ in _fake_backend.CALLS] == ["environment_kwargs"]


async def test_keep_trials_keeps_the_trials_dir(server, config):
    server.config = config.model_copy(update={"keep_trials": True})
    await server.run(body())
    (trial,) = FakeTrial.created
    assert (config.trials_dir / trial.config.trial_name / "agent" / "mini-swe-agent.trajectory.json").is_file()


async def test_an_agent_out_of_turns_is_graded_as_an_agent_error(server):
    FakeTrial.info = {"exit_status": "LimitsExceeded", "model_stats": {"api_calls": 30}}
    FakeTrial.result = harbor_result(reward=0.0)
    answer = await server.run(body())
    assert (answer["reward"], answer["agent_error"], answer["exit_limit"]) == (0.0, 1.0, 1.0)


@pytest.fixture
def tmax(config, tmp_path):
    """A server of vanillux's runs under TMax's rule (reward_requires_submit)."""
    update = {"harness": "vanillux", "agent_kwargs": {}, "reward_requires_submit": True}
    FakeTrial.info = None
    return TrialServer(config.model_copy(update=update), {"django__django-15098": tmp_path / "tasks" / "x"})


async def test_a_run_that_did_not_submit_scores_0_and_its_tests_are_not_run(tmax, config):
    FakeTrial.status = {"exit_status": "step_limit", "steps": 64}
    answer = await tmax.run(body())
    (trial,) = FakeTrial.created
    assert trial.config.verifier.disable and trial.config.agent.import_path == "miles_harbor.vanillux:VanilluxAgent"
    assert trial.config.agent.kwargs == {"max_steps": 30, "max_tokens": 1536}
    assert (answer["reward"], answer["verifier_skipped"], answer["exit_limit"], answer["agent_steps"]) == (
        0.0,
        1.0,
        1.0,
        64.0,
    )
    (record,) = records(config)
    assert record["reward"] == 0.0 and record["exit_status"] == "step_limit"


async def test_a_harness_beside_the_sandbox_leaves_it_no_network_while_it_runs(tmax):
    FakeTrial.status = {"exit_status": "submitted", "steps": 9}
    await tmax.run(body())
    (trial,) = FakeTrial.created
    # the backend was asked for no endpoint kwargs, and to leave the agent's run no network at all (None)
    assert "network_kwargs" not in [name for name, _ in _fake_backend.CALLS]
    assert ("limit_agent_network", (trial.task.config, None)) in _fake_backend.CALLS


async def test_a_submitted_run_is_graded(tmax):
    FakeTrial.status = {"exit_status": "submitted", "steps": 9}
    answer = await tmax.run(body())
    assert (answer["reward"], answer["verifier_skipped"], answer["exit_submitted"]) == (1.0, 0.0, 1.0)
    assert not FakeTrial.created[0].config.verifier.disable


async def test_a_run_whose_end_was_not_known_in_time_is_graded_and_scores_0_unsubmitted(tmax):
    FakeTrial.status = None  # no log as the agent's phase ended: the verifier runs, the rule still holds
    answer = await tmax.run(body())
    assert (answer["reward"], answer["verifier_skipped"], answer["exit_error"]) == (0.0, 0.0, 1.0)


async def test_a_run_that_raised_answers_500_even_with_its_tests_skipped(tmax, config):
    FakeTrial.status = {"exit_status": "model_error", "steps": 3}
    FakeTrial.result = harbor_result(reward=None, exception="ModelCallError")
    with pytest.raises(HTTPException) as raised:
        await tmax.run(body())
    assert raised.value.status_code == 500 and "ModelCallError" in raised.value.detail
    (record,) = records(config)
    assert record["infra"].startswith("ModelCallError")


@pytest.mark.parametrize("rule", [True, False])
async def test_a_model_call_failure_answers_500_never_a_reward(tmax, config, rule):
    # vanillux's ModelCallError (its model server failed MODEL_CALL_ATTEMPTS calls): no reward, with or without TMax's
    # rule; gym_bridge reports the 500 as run_transport_failed and the trainer drops the group (a selftest that points
    # the agents at a closed port gets this answer too)
    tmax.config = tmax.config.model_copy(update={"reward_requires_submit": rule})
    FakeTrial.status = {"exit_status": "model_error", "steps": 0}
    FakeTrial.result = harbor_result(reward=None, exception="ModelCallError")
    transport = httpx.ASGITransport(app=make_app(tmax))
    async with httpx.AsyncClient(transport=transport, base_url="http://server") as client:
        answer = await client.post("/run", json=body())
    assert answer.status_code == 500
    assert answer.json()["detail"].endswith("ModelCallError: it broke it broke it broke ")
    (record,) = records(config)
    assert "reward" not in record and record["infra"].startswith("ModelCallError")


async def test_an_agent_timeout_under_the_rule_scores_0(tmax):
    FakeTrial.status = {"exit_status": "cancelled", "steps": 40}
    FakeTrial.result = harbor_result(exception="AgentTimeoutError")
    answer = await tmax.run(body())
    assert (answer["reward"], answer["verifier_skipped"], answer["exit_timeout"]) == (0.0, 1.0, 1.0)


async def test_an_unsubmitted_run_scores_0_under_the_rule_even_if_its_tests_failed(tmax):
    # its end unknown as its phase ended (the verifier ran), its verifier's failure behind the agent's timeout
    FakeTrial.result = harbor_result(reward=None, exception="AgentTimeoutError")
    answer = await tmax.run(body())
    assert (answer["reward"], answer["verifier_skipped"], answer["verifier_failed"]) == (0.0, 0.0, 0.0)


def test_vanillux_tells_how_it_ended_from_its_status(tmp_path):
    assert VanilluxAgent.outcome(tmp_path) == Outcome(None, False, None)
    (tmp_path / "vanillux.json").write_text(json.dumps({"exit_status": "submitted", "steps": 5}))
    assert VanilluxAgent.outcome(tmp_path) == Outcome("submitted", True, 5)


async def test_a_trial_without_a_reward_answers_500_and_is_recorded(server, config):
    FakeTrial.result = harbor_result(reward=None, exception="AddTestsDirError")
    with pytest.raises(HTTPException) as raised:
        await server.run(body())
    assert raised.value.status_code == 500 and "AddTestsDirError" in raised.value.detail
    (record,) = records(config)
    assert record["infra"].startswith("AddTestsDirError: it broke")


@pytest.mark.parametrize("exception", sorted(trial_server.VERIFIER_FAILURES))
async def test_tests_that_ran_and_did_not_grade_score_0(server, config, exception):
    FakeTrial.result = harbor_result(reward=None, exception=exception)
    answer = await server.run(body())
    assert (answer["reward"], answer["verifier_failed"], answer["exit_submitted"]) == (0.0, 1.0, 1.0)
    (record,) = records(config)
    assert record["reward"] == 0.0 and record["verifier_error"].startswith(f"{exception}: it broke")


async def test_a_graded_trials_verifier_did_not_fail(server):
    assert (await server.run(body()))["verifier_failed"] == 0.0


async def test_reward_files_the_agent_left_are_deleted_before_the_tests_run(server):
    FakeTrial.planted = {"reward.json": '{"reward": 1}', "reward.txt": "1", "notes.txt": "mine"}
    await server.run(body())
    assert FakeTrial.seen_by_tests == ["notes.txt"]


async def test_a_trial_past_trial_timeout_s_is_cancelled_and_answers_500(server, config):
    server.config = config.model_copy(update={"trial_timeout_s": 0.05})
    FakeTrial.gate = asyncio.Event()
    with pytest.raises(HTTPException) as raised:
        await server.run(body())
    assert raised.value.status_code == 500 and FakeTrial.cancelled == 1
    assert "trial over" in records(config)[0]["infra"]
    assert not any(config.trials_dir.iterdir())  # its dir goes as a graded one's does


@pytest.mark.parametrize(
    "bad, status",
    [
        ({"instance_id": "nope"}, 400),
        ({"policy_base_url": ""}, 400),
        ({"policy_base_url": "http://node7:9000/v1"}, 400),
    ],
)
async def test_a_bad_body_is_refused(server, bad, status, monkeypatch):
    # the backend's check of the URL (its ValueError) is a 400 as well, before any trial; the fake refuses nothing by
    # itself, and a binding of the hook made now is the one called (looked up on the module when called)
    def wants_an_address(url):
        if not urlsplit(url).hostname.replace(".", "").isdigit():
            raise ValueError(f"{url}: no IPv4 address")

    monkeypatch.setattr(_fake_backend, "check_policy_base_url", wants_an_address)
    with pytest.raises(HTTPException) as raised:
        await server.run(body(**bad))
    assert raised.value.status_code == status and not FakeTrial.created


async def test_abort_answers_the_runs_in_flight_and_cancels_their_trials(server, caplog):
    caplog.set_level(logging.INFO, logger="miles_harbor.trial_server")
    FakeTrial.gate = asyncio.Event()
    runs = [asyncio.create_task(server.run(body())) for _ in range(3)]
    await until(lambda: len(FakeTrial.created) == 3)
    assert await server.abort() == {"aborted": 3}
    answers = await asyncio.gather(*runs)
    assert all(
        answer == {"reward": 0.0, "aborted": 1.0, "agent_metrics": {"total_tool_time": 0.0}} for answer in answers
    )
    await until(lambda: not server._background)
    assert FakeTrial.cancelled == 3 and not any(server.config.trials_dir.iterdir())
    assert sorted(event for _, event in trial_lines(caplog)) == ["cancel"] * 3 + ["start"] * 3
    # a run after the abort is not aborted
    FakeTrial.gate.set()
    assert (await server.run(body()))["aborted"] == 0.0


async def test_runs_past_max_concurrent_wait_and_an_abort_reaches_them_too(server, config):
    server = TrialServer(config.model_copy(update={"max_concurrent": 1}), server.tasks)
    FakeTrial.gate = asyncio.Event()
    first, second = (asyncio.create_task(server.run(body())) for _ in range(2))
    await until(lambda: FakeTrial.created)
    await asyncio.sleep(0.05)
    assert len(FakeTrial.created) == 1  # the second waits for a slot
    assert await server.abort() == {"aborted": 2}
    assert [answer["aborted"] for answer in await asyncio.gather(first, second)] == [1.0, 1.0]
    await until(lambda: not server._background)
    assert len(FakeTrial.created) == 1  # the queued one never started a trial


async def test_the_app_serves_run_abort_and_health(server):
    transport = httpx.ASGITransport(app=make_app(server))
    async with httpx.AsyncClient(transport=transport, base_url="http://server") as client:
        answer = (await client.post("/run", json=body())).json()
        assert answer["reward"] == 1.0
        FakeTrial.result = harbor_result(reward=None, exception="EnvironmentStartTimeoutError")
        failed = await client.post("/run", json=body())
        assert failed.status_code == 500
        assert (await client.post("/abort", json={})).json() == {"aborted": 0}
        assert (await client.get("/health")).json() == {"tasks": 1, "in_flight": 0, "max_concurrent": 4}


def test_main_serves_on_the_stdlib_event_loop(config, tmp_path, monkeypatch):
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(json.loads(config.model_dump_json())))
    served = {}

    async def load_tasks(*args):
        return {"django__django-15098": tmp_path / "tasks" / "django__django-15098"}

    monkeypatch.setattr(trial_server, "load_tasks", load_tasks)
    monkeypatch.setattr(trial_server.uvicorn, "run", lambda app, **kwargs: served.update(kwargs))
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["trial_server", "--config", str(path), "--port", "8271"])
    trial_server.main()
    # uvloop's children keep copies of their stdio pipes: a daemonizing one (a filesystem daemon) hangs communicate()
    assert served["loop"] == "asyncio" and served["port"] == 8271
    # main asked the backend, through its module, whether it can enforce agent_network and which tasks cannot start
    assert [name for name, _ in _fake_backend.CALLS] == ["refuse_unreachable_agent_network", "unrunnable_tasks"]
    assert _fake_backend.CALLS[0][1] == (config.environment_kwargs, "endpoint")


DEFAULT_BACKEND_SNIPPET = """
import sys
from pathlib import Path

from harbor.models.trial.config import EnvironmentConfig
from miles_harbor.trial_server import ServerConfig, TrialServer, trial_config

root = Path(sys.argv[1])
config = ServerConfig(
    dataset="swebench-verified@1.0", download_dir=root / "tasks", trials_dir=root / "trials", records=root / "r.jsonl"
)
trial = trial_config(config, root / "t", "http://node7:9000/sessions/abc/v1/", step_limit=5, max_tokens=100)
assert trial.environment.kwargs == {}, trial.environment.kwargs
assert trial.environment.import_path == EnvironmentConfig.model_fields["import_path"].default, trial.environment
assert trial.environment.delete and trial.agent.env["OPENAI_BASE_URL"] == "http://node7:9000/sessions/abc/v1"
server = TrialServer(config, {"t": root / "t"})
server.backend.check_policy_base_url("http://node7:9000/v1")  # any http(s) host: no IPv4 rule without a forward
for bad in ("/v1", "ftp://node7/v1", "http:///v1"):
    try:
        server.backend.check_policy_base_url(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
assert server.backend.unrunnable_tasks(config.environment_kwargs, server.tasks) == {}
server.backend.refuse_unreachable_agent_network(config.environment_kwargs, "endpoint")
server.backend.limit_agent_network(None, "http://node7:9000/v1")  # a no-op: nothing is read
loaded = sorted(m for m in sys.modules if m == "_fake_backend")
assert not loaded, loaded
print("OK")
"""


def test_the_default_backend_is_harbors_docker_environment_and_loads_no_sandbox_module(tmp_path):
    """No environment_import_path and no HARBOR_ENVIRONMENT_IMPORT_PATH (S2, D18): the trial's environment is Harbor's
    own (EnvironmentConfig's import_path default), its kwargs the config's alone (no allowlist_ports), every hook its
    default (a policy_base_url needs only an http(s) host), and no sandbox backend module is imported. In a
    subprocess: this module imports _fake_backend itself."""
    env = {k: v for k, v in os.environ.items() if k != "HARBOR_ENVIRONMENT_IMPORT_PATH"}
    done = subprocess.run(
        [sys.executable, "-c", DEFAULT_BACKEND_SNIPPET, str(tmp_path)], capture_output=True, text=True, env=env
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.rstrip().splitlines()[-1] == "OK"


def test_the_backend_comes_from_the_environment_when_the_config_names_none(config, tmp_path, monkeypatch):
    """HARBOR_ENVIRONMENT_IMPORT_PATH stands in for an empty environment_import_path (a config a launcher wrote without
    it): the backend's hooks are then the ones called -- its kwargs hook's result in the kwargs and the import path
    in the trial's environment."""
    bare = config.model_copy(update={"environment_import_path": ""})
    monkeypatch.setenv("HARBOR_ENVIRONMENT_IMPORT_PATH", "_fake_backend:FakeEnvironment")
    _fake_backend.CALLS.clear()
    trial = trial_config(bare, tmp_path / "t", "http://10.0.0.1:9000/v1", step_limit=None, max_tokens=None)
    assert trial.environment.import_path == "_fake_backend:FakeEnvironment"
    assert trial.environment.kwargs == config.environment_kwargs
    assert "environment_kwargs" in [name for name, _ in _fake_backend.CALLS]
    assert trial_server.environment_import_path(bare) == "_fake_backend:FakeEnvironment"
    monkeypatch.delenv("HARBOR_ENVIRONMENT_IMPORT_PATH")
    assert trial_server.environment_import_path(bare) == "" and trial_server.load_backend("").module is None


def test_the_backend_hooks_are_called_by_name(config, tmp_path, monkeypatch):
    """The backend protocol (Backend): each hook is looked up on the backend's module by name when it is called, and
    is called with the server's arguments -- environment_kwargs(config), network_kwargs(policy_base_url),
    check_policy_base_url(url), unrunnable_tasks(kwargs, tasks), refuse_unreachable_agent_network(kwargs,
    agent_network), limit_agent_network(task_config, policy_base_url); a binding of the attribute made later is
    seen; a hook the module lacks is Backend's default (Harbor's docker environment's)."""
    backend = trial_server.load_backend("_fake_backend:FakeEnvironment")
    assert backend.module is _fake_backend and backend.import_path == "_fake_backend:FakeEnvironment"
    url = "http://10.0.0.1:9000/v1"
    tasks = {"t": tmp_path / "t"}
    task = TaskSpec()
    assert backend.environment_kwargs(config) == {"token": "t"}
    assert backend.network_kwargs(url) == {}
    assert backend.check_policy_base_url(url) is None
    assert backend.unrunnable_tasks({"token": "t"}, tasks) == {}
    assert backend.refuse_unreachable_agent_network({"token": "t"}, "endpoint") is None
    assert backend.limit_agent_network(task, url) is None
    assert _fake_backend.CALLS == [
        ("environment_kwargs", (config,)),
        ("network_kwargs", (url,)),
        ("check_policy_base_url", (url,)),
        ("unrunnable_tasks", ({"token": "t"}, tasks)),
        ("refuse_unreachable_agent_network", ({"token": "t"}, "endpoint")),
        ("limit_agent_network", (task, url)),
    ]
    # looked up when called: a binding made after the backend was loaded is the one called
    monkeypatch.setattr(_fake_backend, "network_kwargs", lambda url: {"port": urlsplit(url).port})
    assert backend.network_kwargs(url) == {"port": 9000}
    # a hook the module lacks: Backend's default
    monkeypatch.delattr(_fake_backend, "network_kwargs")
    monkeypatch.delattr(_fake_backend, "check_policy_base_url")
    assert backend.network_kwargs(url) == {}
    with pytest.raises(ValueError, match=r"http\(s\) URL"):
        backend.check_policy_base_url("/v1")
