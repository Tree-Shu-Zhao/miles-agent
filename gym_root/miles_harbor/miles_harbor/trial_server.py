"""Harbor trials for miles' rollouts: POST /run runs one trial of a task, its agent pointed at the rollout's
session server, and answers the trial's reward, as a gym agent server answers for a nemo-gym episode.

    python -m miles_harbor.trial_server --config <server.yaml> --host <ip> --port <port>

miles_agent.gym_bridge is the client (NEMO_GYM_URL): its /run body carries the row's metadata (``instance_id``:
the task, ``max_episode_steps``: the agent's turns), ``policy_base_url`` (the session's /v1, where every model
call of the agent is recorded) and ``responses_create_params`` (``max_output_tokens``: each reply's tokens; the
session server fills the sampling it leaves unset). Every top-level number of the answer reaches wandb as an
``env/*`` metric.

The trial is Harbor's own (Trial.create, trial.run): the task's environment (the sandbox backend below), its agent (a
harness of miles_harbor.harnesses, from its pack), its verifier and its timeouts; the dataset is resolved and
downloaded once, at start, and a task whose sandbox the backend says cannot start (a sandbox backend: its image is
not in its image store) stops the server there (the backend's unrunnable_tasks). What the answer says:
- ``reward``: the verifier's (its ``reward`` key, else its one value). An agent that crashed, ran out of turns
  or of time is graded as Harbor grades it (it verifies after an agent timeout too) and counts as an
  ``agent_error`` (exit != submitted is reported as ``agent_error`` in env_metrics), its reward still in its
  group's baseline. With ``reward_requires_submit`` (TMax's rule) a run that did not submit scores 0 and its
  tests are not run (the verifier is skipped as the agent's run ends, once its log says how it ended);
- tests that ran and did not grade (they outran the verifier's timeout, wrote no reward file, an empty or an unreadable
  one) score 0, as a failing test does (``verifier_failed``): what the agent left behind is what they ran on;
- a trial without a reward otherwise (its environment, the agent's install or the tests' upload failed; the agent's
  run raised, e.g. its model server failed it; it outran ``trial_timeout_s``) answers 500: gym_bridge reports
  ``run_transport_failed`` and the DAPO filter drops the group rather than train on a made-up reward;
- POST /abort (gym_bridge.abort: DAPO has the groups it needs) answers every /run in flight ``aborted`` at once
  (the trainer discards those samples) and cancels their trials, whose sandboxes Harbor still stops.

The sandbox backend is Harbor's ``environment.import_path``: ``environment_import_path`` in the config, else
HARBOR_ENVIRONMENT_IMPORT_PATH, else Harbor's own default environment (docker), with ``environment_kwargs`` as its
kwargs. The backend's module may define the hooks Backend lists (looked up by name when called): a sandbox backend
may define them all; Harbor's docker environment needs none.

The agent's network while it runs (``agent_network``): by default its model endpoint alone, policy_base_url's
address and port as the backend enforces it (a sandbox backend's allowlist: egress off, that one port forwarded;
no DNS, no internet to look an answer up on; Harbor's docker environment: the task's own policy, the hooks being
no-ops), whatever the task's [agent] allows, and none at all for a harness that calls the model from beside the
sandbox (its Endpoint's ``in_sandbox``: vanillux); ``task``: the task's own policy. The environment's setup (the
harness's install) and the verifier keep the task's network either way.

The reward is the tests' alone: /logs/verifier is the trial's verifier dir, which the agent may write while it runs,
and a reward file there (Harbor reads reward.json before reward.txt) is deleted as the verifier starts. A process the
agent left running could still write one while the tests run, as under Harbor's docker environment.

Each trial's record (task, reward, how it ended, its phases' seconds) is a line of ``records``; its directory
(Harbor's logs, the trajectory) is deleted after it unless ``keep_trials``. The log has a ``TRIAL <trial> <event>
<seconds>`` line as each trial reaches an event of its lifecycle (Harbor's TrialEvent: start, environment-start,
agent-start, agent-end, verification-start, end, cancel), its seconds since its start.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import importlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal
from urllib.parse import urlsplit

import uvicorn
import yaml
from fastapi import Body, FastAPI, HTTPException
from harbor.models.job.config import DatasetConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial
from pydantic import BaseModel, Field

from miles_harbor.harnesses import HARNESSES, Outcome

logger = logging.getLogger("miles_harbor.trial_server")

AGENT_TIMEOUT = "AgentTimeoutError"
# what the agent's run may raise and still be graded (Harbor's SingleStepTrial verifies after these); anything else it
# raised ends the trial without a verifier
GRADED_EXCEPTIONS = {AGENT_TIMEOUT, "NonZeroAgentExitCodeError"}
# what Harbor's verifier raises when the tests ran and did not grade: a failing test's 0 (TMax scores a test run that
# outran its timeout 0); its other failures (the tests' upload: AddTestsDirError; a sandbox gone) are the trial's
VERIFIER_FAILURES = {
    "VerifierTimeoutError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
}
# the harnesses' names (Outcome.exit_status) for a run that ended at a limit, as the answer's exit_* metrics count them
EXITS = {"LimitsExceeded": "limit", "step_limit": "limit", "context": "context", "no_tool_call": "no_tool_call"}
# task dirs read at once at start (local_named_tasks): each read waits on the dataset's
# filesystem (a networked one: about a millisecond an open), little on the interpreter
LOAD_THREADS = 32


class ServerConfig(BaseModel):
    # HARBOR_DATASET as scripts/harbor_run.sh spells it: a Hub package <org>/<name>[@<ref>], or the registry's
    # <name>@<version>
    dataset: str
    download_dir: Path
    # only these tasks of the dataset (Harbor's task_names: names or globs; every plain name must be one)
    task_names: list[str] | None = None
    harness: str = "mini-swe-agent"
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)
    # the model name the harness sends (the session server serves one)
    model: str = "model"
    # the sandbox backend, Harbor's environment.import_path (<module>:<Class>), whose module may define the hooks
    # Backend lists; "" (the default): HARBOR_ENVIRONMENT_IMPORT_PATH, else Harbor's own default environment (docker)
    environment_import_path: str = ""
    # the backend's environment.kwargs (a sandbox backend's)
    environment_kwargs: dict[str, Any] = Field(default_factory=dict)
    trials_dir: Path
    records: Path
    max_concurrent: int = 64
    # the agent's run, instead of the task's own timeout (Harbor's override_timeout_sec); None: the task's
    agent_timeout_s: float | None = None
    # a trial past this (Harbor's own timeouts bound its phases) is cancelled and answers 500
    trial_timeout_s: float = 7200.0
    keep_trials: bool = False
    # the agent's network while it runs (the module docstring): "endpoint", its model endpoint alone; "task", the
    # task's own policy
    agent_network: Literal["endpoint", "task"] = "endpoint"
    # TMax's rule: a run that did not submit (Outcome.submitted) scores 0, its tests not run
    reward_requires_submit: bool = False


def resolve_dataset(dataset: str, download_dir: Path, task_names: list[str] | None) -> DatasetConfig:
    """``dataset`` as scripts/harbor_run.sh spells HARBOR_DATASET."""
    if dataset.startswith("/"):  # a directory of task dirs on this node (Harbor's local dataset)
        return DatasetConfig(path=Path(dataset), task_names=task_names)
    if "/" in dataset:  # a Hub package: <org>/<name>[@<ref>]
        name, _, ref = dataset.partition("@")
        fields: dict[str, Any] = {"name": name, **({"ref": ref} if ref else {})}
    else:  # the registry: <name>@<version>
        name, _, version = dataset.partition("@")
        fields = {"name": name, **({"version": version} if version else {})}
    return DatasetConfig(**fields, download_dir=download_dir, task_names=task_names)


def missing_tasks(task_names: list[str] | None, known: set[str]) -> list[str]:
    """The plain names (not globs) of ``task_names`` that name none of the ``known`` tasks: a training row whose task
    the server would refuse."""
    return sorted(n for n in task_names or [] if not any(c in n for c in "*?[") and n not in known)


def local_named_tasks(dataset_name: str, task_names: list[str] | None) -> dict[str, Path] | None:
    """A local dataset's ``task_names`` (plain names, no globs) when each is the name of a task dir of it, by that
    name: their dirs by name, as Harbor's resolution gives them, read from those dirs alone. Harbor's validates every
    dir of the dataset and matches each against each name, resolving the dir's path once a name
    (DatasetConfig._filter_task_ids): 1,078 of TMax-15K's 14,601 on a networked filesystem kept a trial server from
    serving for over 10 minutes, and the cost grows with the names. None when a name is not so: Harbor's resolution
    decides."""
    if not dataset_name.startswith("/") or not task_names or any(c in n for n in task_names for c in "*?["):
        return None
    root = Path(dataset_name)
    names = list(dict.fromkeys(task_names))

    def named(name: str) -> bool:
        task_dir = root / name
        try:
            return "/" not in name and Task.is_valid_dir(task_dir) and Task(task_dir=task_dir).short_name == name
        except (OSError, ValueError, RuntimeError):
            return False

    with ThreadPoolExecutor(LOAD_THREADS) as pool:
        if not all(pool.map(named, names)):
            return None
    return {name: root / name for name in names}


async def load_tasks(dataset_name: str, download_dir: Path, task_names: list[str] | None) -> dict[str, Path]:
    """The dataset's tasks (``dataset_name`` as resolve_dataset takes it; only ``task_names``), downloaded: their
    local dirs by name (a Hub task's without its org)."""
    from harbor.job_plan import JobPlan

    if (named := local_named_tasks(dataset_name, task_names)) is not None:
        return named
    dataset = resolve_dataset(dataset_name, download_dir, task_names)
    downloads = await JobPlan.cache_tasks(await dataset.get_task_configs())
    tasks: dict[str, Path] = {}
    known: set[str] = set()
    for result in downloads.values():
        task = Task(task_dir=result.path)
        name = task.short_name
        if name in tasks and tasks[name] != result.path:
            raise ValueError(f"two tasks of {dataset_name} are named {name}: {tasks[name]} and {result.path}")
        tasks[name] = result.path
        known |= {name, task.name}
    if not tasks:
        raise ValueError(f"{dataset_name} has no tasks{' named ' + str(task_names) if task_names else ''}")
    if missing := missing_tasks(task_names, known):
        raise ValueError(f"{len(missing)} task_names are no task of {dataset_name}: {missing[:20]}")
    return tasks


def environment_import_path(config: ServerConfig) -> str:
    """The backend's import path: the config's ``environment_import_path``, else HARBOR_ENVIRONMENT_IMPORT_PATH, else
    "" (Harbor's own default environment, EnvironmentConfig's import_path default)."""
    return config.environment_import_path or os.environ.get("HARBOR_ENVIRONMENT_IMPORT_PATH", "")


def check_url_shape(url: str) -> None:
    """The default backend's check of a policy_base_url (Backend.check_policy_base_url): an http(s) URL with a host,
    which any agent can be pointed at; a sandbox backend may want an IPv4 address and port instead."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"policy_base_url: {url!r} is not an http(s) URL with a host")


class Backend:
    """The sandbox backend of the trials: Harbor's ``environment.import_path`` (``import_path``; "": Harbor's default
    environment) and the hooks this server calls on its module, each looked up BY NAME WHEN CALLED (a module may
    define any subset; the import is cached in sys.modules, so a binding of the attribute made later is seen), with
    these defaults, every one a no-op (Harbor's docker environment needs none of them):
      environment_kwargs(config) -> dict          EnvironmentConfig.kwargs: a copy of config.environment_kwargs
      network_kwargs(policy_base_url) -> dict     more kwargs for an agent that calls its endpoint from inside the
                                                  sandbox under agent_network=endpoint
      check_policy_base_url(url) -> None          ValueError: the /run is refused with 400 before any trial; the
                                                  default wants an http(s) URL with a host (check_url_shape)
      unrunnable_tasks(kwargs, tasks) -> dict     the tasks (local dirs by name) whose sandbox cannot start, each
                                                  with why; main stops there
      refuse_unreachable_agent_network(kwargs, agent_network) -> None
                                                  SystemExit when the backend cannot enforce agent_network=endpoint
      limit_agent_network(task_config, policy_base_url) -> None
                                                  the trial's task config limited to the endpoint's address while
                                                  its agent runs (None: no network at all)
    A sandbox backend may define them all."""

    def __init__(self, import_path: str):
        self.import_path = import_path
        self.module: ModuleType | None = None
        if import_path:
            self.module = importlib.import_module(import_path.split(":", 1)[0])

    def _hook(self, name: str, default: Callable[..., Any]) -> Callable[..., Any]:
        return getattr(self.module, name, default) if self.module is not None else default

    def environment_kwargs(self, config: ServerConfig) -> dict[str, Any]:
        return dict(self._hook("environment_kwargs", lambda config: dict(config.environment_kwargs))(config))

    def network_kwargs(self, policy_base_url: str) -> dict[str, Any]:
        return dict(self._hook("network_kwargs", lambda url: {})(policy_base_url))

    def check_policy_base_url(self, url: str) -> None:
        self._hook("check_policy_base_url", check_url_shape)(url)

    def unrunnable_tasks(self, kwargs: dict[str, Any], tasks: dict[str, Path]) -> dict[str, str]:
        return dict(self._hook("unrunnable_tasks", lambda kwargs, tasks: {})(kwargs, tasks))

    def refuse_unreachable_agent_network(self, kwargs: dict[str, Any], agent_network: str) -> None:
        self._hook("refuse_unreachable_agent_network", lambda kwargs, network: None)(kwargs, agent_network)

    def limit_agent_network(self, task_config: Any, policy_base_url: str | None) -> None:
        self._hook("limit_agent_network", lambda task_config, url: None)(task_config, policy_base_url)


def load_backend(import_path: str) -> Backend:
    """The backend ``import_path`` names (environment_import_path's answer): its module is imported here and nowhere
    else in this server; "" is Harbor's default environment with every hook its default."""
    return Backend(import_path)


def trial_config(
    config: ServerConfig,
    task_dir: Path,
    policy_base_url: str,
    *,
    step_limit: int | None,
    max_tokens: int | None,
    backend: Backend | None = None,
) -> TrialConfig:
    """One trial of the task at ``task_dir``, its agent calling the session at ``policy_base_url``, in a sandbox of
    ``backend`` (default: the config's, load_backend)."""
    harness = HARNESSES[config.harness]
    backend = backend or load_backend(environment_import_path(config))
    environment_kwargs = backend.environment_kwargs(config)
    if config.agent_network == "endpoint" and harness.ENDPOINT.in_sandbox:
        # the agent calls its endpoint from inside the sandbox: the backend's way there
        environment_kwargs.update(backend.network_kwargs(policy_base_url))
    # Harbor's own environment unless the backend names one (EnvironmentConfig's import_path default applies)
    environment: dict[str, Any] = {"kwargs": environment_kwargs, "delete": True}
    if backend.import_path:
        environment["import_path"] = backend.import_path
    return TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=config.trials_dir,
        agent=AgentConfig(
            import_path=f"{harness.__module__}:{harness.__name__}",
            kwargs=harness.rollout_kwargs(config.agent_kwargs, step_limit=step_limit, max_tokens=max_tokens),
            override_timeout_sec=config.agent_timeout_s,
            # the session server checks no key; the harnesses want one
            **harness.ENDPOINT.agent_config(policy_base_url, "EMPTY", config.model),
        ),
        environment=EnvironmentConfig(**environment),
    )


def _seconds(timing: Any) -> float | None:
    if timing is None or timing.started_at is None or timing.finished_at is None:
        return None
    return (timing.finished_at - timing.started_at).total_seconds()


def trial_reward(result: Any) -> float | None:
    """The verifier's reward (its ``reward`` key, else its only value); None: the trial has none."""
    rewards = getattr(result.verifier_result, "rewards", None) or {}
    if "reward" in rewards:
        return float(rewards["reward"])
    return float(next(iter(rewards.values()))) if len(rewards) == 1 else None


EXIT_NAMES = ("submitted", "limit", "context", "no_tool_call", "timeout", "error")


def episode_metrics(result: Any, outcome: Outcome) -> dict[str, float]:
    """The numbers of a graded trial's answer: how its agent ended, its steps and tokens, its phases' seconds."""
    exception = result.exception_info.exception_type if result.exception_info else None
    if exception == AGENT_TIMEOUT:  # Harbor stopped the agent: whatever its log last said
        ended = "timeout"
    elif outcome.submitted:
        ended = "submitted"
    else:
        ended = EXITS.get(outcome.exit_status or "", "error")
    metrics = {
        "agent_error": float(ended != "submitted"),
        **{f"exit_{name}": float(ended == name) for name in EXIT_NAMES},
    }
    if outcome.steps is not None:
        metrics["agent_steps"] = float(outcome.steps)
    agent = result.agent_result
    for key, name in (("n_input_tokens", "agent_input_tokens"), ("n_output_tokens", "agent_output_tokens")):
        if agent is not None and getattr(agent, key, None) is not None:
            metrics[name] = float(getattr(agent, key))
    # a reply's length (thinking + action), which a mean of the two above over a batch does not give
    if metrics.get("agent_steps") and "agent_output_tokens" in metrics:
        metrics["agent_output_tokens_per_step"] = metrics["agent_output_tokens"] / metrics["agent_steps"]
    for key, name in (
        ("environment_setup", "env_setup_s"),
        ("agent_setup", "agent_setup_s"),
        ("agent_execution", "agent_s"),
        ("verifier", "verifier_s"),
    ):
        if (seconds := _seconds(getattr(result, key, None))) is not None:
            metrics[name] = seconds
    return metrics


def aborted_answer() -> dict[str, Any]:
    return {"reward": 0.0, "aborted": 1.0, "agent_metrics": {"total_tool_time": 0.0}}


@dataclasses.dataclass(eq=False)  # by identity: a member of TrialServer._runs
class _InFlight:
    stop: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)


class TrialServer:
    def __init__(self, config: ServerConfig, tasks: dict[str, Path]):
        if config.harness not in HARNESSES:
            raise ValueError(f"no harness {config.harness!r} (miles_harbor.harnesses: {sorted(HARNESSES)})")
        self.config = config
        self.tasks = tasks
        self.backend = load_backend(environment_import_path(config))
        self._slots = asyncio.Semaphore(config.max_concurrent)
        self._runs: set[_InFlight] = set()
        # trials of answered (aborted) runs, still stopping; held so the event loop does not drop them
        self._background: set[asyncio.Task] = set()
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="records")
        config.trials_dir.mkdir(parents=True, exist_ok=True)
        config.records.parent.mkdir(parents=True, exist_ok=True)

    async def run(self, body: dict[str, Any]) -> dict[str, Any]:
        task = body.get("instance_id")
        if task not in self.tasks:
            raise HTTPException(status_code=400, detail=f"no task {task!r} in {self.config.dataset}")
        if not body.get("policy_base_url"):
            raise HTTPException(status_code=400, detail="no policy_base_url in the /run body")
        if self.config.agent_network == "endpoint":  # before any trial: can the backend point an agent at it?
            try:
                self.backend.check_policy_base_url(body["policy_base_url"])
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
        inflight = _InFlight()
        self._runs.add(inflight)
        episode = asyncio.create_task(self._episode(task, body))
        stopped = asyncio.create_task(inflight.stop.wait())
        try:
            await asyncio.wait({episode, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopped.cancel()
            self._runs.discard(inflight)
            if not episode.done():  # aborted, or this request was cancelled: the trial stops its sandbox
                episode.cancel()
                self._background.add(episode)
                episode.add_done_callback(self._forget)
        if episode.done() and not episode.cancelled():
            return episode.result()
        return aborted_answer()

    def _forget(self, episode: asyncio.Task) -> None:
        self._background.discard(episode)
        if not episode.cancelled() and episode.exception() is not None:
            logger.info("an aborted trial ended with %r", episode.exception())

    async def abort(self) -> dict[str, int]:
        """Answer every /run in flight ``aborted`` now; runs that start later are unaffected."""
        runs = [run for run in self._runs if not run.stop.is_set()]
        for run in runs:
            run.stop.set()
        return {"aborted": len(runs)}

    async def _episode(self, task: str, body: dict[str, Any]) -> dict[str, Any]:
        params = body.get("responses_create_params") or {}
        steps = body.get("max_episode_steps")
        config = trial_config(
            self.config,
            self.tasks[task],
            body["policy_base_url"],
            step_limit=int(steps) if steps else None,
            max_tokens=params.get("max_output_tokens"),
            backend=self.backend,
        )
        harness = HARNESSES[self.config.harness]
        trial_dir = self.config.trials_dir / config.trial_name
        # reward_requires_submit: the verifier is skipped once the agent's log says it ended without submitting
        skipped = False

        async def skip_unsubmitted(event: TrialHookEvent) -> None:
            nonlocal skipped
            ended = harness.outcome(trial_dir / "agent")
            if ended.exit_status is not None and not ended.submitted:
                # Harbor reads it as the agent's phase is over (SingleStepTrial._run_verifier)
                event.config.verifier.disable = skipped = True

        async def clear_rewards(event: TrialHookEvent) -> None:
            paths = TrialPaths(trial_dir)
            for path in (paths.reward_json_path, paths.reward_text_path):
                path.unlink(missing_ok=True)

        t0 = time.monotonic()
        async with self._slots:
            queue_s = time.monotonic() - t0
            trial = await Trial.create(config)
            started = time.monotonic()

            async def log_event(event: TrialHookEvent) -> None:
                logger.info("TRIAL %s %s %.1f", config.trial_name, event.event.value, time.monotonic() - started)

            for event in TrialEvent:
                trial.add_hook(event, log_event)
            trial.add_hook(TrialEvent.VERIFICATION_START, clear_rewards)
            if self.config.reward_requires_submit:
                trial.add_hook(TrialEvent.AGENT_END, skip_unsubmitted)
            if self.config.agent_network == "endpoint":
                # the agent's run reaches its endpoint alone (a harness beside the sandbox: the sandbox, nothing)
                self.backend.limit_agent_network(
                    trial.task.config, body["policy_base_url"] if harness.ENDPOINT.in_sandbox else None
                )
            try:
                result = await asyncio.wait_for(trial.run(), timeout=self.config.trial_timeout_s)
            except TimeoutError:
                over = f"over trial_timeout_s {self.config.trial_timeout_s:.0f}"
                await self._record(task, config.trial_name, {"infra": f"trial {over}"})
                await self._done(trial_dir)
                raise HTTPException(status_code=500, detail=f"{config.trial_name}: {over}") from None
            except asyncio.CancelledError:  # aborted: its sandbox stopped as the trial was cancelled
                await self._done(trial_dir)
                raise
        trial_s = time.monotonic() - t0 - queue_s
        reward = trial_reward(result)
        exception = result.exception_info
        kind = exception.exception_type if exception else None
        detail = f"{kind}: {exception.exception_message[-2000:]}" if exception else "no reward"
        outcome = harness.outcome(trial_dir / "agent")
        # TMax's rule scores an unsubmitted run 0 whatever its tests did (skipped, or run as its end was not known yet)
        unsubmitted = self.config.reward_requires_submit and not outcome.submitted
        verifier_failed = reward is None and kind in VERIFIER_FAILURES
        if reward is None and (verifier_failed or (unsubmitted and (kind is None or kind in GRADED_EXCEPTIONS))):
            reward = 0.0
        if reward is None:  # the environment, the install, the agent or the tests' upload failed: none to train on
            await self._record(task, config.trial_name, {"infra": detail, "queue_s": queue_s, "trial_s": trial_s})
            await self._done(trial_dir)
            raise HTTPException(status_code=500, detail=f"{config.trial_name}: {detail}")
        if unsubmitted:
            reward = 0.0
        metrics = {
            **episode_metrics(result, outcome),
            "verifier_skipped": float(skipped),
            "verifier_failed": float(verifier_failed),
        }
        await self._record(
            task,
            config.trial_name,
            {
                "reward": reward,
                "exit_status": outcome.exit_status,
                "exception": kind,
                **({"verifier_error": detail} if verifier_failed else {}),
                "queue_s": queue_s,
                "trial_s": trial_s,
                **metrics,
            },
        )
        await self._done(trial_dir)
        return {
            "reward": reward,
            **metrics,
            "aborted": 0.0,
            "queue_s": queue_s,
            "trial_s": trial_s,
            # miles' Sample.non_generation_time: the trial outside its agent's run
            "agent_metrics": {"total_tool_time": max(trial_s - metrics.get("agent_s", 0.0), 0.0)},
        }

    async def _record(self, task: str, trial: str, fields: dict[str, Any]) -> None:
        line = {"time": datetime.now().isoformat(timespec="seconds"), "task": task, "trial": trial, **fields}
        # one writer thread: a stall of the records' filesystem (a shared one) holds no trial's model calls
        await asyncio.get_running_loop().run_in_executor(self._writer, self._append, json.dumps(line) + "\n")

    def _append(self, text: str) -> None:
        with self.config.records.open("a") as out:
            out.write(text)

    async def _done(self, trial_dir: Path) -> None:
        if not self.config.keep_trials:
            await asyncio.to_thread(shutil.rmtree, trial_dir, True)


def make_app(server: TrialServer) -> FastAPI:
    app = FastAPI(title="miles_harbor trial server")

    @app.post("/run")
    async def run(body: dict[str, Any] = Body()) -> dict[str, Any]:  # noqa: B008
        return await server.run(body)

    @app.post("/abort")
    async def abort() -> dict[str, int]:
        return await server.abort()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "tasks": len(server.tasks),
            "in_flight": len(server._runs),
            "max_concurrent": server.config.max_concurrent,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    # force: an import has already given the root logger a handler (WARNING, no times), which would make this a no-op
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    config = ServerConfig.model_validate(yaml.safe_load(args.config.read_text()))
    backend = load_backend(environment_import_path(config))
    backend.refuse_unreachable_agent_network(config.environment_kwargs, config.agent_network)
    tasks = asyncio.run(load_tasks(config.dataset, config.download_dir, config.task_names))
    if unrunnable := backend.unrunnable_tasks(config.environment_kwargs, tasks):
        listed = "; ".join(f"{name}: {why}" for name, why in list(unrunnable.items())[:10])
        raise SystemExit(f"FATAL: {len(unrunnable)} of the {len(tasks)} tasks cannot start a sandbox: {listed}")
    print(
        f"TRIAL_SERVER_TASKS {len(tasks)} of {config.dataset} harness={config.harness}"
        f" environment={backend.import_path or 'default'}",
        flush=True,
    )
    # The stdlib event loop, not uvloop (uvicorn's "auto" pick when installed; a server that forks daemons keeps off it
    # too): uvloop leaves in each child inheritable copies of its stdio pipes (libuv dup2s them onto 0-2 and keeps
    # the originals across exec), and a child that daemonizes keeps them: a sandbox backend's filesystem daemon did,
    # the provider's communicate() never saw their EOF, and every template mount hung until its create timeout.
    # The stdlib loop's children (subprocess, close_fds) get 0-2 alone.
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(
            make_app(TrialServer(config, tasks)), host=args.host, port=args.port, access_log=False, loop="asyncio"
        )


if __name__ == "__main__":
    main()
