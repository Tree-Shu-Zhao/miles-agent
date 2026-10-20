"""A sandbox backend for the trial server's tests (HARBOR_ENVIRONMENT_IMPORT_PATH=_fake_backend:FakeEnvironment):
the environment class the import path names (never instantiated here: the tests' stand-in for Harbor's Trial builds
no environment) and the six hooks of trial_server.Backend's protocol, each recording ``(name, args)`` in CALLS and
answering the neutral value -- the environment's kwargs as the config has them, nothing more for the endpoint, every
URL accepted, every task runnable, every agent_network enforceable, the agent's network left as the task has it. The
test module imports harbor; this one imports nothing of it, so it loads wherever the tests do (this directory is on
sys.path as pytest collects them)."""

CALLS: list[tuple[str, tuple]] = []


class FakeEnvironment:
    """The environment class of the import path; records its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def environment_kwargs(config) -> dict:
    CALLS.append(("environment_kwargs", (config,)))
    return dict(config.environment_kwargs)


def network_kwargs(policy_base_url) -> dict:
    CALLS.append(("network_kwargs", (policy_base_url,)))
    return {}


def check_policy_base_url(url) -> None:
    CALLS.append(("check_policy_base_url", (url,)))


def unrunnable_tasks(kwargs, tasks) -> dict:
    CALLS.append(("unrunnable_tasks", (kwargs, tasks)))
    return {}


def refuse_unreachable_agent_network(kwargs, agent_network) -> None:
    CALLS.append(("refuse_unreachable_agent_network", (kwargs, agent_network)))


def limit_agent_network(task_config, policy_base_url) -> None:
    CALLS.append(("limit_agent_network", (task_config, policy_base_url)))
