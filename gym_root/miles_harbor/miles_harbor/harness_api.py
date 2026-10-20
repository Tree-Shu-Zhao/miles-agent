"""What a harness of miles_harbor.harnesses declares to the trial server: how it is pointed at a model server
(Endpoint) and how its run ended (Outcome)."""

import dataclasses
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class Endpoint:
    """A harness pointed at an OpenAI-compatible server: AgentConfig.model_name from the served model's name, and
    AgentConfig.env from the server's base URL (ending in /v1) and API key."""

    model_name: Callable[[str], str]
    env: Callable[[str, str], dict[str, str]]
    # where its model calls come from: a process in the sandbox (the server is reached from there), else the trial's
    # own process, beside the sandbox (the sandbox needs no way to the server)
    in_sandbox: bool = True

    def agent_config(self, base_url: str, api_key: str, model: str) -> dict[str, object]:
        """The model_name and env entries of a Harbor agent config."""
        return {"model_name": self.model_name(model), "env": self.env(base_url.rstrip("/"), api_key)}


@dataclasses.dataclass(frozen=True)
class Outcome:
    """How a harness's run ended, as its own log tells it."""

    # the harness's name for it (mini-swe-agent: Submitted, LimitsExceeded, an exception's name); None: no log
    exit_status: str | None
    # it ended by handing in its work; anything else is reported as agent_error (not submitted)
    submitted: bool
    # its model calls; None: unknown
    steps: int | None
