"""mini-swe-agent's local environment with its docker environment's ``interpreter``, for the mini that Harbor runs
inside the task's container (there it is ``environment_class: local``, not docker), so that a config written for
the docker environment runs its commands as there: each one through ``interpreter`` (DockerEnvironment: ``docker
exec ... <interpreter...> <command>``) instead of the local environment's ``/bin/sh -c <command>``.

mini-swe-agent's SWE-bench config (config/benchmarks/swebench.yaml) depends on it: its ``bash -c`` reads BASH_ENV
(/root/.bashrc), whose ``conda activate testbed`` puts the task's python first on PATH. /bin/sh (dash on the
SWE-bench images) never reads BASH_ENV, so under the plain local environment every command runs in conda's base env:
the repository's dependencies missing, its tests not runnable, until the agent pip-installs into the wrong python.

Shipped in the mini-swe-agent pack (gym_root/miles_harbor/packs/harbor_packs.sh, beside this file: a ``+miles``
local version of the pack) as a top-level module of its python, named in a config by
``environment_class: miles_mini_ext.InterpreterLocalEnvironment``.
"""

import shlex

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


class InterpreterLocalEnvironmentConfig(LocalEnvironmentConfig):
    interpreter: list[str] = ["bash", "-lc"]
    """DockerEnvironmentConfig's ``interpreter`` and its default: the command is appended to it as one argument."""


class InterpreterLocalEnvironment(LocalEnvironment):
    def __init__(self, *, config_class: type = InterpreterLocalEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        # the local environment's /bin/sh -c execs the interpreter with the command as its one argument (the shell's
        # pid, process group and cwd, so its timeout still kills the lot); the action itself, which the trajectory
        # keeps, is not touched
        command = "exec " + shlex.join([*self.config.interpreter, action.get("command", "")])
        return super().execute({**action, "command": command}, cwd, timeout=timeout)
