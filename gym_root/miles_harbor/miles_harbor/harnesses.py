"""Harbor's installed agents, installed from a pack built ahead instead of from the internet.

    agents: [{import_path: "miles_harbor.harnesses:MiniSweAgent", model_name: ..., kwargs: {...}}]

Harbor's install() of an agent fetches it into every trial's container (uv, npm, curl | sh). Here
a harness is built once per version into a pack, which every sandbox sees read-only at PACK_ROOT
(HARBOR_PACK_ROOT, default ~/.cache/miles-harness; a sandbox backend may mount a pack
there). install() keeps the system packages
Harbor's install() puts in the image, so the agent works in the image it has on docker, and links
the pack's executables where Harbor's run() looks for them. Everything else is Harbor's own: the
options, the config, run(), the trajectory.

A pack is ``<PACK_ROOT>/<name>/``: VERSION (the harness's version) and one tree per C library,
``glibc/`` and ``musl/`` (the image's, told apart as Harbor's codex install() does), each holding
what the links point to; a local version, ``<version>+miles<N>``, is the harness and miles' own
extensions to it (gym_root/miles_harbor/packs/harbor_packs.sh). A harness is one class here, Harbor's agent and
its Pack; HARNESSES maps the names Harbor knows them by to the classes. A harness that runs beside the
sandbox, in the trial's process, has no pack and a module of its own (miles_harbor.vanillux: TMax's).

A harness's ENDPOINT is how it is pointed at an OpenAI-compatible model server (an sglang endpoint
for an evaluation, miles' session server for a rollout): the model name Harbor hands it and the env
its run() reads the server and key from. For a rollout (miles_harbor.trial_server) a harness also
says how its options take a rollout's limits (rollout_kwargs: its turns, its tokens per reply) and
how its run ended (outcome: from its own log, which Harbor does not read for it).
"""

import copy
import dataclasses
import json
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import yaml
from harbor.agents.installed.mini_swe_agent import MiniSweAgent as HarborMiniSweAgent
from harbor.environments.base import BaseEnvironment

from miles_harbor.harness_api import Endpoint, Outcome

# Where the packs are: HARBOR_PACK_ROOT, else ~/.cache/miles-harness; under Harbor's docker environment it is a path
# of the image.
PACK_ROOT = os.environ.get("HARBOR_PACK_ROOT", str(Path.home() / ".cache" / "miles-harness"))

# the image's C library: musl (Alpine) or glibc, as Harbor's codex install() tells them apart
MUSL_CHECK = "ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]"
# a glibc tree's GLIBC_MIN (the oldest glibc its wheels take: gym_root/miles_harbor/packs/harbor_packs.sh)
# against the image's
GLIBC_CHECK = """\
if [ -f {tree}/GLIBC_MIN ]; then
  min=$(cat {tree}/GLIBC_MIN)
  if have=$(getconf GNU_LIBC_VERSION 2>/dev/null); then have=${{have#glibc }}
  else have=$(ldd --version 2>&1 | head -n 1) || true; have=${{have##* }}; fi
  hmaj=${{have%%.*}}; hmin=${{have#*.}}; hmin=${{hmin%%[!0-9]*}}; mmaj=${{min%%.*}}; mmin=${{min#*.}}
  if ! {{ [ "$hmaj" -gt "$mmaj" ] || {{ [ "$hmaj" -eq "$mmaj" ] && [ "$hmin" -ge "$mmin" ]; }}; }} 2>/dev/null; then
    echo "the {name} pack needs glibc >= $min; the image has ${{have:-no glibc found}}"; exit 1
  fi
fi"""


@dataclasses.dataclass(frozen=True)
class Pack:
    name: str
    # Harbor's install() of the agent ensures these (its names: apt / dnf / yum / apk packages each)
    system_dependencies: tuple[str, ...]
    # (an executable in the pack's tree, the path Harbor's run() runs it by; $HOME is the agent user's)
    links: tuple[tuple[str, str], ...]

    @property
    def home(self) -> str:
        return f"{PACK_ROOT}/{self.name}"

    def tree(self, libc: str) -> str:
        return f"{self.home}/{libc}"

    @property
    def version_file(self) -> str:
        return f"{self.home}/VERSION"


class Preinstalled:
    """install() from the class's PACK (the module docstring); first in the bases, before Harbor's agent."""

    PACK: ClassVar[Pack]

    async def install(self, environment: BaseEnvironment) -> None:
        pack = self.PACK
        await self.ensure_system_dependencies(environment, pack.system_dependencies)
        musl = await environment.exec(command=MUSL_CHECK)
        libc = "musl" if musl.return_code == 0 else "glibc"
        tree = pack.tree(libc)
        asked = shlex.quote(self._version or "")
        script = [
            "set -eu",
            f'v=$(cat {pack.version_file} 2>/dev/null) || {{ echo "no {pack.name} pack at {pack.home}"; exit 1; }}',
            f'[ -z {asked} ] || [ "$v" = {asked} ] || {{ echo "the {pack.name} pack is $v, not {asked}"; exit 1; }}',
        ]
        if libc == "glibc":
            script.append(GLIBC_CHECK.format(tree=tree, name=pack.name))
        for source, link in pack.links:
            target = f"{tree}/{source}"
            script += [
                f'[ -x {target} ] || {{ echo "{target}: no such executable in the pack"; exit 1; }}',
                f'mkdir -p "$(dirname "{link}")"',
                f'ln -sfn {target} "{link}"',
            ]
        await self.exec_as_agent(environment, command="\n".join(script))

    def get_version_command(self) -> str | None:
        return f"cat {self.PACK.version_file}"


def _mini_swe_agent_env(base_url: str, api_key: str) -> dict[str, str]:
    # litellm's openai provider reads either base URL name, Harbor's run() takes the key from MSWEA_API_KEY, and its
    # cost tracking would fail on a model litellm has no price for (Harbor's run() sets this too)
    return {
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_BASE": base_url,
        "OPENAI_API_KEY": api_key,
        "MSWEA_API_KEY": api_key,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }


class MiniSweAgent(Preinstalled, HarborMiniSweAgent):
    PACK = Pack(
        name="mini-swe-agent",
        system_dependencies=("curl", "bash", "build_tools", "git"),
        links=(("bin/mini-swe-agent", "$HOME/.local/bin/mini-swe-agent"),),
    )
    # litellm's openai provider: chat completions at the base URL
    ENDPOINT = Endpoint(model_name=lambda model: f"openai/{model}", env=_mini_swe_agent_env)

    @staticmethod
    def rollout_kwargs(kwargs: Mapping[str, Any], *, step_limit: int | None, max_tokens: int | None) -> dict:
        """The agent's kwargs with a rollout's limits: its config's ``agent.step_limit`` (the config file, read
        here, becomes Harbor's inline ``config``) and Harbor's ``max_tokens`` (each reply's)."""
        kwargs = dict(kwargs)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if step_limit is not None:
            config = kwargs.pop("config", None)
            if config is None and kwargs.get("config_file"):
                config = yaml.safe_load(Path(kwargs["config_file"]).read_text())
            kwargs.pop("config_file", None)
            config = copy.deepcopy(config or {})
            config.setdefault("agent", {})["step_limit"] = step_limit
            kwargs["config"] = config
        return kwargs

    @staticmethod
    def outcome(agent_dir: Path) -> Outcome:
        """From the trajectory mini-swe-agent writes after every step: its exit status, and its model calls."""
        try:
            info = json.loads((agent_dir / "mini-swe-agent.trajectory.json").read_text()).get("info") or {}
        except (OSError, ValueError, AttributeError):
            return Outcome(exit_status=None, submitted=False, steps=None)
        status = info.get("exit_status")
        calls = (info.get("model_stats") or {}).get("api_calls")
        return Outcome(
            exit_status=status if isinstance(status, str) else None,
            submitted=status == "Submitted",
            steps=calls if isinstance(calls, int) else None,
        )


# after the classes it imports from here: a harness of its own module (vanillux) takes Endpoint and Outcome from
# harness_api, not from this module
from miles_harbor.vanillux import VanilluxAgent  # noqa: E402

HARNESSES: dict[str, type] = {cls.name(): cls for cls in (MiniSweAgent, VanilluxAgent)}
__all__ = ["HARNESSES", "Endpoint", "MiniSweAgent", "Outcome", "Pack", "Preinstalled", "VanilluxAgent"]
