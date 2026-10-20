"""miles_harbor.harnesses: Harbor's agents installed from a pack (a pack tree under tmp_path for PACK_ROOT;
the install script run by a real bash, Harbor's own package checks answered by the fake)."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")  # the Harbor package is in gym_root/miles_harbor's venv, not in every checkout

import miles_harbor.harnesses as harnesses  # noqa: E402
from harbor.agents.factory import AgentFactory  # noqa: E402
from harbor.agents.installed.mini_swe_agent import MiniSweAgent as HarborMiniSweAgent  # noqa: E402
from harbor.models.trial.config import AgentConfig  # noqa: E402
from miles_harbor.harnesses import HARNESSES, MUSL_CHECK, MiniSweAgent, VanilluxAgent  # noqa: E402


class FakeEnvironment:
    """Answers Harbor's dependency checks (``command -v``: present unless ``missing``), its package manager
    probe and MUSL_CHECK; runs anything else with bash, HOME at ``home``."""

    def __init__(self, home, musl=False, missing=False, glibc=None):
        self.home, self.musl, self.missing = home, musl, missing
        self.commands: list[tuple[str, object]] = []
        self.path = "/usr/bin:/bin"
        if glibc:  # the image's glibc, as getconf tells it
            fakebin = home.parent / f"fakebin-{home.name}"
            fakebin.mkdir(parents=True, exist_ok=True)
            (fakebin / "getconf").write_text(f'#!/bin/sh\n[ "$1" = GNU_LIBC_VERSION ] && echo "glibc {glibc}"\n')
            (fakebin / "getconf").chmod(0o755)
            self.path = f"{fakebin}:{self.path}"

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        command = command.removeprefix("set -o pipefail; ")  # what Harbor's agents put in front of theirs
        self.commands.append((command, user))
        if command.startswith("command -v "):
            return SimpleNamespace(stdout="", stderr="", return_code=1 if self.missing else 0)
        if command.startswith("for manager in apt-get"):
            return SimpleNamespace(stdout="apt-get", stderr="", return_code=0)
        if command == MUSL_CHECK:
            return SimpleNamespace(stdout="", stderr="", return_code=0 if self.musl else 1)
        if command.startswith("apt-get "):
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        done = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            env={"HOME": str(self.home), "PATH": self.path},
        )
        return SimpleNamespace(stdout=done.stdout + done.stderr, stderr=None, return_code=done.returncode)


@pytest.fixture
def pack(monkeypatch, tmp_path):
    root = tmp_path / "packs"
    monkeypatch.setattr(harnesses, "PACK_ROOT", str(root))

    def make(name="mini-swe-agent", version="2.4.6", libcs=("glibc", "musl"), glibc_min=None):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "VERSION").write_text(version + "\n")
        for libc in libcs:
            exe = root / name / libc / "bin" / "mini-swe-agent"
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_text(f"#!/bin/sh\necho mini {libc}\n")
            exe.chmod(0o755)
        if glibc_min:
            (root / name / "glibc" / "GLIBC_MIN").write_text(glibc_min + "\n")
        return root / name

    return make


def agent(tmp_path, **kwargs):
    return MiniSweAgent(logs_dir=tmp_path / "logs", model_name="openai/x", **kwargs)


@pytest.mark.parametrize("musl", [False, True])
async def test_install_links_the_packs_tree_for_the_images_libc(tmp_path, pack, musl):
    tree = pack()
    home = tmp_path / "home"
    env = FakeEnvironment(home, musl=musl)
    await agent(tmp_path).install(env)
    link = home / ".local" / "bin" / "mini-swe-agent"
    assert link.is_symlink() and link.resolve() == (tree / ("musl" if musl else "glibc") / "bin" / "mini-swe-agent")
    assert (
        subprocess.run([str(link)], capture_output=True, text=True).stdout == f"mini {'musl' if musl else 'glibc'}\n"
    )
    assert not any(c.startswith(("apt-get", "uv ", "curl")) for c, _ in env.commands)  # nothing fetched


async def test_harbors_system_dependencies_are_still_installed(tmp_path, pack):
    pack()
    env = FakeEnvironment(tmp_path / "home", missing=True)
    await agent(tmp_path).install(env)
    (apt,) = [(c, user) for c, user in env.commands if c.startswith("apt-get ")]
    assert apt[1] == "root" and "build-essential" in apt[0] and "git" in apt[0] and "curl" in apt[0]


async def test_a_version_asked_for_must_be_the_packs_and_a_missing_pack_fails_the_install(tmp_path, pack):
    pack(version="2.4.6")
    await agent(tmp_path, version="2.4.6").install(FakeEnvironment(tmp_path / "h1"))
    with pytest.raises(RuntimeError, match="the mini-swe-agent pack is 2.4.6, not 2.5.0"):
        await agent(tmp_path, version="2.5.0").install(FakeEnvironment(tmp_path / "h2"))
    pack(name="other")
    harnesses_root = tmp_path / "packs"
    (harnesses_root / "mini-swe-agent" / "VERSION").rename(harnesses_root / "VERSION.moved")
    with pytest.raises(RuntimeError, match="no mini-swe-agent pack at"):
        await agent(tmp_path).install(FakeEnvironment(tmp_path / "h3"))


async def test_a_pack_without_the_images_libc_fails_the_install(tmp_path, pack):
    pack(libcs=("glibc",))
    with pytest.raises(RuntimeError, match="musl/bin/mini-swe-agent: no such executable in the pack"):
        await agent(tmp_path).install(FakeEnvironment(tmp_path / "home", musl=True))


async def test_the_version_comes_from_the_pack(tmp_path, pack):
    pack(version="2.4.6")
    mini = agent(tmp_path)
    result = await FakeEnvironment(tmp_path / "home").exec(mini.get_version_command())
    assert mini.parse_version(result.stdout) == "2.4.6"


def test_harbor_loads_the_harness_by_import_path_and_everything_but_install_is_harbors(tmp_path):
    config = AgentConfig(import_path="miles_harbor.harnesses:MiniSweAgent", model_name="openai/x")
    loaded = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path / "logs")
    assert type(loaded) is MiniSweAgent and isinstance(loaded, HarborMiniSweAgent)
    assert MiniSweAgent.run is HarborMiniSweAgent.run and MiniSweAgent.name() == "mini-swe-agent"
    assert HARNESSES == {"mini-swe-agent": MiniSweAgent, "vanillux": VanilluxAgent}


@pytest.mark.parametrize(
    ("image", "ok"), [("2.31", True), ("2.28", True), ("3.0", True), ("2.27", False), ("1.99", False)]
)
async def test_a_glibc_tree_needs_the_glibc_its_wheels_take(tmp_path, pack, image, ok):
    pack(glibc_min="2.28")
    install = agent(tmp_path).install(FakeEnvironment(tmp_path / "home", glibc=image))
    if ok:
        await install
    else:
        with pytest.raises(RuntimeError, match=f"the mini-swe-agent pack needs glibc >= 2.28; the image has {image}"):
            await install


async def test_a_musl_image_skips_the_glibc_check(tmp_path, pack):
    pack(glibc_min="2.28")
    await agent(tmp_path).install(FakeEnvironment(tmp_path / "home", musl=True))


def test_the_endpoint_binding_is_what_harbors_mini_swe_agent_resolves_its_model_connection_from(tmp_path):
    url = "http://10.1.2.3:7000/v1"
    config = MiniSweAgent.ENDPOINT.agent_config(url + "/", "EMPTY", "Qwen3.5-4B")
    assert config["model_name"] == "openai/Qwen3.5-4B"
    loaded = AgentFactory.create_agent_from_config(
        AgentConfig(import_path="miles_harbor.harnesses:MiniSweAgent", **config), logs_dir=tmp_path / "logs"
    )
    access = loaded.model_connection
    assert access.api_key == "EMPTY"
    assert access.env["OPENAI_BASE_URL"] == access.env["OPENAI_API_BASE"] == url
    assert config["env"]["MSWEA_COST_TRACKING"] == "ignore_errors"


def test_pack_root_comes_from_the_environment(tmp_path):
    """PACK_ROOT (D18): HARBOR_PACK_ROOT, else ~/.cache/miles-harness, read when the module is imported -- a subprocess
    per value, no reload."""
    code = "from miles_harbor.harnesses import Pack; print(Pack('mini-swe-agent', (), ()).home)"
    clean = {k: v for k, v in os.environ.items() if k not in ("HARBOR_PACK_ROOT", "HOME")}

    def home(**env):
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**clean, **env}
        )
        return done.stdout.strip()

    assert home(HARBOR_PACK_ROOT="/x", HOME=str(tmp_path)) == "/x/mini-swe-agent"
    assert home(HOME=str(tmp_path)) == f"{tmp_path}/.cache/miles-harness/mini-swe-agent"
