"""The Harbor harness-pack installer and the file it installs live under gym_root/miles_harbor/packs/ (a git mv out of
the old infra directory; DESIGN §6 I2a step 4, S9): both files are there; the installer parses; no text file of the
tree names the old paths (the pattern is built at run time so that this file does not name them either; only
reports/, the private run ledgers, may).

One contract on both sides (N6): the pack root is HARBOR_PACK_ROOT for the runtime (harnesses.py reads the packs
from it) and for the installer (harbor_packs.sh writes under it; default $HOME/.cache/miles-harness on both sides), so
a wrapper that mounts the pack elsewhere puts HARBOR_PACK_ROOT=<its mount point> into the installer's environment.
The installer reads miles_mini_ext.py from its own directory, so it runs from /installer/ under any wrapper and from
the repo alike. mini_swebench.yaml keeps `environment_class: miles_mini_ext.InterpreterLocalEnvironment` (N1): the
installer copies the file as a TOP-LEVEL module of the pack's python, so the name mini-swe-agent imports does not
follow the source file's place in the repo."""

import re
import shutil
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PACKS = REPO / "gym_root" / "miles_harbor" / "packs"
INSTALLER = PACKS / "harbor_packs.sh"
EXTENSION = PACKS / "miles_mini_ext.py"
HARNESSES = REPO / "gym_root" / "miles_harbor" / "miles_harbor" / "harnesses.py"
MINI_CONFIG = REPO / "gym_root" / "miles_harbor" / "configs" / "mini_swebench.yaml"
MINI_TEST = REPO / "gym_root" / "miles_harbor" / "tests" / "test_mini_ext.py"
BASH = shutil.which("bash")
# the old directory, in two pieces: the grep below must find no file naming it
OLD_DIR = "/".join(["infra", "harnesses"])
MOVED = ("harbor_packs.sh", "miles_mini_ext.py")


def _lines_naming_the_old_paths():
    pattern = "|".join(f"{OLD_DIR}/{name}" for name in MOVED)
    includes = [f"--include=*.{ext}" for ext in ("py", "sh", "sb" "atch", "md", "toml", "yaml", "yml", "txt", "cfg", "ini", "example")]
    excludes = [f"--exclude-dir={d}" for d in ("reports", "3rdparty", ".git", "__pycache__", "outputs", ".venv", ".claude", "node_modules")]
    proc = subprocess.run(["grep", "-rnIE", *includes, *excludes, "--", pattern, "."], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    return proc.stdout.splitlines()


def test_the_pack_installer_and_its_files_live_under_miles_harbor_packs():
    assert INSTALLER.is_file() and EXTENSION.is_file()
    for name in MOVED:
        assert not (REPO / OLD_DIR / name).exists(), name
    proc = subprocess.run([BASH, "-n", str(INSTALLER)], capture_output=True, text=True)
    assert proc.returncode == 0, f"{INSTALLER}: {proc.stderr}"
    assert _lines_naming_the_old_paths() == []
    # the N6 contract, statically: the installer writes under HARBOR_PACK_ROOT (default = the runtime's, harnesses.py)
    installer = INSTALLER.read_text()
    assert "PACK=${HARBOR_PACK_ROOT:-$HOME/.cache/miles-harness}/$NAME" in installer
    assert len(re.findall(r"(?m)^PACK=", installer)) == 1


def test_installer_and_runtime_share_the_pack_root_and_the_extension_travels_with_it():
    installer = INSTALLER.read_text()
    runtime = HARNESSES.read_text()
    # one variable name on both sides: the runtime reads the packs from HARBOR_PACK_ROOT, the installer writes there
    assert 'os.environ.get("HARBOR_PACK_ROOT"' in runtime
    assert "${HARBOR_PACK_ROOT:-" in installer
    # the installer reads its extension from its own directory (the copy and the recorded sha256): no /installer/ path
    # is baked in, so it runs from /installer/ under a wrapper and from gym_root/miles_harbor/packs/ in the repo
    assert 'cp "$(dirname "$0")/miles_mini_ext.py" "$site/"' in installer
    assert 'sha256sum "$(dirname "$0")/miles_mini_ext.py"' in installer
    assert "/installer/" not in installer
    assert "HARBOR_PACK_ROOT" in installer.split("set -euo pipefail")[0], "the header states the contract"
    # N1: the extension is a top-level module of the pack's python, so the config's import name stays the bare module
    # and the extension's own test imports it from packs/
    config = yaml.safe_load(MINI_CONFIG.read_text())
    assert config["environment"]["environment_class"] == "miles_mini_ext.InterpreterLocalEnvironment"
    assert 'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packs"))' in MINI_TEST.read_text()
    assert "gym_root/miles_harbor/packs/harbor_packs.sh" in EXTENSION.read_text()
    assert "gym_root/miles_harbor/packs/harbor_packs.sh" in runtime
    # packs/ is not a python package of miles_harbor (its pyproject installs miles_harbor only): the module reaches the
    # pack's python through the installer's copy, never through an import of miles_harbor
    assert not (PACKS / "__init__.py").exists()
