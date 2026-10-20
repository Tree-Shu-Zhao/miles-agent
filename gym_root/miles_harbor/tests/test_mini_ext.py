"""gym_root/miles_harbor/packs/miles_mini_ext.py: mini-swe-agent's local environment running each command through its
docker environment's interpreter, so that mini's SWE-bench config (bash -c, BASH_ENV) runs inside the task's container
as it does through docker (mini-swe-agent itself, as the pack has it, from the test venv)."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

minisweagent = pytest.importorskip("minisweagent")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packs"))
from miles_mini_ext import InterpreterLocalEnvironment, InterpreterLocalEnvironmentConfig  # noqa: E402
from minisweagent.environments import get_environment  # noqa: E402
from minisweagent.environments.docker import DockerEnvironmentConfig  # noqa: E402
from minisweagent.exceptions import Submitted  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mini_swebench.yaml"


def mini_environment(tmp_path, environment_class=None, **env):
    """The environment mini_swebench.yaml makes, its cwd tmp_path and ``env`` over its own."""
    config = yaml.safe_load(CONFIG.read_text())["environment"]
    config["cwd"] = str(tmp_path)
    config["env"] = {**config["env"], **env}
    if environment_class:
        config["environment_class"] = environment_class
    return get_environment(config)


@pytest.fixture
def bashrc(tmp_path):
    rc = tmp_path / "bashrc"
    rc.write_text("export ACTIVE_ENV=testbed\nactivated() { echo function-from-bashrc; }\n")
    return rc


def test_the_swebench_config_runs_bash_which_reads_its_bash_env(tmp_path, bashrc):
    env = mini_environment(tmp_path, BASH_ENV=str(bashrc))
    assert isinstance(env, InterpreterLocalEnvironment) and env.config.interpreter == ["bash", "-c"]
    out = env.execute({"command": 'echo "$ACTIVE_ENV"; [[ -n $BASH_VERSION ]] && echo bash; activated; pwd -P'})
    assert out["returncode"] == 0, out
    assert out["output"].splitlines() == ["testbed", "bash", "function-from-bashrc", str(tmp_path.resolve())]


def test_the_plain_local_environment_never_reads_bash_env(tmp_path, bashrc):
    # what the class is for: /bin/sh -c (dash, or bash as sh) ignores BASH_ENV, so testbed is never activated
    env = mini_environment(tmp_path, environment_class="local", BASH_ENV=str(bashrc))
    assert env.execute({"command": 'echo "[$ACTIVE_ENV]"'})["output"] == "[]\n"


@pytest.mark.parametrize(
    "command",
    [
        "printf '%s\\n' \"it's\" '$HOME' \"a  b\" `echo sub`",
        "cat <<'EOF'\nline 1\n  $not_expanded \\\nEOF\necho after",
        "echo out; echo err >&2; exit 3",
        "",
    ],
)
def test_a_command_reaches_the_interpreter_as_it_was_written(tmp_path, command):
    env = InterpreterLocalEnvironment(cwd=str(tmp_path), interpreter=["bash", "-c"])
    direct = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = env.execute({"command": command})
    assert (out["output"], out["returncode"]) == (direct.stdout, direct.returncode)


def test_the_action_the_trajectory_keeps_is_not_rewritten(tmp_path):
    action = {"command": "echo hi", "tool_call_id": "call_1"}
    InterpreterLocalEnvironment(cwd=str(tmp_path), interpreter=["bash", "-c"]).execute(action)
    assert action == {"command": "echo hi", "tool_call_id": "call_1"}


def test_the_submission_still_ends_the_run(tmp_path):
    env = InterpreterLocalEnvironment(cwd=str(tmp_path), interpreter=["bash", "-c"])
    with pytest.raises(Submitted) as raised:
        env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; printf 'the patch\\n'"})
    assert raised.value.messages[0]["extra"]["submission"] == "the patch\n"


def test_a_timeout_kills_the_interpreter_and_what_it_started(tmp_path):
    env = InterpreterLocalEnvironment(cwd=str(tmp_path), interpreter=["bash", "-c"], timeout=1)
    start = time.monotonic()
    out = env.execute({"command": "sleep 30 & sleep 30; echo never"})
    # a background child left holding the output pipe would keep the read open for its 30 s
    assert out["returncode"] == -1 and time.monotonic() - start < 10
    # macOS's killpg now and then leaves the group's leader alive (mini's own `/bin/sh -c` too: 4 of 6 runs), which
    # then runs on; Linux's kills the whole group
    if sys.platform != "darwin":
        assert "never" not in out["output"]


def test_the_default_interpreter_is_the_docker_environments():
    ours = InterpreterLocalEnvironmentConfig.model_fields["interpreter"].default
    assert ours == DockerEnvironmentConfig.model_fields["interpreter"].default == ["bash", "-lc"]


def test_the_swebench_config_is_minis_but_for_its_environment_class():
    if minisweagent.__version__ != "2.4.6":
        pytest.skip(f"mini_swebench.yaml is 2.4.6's; the test venv has mini-swe-agent {minisweagent.__version__}")
    packaged = Path(minisweagent.__file__).parent / "config" / "benchmarks" / "swebench.yaml"
    theirs, ours = yaml.safe_load(packaged.read_text()), yaml.safe_load(CONFIG.read_text())
    assert theirs["environment"].pop("environment_class") == "docker"
    assert ours["environment"].pop("environment_class") == "miles_mini_ext.InterpreterLocalEnvironment"
    assert ours == theirs
