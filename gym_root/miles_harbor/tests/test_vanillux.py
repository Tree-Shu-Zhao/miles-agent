"""miles_harbor.vanillux: TMax's harness against a fake OpenAI-compatible server (replies scripted per test, every
request recorded) and a fake environment that runs each command with a real bash in a scratch root."""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")  # the Harbor package is in gym_root/miles_harbor's venv, not in every checkout

import miles_harbor.vanillux as vanillux  # noqa: E402
from harbor.agents.factory import AgentFactory  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402
from harbor.models.trial.config import AgentConfig  # noqa: E402
from miles_harbor.vanillux import (  # noqa: E402
    SUBMIT_MARKER,
    VanilluxAgent,
    bash_command,
    observation,
    tool_calls,
    truncate_observation,
)


def call(command, call_id="c1", name="bash"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"command": command})},
    }


def reply(*calls, content="THOUGHT: x", reasoning="thinking", prompt_tokens=100, completion_tokens=10):
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        message["tool_calls"] = list(calls)
    return {
        "id": "r",
        "object": "chat.completion",
        "created": 0,
        "model": "model",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class Server:
    """Answers each POST with the next scripted reply (an int: that HTTP status), recording the request bodies."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                answer = server.replies.pop(0) if server.replies else 400  # unscripted: the run ends at once
                status, body = (
                    (answer, {"error": {"message": "too long"}}) if isinstance(answer, int) else (200, answer)
                )
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"


# the sandbox's paths the harness and the tests use, each where a path begins
SANDBOX_PATHS = re.compile(r"(?<![\w./-])/(?:tmp/|workspace\b|output\b|logs/|root\b|app\b)")

# GNU timeout where the host has none (macOS): the command in a process group of its own, killed whole at the limit
TIMEOUT_SHIM = f"""#!{sys.executable}
import os, signal, subprocess, sys
args = sys.argv[1:]
while args[0].startswith("--"):
    args.pop(0)
limit = float(args.pop(0))
child = subprocess.Popen(args, start_new_session=True)
try:
    sys.exit(child.wait(timeout=limit))
except subprocess.TimeoutExpired:
    os.killpg(child.pid, signal.SIGKILL)
    child.wait()
    sys.exit(124)
"""


class Environment:
    """Runs commands with bash, the sandbox's / at ``root`` (its absolute paths in a command rewritten under it)."""

    def __init__(self, root):
        self.root, self.commands = root, []
        self.env = dict(os.environ)
        if shutil.which("timeout") is None:
            shim = root.parent / "bin" / "timeout"
            shim.parent.mkdir(exist_ok=True)
            shim.write_text(TIMEOUT_SHIM)
            shim.chmod(0o755)
            self.env["PATH"] = f"{shim.parent}{os.pathsep}{self.env['PATH']}"

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append((command, timeout_sec))
        command = SANDBOX_PATHS.sub(lambda m: f"{self.root}{m.group(0)}", command)
        done = subprocess.run(
            ["bash", "-c", "exec 2>&1\n" + command], capture_output=True, text=True, cwd=self.root, env=self.env
        )
        return SimpleNamespace(stdout=done.stdout or None, stderr=None, return_code=done.returncode)


@pytest.fixture
def root(tmp_path):
    for d in ("tmp", "logs"):
        (tmp_path / "sb" / d).mkdir(parents=True)
    return tmp_path / "sb"


async def run_agent(tmp_path, root, replies, **kwargs):
    server = Server(replies)
    agent = VanilluxAgent(
        logs_dir=tmp_path / "agent",
        model_name="model",
        extra_env={"OPENAI_BASE_URL": server.url, "OPENAI_API_KEY": "EMPTY"},
        **kwargs,
    )
    environment = Environment(root)
    await agent.setup(environment)
    context = AgentContext()
    await agent.run("  Do the task.  ", environment, context)
    server.httpd.shutdown()
    return server, environment, context, VanilluxAgent.outcome(tmp_path / "agent")


async def test_a_run_keeps_the_shell_and_replays_each_reply_as_the_server_sent_it(tmp_path, root):
    replies = [
        reply(call("cd /workspace && export X=42 && echo hi")),
        reply(call("pwd; echo $X; ls /nonexistent; (exit 3)", "c2")),
        reply(call(f"echo {SUBMIT_MARKER}", "c3")),
    ]
    server, _, context, outcome = await run_agent(tmp_path, root, replies)
    assert outcome.submitted and outcome.exit_status == "submitted" and outcome.steps == 3
    first, second, third = server.requests
    assert first["messages"][0]["content"] == vanillux.SYSTEM_TEMPLATE
    assert first["messages"][1]["content"].startswith("Please solve this task:\n\nDo the task.\n\nYou can execute")
    assert first["tools"] == [vanillux.BASH_TOOL] and first["max_tokens"] == 16384
    assert "temperature" not in first and "top_p" not in first  # the server's sampling
    # the reply as sent, reasoning included, then the step's observation
    assert second["messages"][2] == {
        "role": "assistant",
        "content": "THOUGHT: x",
        "reasoning_content": "thinking",
        "tool_calls": [call("cd /workspace && export X=42 && echo hi")],
    }
    assert second["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "hi\n\n\n(exit_code=0)"}
    # the cwd and the export persisted; the command's exit code
    obs = third["messages"][5]["content"]
    assert obs.startswith(f"{root}/workspace\n42\n") and "No such file" in obs and obs.endswith("(exit_code=3)")
    # the conversation's tokens so far (the server's 110) and the estimate of the observation: under 16384 left
    assert second["max_tokens"] == 16384 and context.n_output_tokens == 30 and context.n_input_tokens == 300
    trajectory = json.loads((tmp_path / "agent" / "trajectory.json").read_text())
    assert len(trajectory) == 7 and trajectory[-1]["tool_calls"][0]["id"] == "c3"


async def test_a_reply_without_a_tool_call_ends_the_episode_unless_format_errors_are_answered(tmp_path, root):
    _, _, _, outcome = await run_agent(tmp_path, root, [reply()])
    assert (outcome.exit_status, outcome.submitted, outcome.steps) == ("no_tool_call", False, 0)
    server, _, _, outcome = await run_agent(
        tmp_path, root, [reply(), reply(call(f"echo {SUBMIT_MARKER}"))], format_error_feedback=True
    )
    assert outcome.submitted and server.requests[1]["messages"][-1] == {
        "role": "user",
        "content": vanillux.format_error_message(vanillux.NO_TOOL_CALL_ERROR),
    }


async def test_every_bash_call_of_a_reply_is_a_step_and_the_step_limit_ends_the_run(tmp_path, root):
    # TMax's rollout: a reply's calls run in order, each a step, each answered under its id
    server, _, _, outcome = await run_agent(
        tmp_path, root, [reply(call("echo a"), call("echo b", "c2")), reply(call("echo c", "c3"))], max_steps=3
    )
    assert (outcome.exit_status, outcome.steps) == ("step_limit", 3) and len(server.requests) == 2
    assert server.requests[1]["messages"][-2:] == [
        {"role": "tool", "tool_call_id": "c1", "content": "a\n\n\n(exit_code=0)"},
        {"role": "tool", "tool_call_id": "c2", "content": "b\n\n\n(exit_code=0)"},
    ]
    # the calls past the limit do not run; a submit in a later call submits
    _, environment, _, outcome = await run_agent(
        tmp_path, root, [reply(call("echo a"), call("echo b", "c2"), call("echo z", "c3"))], max_steps=2
    )
    assert (outcome.exit_status, outcome.steps) == ("step_limit", 2) and len(environment.commands) == 3  # setup too
    _, _, _, outcome = await run_agent(tmp_path, root, [reply(call("echo a"), call(f"echo {SUBMIT_MARKER}", "c2"))])
    assert (outcome.exit_status, outcome.steps) == ("submitted", 2)


async def test_another_tool_ends_the_episode_unless_format_errors_are_answered(tmp_path, root):
    # TMax trains on the bash calls alone: a reply with only another tool's call has none
    _, _, _, outcome = await run_agent(tmp_path, root, [reply(call("x", name="python"))])
    assert (outcome.exit_status, outcome.steps) == ("no_tool_call", 0)
    # beside a bash call it is passed over
    replies = [reply(call("x", name="python"), call("echo a", "c2")), reply(call("echo b", "c3"))]
    server, _, _, outcome = await run_agent(tmp_path, root, replies, max_steps=2)
    assert (outcome.exit_status, outcome.steps) == ("step_limit", 2)
    answer = {"role": "tool", "tool_call_id": "c2", "content": "a\n\n\n(exit_code=0)"}
    assert server.requests[1]["messages"][-1] == answer
    # answered (evaluation): the format error under its id, a step, then the reply's bash calls
    replies = [reply(call("x", name="python"), call("echo a", "c2")), reply(call("echo b", "c3"))]
    server, _, _, outcome = await run_agent(tmp_path, root, replies, max_steps=2, format_error_feedback=True)
    assert (outcome.exit_status, outcome.steps) == ("step_limit", 2) and len(server.requests) == 1
    assert server.requests[0]["messages"][-1]["role"] == "user"
    trajectory = json.loads((tmp_path / "agent" / "trajectory.json").read_text())
    assert trajectory[-2:] == [
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": vanillux.format_error_message("Unknown tool 'python'. The only available tool is `bash`."),
        },
        answer,
    ]


async def test_the_context_budget_shrinks_each_reply_and_ends_the_run(tmp_path, root):
    # the response budget: 1000 + 65536 - (1000 + 60000) - (len("hi\n\n\n(exit_code=0)") / 2.5 + 1 + 32) = 5496
    replies = [
        reply(call("echo hi"), prompt_tokens=1000, completion_tokens=60000),
        reply(call("echo hi"), prompt_tokens=66000, completion_tokens=400),
    ]
    server, _, _, outcome = await run_agent(tmp_path, root, replies)
    assert server.requests[1]["max_tokens"] == 5496 and len(server.requests) == 2
    assert (outcome.exit_status, outcome.steps) == ("context", 2)
    # the context's, for a first prompt past 2,048 tokens: 67584 - (3000 + 60000) - 40 = 4544
    replies = [reply(call("echo hi"), prompt_tokens=3000, completion_tokens=60000)]
    server, _, _, outcome = await run_agent(tmp_path, root, replies)
    assert server.requests[1]["max_tokens"] == 4544
    # a request the server refuses as too long ends it too
    _, _, _, outcome = await run_agent(tmp_path, root, [400])
    assert (outcome.exit_status, outcome.steps) == ("context", 0)


async def test_a_command_past_its_timeout_is_told_so(tmp_path, root):
    server, environment, _, outcome = await run_agent(
        tmp_path, root, [reply(call("echo started; sleep 5")), reply(call(f"echo {SUBMIT_MARKER}"))], command_timeout=1
    )
    assert outcome.submitted
    assert server.requests[1]["messages"][-1]["content"] == (
        "started\n\nCommand timed out after 1s.\n\n\n(exit_code=124)"
    )
    assert environment.commands[1] == (bash_command("echo started; sleep 5", 1), 31)


async def test_a_command_with_a_nul_byte_is_a_failed_step_as_tmaxs_rollout_answers_it(tmp_path, root):
    # no process takes a NUL in an argument (the exec raised ValueError, a trial lost); TMax's
    # rollout answers a step that raised with the error and goes on
    replies = [reply(call("printf 'a\x00b' > /tmp/x")), reply(call(f"echo {SUBMIT_MARKER}", "c2"))]
    server, environment, _, outcome = await run_agent(tmp_path, root, replies)
    assert (outcome.exit_status, outcome.steps) == ("submitted", 2)
    assert server.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Step 'bash' failed: embedded null byte. Args: {'command': \"printf 'a\\x00b' > /tmp/x\"}",
    }
    assert len(environment.commands) == 2 and not any("\x00" in c for c, _ in environment.commands)  # setup, submit


def test_observations_cut_long_outputs_as_mini_swe_agent_does():
    out = "a" * 6000 + "b" * 6000
    cut = truncate_observation(out)
    assert cut.startswith(vanillux.OBS_TOO_LONG_HINT) and "---- 2000 chars elided ----" in cut
    assert cut.endswith("b" * 5000) and observation("", 0) == "(no output)\n\n(exit_code=0)"


def test_a_replys_calls_are_its_steps():
    message = {"tool_calls": [call("ls"), call("x", "c2", name="python"), call("pwd", "c3")]}
    assert tool_calls(message) == ([("c1", {"command": "ls"}), ("c3", {"command": "pwd"})], [("c2", "python")])
    bad = {"tool_calls": [{"id": "c", "function": {"name": "bash", "arguments": "{oops"}}]}
    assert tool_calls(bad) == ([], []) and tool_calls({"content": "x"}) == ([], [])


def test_harbor_loads_it_by_import_path_with_the_endpoint_env(tmp_path):
    config = AgentConfig(
        import_path="miles_harbor.vanillux:VanilluxAgent",
        kwargs=VanilluxAgent.rollout_kwargs({"command_timeout": 60}, step_limit=64, max_tokens=8192),
        **VanilluxAgent.ENDPOINT.agent_config("http://h:1/v1/", "EMPTY", "model"),
    )
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert type(agent) is VanilluxAgent and agent.model_name == "model"
    assert (agent.max_steps, agent.max_tokens, agent.command_timeout) == (64, 8192, 60)
    # the endpoint is this process's, not exported into the sandbox's commands (Harbor's exec env is extra_env)
    assert agent.extra_env == {} and agent._get_env("OPENAI_BASE_URL") == "http://h:1/v1"
    assert not VanilluxAgent.ENDPOINT.in_sandbox
