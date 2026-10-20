"""TMax's harness (AllenAI, arXiv 2606.23321): the "vanillux" bash agent its RL ran in, open-instruct's
swerl_vanillux_sandbox (hamishivi/tmax @6d3d606c, training/open-instruct/open_instruct/environments/), as a Harbor
agent that runs beside the sandbox and works in it through Harbor's environment.exec:

    agents: [{import_path: "miles_harbor.vanillux:VanilluxAgent", model_name: ..., kwargs: {...}}]

What the model sees is TMax's training environment's, verbatim:
- the system prompt and the first user message: its vanillux_prompts.yaml (mini-swe-agent v2.2.x's, for a native
  tool call), the task's instruction in ``instance_template``;
- one tool, ``bash``, its schema the environment's (_BASH_TOOL);
- a persistent shell: each command runs in a fresh ``bash`` through a wrapper that sources the last command's
  exported environment and cds to its directory, and saves both after it (cwd /app to begin with, linked to
  /workspace when the image has no /app); ``timeout --signal=TERM --kill-after=10 <command_timeout>`` around it, a
  timeout told as ``Command timed out after <n>s.``;
- an observation: the output (head and tail 5,000 characters with mini-swe-agent's too-long hint past 10,000),
  ``(no output)`` for none, then ``(exit_code=N)``; a command with a NUL byte, which no process takes as an argument
  (TMax's docker exec raised on it), gets what TMax's rollout answers a step that raised, ``Step 'bash' failed:
  <error>. Args: <arguments>``, and the episode goes on;
- every ``bash`` call of a reply runs, in order, each one step, as TMax's rollout runs a reply's calls (its
  process_request); a reply without one (no call, calls of another tool, arguments that are no JSON object) ends
  the episode, as TMax trains (format-error feedback off). ``format_error_feedback`` instead answers it with the
  format-error message, as TMax's evaluation agent (Vanillux2Agent) does, up to ``max_format_errors`` in a row: a
  user message for a reply without a call, a tool message for another tool's call (the reply's ``bash`` calls run
  after it); each such answer counts as a step;
- ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` in a command's output submits: the agent stops and Harbor's
  verifier runs the task's tests in the same sandbox (TMax runs them then too, never before: the tests are not in
  the sandbox while the agent works). TMax scores an episode that never submitted 0; Harbor grades every run, so a
  rollout that wants TMax's rule asks the trial server for it (reward_requires_submit);
- at most ``max_steps`` steps (TMax: 64 tool calls; a reply's calls past the limit do not run), each reply at most
  ``max_tokens`` (16,384), the replies and tool output after the first prompt at most ``max_response_tokens``
  (65,536) and the whole conversation at most ``max_context_tokens`` (67,584: a 2,048-token prompt and the 65,536):
  each call's max_tokens is what the budgets have left, the conversation's tokens counted from the server's usage
  and the new observations' estimated (UTF-8 bytes / OBS_BYTES_PER_TOKEN, an overestimate), and the episode ends
  once less than MIN_REPLY_TOKENS is left, or the server refuses a request as too long.

Where it differs from TMax's training environment: the sandbox's output is one stream, stdout and stderr
interleaved as written (Harbor's exec, as docker compose exec gives it), where TMax read them apart and put stderr
after stdout; the timeout notice follows the output instead of leading stderr.

The model is called at the server Harbor's agent config names (its env: OPENAI_BASE_URL, OPENAI_API_KEY; the
model_name as the server knows it), through the OpenAI client, from this process: the env is not exported into the
sandbox's commands, and the sandbox gets no way to the server (the trial server's agent_network). Each reply goes back
into the conversation as the server returned it (content, reasoning_content, tool_calls), so a session server's
token-in/token-out history continues. Sampling the call leaves unset (``temperature`` / ``top_p`` None) is the
server's: a training rollout's session server samples as the run does. Its run is in ``<logs>/vanillux.json`` (how it
ended, its steps, tokens) and ``<logs>/trajectory.json`` (the messages), which ``outcome`` reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import openai
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from miles_harbor.harness_api import Endpoint, Outcome

logger = logging.getLogger(__name__)

VERSION = "tmax-6d3d606c+miles1"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# vanillux_prompts.yaml (open_instruct/environments), verbatim; the training system prompt
# (scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt) is its system_template
SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

Your response must include a THOUGHT section before your action where you
explain your reasoning. After the THOUGHT, you must call the `bash` tool
with EXACTLY ONE bash command (multiple commands chained with `&&` or `||`
count as a single action).

Failure to follow these rules — calling no tool, calling a tool other than
`bash`, or omitting the THOUGHT — will cause your response to be rejected.
"""
INSTANCE_TEMPLATE = """\
Please solve this task:

{{task}}

You can execute bash commands and edit files (with `sed`, `cat > file << 'EOF'`,
etc.) to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your
changes and any possible problems.

1. Analyze the codebase / environment by finding and reading relevant files.
2. If applicable, create a script to reproduce the issue or expected behaviour.
3. Implement the change(s) by editing the source code or environment state.
4. Verify your fix works by running your script (or relevant test) again.
5. Test edge cases to ensure your fix is robust.
6. Submit your changes and finish your work by issuing the following command:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine it with any other command. After this command, you cannot
   continue working on this task.

## Important Rules

1. Every response must contain exactly one tool call to `bash`.
2. Directory and environment-variable changes ARE persistent across calls
   in this harness — you can `cd` and `export` and subsequent commands will
   see the change. (This differs from upstream mini-swe-agent's subshell
   model; treat the shell as a long-running login shell.)
3. Long-running commands: wrap with `timeout`, e.g. `timeout 30 <command>`.
4. Interactive commands are not possible. Use `yes`/`no` piping or
   non-interactive flags as appropriate.
5. Output may be truncated. Use `head`, `tail`, `grep`, `sed -n 'A,Bp'`,
   etc. to filter large outputs.

## Useful command examples

### Create a new file:
`cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF`

### Edit files with sed:
`sed -i 's/old_string/new_string/g' filename.py`        # all occurrences
`sed -i '1s/old_string/new_string/' filename.py`         # first on line 1
`sed -i '1,10s/old_string/new_string/g' filename.py`     # lines 1-10

### View file content:
`nl -ba filename.py | sed -n '10,20p'`
"""
OBS_MAX_CHARS, OBS_HEAD_CHARS, OBS_TAIL_CHARS = 10000, 5000, 5000
OBS_TOO_LONG_HINT = """\
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a
smaller number of lines selectively. If you're using grep or find and it
produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you
can redirect output to a file and then search in that file.
"""
FORMAT_ERROR_TEMPLATE = """\
Format error: {{error}}

Please always provide EXACTLY ONE call to the `bash` tool. If you want to
end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
via the `bash` tool, with no other content in the command.
"""
NO_TOOL_CALL_ERROR = "Your last response did not include a valid `bash` tool call."

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell. "
            "Working directory and environment variables are preserved between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

# swerl_vanillux_sandbox's persistent shell
WRAPPER_PATH = "/tmp/.swerl_vanillux_bash_wrapper.sh"
CWD_PATH = "/tmp/.swerl_vanillux_cwd"
ENV_PATH = "/tmp/.swerl_vanillux_env"
WRAPPER = f"""#!/bin/bash
set -a
source {shlex.quote(ENV_PATH)} 2>/dev/null || true
set +a
_cwd="$(cat {shlex.quote(CWD_PATH)} 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > {shlex.quote(ENV_PATH)}
pwd > {shlex.quote(CWD_PATH)}
exit $_exit_code
"""
# reset: the dirs, /app (the working dir's link when the image has none), the shell's first cwd and env, the wrapper
PREPARE = (
    "mkdir -p /workspace /output /logs/verifier /root && "
    "cd /workspace && "
    '{ [ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; }; } && '
    f"printf '%s\\n' /app > {shlex.quote(CWD_PATH)} && "
    f": > {shlex.quote(ENV_PATH)} && "
    f"cat > {shlex.quote(WRAPPER_PATH)} <<'VANILLUX_WRAPPER_EOF'\n{WRAPPER}VANILLUX_WRAPPER_EOF\n"
    f"chmod +x {shlex.quote(WRAPPER_PATH)}"
)

# the context budget: an observation's tokens are estimated from its UTF-8 bytes (Qwen's tokenizer gives ~3-4 bytes a
# token for English and code; 2.5 overestimates), the request's template tokens around it, the least reply worth asking
OBS_BYTES_PER_TOKEN = 2.5
TEMPLATE_TOKENS = 32
MIN_REPLY_TOKENS = 256
MODEL_CALL_ATTEMPTS = 4
STATUS_FILE = "vanillux.json"
TRAJECTORY_FILE = "trajectory.json"


def render_instance(task: str) -> str:
    return INSTANCE_TEMPLATE.replace("{{task}}", task)


def format_error_message(error: str) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


def truncate_observation(output: str) -> str:
    """mini-swe-agent's head/tail cut of a long output, as swerl_vanillux_sandbox applies it."""
    if len(output) <= OBS_MAX_CHARS:
        return output
    elided = len(output) - OBS_HEAD_CHARS - OBS_TAIL_CHARS
    return (
        f"{OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({OBS_HEAD_CHARS} chars) ----\n"
        f"{output[:OBS_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({OBS_TAIL_CHARS} chars) ----\n"
        f"{output[-OBS_TAIL_CHARS:]}"
    )


def observation(output: str, exit_code: int) -> str:
    return f"{truncate_observation(output) if output else '(no output)'}\n\n(exit_code={exit_code})"


def step_failed(error: str, args: Mapping[str, Any]) -> str:
    """TMax's rollout's answer to a step that raised (its process_request), the episode going on."""
    return f"Step 'bash' failed: {error}. Args: {dict(args)}"


def bash_command(command: str, timeout_s: int) -> str:
    """The sandbox command of one step: the wrapper under ``timeout``, as swerl_vanillux_sandbox runs it."""
    inner = f"bash {shlex.quote(WRAPPER_PATH)} {shlex.quote(command)}"
    return f"timeout --signal=TERM --kill-after=10 {timeout_s} bash -c {shlex.quote(inner)}"


def tool_calls(message: Mapping[str, Any]) -> tuple[list[tuple[str | None, dict]], list[tuple[str | None, str]]]:
    """A reply's steps: its ``bash`` calls in order, (call id, arguments) each, and its calls of other tools, (call id,
    name) each. A call whose arguments are no JSON object is neither (TMax's parser yields no call for it)."""
    bash, other = [], []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                continue
        else:
            args = raw or {}
        if not isinstance(args, dict):
            continue
        if function.get("name") == "bash":
            bash.append((call.get("id"), args))
        else:
            other.append((call.get("id"), str(function.get("name"))))
    return bash, other


def estimated_tokens(text: str) -> int:
    return int(len(text.encode("utf-8")) / OBS_BYTES_PER_TOKEN) + 1


def _openai_env(base_url: str, api_key: str) -> dict[str, str]:
    return {"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": api_key}


class ModelCallError(RuntimeError):
    """The model server failed a call MODEL_CALL_ATTEMPTS times: the episode has no outcome to train on."""


class VanilluxAgent(BaseAgent):
    # the OpenAI client at the server's /v1, the model by the name the server serves, called from this process
    ENDPOINT = Endpoint(model_name=lambda model: model, env=_openai_env, in_sandbox=False)

    @staticmethod
    def name() -> str:
        return "vanillux"

    def version(self) -> str:
        return VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = 64,
        max_tokens: int = 16384,
        max_response_tokens: int = 65536,
        max_context_tokens: int = 67584,
        temperature: float | None = None,
        top_p: float | None = None,
        command_timeout: int = 120,
        format_error_feedback: bool = False,
        max_format_errors: int = 64,
        request_timeout_s: float = 3600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = int(max_steps)
        self.max_tokens = int(max_tokens)
        self.max_response_tokens = int(max_response_tokens)
        self.max_context_tokens = int(max_context_tokens)
        self.temperature = temperature
        self.top_p = top_p
        self.command_timeout = int(command_timeout)
        self.format_error_feedback = bool(format_error_feedback)
        self.max_format_errors = int(max_format_errors)
        self.request_timeout_s = float(request_timeout_s)

    @property
    def extra_env(self) -> dict[str, str]:
        """None: Harbor exports an agent's env into every command of its run, and this agent's (the server's URL and
        key) is its own process's; its run reads it from there (_get_env)."""
        return {}

    @staticmethod
    def rollout_kwargs(kwargs: Mapping[str, Any], *, step_limit: int | None, max_tokens: int | None) -> dict:
        """The agent's kwargs with a rollout's limits: its steps (tool calls) and each reply's tokens."""
        kwargs = dict(kwargs)
        if step_limit is not None:
            kwargs["max_steps"] = step_limit
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    @staticmethod
    def outcome(agent_dir: Path) -> Outcome:
        """From the status the run writes as it ends (STATUS_FILE)."""
        try:
            status = json.loads((agent_dir / STATUS_FILE).read_text())
        except (OSError, ValueError):
            return Outcome(exit_status=None, submitted=False, steps=None)
        steps = status.get("steps")
        return Outcome(
            exit_status=status.get("exit_status"),
            submitted=status.get("exit_status") == "submitted",
            steps=steps if isinstance(steps, int) else None,
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(command=PREPARE, timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"vanillux: the shell's setup failed (rc={result.return_code}): {result.stdout}")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": render_instance(instruction.strip())},
        ]
        status: dict[str, Any] = {
            "exit_status": "step_limit",
            "steps": 0,
            "model_calls": 0,
            "format_errors": 0,
            "timeouts": 0,
        }
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        # the conversation's tokens after the last reply (the server's count), what was added since (estimated), and
        # where the response budget ends (the first prompt's tokens plus max_response_tokens)
        context_tokens, pending, limit = None, 0, self.max_context_tokens
        format_errors = 0
        client = openai.AsyncOpenAI(
            base_url=self._get_env("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            api_key=self._get_env("OPENAI_API_KEY") or "EMPTY",
            timeout=self.request_timeout_s,
            max_retries=0,
        )
        started = time.monotonic()
        try:
            while status["steps"] < self.max_steps:
                reply_tokens = self.max_tokens
                if context_tokens is not None:
                    reply_tokens = min(reply_tokens, limit - context_tokens - pending)
                if reply_tokens < MIN_REPLY_TOKENS:
                    status["exit_status"] = "context"
                    break
                try:
                    response = await self._query(client, messages, reply_tokens)
                except openai.BadRequestError as e:  # the server refuses the request: the context is full
                    status.update(exit_status="context", error=str(e)[-500:])
                    break
                status["model_calls"] += 1
                if response.usage is not None:
                    usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                    usage["completion_tokens"] += response.usage.completion_tokens or 0
                    if context_tokens is None:
                        limit = min(limit, (response.usage.prompt_tokens or 0) + self.max_response_tokens)
                    context_tokens = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
                    pending = 0
                message = response.choices[0].message.model_dump(exclude_none=True)
                messages.append(message)
                calls, others = tool_calls(message)
                added: list[str] = []
                if others or not calls:
                    status["format_errors"] += 1
                if not calls:
                    format_errors += 1
                    if not self.format_error_feedback or format_errors >= self.max_format_errors:
                        status["exit_status"] = "no_tool_call"
                        break
                else:
                    format_errors = 0
                # TMax's feedback (evaluation): one answer a reply, a step, before the reply's bash calls run
                if self.format_error_feedback and (others or not calls):
                    status["steps"] += 1
                    if others:
                        call_id, name = others[0]
                        error = f"Unknown tool '{name}'. The only available tool is `bash`."
                        added.append(self._answer(messages, call_id, format_error_message(error)))
                    else:
                        added.append(self._answer(messages, None, format_error_message(NO_TOOL_CALL_ERROR)))
                for call_id, args in calls:
                    if status["steps"] >= self.max_steps:
                        break
                    status["steps"] += 1
                    command = args.get("command", "")
                    if not isinstance(command, str) or not command:
                        error = format_error_message("'command' parameter is required.")
                        added.append(self._answer(messages, call_id, error))
                        continue
                    if "\x00" in command:  # no process takes it as an argument: TMax's exec raised on it
                        added.append(self._answer(messages, call_id, step_failed("embedded null byte", args)))
                        continue
                    output, exit_code = await self._bash(environment, command)
                    if exit_code == 124:
                        status["timeouts"] += 1
                    if SUBMIT_MARKER in output:
                        status["exit_status"] = "submitted"
                        break
                    added.append(self._answer(messages, call_id, observation(output, exit_code)))
                if status["exit_status"] == "submitted":
                    break
                pending += sum(estimated_tokens(text) + TEMPLATE_TOKENS for text in added)
        except ModelCallError as e:
            status.update(exit_status="model_error", error=str(e)[-500:])
            raise
        except asyncio.CancelledError:  # Harbor's timeout of the agent, or the trial cancelled
            status["exit_status"] = "cancelled"
            raise
        except Exception as e:  # the sandbox failed it: no ending of the agent's own
            status.update(exit_status="error", error=f"{type(e).__name__}: {e}"[-500:])
            raise
        finally:
            await client.close()
            status["agent_s"] = round(time.monotonic() - started, 1)
            status.update(usage)
            context.n_input_tokens = usage["prompt_tokens"]
            context.n_output_tokens = usage["completion_tokens"]
            context.metadata = {**(context.metadata or {}), "vanillux": status}
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / TRAJECTORY_FILE).write_text(json.dumps(messages, indent=1, default=str) + "\n")
            (self.logs_dir / STATUS_FILE).write_text(json.dumps(status) + "\n")

    async def _query(self, client: openai.AsyncOpenAI, messages: list[dict[str, Any]], max_tokens: int) -> Any:
        sampling = {k: v for k, v in (("temperature", self.temperature), ("top_p", self.top_p)) if v is not None}
        for attempt in range(1, MODEL_CALL_ATTEMPTS + 1):
            try:
                return await client.chat.completions.create(
                    model=self.model_name, messages=messages, tools=[BASH_TOOL], max_tokens=max_tokens, **sampling
                )
            except openai.BadRequestError:
                raise
            except (openai.APIConnectionError, openai.APIStatusError) as e:
                if attempt == MODEL_CALL_ATTEMPTS:
                    raise ModelCallError(f"{type(e).__name__}: {e}") from e
                logger.warning("vanillux: model call %d/%d failed: %s", attempt, MODEL_CALL_ATTEMPTS, e)
                await asyncio.sleep(2.0 * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    async def _bash(self, environment: BaseEnvironment, command: str) -> tuple[str, int]:
        """One step's command in the sandbox: its output (one stream) and exit code; a timeout as TMax tells it."""
        notice = f"Command timed out after {self.command_timeout}s.\n"
        try:
            # Harbor's own timeout a margin past timeout's TERM and KILL, in case the sandbox stops answering
            result = await environment.exec(
                command=bash_command(command, self.command_timeout), timeout_sec=self.command_timeout + 30
            )
        except RuntimeError as e:
            if "timed out" not in str(e):
                raise
            return notice, 124
        output = result.stdout or ""
        if result.return_code == 124:
            output = f"{output}\n{notice}" if output else notice
        return output, result.return_code

    @staticmethod
    def _answer(messages: list[dict[str, Any]], call_id: str | None, content: str) -> str:
        """The step's observation: a tool message for the reply's call, else a user message."""
        if call_id:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
        else:
            messages.append({"role": "user", "content": content})
        return content
