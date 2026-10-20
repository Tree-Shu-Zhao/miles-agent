"""The recipes' prompts live in prompts/<env>/<name>.py; this renders one for a model.

A prompt module holds words only:

    SYSTEM             the system message. ``{example_call}`` marks where one example
                       call of the row's action tool goes, in the model's own markup;
                       ``{example_call:<label>}`` marks one of several.
    EXAMPLE_ARGUMENTS  that example call's arguments (required with ``{example_call}``).
    EXAMPLE_CALLS      ``{<label>: arguments}`` for the labelled placeholders.
    TURNS              optional demonstration turns after the system message: user,
                       assistant (``content`` plus ``tool_calls`` as
                       ``[{"name", "arguments"}]``) and tool (``content``) messages.

The recipe's prepare_data.py owns the rows, the tool schema and the split layout; it
calls load() for the module and render() for the messages. The rendered messages are
role/content only, because the agents forward nothing else (agent_utils.chat_dialect):
every call and every tool result is written as the policy model's chat template would
write it, and render() re-renders the whole prompt both ways to prove the bytes equal,
so a prompt file never spells out a markup and one file serves every model family.

Selection: ``--prompt <name>`` on prepare_data.py (``PROMPT=<name>`` on the training entry); each
recipe names its default. Runs under the bare container python3 like prepare_data.py.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .tool_call_markup import RenderChat, chat_tool, render_tool_call

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
ROLES = ("user", "assistant", "tool")
_SENTINEL = "\x00prompt-turn\x00"
_PLACEHOLDER = re.compile(r"\{example_call(?::([A-Za-z0-9_]+))?\}")


@dataclass(frozen=True)
class Prompt:
    env: str
    name: str
    system: str
    example_arguments: Mapping | None
    example_calls: Mapping[str, Mapping]
    turns: tuple[Mapping, ...]
    path: Path


def available(env: str, prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    """The prompt names of an env (its ``prompts/<env>/*.py``, underscore-prefixed files aside)."""
    env_dir = prompts_dir / env
    if not env_dir.is_dir():
        return []
    return sorted(p.stem for p in env_dir.glob("*.py") if not p.stem.startswith("_"))


def load(env: str, name: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    """``prompts/<env>/<name>.py`` as a Prompt, its attributes checked."""
    path = prompts_dir / env / f"{name}.py"
    if not path.is_file():
        have = ", ".join(available(env, prompts_dir)) or "none"
        raise FileNotFoundError(f"no prompt {name!r} for {env}: {path} does not exist (available: {have})")
    spec = importlib.util.spec_from_file_location(f"prompts.{env}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    system = getattr(module, "SYSTEM", None)
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"{path}: SYSTEM must be a non-empty string")
    example = getattr(module, "EXAMPLE_ARGUMENTS", None)
    labels = {m.group(1) for m in _PLACEHOLDER.finditer(system)}
    if (None in labels) != (example is not None):
        raise ValueError(f"{path}: SYSTEM's {{example_call}} and EXAMPLE_ARGUMENTS go together")
    if example is not None and not isinstance(example, Mapping):
        raise ValueError(f"{path}: EXAMPLE_ARGUMENTS must be a mapping of the call's arguments")
    calls = getattr(module, "EXAMPLE_CALLS", None) or {}
    if not isinstance(calls, Mapping) or not all(isinstance(v, Mapping) for v in calls.values()):
        raise ValueError(f"{path}: EXAMPLE_CALLS must map labels to argument mappings")
    if set(calls) != labels - {None}:
        raise ValueError(
            f"{path}: SYSTEM's labelled placeholders {sorted(labels - {None})} "
            f"and EXAMPLE_CALLS {sorted(calls)} differ"
        )
    turns = tuple(getattr(module, "TURNS", None) or ())
    for i, turn in enumerate(turns):
        _check_turn(turn, f"{path}: TURNS[{i}]")
    return Prompt(
        env=env, name=name, system=system, example_arguments=example, example_calls=dict(calls), turns=turns, path=path
    )


def _check_turn(turn, where: str) -> None:
    if not isinstance(turn, Mapping) or turn.get("role") not in ROLES:
        raise ValueError(f"{where}: a turn is a mapping with role in {ROLES}")
    if not isinstance(turn.get("content", ""), str):
        raise ValueError(f"{where}: content must be a string")
    calls = turn.get("tool_calls")
    if calls is not None and turn["role"] != "assistant":
        raise ValueError(f"{where}: only assistant turns carry tool_calls")
    if turn["role"] == "assistant" and not (turn.get("content") or calls):
        raise ValueError(f"{where}: an assistant turn needs content or tool_calls")
    for call in calls or ():
        if (
            not isinstance(call, Mapping)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), Mapping)
        ):
            raise ValueError(f"{where}: a tool call is {{'name': str, 'arguments': mapping}}")
    if turn["role"] != "assistant" and not turn.get("content"):
        raise ValueError(f"{where}: a {turn['role']} turn needs content")


def render(prompt: Prompt, render_chat: RenderChat, tools: Sequence[Mapping]) -> list[dict]:
    """The prompt's messages for one model, role/content only, in the chat template's own markup.

    ``tools`` are the row's tools in Responses form (``metadata["tools"]``); the example
    call and the demonstration calls must name one of them.
    """
    chat_tools = [chat_tool(t) for t in tools]
    by_name = {t["function"]["name"]: t for t in chat_tools}

    def check_named(name: str) -> None:
        if name not in by_name:
            raise ValueError(f"{prompt.path}: the prompt calls {name!r} but the row's tools are {sorted(by_name)}")

    system = prompt.system
    if prompt.example_arguments is not None or prompt.example_calls:
        if len(by_name) != 1:
            raise ValueError(f"{prompt.path}: {{example_call}} needs exactly one tool, the row has {sorted(by_name)}")
        (tool,) = chat_tools
        if prompt.example_arguments is not None:
            system = system.replace("{example_call}", render_tool_call(render_chat, tool, prompt.example_arguments))
        for label, arguments in prompt.example_calls.items():
            system = system.replace(f"{{example_call:{label}}}", render_tool_call(render_chat, tool, arguments))
    messages = [{"role": "system", "content": system}]
    if not prompt.turns:
        return messages

    structured: list[dict] = []
    flat: list[dict] = []
    for turn in prompt.turns:
        role, content = turn["role"], turn.get("content") or ""
        if role == "assistant" and turn.get("tool_calls"):
            calls = [dict(c) for c in turn["tool_calls"]]
            for c in calls:
                check_named(c["name"])
            structured.append({"role": "assistant", "content": content, "tool_calls": _chat_calls(calls)})
            flat.append({"role": "assistant", "content": _assistant_text(render_chat, chat_tools, content, calls)})
        elif role == "tool":
            structured.append({"role": "tool", "content": content})
            flat.append({"role": "user", "content": _tool_result_text(render_chat, chat_tools, content)})
        else:
            structured.append({"role": role, "content": content})
            flat.append({"role": role, "content": content})

    # The proof: the flat turns render to the bytes the template writes for the structured
    # ones, in the prompt's own context and followed by the episode's first user turn.
    tail = [{"role": "user", "content": "next"}]
    if render_chat(messages + structured + tail, chat_tools) != render_chat(messages + flat + tail, chat_tools):
        raise ValueError(
            f"{prompt.path}: this chat template renders the demonstration turns differently from their text "
            "(e.g. tool results outside a user turn); the prompt's TURNS cannot be flattened for it"
        )
    return messages + flat


def _chat_calls(calls: Sequence[Mapping]) -> list[dict]:
    return [{"type": "function", "function": {"name": c["name"], "arguments": dict(c["arguments"])}} for c in calls]


def _assistant_text(render_chat: RenderChat, chat_tools: list[dict], content: str, calls: Sequence[Mapping]) -> str:
    """``content`` followed by its calls, separated and written as the template writes an
    assistant turn that carries tool_calls (Qwen3 puts one newline between content and
    the first call, Qwen3.5 two; render_tool_call strips that, this keeps it).

    Renders one conversation twice, its assistant turn's content a sentinel, without and
    with the calls: the calls add their block between the content and the turn's end.
    """
    head = [{"role": "user", "content": "start"}]
    tail = [{"role": "user", "content": "next"}]
    plain = render_chat(head + [{"role": "assistant", "content": _SENTINEL}] + tail, chat_tools)
    called = render_chat(
        head + [{"role": "assistant", "content": _SENTINEL, "tool_calls": _chat_calls(calls)}] + tail, chat_tools
    )
    plain, called = plain.split(_SENTINEL, 1)[1], called.split(_SENTINEL, 1)[1]
    if not called.endswith(plain) or len(called) == len(plain):
        raise ValueError("the chat template does not render the tool calls inside the assistant turn")
    block = called[: len(called) - len(plain)]
    return content + block if content else block.lstrip()


def _tool_result_text(render_chat: RenderChat, chat_tools: list[dict], content: str) -> str:
    """``content`` wrapped as the template wraps a tool result inside a user turn.

    Renders one conversation twice, its third turn a tool result and a user message
    carrying a sentinel: the user render splits at the sentinel into the bytes
    around any user content, and the tool render minus those bytes is the wrapped
    result.
    """
    head = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"}]
    tail = [{"role": "user", "content": "next"}]
    as_tool = render_chat(head + [{"role": "tool", "content": content}] + tail, chat_tools)
    as_user = render_chat(head + [{"role": "user", "content": _SENTINEL}] + tail, chat_tools)
    before, after = as_user.split(_SENTINEL, 1)
    if not (as_tool.startswith(before) and as_tool.endswith(after)) or len(as_tool) < len(before) + len(after):
        raise ValueError("the chat template does not render tool results inside a user turn")
    return as_tool[len(before) : len(as_tool) - len(after)]
