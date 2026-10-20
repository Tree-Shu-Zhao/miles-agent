"""miles_agent/prompts.py: prompts/<env>/<name>.py loaded and rendered in the model's markup.

Rendered through miles' fixed Qwen templates (tests/conftest.py template_render_chat):
the demonstration turns of a prompt must come out byte-identical to what the template
writes for structured tool calls and tool results.
"""

from pathlib import Path

import pytest

from miles_agent import prompts

REPO = Path(__file__).resolve().parent.parent
ACT = {
    "type": "function",
    "name": "act",
    "description": "d",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    "strict": True,
}
HERMES = '<tool_call>\n{"name": "act", "arguments": {"command": "go to desk 1"}}\n</tool_call>'
XML = "<tool_call>\n<function=act>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>"


def _write(tmp_path: Path, env: str, name: str, body: str) -> Path:
    (tmp_path / env).mkdir(exist_ok=True)
    (tmp_path / env / f"{name}.py").write_text(body)
    return tmp_path


ONESHOT = """
SYSTEM = "Reason briefly, then call act. Example:\\n{example_call}"
EXAMPLE_ARGUMENTS = {"command": "go to desk 1"}
TURNS = [
    {"role": "user", "content": "You are in a room. Your task is to: put a mug on the desk."},
    {"role": "assistant", "content": "I have not seen a mug yet, so I will check the countertop.",
     "tool_calls": [{"name": "act", "arguments": {"command": "go to countertop 1"}}]},
    {"role": "tool", "content": "You arrive at countertop 1. You see a mug 1."},
    {"role": "assistant", "tool_calls": [{"name": "act", "arguments": {"command": "take mug 1 from countertop 1"}}]},
    {"role": "tool", "content": "[example ended; a new task follows]"},
]
"""


def test_repo_prompts_load_and_name_no_markup():
    assert prompts.available("alfworld") == ["default", "r2s_oneshot"]
    assert prompts.available("scienceworld") == ["default", "r2s_oneshot"]
    assert prompts.available("webshop") == ["none", "r2s_oneshot", "think"]
    assert prompts.available("no_such_env") == []
    for env in ("alfworld", "scienceworld", "webshop"):
        for name in prompts.available(env):
            p = prompts.load(env, name)
            assert "<tool_call>" not in p.system and "<function=" not in p.system
            for turn in p.turns:
                assert "<tool_call>" not in (turn.get("content") or "")
    for env in ("alfworld", "webshop"):
        for name in ("default", "think", "none"):
            if name in prompts.available(env):
                p = prompts.load(env, name)
                assert p.example_arguments == {"command": p.example_arguments["command"]}
                assert p.turns == () and p.example_calls == {}
    # the scienceworld system message shows commands in words: no example call, no turns
    sw = prompts.load("scienceworld", "default")
    assert sw.example_arguments is None and sw.turns == () and sw.example_calls == {}
    assert sw.system == prompts.load("scienceworld", "r2s_oneshot").system


@pytest.mark.parametrize(
    ("template", "call_marker"), [("qwen3_fixed.jinja", '"arguments"'), ("qwen3.5_fixed.jinja", "<function=act>")]
)
def test_r2s_oneshot_prompts_render_the_papers_demonstrations(template_render_chat, template, call_marker):
    """The R2S paper's one-shot prompts (the campaign branch's) as single
    prompt files: the paper wrote one file per family; rendered under the Qwen3 and Qwen3.5 templates these
    equal its hermes and qwen3_coder rows (checked against the campaign's eval274 / canonical rows, 2026-10-17)."""
    render_chat = template_render_chat(template)
    alf = prompts.render(prompts.load("alfworld", "r2s_oneshot"), render_chat, [ACT])
    assert [m["role"] for m in alf] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "{example_call" not in alf[0]["content"] and "<tool_call>" not in alf[0]["content"]
    for m in (alf[2], alf[4]):
        assert m["content"].count("<tool_call>") == 1 and call_marker in m["content"]
        assert not m["content"].startswith("<tool_call>")  # deliberation first
    assert alf[5]["content"] == "<tool_response>\n[example ended; a new task follows]\n</tool_response>"
    sw = prompts.render(prompts.load("scienceworld", "r2s_oneshot"), render_chat, [ACT])
    assert [m["role"] for m in sw] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "<tool_call>" not in sw[0]["content"] and sw[0]["content"].endswith("more than once per turn.")
    assert sw[1]["content"].startswith("Your ScienceWorld task is: Your task is to focus on the thermometer")
    for m, command in ((sw[2], "open door to kitchen"), (sw[4], "go to kitchen")):
        assert m["content"].count("<tool_call>") == 1 and call_marker in m["content"] and command in m["content"]
        assert not m["content"].startswith("<tool_call>")  # deliberation first
    assert sw[3]["content"].startswith("<tool_response>\nThe door is now open.\nAvailable action commands: [")
    assert sw[5]["content"] == alf[5]["content"]
    ws = prompts.render(prompts.load("webshop", "r2s_oneshot"), render_chat, [ACT])
    assert [m["role"] for m in ws] == ["system"]
    system = ws[0]["content"]
    assert system.count("<tool_call>") == 4 and system.count(call_marker) == 4 and "{example_call" not in system
    assert system.count("Reply: ") == 4 and system.endswith("End of example. Now here is your task.")


def test_load_refuses_a_missing_or_malformed_prompt(tmp_path):
    with pytest.raises(FileNotFoundError, match="available: none"):
        prompts.load("alfworld", "nope", prompts_dir=tmp_path)
    _write(tmp_path, "alfworld", "a", 'SYSTEM = "x {example_call}"\n')
    with pytest.raises(ValueError, match="go together"):
        prompts.load("alfworld", "a", prompts_dir=tmp_path)
    _write(tmp_path, "alfworld", "b", 'SYSTEM = "x"\nTURNS = [{"role": "narrator", "content": "y"}]\n')
    with pytest.raises(ValueError, match="role in"):
        prompts.load("alfworld", "b", prompts_dir=tmp_path)
    _write(tmp_path, "alfworld", "c", 'SYSTEM = "x"\nTURNS = [{"role": "user", "content": "y", "tool_calls": []}]\n')
    with pytest.raises(ValueError, match="only assistant turns"):
        prompts.load("alfworld", "c", prompts_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="available: a, b, c"):
        prompts.load("alfworld", "d", prompts_dir=tmp_path)


@pytest.mark.parametrize(
    ("template", "markup"),
    [("qwen3_fixed.jinja", HERMES), ("qwen3.5_fixed.jinja", XML), ("qwen3.6_fixed.jinja", XML)],
)
def test_example_call_is_the_templates_markup(template_render_chat, template, markup):
    p = prompts.load("alfworld", "default")
    messages = prompts.render(p, template_render_chat(template), [ACT])
    assert [m["role"] for m in messages] == ["system"]
    assert markup in messages[0]["content"] and "{example_call}" not in messages[0]["content"]


@pytest.mark.parametrize("template", ["qwen3_fixed.jinja", "qwen3.5_fixed.jinja", "qwen3.6_fixed.jinja"])
def test_demonstration_turns_render_as_the_template_would(template_render_chat, template, tmp_path):
    render_chat = template_render_chat(template)
    p = prompts.load("alfworld", "oneshot", prompts_dir=_write(tmp_path, "alfworld", "oneshot", ONESHOT))
    messages = prompts.render(p, render_chat, [ACT])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert set().union(*(m.keys() for m in messages)) == {"role", "content"}
    # the structured turns, as a session would hold them
    structured = [messages[0]] + [
        dict(t, tool_calls=[{"type": "function", "function": c} for c in t["tool_calls"]])
        if t.get("tool_calls")
        else dict(t)
        for t in p.turns
    ]
    for t in structured:
        t.setdefault("content", "")
    tail = [{"role": "user", "content": "the first real observation"}]
    chat_tools = [prompts.chat_tool(ACT)]
    assert render_chat(messages + tail, chat_tools) == render_chat(structured + tail, chat_tools)
    # the first assistant demonstration: deliberation, then the call in the model's markup
    first = messages[2]["content"]
    assert first.startswith("I have not seen a mug yet") and first.count("<tool_call>") == 1
    assert ("go to countertop 1" in first) and ("<function=act>" in first) == (template != "qwen3_fixed.jinja")
    # tool results ride in user turns the way the template wraps them
    assert messages[3]["content"] == "<tool_response>\nYou arrive at countertop 1. You see a mug 1.\n</tool_response>"


@pytest.mark.parametrize(
    ("template", "markup"),
    [("qwen3_fixed.jinja", HERMES), ("qwen3.5_fixed.jinja", XML)],
)
def test_labelled_example_calls_in_the_system_text(template_render_chat, template, markup, tmp_path):
    body = (
        'SYSTEM = "First {example_call:a}, then {example_call:b}; e.g. {example_call}."\n'
        'EXAMPLE_ARGUMENTS = {"command": "go to desk 1"}\n'
        'EXAMPLE_CALLS = {"a": {"command": "look"}, "b": {"command": "inventory"}}\n'
    )
    p = prompts.load("alfworld", "multi", prompts_dir=_write(tmp_path, "alfworld", "multi", body))
    system = prompts.render(p, template_render_chat(template), [ACT])[0]["content"]
    assert system.count("<tool_call>") == 3 and "{example_call" not in system
    assert system.endswith(f"e.g. {markup}.") and system.index("look") < system.index("inventory")
    _write(tmp_path, "alfworld", "orphan", 'SYSTEM = "x {example_call:a}"\nEXAMPLE_CALLS = {"a": {}, "b": {}}\n')
    with pytest.raises(ValueError, match="differ"):
        prompts.load("alfworld", "orphan", prompts_dir=tmp_path)


def test_render_refuses_a_call_to_an_unknown_tool(template_render_chat, tmp_path):
    body = ONESHOT.replace(
        '"name": "act", "arguments": {"command": "go to countertop 1"}', '"name": "fly", "arguments": {}'
    )
    p = prompts.load("alfworld", "bad", prompts_dir=_write(tmp_path, "alfworld", "bad", body))
    with pytest.raises(ValueError, match="calls 'fly'"):
        prompts.render(p, template_render_chat("qwen3_fixed.jinja"), [ACT])


def test_render_refuses_a_template_that_cannot_flatten(tmp_path):
    def tool_role_template(messages, tools):
        return "".join(
            f"<{m['role']}>{m.get('content', '')}{'[call]' if m.get('tool_calls') else ''}</{m['role']}>"
            for m in messages
        )

    p = prompts.load("alfworld", "oneshot", prompts_dir=_write(tmp_path, "alfworld", "oneshot", ONESHOT))
    with pytest.raises(ValueError, match="does not render tool results inside a user turn"):
        prompts.render(p, tool_role_template, [ACT])
