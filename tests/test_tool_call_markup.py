"""miles_agent/tool_call_markup.py and the recipes' system prompts built on it.

The tool-call markup belongs to the model: a recipe's example call is rendered
by the policy model's chat template, never spelled out, and the token-space
facts the OPD hook and the R2S label stage need (the marker that opens a call,
the generation prompt, a user message) come from the same render-twice-and-diff.
Rendered here through miles' fixed Qwen templates (tests/conftest.py
template_render_chat) and a dict-backed stand-in tokenizer.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from conftest import MILES_TEMPLATES
from miles_agent.tool_call_markup import (
    chat_tool,
    generation_prompt_ids,
    marker_token_id,
    render_tool_call,
    tito_render_chat,
    user_message_ids,
)

REPO = Path(__file__).resolve().parent.parent

# The rows' act tool in Responses form (metadata.tools) and in chat-completions form.
ACT_FLAT = {
    "type": "function",
    "name": "act",
    "description": "d",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}
ACT = chat_tool(ACT_FLAT)
# The example every pre-Qwen3.5 alfworld run trained on, written out by hand then.
HERMES_GO_TO_DESK = '<tool_call>\n{"name": "act", "arguments": {"command": "go to desk 1"}}\n</tool_call>'
XML_GO_TO_DESK = "<tool_call>\n<function=act>\n<parameter=command>\ngo to desk 1\n</parameter>\n</function>\n</tool_call>"

# Qwen's added tokens the fixed templates write (ChatML specials, the think and call tags); ids arbitrary.
ADDED = {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4, "<tool_call>": 5, "</tool_call>": 6}
# The pin's registry rows these tests stand in for (tito_tokenizer.py: Qwen3 :382, Qwen35 :434, Nemotron3
# :579 ships no fixed jinja and keeps the HF-native template).
FIXED_TEMPLATES = {"qwen3": "qwen3_fixed.jinja", "qwen35": "qwen3.5_fixed.jinja", "nemotron3": None}
FAMILY_KWARGS = {
    "qwen3": {"clear_thinking": False},
    "qwen35": {"preserve_thinking": True},
    "nemotron3": {"truncate_history_thinking": False},
}


class FakeTokenizer:
    """Dict-backed stand-in for a HF tokenizer: greedy longest match over the added vocabulary, one id per
    other character, and apply_chat_template through tests/conftest.py's sandboxed jinja (the template it is
    handed, else its own, as transformers does)."""

    name_or_path = "fake-tokenizer"

    def __init__(self, template_render, added=ADDED, chat_template=None):
        self._render, self.added, self.chat_template = template_render, dict(added), chat_template

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            tag = next((t for t in sorted(self.added, key=len, reverse=True) if text.startswith(t, i)), None)
            ids.append(self.added[tag] if tag else 1000 + ord(text[i]))
            i += len(tag) if tag else 1
        return ids

    def decode(self, ids):
        by_id = {v: k for k, v in self.added.items()}
        return "".join(by_id[i] if i in by_id else chr(i - 1000) for i in ids)

    def get_added_vocab(self):
        return dict(self.added)

    def apply_chat_template(
        self, messages, tools=None, chat_template=None, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        assert tokenize is False
        text = chat_template if chat_template is not None else self.chat_template
        render = self._render(text)
        return render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


@pytest.fixture
def tito_stub(monkeypatch):
    """miles' tito_tokenizer surface tito_render_chat reads (resolve_fixed_chat_template), without miles; the
    real one is pinned by tests/test_contracts.py where miles imports."""
    mod = types.ModuleType("miles.utils.chat_template_utils.tito_tokenizer")

    def resolve_fixed_chat_template(tito_model):
        if tito_model not in FIXED_TEMPLATES:
            raise ValueError(f"{tito_model!r} is not a valid TITOTokenizerType")
        name = FIXED_TEMPLATES[tito_model]
        return (str(MILES_TEMPLATES / name) if name else None, dict(FAMILY_KWARGS[tito_model]))

    mod.resolve_fixed_chat_template = resolve_fixed_chat_template
    monkeypatch.setitem(sys.modules, "miles.utils.chat_template_utils.tito_tokenizer", mod)
    return mod


def _load(recipe: str):
    spec = importlib.util.spec_from_file_location(f"_{recipe}_prepare_data", REPO / "recipes" / recipe / "prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chat_tool_nests_the_responses_form():
    flat = {"type": "function", "name": "act", "description": "d", "parameters": {"type": "object"}, "strict": True}
    assert chat_tool(flat) == {
        "type": "function",
        "function": {"name": "act", "description": "d", "parameters": {"type": "object"}},
    }


@pytest.mark.parametrize(
    ("template", "markup"),
    [
        ("qwen3_fixed.jinja", HERMES_GO_TO_DESK),  # Qwen3 (and Qwen2.5): hermes JSON
        ("qwen3.5_fixed.jinja", XML_GO_TO_DESK),  # Qwen3.5: qwen3_coder XML
        ("qwen3.6_fixed.jinja", XML_GO_TO_DESK),
    ],
)
def test_render_tool_call_is_the_templates_own_markup(template_render_chat, template, markup):
    assert render_tool_call(template_render_chat(template), ACT, {"command": "go to desk 1"}) == markup


def test_render_tool_call_refuses_a_template_without_calls():
    def ignores_calls(messages, tools):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    with pytest.raises(ValueError, match="does not render the tool call"):
        render_tool_call(ignores_calls, ACT, {"command": "x"})


def test_alfworld_prompt_names_no_markup_and_shows_the_models(template_render_chat):
    from miles_agent import prompts

    pd = _load("alfworld")
    system = prompts.load("alfworld", pd.DEFAULT_PROMPT).system
    assert "<tool_call>" not in system and "{example_call}" in system
    hermes = pd.prompt_messages(pd.DEFAULT_PROMPT, template_render_chat("qwen3_fixed.jinja"))[0]["content"]
    xml = pd.prompt_messages(pd.DEFAULT_PROMPT, template_render_chat("qwen3.5_fixed.jinja"))[0]["content"]
    assert HERMES_GO_TO_DESK in hermes and "<function=" not in hermes
    assert XML_GO_TO_DESK in xml and '"arguments"' not in xml
    for prompt in (hermes, xml):
        assert prompt.startswith("You are an agent in a household environment")
        assert "{example_call}" not in prompt


# ------------------------------------------------------------ token-space facts


def test_tito_render_chat_reads_the_fixed_template(tito_stub, template_render, template_render_chat):
    """tito_render_chat renders through the TITO family's fixed template file (miles' TEMPLATE_DIR) with the
    launch's kwargs (the launcher's JSON string or miles' parsed dict) plus the family's registered ones, as
    miles merges them at startup; nemotron3 registers no file and keeps the tokenizer's own template."""
    seen = []

    class Recording(FakeTokenizer):
        def apply_chat_template(self, messages, tools=None, chat_template=None, **kwargs):
            seen.append((chat_template, kwargs))
            return super().apply_chat_template(messages, tools=tools, chat_template=chat_template, **kwargs)

    messages = [{"role": "user", "content": "hi"}]
    tokenizer = Recording(template_render)
    render = tito_render_chat("qwen35", tokenizer, '{"enable_thinking": false}')
    assert render(messages, [ACT]) == template_render_chat("qwen3.5_fixed.jinja")(messages, [ACT])
    template, kwargs = seen[-1]
    assert template == (MILES_TEMPLATES / "qwen3.5_fixed.jinja").read_text()
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": False,
        "preserve_thinking": True,
    }
    # the kwargs reach the template: thinking off changes the generation prompt, which is appended
    plain = render(messages, [])
    off = render(messages, [], add_generation_prompt=True)
    default = tito_render_chat("qwen35", tokenizer, {})(messages, [], add_generation_prompt=True)
    assert off.startswith(plain) and off != default
    # miles' parsed dict (input.args.apply_chat_template_kwargs) and the launcher's JSON string render alike
    as_dict = tito_render_chat("qwen3", tokenizer, {"enable_thinking": False})
    as_json = tito_render_chat("qwen3", tokenizer, '{"enable_thinking": false}')
    assert as_dict(messages, [], add_generation_prompt=True) == as_json(messages, [], add_generation_prompt=True)
    assert seen[-1][0] == (MILES_TEMPLATES / "qwen3_fixed.jinja").read_text()
    assert seen[-1][1]["clear_thinking"] is False
    # nemotron3: no bundled file, so the tokenizer's own template (a Qwen3.5 text stands in for it here)
    own = Recording(template_render, chat_template=(MILES_TEMPLATES / "qwen3.5_fixed.jinja").read_text())
    native = template_render_chat("qwen3.5_fixed.jinja")
    assert tito_render_chat("nemotron3", own, "")(messages, [ACT]) == native(messages, [ACT])
    assert seen[-1][0] is None and seen[-1][1]["truncate_history_thinking"] is False
    with pytest.raises(ValueError, match="not a valid TITOTokenizerType"):
        tito_render_chat("qwen2", tokenizer)


@pytest.mark.parametrize(("tito", "template"), [("qwen3", "qwen3_fixed.jinja"), ("qwen35", "qwen3.5_fixed.jinja")])
def test_generation_prompt_ids_and_user_message_ids_by_diff(
    tito_stub, template_render, template_render_chat, tito, template
):
    """Render twice and diff: the generation prompt (with/without add_generation_prompt) and one user
    message (with/without the trailing turn), as the template writes them, encoded without special tokens.
    Under the engines' render the user block spliced in front of the generation prompt is the hinted prompt
    of the R2S label stage: token-equal to rendering the hint as a message, no re-tokenization (design D9)."""
    tokenizer = FakeTokenizer(template_render)
    bare = template_render_chat(template)  # the template's own defaults
    engines = tito_render_chat(tito, tokenizer, '{"enable_thinking": false}')  # the campaign launch's render
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "obs 0"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "act", "arguments": {"command": "look"}}}],
        },
        {"role": "tool", "content": "obs 1"},
    ]
    hint = "Hint: go to desk 1\nthen open it"

    gen_ids = generation_prompt_ids(bare, tokenizer)
    hint_ids = user_message_ids(bare, tokenizer, hint)
    assert gen_ids and hint_ids and gen_ids[0] == ADDED["<|im_start|>"] == hint_ids[0]
    primed = bare(history, [ACT], add_generation_prompt=True)
    assert primed == bare(history, [ACT]) + tokenizer.decode(gen_ids)
    assert hint in tokenizer.decode(hint_ids)
    # thinking off (the campaign) lengthens the generation prompt by an empty think block; the user block is the same
    off_ids = generation_prompt_ids(engines, tokenizer)
    assert len(off_ids) > len(gen_ids) and off_ids[: len(gen_ids)] == gen_ids
    assert user_message_ids(engines, tokenizer, hint) == hint_ids
    # the splice: prompt tokens minus the generation prompt, the hint block, the generation prompt. It holds under
    # the engines' render, whose family kwargs keep every assistant turn's bytes fixed when a user message
    # follows; the template's own defaults would rewrite the demo turn's think block instead.
    prompt = tokenizer.encode(engines(history, [ACT], add_generation_prompt=True))
    ins = len(prompt) - len(off_ids)
    assert prompt[ins:] == off_ids
    hinted = [{"role": "user", "content": hint}]
    spliced = prompt[:ins] + hint_ids + prompt[ins:]
    assert tokenizer.encode(engines(history + hinted, [ACT], add_generation_prompt=True)) == spliced
    assert bare(history + hinted, [ACT]) != bare(history, [ACT]) + tokenizer.decode(hint_ids)

    def rewrites(messages, tools, *, add_generation_prompt=False):
        return "same text whatever is asked"

    with pytest.raises(ValueError, match="does not append the generation prompt"):
        generation_prompt_ids(rewrites, tokenizer)
    with pytest.raises(ValueError, match="does not append a user message"):
        user_message_ids(rewrites, tokenizer, hint)


@pytest.mark.parametrize(
    ("tito", "template"), [("qwen3", "qwen3_fixed.jinja"), ("qwen35", "qwen3.5_fixed.jinja"), ("nemotron3", None)]
)
def test_marker_token_id_per_family(tito_stub, template_render, template_render_chat, tito, template):
    """The marker is the first token of the family's rendered call and must be one whole added token of the
    tokenizer: hermes JSON (qwen3) and the XML families (qwen35; nemotron3 through the tokenizer's own
    template) each name theirs, and a tokenizer that spells the opening in pieces is refused."""
    # nemotron3 ships no fixed jinja: the fake tokenizer's own template stands in for the checkpoint's
    own = (MILES_TEMPLATES / "qwen3.5_fixed.jinja").read_text() if template is None else None
    tokenizer = FakeTokenizer(template_render, chat_template=own)
    render = tito_render_chat(tito, tokenizer, "")
    marker = marker_token_id(tokenizer, render, ACT_FLAT)
    markup = render_tool_call(render, ACT, {"command": "look"})
    assert tokenizer.encode(markup)[0] == marker and marker in ADDED.values()
    assert tokenizer.decode([marker]) == markup.splitlines()[0]
    if template is not None:
        assert marker_token_id(tokenizer, template_render_chat(template), ACT_FLAT) == marker
    in_pieces = {k: v for k, v in ADDED.items() if v != marker}
    split = FakeTokenizer(template_render, added=in_pieces, chat_template=own)
    with pytest.raises(ValueError, match="whole added token") as refused:
        marker_token_id(split, tito_render_chat(tito, split, ""), ACT_FLAT)
    assert markup.splitlines()[0] in str(refused.value) and split.name_or_path in str(refused.value)
