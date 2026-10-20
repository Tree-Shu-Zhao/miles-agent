"""A model's own tool-call markup, rendered by its chat template.

How a model writes a tool call (hermes JSON for Qwen2.5 / Qwen3, the XML of
Qwen3.5, ...) is a property of the model: its chat template renders the markup
and sglang's tool-call parser reads it back. A prompt that demonstrates a call
must show exactly that markup, so the recipes' prepare_data.py ask the policy
model's template for it instead of spelling it out, and nothing else in the
repo depends on a markup: envs, agents and rewards see structured calls only.

The same render-twice-and-diff yields the token-space facts a training stage
needs without naming a token: the id that opens a call (marker_token_id, the
OPD hook's turn segmentation) and the ids of the generation prompt and of a
user message (generation_prompt_ids, user_message_ids: where the R2S label
stage splices its hint into a prompt's tokens). tito_render_chat renders as the
training engines do, through miles' fixed template of the TITO family.

Runs under the bare container python3 (prepare_data.py); transformers and
miles come from the miles image, and tests pass their own renderer and
tokenizer.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class RenderChat(Protocol):
    """(messages, tools) -> the text the model sees, e.g. a tokenizer's apply_chat_template(..., tokenize=False).
    Tools in chat-completions form; add_generation_prompt=True ends the text with the header the model
    generates after."""

    def __call__(self, messages: list[dict], tools: list[dict], *, add_generation_prompt: bool = False) -> str: ...


_SENTINEL = "\x00tool-call-markup\x00"


def chat_tool(tool: Mapping) -> dict:
    """Chat-completions form of a Responses-form flat tool (the rows' metadata.tools)."""
    return {
        "type": "function",
        "function": {key: tool[key] for key in ("name", "description", "parameters") if key in tool},
    }


def render_tool_call(render_chat: RenderChat, tool: Mapping, arguments: Mapping) -> str:
    """One call of ``tool`` (chat-completions form) with ``arguments``, in the model's markup.

    Renders one conversation twice, its assistant turn without and with the
    call: the call adds its markup between the turn's content and the turn's
    end, so the tail of the plain render is a suffix of the other. No
    per-family code; a template that renders calls elsewhere is refused.
    """
    name = tool["function"]["name"]

    def after_content(with_call: bool) -> str:
        assistant = {"role": "assistant", "content": _SENTINEL}
        if with_call:
            assistant["tool_calls"] = [{"type": "function", "function": {"name": name, "arguments": dict(arguments)}}]
        messages = [{"role": "user", "content": "start"}, assistant, {"role": "user", "content": "next"}]
        return render_chat(messages, [dict(tool)]).split(_SENTINEL, 1)[1]

    plain, called = after_content(False), after_content(True)
    if not called.endswith(plain) or len(called) == len(plain):
        raise ValueError("the chat template does not render the tool call inside the assistant turn")
    return called[: len(called) - len(plain)].strip()


def hf_render_chat(model_dir: str | Path) -> RenderChat:
    """RenderChat over a HuggingFace checkpoint's own chat template."""
    from transformers import AutoTokenizer  # heavy; only prepare_data.py needs it

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    def render(messages: list[dict], tools: list[dict], *, add_generation_prompt: bool = False) -> str:
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return render


def tito_render_chat(tito_model: str, tokenizer, chat_template_kwargs: str | Mapping = "") -> RenderChat:
    """RenderChat over what the training engines render for a launch: miles' fixed template of the TITO family
    (--tito-model) through ``tokenizer``, with the launch's --apply-chat-template-kwargs (the launcher's JSON
    string, "" = none, or miles' parsed dict) and the family's registered kwargs folded in the way miles does at
    startup (arguments.py, the family's values winning). A family that registers no template file (nemotron3)
    keeps the tokenizer's own template, as the engines do."""
    from miles.utils.chat_template_utils.tito_tokenizer import resolve_fixed_chat_template  # the miles image only

    path, family_kwargs = resolve_fixed_chat_template(tito_model)
    template = Path(path).read_text(encoding="utf-8") if path else None
    if isinstance(chat_template_kwargs, Mapping):
        kwargs = dict(chat_template_kwargs)
    else:
        kwargs = json.loads(chat_template_kwargs or "{}")
    kwargs.update(family_kwargs)

    def render(messages: list[dict], tools: list[dict], *, add_generation_prompt: bool = False) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    return render


def marker_token_id(tokenizer, render_chat: RenderChat, tool: Mapping) -> int:
    """The id of the token that opens a call of ``tool`` (Responses form) in the model's markup: the first token
    of a rendered call, which must be one whole added token of ``tokenizer``. The OPD hook segments a turn's
    deliberation from its action at it; a tokenizer that spells the opening in pieces cannot be segmented."""
    markup = render_tool_call(render_chat, chat_tool(tool), {"command": "look"})
    first = tokenizer.encode(markup, add_special_tokens=False)[0]
    piece = tokenizer.decode([first])
    if tokenizer.get_added_vocab().get(piece) != first:
        name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
        raise ValueError(
            f"{name}: the call markup {markup.splitlines()[0]!r} opens with token {first} = {piece!r}, not with one "
            "whole added token of the tokenizer; OPD turns cannot be segmented for this model"
        )
    return first


def generation_prompt_ids(render_chat: RenderChat, tokenizer) -> list[int]:
    """The tokens the template appends for the model to continue (add_generation_prompt): the tail of every
    prompt the engines tokenize, in front of which the R2S label stage splices its hint."""
    plain = render_chat(_lone_user_turn(), [])
    primed = render_chat(_lone_user_turn(), [], add_generation_prompt=True)
    return _appended_ids(tokenizer, plain, primed, "the generation prompt")


def user_message_ids(render_chat: RenderChat, tokenizer, text: str) -> list[int]:
    """The tokens of one user message carrying ``text``, as the template writes it after a turn."""
    plain = render_chat(_lone_user_turn(), [])
    extended = render_chat(_lone_user_turn() + [{"role": "user", "content": text}], [])
    return _appended_ids(tokenizer, plain, extended, "a user message")


def _lone_user_turn() -> list[dict]:
    """The conversation both diffs start from: one user turn and nothing else. With an assistant turn in it the
    bytes before the appended part would change (the Qwen templates keep a think block on the turns after the
    last user query only)."""
    return [{"role": "user", "content": "start"}]


def _appended_ids(tokenizer, plain: str, longer: str, what: str) -> list[int]:
    """Encode what ``longer`` adds after ``plain``; a template that rewrites instead of appending is refused."""
    if not longer.startswith(plain) or len(longer) == len(plain):
        raise ValueError(f"the chat template does not append {what} after the conversation")
    return tokenizer.encode(longer[len(plain) :], add_special_tokens=False)
