"""Observation / admissible-action formatting for the WebShop episode.

Training-semantics presentation for the WebShop episode (the webshop_env
adapter owns this; the py3.10 engine owns env mechanics). Wording follows the
TCOD WebShop protocol (trinity/common/workflows/envs/TCOD/webshop/utils.py,
itself the verl-agent template): ``Your current observation is: <page>.`` /
``You are now at step k and your current observation is: <page>.`` followed by
``Your admissible actions of the current situation are: [ ... ]`` with one
``search[<query>]`` / ``click[<button>]`` command per line, and TCOD's exact
error sentences for a non-admissible command. Act-tool command extraction is
the shared ``agent_utils.session_gym.extract_command``.

Two page renderings are understood (``observation_mode``, must match the
engine's --observation-mode):

- ``text_rich`` (TCOD / original WebShop baselines): one visible text per
  line, buttons as ``[button] X [button_]``; the page is passed through
  verbatim (TCOD's ``format_observation`` = ``strip()``), instruction included.
- ``text`` (verl-agent): ``[SEP]``-joined; everything up to and including the
  instruction is stripped and the remaining parts are single-quoted (byte-
  compatible with the verl-agent text env).
"""

import re
from typing import Any

from agent_utils.session_gym import truncate_obs

OBSERVATION_MODES = ("text", "text_rich")


class WebshopFormatError(ValueError):
    """The engine returned a payload the adapter cannot interpret (protocol
    fault, not a model mistake) — surfaced as an episode drop, never a crash."""


def extract_task(text_obs: str) -> str:
    """Goal instruction from the initial WebShop observation, either rendering.

    ``text``: ``... [SEP] Instruction: [SEP] <instruction> [SEP] Search``.
    ``text_rich``: an ``Instruction:`` line followed by the instruction line(s)
    up to the first ``[button]`` line (TCOD's ``_extract_task_description``).
    Both key off the ``Instruction:`` marker so a banner change cannot shift
    the index.
    """
    if " [SEP] " in text_obs:
        parts = text_obs.split(" [SEP] ")
        for i, part in enumerate(parts):
            if part.strip() == "Instruction:" and i + 1 < len(parts):
                return parts[i + 1]
        raise WebshopFormatError(f"no Instruction marker in initial webshop obs: {text_obs[:200]!r}")
    lines = [line.strip() for line in text_obs.splitlines()]
    if "Instruction:" in lines:
        task_lines: list[str] = []
        for line in lines[lines.index("Instruction:") + 1 :]:
            if not line:
                continue
            if line.startswith("[button]"):
                break
            task_lines.append(line)
        if task_lines:
            return " ".join(task_lines)
    raise WebshopFormatError(f"no Instruction marker in initial webshop obs: {text_obs[:200]!r}")


def format_obs(text_obs: str, task: str, observation_mode: str = "text_rich") -> str:
    """Model-facing page text for one rendering (see module docstring)."""
    if observation_mode == "text_rich":
        return text_obs.strip()
    if observation_mode != "text":
        raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}, got {observation_mode!r}")
    parts = text_obs.split(" [SEP] ")
    try:
        index = parts.index(task)
    except ValueError:
        return text_obs
    return " [SEP] ".join(f"'{p}'" for p in parts[index + 1 :])


def available_commands(avail: dict[str, Any]) -> list[str]:
    """Available-actions dict -> WebShop command strings (TCOD's
    ``_format_available_actions``: ``search[<query>]`` when the search bar is
    present, then ``click[<clickable>]``). A malformed payload is an
    engine-protocol fault (drop the episode), not a model mistake."""
    if not isinstance(avail, dict):
        raise WebshopFormatError("engine response missing info['available_actions']")
    for key in avail:
        if key not in ("has_search_bar", "clickables"):
            raise WebshopFormatError(f"unknown key in available actions: {key!r}")
    commands: list[str] = []
    has_search_bar = bool(avail.get("has_search_bar"))
    if has_search_bar:
        commands.append("search[<query>]")
    for txt in avail.get("clickables", []):
        if has_search_bar and str(txt).lower() == "search":
            # The search-bar submit button: upstream explicitly no-ops
            # click[search] (the query goes through search[...]), so
            # advertising it would only offer a wasted step.
            continue
        commands.append(f"click[{txt}]")
    return commands


def render_page(page: str, avail: dict[str, Any], next_step: int, max_commands: int) -> str:
    """Wrap an already-formatted page (or an error sentence) in the TCOD
    observation block. ``next_step`` is the 1-based index of the action the
    model is about to take; the opening observation (1) uses TCOD's no-history
    wording, later ones announce the step so the 15-step budget stays visible
    (the transcript itself is the history under Miles' one-Sample-per-episode
    model)."""
    commands = "\n".join(available_commands(avail)[:max_commands])
    lead = (
        "Your current observation is"
        if next_step <= 1
        else f"You are now at step {next_step} and your current observation is"
    )
    return f"{lead}: {page}.\nYour admissible actions of the current situation are:\n[\n{commands}\n]."


def render_observation(
    text_obs: str,
    task: str,
    avail: dict[str, Any],
    max_obs_chars: int,
    max_commands: int,
    *,
    observation_mode: str = "text_rich",
    next_step: int = 1,
) -> str:
    """Model-facing observation block: current page + admissible commands."""
    page = truncate_obs(format_obs(text_obs, task, observation_mode), max_obs_chars)
    return render_page(page, avail, next_step, max_commands)


# Reserved first tokens of a WebShop search query (``search[<a> ...]`` etc.).
# Upstream web_agent_site routes them to attribute/category/query/random
# lookups that crash (KeyError) on the shipped catalog instead of no-op'ing
# like every other unparseable action. The engine glue
# (env_servers/webshop/_webshop_engine.py) neutralizes such a search into a
# no-op before env.step; the adapter counts it as an invalid action. The two
# tuples must stay identical.
RESERVED_SEARCH_TOKENS = ("<a>", "<c>", "<q>", "<r>")
# Upstream parse_action pattern (greedy name, bracketed argument).
_ACTION_RE = re.compile(r"(.+)\[(.+)\]")


def is_reserved_search(command: str) -> bool:
    """Whether ``command`` is a search whose first query token is reserved.

    Mirrors the upstream parse so the check fires exactly when the engine
    would: ``parse_action`` regex, argument lowercased, ``keywords =
    arg.split(" ")`` and ``keywords[0]`` compared to the tokens (a leading
    space inside the brackets defuses it upstream too). The action name is
    matched case-insensitively (upstream no-ops ``Search[...]`` anyway, so
    the page is unchanged either way).
    """
    m = _ACTION_RE.match(command.strip())
    if m is None:
        return False
    name, arg = m.groups()
    return name.strip().lower() == "search" and arg.lower().split(" ")[0] in RESERVED_SEARCH_TOKENS


# TCOD's validate_action error sentences (verbatim where the transport allows;
# the last one names the act tool instead of <action> tags).
INVALID_SEARCH_EMPTY = "Invalid action, please type in the query you want to search in the square brackets here."
INVALID_SEARCH_NO_BAR = (
    "Can not perform search action without search bar. Please click the Back to Search button first."
)
INVALID_CLICK_PREFIX = (
    "Incorrect action format, make sure you have the correct button name that is within the current page. "
    "The buttons you can click now are: "
)
INVALID_GENERIC = (
    "Invalid action. Call the act tool with exactly one admissible command from the list: "
    "search[<query>] (only when the search bar is present) or click[<button>]."
)


def invalid_action_message(command: str, avail: dict[str, Any]) -> str:
    """TCOD's error text for a non-admissible command (its ``validate_action``
    branches): empty search, search without a search bar, click on a button
    that is not on the page, anything else. The caller decided the command is
    not admissible; this only picks the sentence."""
    cmd = command.strip()
    m = _ACTION_RE.match(cmd)
    if m is None:  # TCOD: action_name = the whole action, no argument
        name, arg = cmd.lower(), None
    else:
        name, arg = m.group(1).strip().lower(), m.group(2)
    if name == "search":
        if arg is None or not arg.strip():
            return INVALID_SEARCH_EMPTY
        if not avail.get("has_search_bar"):
            return INVALID_SEARCH_NO_BAR
        return INVALID_GENERIC  # a search the engine cannot run (reserved first token)
    if name == "click":
        buttons = [c for c in available_commands(avail) if c.startswith("click[")]
        return INVALID_CLICK_PREFIX + str([b[len("click[") : -1] for b in buttons])
    return INVALID_GENERIC
