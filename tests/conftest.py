import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Mirror the training entry's import surface (run.py's PYTHONPATH): the repo root
# exposes the miles_agent package (hooks load it by dotted path), and the
# recipe dir exposes bare recipe modules (REWARD_SHAPING_PATH=reward.shape).
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "recipes" / "search_r1"))
# gym_root exposes the shared agent_utils package the episode-driver agents
# import (its resources_servers/ and responses_api_agents/ dirs are namespace
# portions that merge with, not shadow, nemo-gym's own).
sys.path.insert(0, str(REPO / "gym_root"))


# miles' fixed chat templates: real Qwen-family templates the tests can render
# without transformers or a checkpoint (prepare_data.py renders the
# policy model's own template through transformers instead).
MILES_TEMPLATES = REPO / "3rdparty" / "miles" / "miles" / "utils" / "chat_template_utils" / "templates"


@pytest.fixture(scope="session")
def template_render():
    """template text -> render(**variables), rendered the way transformers'
    apply_chat_template does (sandboxed jinja, trim/lstrip blocks, loopcontrols,
    tojson without ASCII escaping, raise_exception); the caller sets messages,
    tools, add_generation_prompt and the template kwargs (enable_thinking)."""
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, indent=None, ensure_ascii=False, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception

    def make(text: str):
        template = env.from_string(text)
        return lambda **variables: template.render(**variables)

    return make


@pytest.fixture(scope="session")
def template_render_chat(template_render):
    """template file name -> RenderChat (miles_agent.tool_call_markup) over miles'
    fixed templates; the generation prompt only when asked for."""

    def make(template_name: str):
        path = MILES_TEMPLATES / template_name
        if not path.is_file():
            pytest.skip(f"{path} not present (3rdparty/miles not checked out)")
        render = template_render(path.read_text())

        def render_chat(messages, tools, *, add_generation_prompt=False):
            return render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)

        return render_chat

    return make
