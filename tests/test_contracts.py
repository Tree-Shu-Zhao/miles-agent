"""Smoke assertions on the Miles + NeMo Gym coupling surfaces.

This is the ENTIRE list of upstream behaviors miles-agent depends on. Run
after every submodule bump; if these pass, the overlay still fits.
"""

import importlib.util
import inspect
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_agent_function_signature():
    """Miles' agentic_tool_call.generate calls the agent function with keyword
    args (base_url=, prompt=, request_kwargs=, metadata=) — gym_bridge.run
    must accept exactly that surface."""
    from miles_agent import gym_bridge

    params = inspect.signature(gym_bridge.run).parameters
    for kw in ("base_url", "prompt", "request_kwargs", "metadata"):
        assert kw in params, f"gym_bridge.run lost parameter {kw!r}"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "gym_bridge.run must absorb future keyword args (**kwargs)"
    )
    assert inspect.iscoroutinefunction(gym_bridge.run)


def test_agent_abort_hook_signature():
    """miles.utils.misc.call_agent_abort_hook awaits ``<agent module>.abort(args)``."""
    from miles_agent import gym_bridge

    assert inspect.iscoroutinefunction(gym_bridge.abort)
    assert list(inspect.signature(gym_bridge.abort).parameters) == ["args"]


def test_miles_agentic_generate_exists():
    """The built-in generate path the launch script wires up."""
    mod = pytest.importorskip(
        "miles.rollout.generate_hub.agentic_tool_call",
        reason="miles (and torch) not importable here",
    )
    assert inspect.iscoroutinefunction(mod.generate)


def test_reward_routing_contract():
    """reward_routing fans the env verify score into {"env", "train"} and
    stamps metadata["raw_reward"]; with no shaping plugin, train == env."""
    import asyncio
    from types import SimpleNamespace

    pytest.importorskip("miles.utils.types", reason="miles not importable here")
    from miles_agent import reward_routing

    params = inspect.signature(reward_routing.reward_func).parameters
    assert list(params)[:2] == ["args", "samples"]

    sample = SimpleNamespace(metadata={"reward": 1.0}, response="<answer>x</answer>")
    result = asyncio.run(reward_routing.reward_func(None, sample))
    assert result == {"env": 1.0, "train": 1.0}
    assert sample.metadata["raw_reward"] == 1.0

    # /run transport failure: no metadata["reward"] -> env score 0.0
    failed = SimpleNamespace(metadata={}, response=None)
    assert asyncio.run(reward_routing.reward_func(None, failed))["env"] == 0.0


def test_recipe_shaping_signature():
    """search_r1's shaping plugin (REWARD_SHAPING_PATH=reward.shape): with the
    coefficient env vars unset it must be the identity."""
    from types import SimpleNamespace

    import reward  # bare recipe module, recipes/search_r1/reward.py

    sample = SimpleNamespace(metadata={"env_metrics": {"tool_call_count": 3.0}}, response="<answer>x</answer>")
    assert reward.shape(0.5, sample) == 0.5


def test_search_r1_budget_term_reads_the_agents_call_count(monkeypatch):
    """The search budget counts the calls tool_call_agent reports, whatever the
    policy model's markup; the response text is never searched for markup."""
    from types import SimpleNamespace

    for name in ("REWARD_FORMAT_COEF", "REWARD_BUDGET_COEF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REWARD_BUDGET_COEF", "0.25")
    mod = _load_module("search_r1_reward_budget", REPO / "recipes" / "search_r1" / "reward.py")
    xml_calls = "<tool_call>\n<function=search>\n<parameter=query>\nq\n</parameter>\n</function>\n</tool_call>" * 3
    assert mod.shape(1.0, SimpleNamespace(metadata={"env_metrics": {"tool_call_count": 2.0}}, response=xml_calls)) == 0.5
    assert mod.shape(1.0, SimpleNamespace(metadata={}, response=xml_calls)) == 1.0  # /run failed: no counters


def _load_tool_call_agent_module():
    path = REPO / "gym_root" / "responses_api_agents" / "tool_call_agent" / "app.py"
    spec = importlib.util.spec_from_file_location("tool_call_agent_app", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tool_call_agent_run_request_schema():
    """policy_base_url must ride the /run body (the #2166 pattern), and dataset
    extras (expected_answer) must survive validation."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    mod = _load_tool_call_agent_module()

    body = mod.ToolCallAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "policy_base_url": "http://trainer:30000/sessions/abc/v1",
            "expected_answer": ["Paris"],
        }
    )
    assert body.policy_base_url.endswith("/v1")
    assert body.model_extra["expected_answer"] == ["Paris"]


def test_tool_call_agent_reports_the_calls_it_got(monkeypatch):
    """tool_call_count = the calls the engine's parser handed the agent, bad
    arguments included: the structured count search_r1's budget term reads."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    import asyncio
    import json
    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient

    mod = _load_tool_call_agent_module()

    class Resp:
        def __init__(self, payload):
            self._body = json.dumps(payload).encode()
            self.status, self.ok, self.cookies, self.headers = 200, True, {}, {}

        async def read(self):
            return self._body

        @property
        def content(self):
            return self

    def call(i, arguments):
        return {"id": f"tc{i}", "type": "function", "function": {"name": "search", "arguments": arguments}}

    replies = iter(
        [
            {"role": "assistant", "content": "", "tool_calls": [call(1, '{"query": "a"}'), call(2, "not json")]},
            {"role": "assistant", "content": "", "tool_calls": [call(3, '{"query": "b"}')]},
            {"role": "assistant", "content": "<answer>x</answer>"},
        ]
    )

    async def policy(method, url, **kwargs):
        return Resp({"choices": [{"message": next(replies)}]})

    async def resources(server_name, url_path, json=None, cookies=None):
        return Resp({"reward": 1.0} if url_path == "/verify" else {"result": "doc"})

    monkeypatch.setattr(mod, "http_request", policy)
    client = MagicMock(spec=ServerClient)
    client.post = resources
    config = mod.ToolCallAgentConfig(
        host="0.0.0.0",
        port=8200,
        entrypoint="",
        name="tool_call_agent",
        resources_server={"type": "resources_servers", "name": "wiki_search"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
    )
    agent = mod.ToolCallAgent(config=config, server_client=client)
    body = mod.ToolCallAgentRunRequest.model_validate(
        {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}, "policy_base_url": "http://p/v1"}
    )
    result = asyncio.run(agent.run(None, body))
    assert result["reward"] == 1.0
    assert result["tool_call_count"] == 3 and result["tool_errors_model_bad_args"] == 1


def test_agent_function_body_validates_against_gym_schema():
    """Cross-side contract: the miles-side adapter's /run body must validate
    against the gym-side ToolCallAgentRunRequest, dialect translation included
    (max_tokens -> max_output_tokens, session URL gains /v1)."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    import json

    from miles_agent import gym_bridge

    example = REPO / "gym_root" / "resources_servers" / "wiki_search" / "data" / "example.jsonl"
    if not example.is_file():
        pytest.skip("example.jsonl not present (ships via src tar, not git)")
    row = json.loads(example.read_text().splitlines()[0])
    params = row["responses_create_params"]
    body = gym_bridge.build_run_body(
        "http://trainer:30000/sessions/abc",
        params["input"],
        {"temperature": 1.0, "max_tokens": 512, "top_p": None},
        {
            "tools": params["tools"],
            "expected_answer": row["expected_answer"],
            "data_source": row.get("data_source"),
            "custom_grader_field": {"nested": True},  # arbitrary extras must flow through
        },
    )

    mod = _load_tool_call_agent_module()
    validated = mod.ToolCallAgentRunRequest.model_validate(body)
    assert validated.policy_base_url == "http://trainer:30000/sessions/abc/v1"
    assert validated.responses_create_params.max_output_tokens == 512
    assert validated.responses_create_params.temperature == 1.0
    assert validated.model_extra["expected_answer"] == row["expected_answer"]
    assert validated.model_extra["custom_grader_field"] == {"nested": True}
    assert "tools" not in validated.model_extra  # consumed into responses_create_params, not flattened
    # Validated tool items may stay dicts or become models depending on the
    # union member pydantic picks — the same reason tool_call_agent's
    # _to_chat_tools handles both shapes.
    tool_names = [t["name"] if isinstance(t, dict) else t.name for t in validated.responses_create_params.tools]
    assert tool_names == ["search"]


def test_gym_verify_response_has_reward():
    """The Miles adapter forwards response['reward'] — pin the gym schema."""
    base = pytest.importorskip("nemo_gym.base_resources_server", reason="nemo-gym not importable here")
    assert "reward" in base.BaseVerifyResponse.model_fields


def test_gymnasium_server_contract():
    """alfworld_env subclasses nemo-gym's GymnasiumServer and env_loop_agent
    drives its wire protocol — pin the import path, the /reset and /step
    schemas (extra="allow" so game_file rides along), and the auto-close-on-
    terminal behavior the agent relies on to skip /close."""
    gym_base = pytest.importorskip(
        "resources_servers.gymnasium.base", reason="nemo-gym not importable here (uv sync --group gym)"
    )
    import inspect as _inspect

    for field in ("observation", "reward", "terminated", "truncated", "info"):
        assert field in gym_base.EnvStepResponse.model_fields
    for req in (gym_base.EnvResetRequest, gym_base.EnvStepRequest):
        assert req.model_config.get("extra") == "allow", f"{req.__name__} must tolerate dataset extras"
        assert "responses_create_params" in req.model_fields
    assert "response" in gym_base.EnvStepRequest.model_fields
    assert callable(gym_base.extract_text)
    assert callable(gym_base.GymnasiumServer.close_session)
    # env_loop_agent never POSTs /close after a terminal step because the base
    # step endpoint closes the session itself.
    step_endpoint_src = _inspect.getsource(gym_base.GymnasiumServer._step_endpoint)
    assert "close_session" in step_endpoint_src


def test_stateful_servers_run_to_completion_on_disconnect():
    """nemo-gym >= 0.7.0 cancels a request whose client disconnects, installed
    through the overridable SimpleServer.setup_cancellation_middleware hook. The
    session env servers and the episode agents release sessions / pool envs /
    sandbox slots in cleanup a cancellation skips, so they opt out
    (agent_utils/disconnect.py) — pin both the hook and the opt-out."""
    server_utils = pytest.importorskip("nemo_gym.server_utils", reason="nemo-gym not importable here")
    from fastapi import FastAPI

    upstream = FastAPI()
    server_utils.SimpleServer.setup_cancellation_middleware(None, upstream)
    assert [m.cls for m in upstream.user_middleware] == [server_utils.ClientDisconnectCancellationMiddleware], (
        "nemo-gym no longer installs its disconnect cancellation through setup_cancellation_middleware: "
        "re-check gym_root/agent_utils/disconnect.py"
    )

    from agent_utils.disconnect import RunToCompletionOnDisconnect
    from agent_utils.session_gym import SessionGymnasiumServer

    env_loop = _load_env_loop_agent_module()
    for cls in (SessionGymnasiumServer, env_loop.EnvLoopAgent):
        assert cls.setup_cancellation_middleware is RunToCompletionOnDisconnect.setup_cancellation_middleware, cls
    ours = FastAPI()
    RunToCompletionOnDisconnect().setup_cancellation_middleware(ours)
    assert ours.user_middleware == []


def test_miles_launch_api_surface():
    """The miles launch API recipes call (stubbed in tests/test_launcher.py): miles
    >= 0.1.2 moved execute_train / convert_checkpoint from command_utils onto the
    backend an ExecuteTrainConfig creates. Runs wherever miles imports (the
    container step of scripts/smoke_tests.sh)."""
    U = pytest.importorskip("miles.utils.external_utils.command_utils", reason="miles not importable here")
    from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

    for name in ("ExecuteTrainConfig", "dataclass_cli", "get_default_wandb_args", "repo_base_dir"):
        assert hasattr(U, name), name
    assert callable(U.ExecuteTrainConfig.create_backend)
    inspect.signature(BaseCommandBackend.execute_train).bind(
        None,
        train_args="",
        num_gpus_per_node=4,
        megatron_model_type="m",
        megatron_path="/root/Megatron-LM",
        train_script="train.py",
        extra_env_vars={},
    )
    inspect.signature(BaseCommandBackend.convert_checkpoint).bind(
        None,
        model_name="m",
        megatron_model_type="m",
        num_gpus_per_node=4,
        dir_dst="/root",
        hf_checkpoint="/hf",
        megatron_path="/root/Megatron-LM",
    )


def test_miles_tito_families_bind_the_parsers_we_launch():
    """miles_agent/launcher.py takes the engines' tool-call parser from the TITO
    family (tool_call_parser) and MODEL_FAMILIES names the families: both must
    exist in the pin with the parser for the family's markup."""
    ctu = pytest.importorskip("miles.utils.chat_template_utils", reason="miles not importable here")
    from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType

    assert ctu.resolve_reasoning_and_tool_call_parser("qwen35")[1] == "qwen3_coder"
    assert ctu.resolve_reasoning_and_tool_call_parser("qwen3")[1] == "qwen25"  # sglang's hermes parser
    src = (REPO / "miles_agent" / "launcher.py").read_text()  # MODEL_FAMILIES, without the launcher's imports
    for tito in re.findall(r'tito_model="([a-z0-9_]+)"', src):
        TITOTokenizerType(tito)  # raises on a family the pin does not register


def test_miles_tito_fixed_templates_we_serve():
    """launcher.write_serve_template (scripts/serve_model.sh and every server that renders as the engines do) reads
    the family's fixed template and parsers off the TITO class - TEMPLATE_DIR / FIXED_TEMPLATE.template,
    .extra_kwargs, tool_call_parser, reasoning_parser - and rewrites the one enable_thinking guard of its
    generation prompt: pin that surface for the families served (tests/test_launcher.py stubs it the same way)."""
    pytest.importorskip("miles.utils.chat_template_utils", reason="miles not importable here")
    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, TITOTokenizerType

    for tito in ("qwen3", "qwen35"):
        cls = TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(tito))
        assert cls.tool_call_parser and cls.reasoning_parser, tito
        fixed = cls.FIXED_TEMPLATE
        assert isinstance(fixed.extra_kwargs, dict)
        text = (TEMPLATE_DIR / fixed.template).read_text()
        assert text.count("{%- if enable_thinking is defined and enable_thinking is false %}") == 1, tito


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_env_loop_agent_module():
    path = REPO / "gym_root" / "responses_api_agents" / "env_loop_agent" / "app.py"
    spec = importlib.util.spec_from_file_location("env_loop_agent_app_contracts", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_env_loop_agent_body_validates_against_gym_schema():
    """Cross-side contract for the alfworld arm: the same miles-side adapter's
    /run body must validate against EnvLoopAgentRunRequest with the episode
    extras (game_file, max_episode_steps) surviving as model_extra — that is
    how they reach the resources server's /reset."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    from miles_agent import gym_bridge

    act_tool = {
        "type": "function",
        "name": "act",
        "description": "Execute one admissible command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    body = gym_bridge.build_run_body(
        "http://trainer:30000/sessions/abc",
        [{"role": "system", "content": "You are an agent."}],
        {"temperature": 1.0, "max_tokens": 512, "top_p": None},
        {
            "tools": [act_tool],
            "game_file": "json_2.1.1/train/x/game.tw-pddl",
            "max_episode_steps": 30,
            "data_source": "alfworld/pick_and_place",
        },
    )

    mod = _load_env_loop_agent_module()
    validated = mod.EnvLoopAgentRunRequest.model_validate(body)
    assert validated.policy_base_url == "http://trainer:30000/sessions/abc/v1"
    assert validated.responses_create_params.max_output_tokens == 512
    assert validated.model_extra["game_file"] == "json_2.1.1/train/x/game.tw-pddl"
    assert validated.model_extra["max_episode_steps"] == 30
    # The action tool rides inside responses_create_params, not model_extra.
    assert "tools" not in validated.model_extra
    tool_names = [t["name"] if isinstance(t, dict) else t.name for t in validated.responses_create_params.tools]
    assert tool_names == ["act"]


def test_alfworld_example_rows_validate():
    """The committed-by-tar dataset rows must round-trip the run-request schema
    exactly like real training rows built by prepare_data.py."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    import json

    example = REPO / "gym_root" / "resources_servers" / "alfworld_env" / "data" / "example.jsonl"
    if not example.is_file():
        pytest.skip("example.jsonl not present (ships via src tar, not git)")

    from miles_agent import gym_bridge

    mod = _load_env_loop_agent_module()
    for line in example.read_text().splitlines():
        row = json.loads(line)
        # Gym-native rows keep tools inside responses_create_params; miles rows
        # carry them in metadata — reconstruct the miles shape here.
        metadata = {k: v for k, v in row.items() if k != "responses_create_params"}
        metadata["tools"] = row["responses_create_params"].get("tools") or []
        body = gym_bridge.build_run_body(
            "http://trainer:30000/sessions/abc", row["responses_create_params"]["input"], {}, metadata
        )
        validated = mod.EnvLoopAgentRunRequest.model_validate(body)
        assert validated.model_extra["game_file"].endswith("game.tw-pddl")
        tool_names = [t["name"] if isinstance(t, dict) else t.name for t in validated.responses_create_params.tools]
        assert tool_names == ["act"]


def test_scienceworld_example_rows_validate():
    """scienceworld_env's committed example rows (recipe-built) round-trip the
    run-request schema with task_name/var_num surviving as model_extra."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    import json

    example = REPO / "gym_root" / "resources_servers" / "scienceworld_env" / "data" / "example.jsonl"
    from miles_agent import gym_bridge

    mod = _load_env_loop_agent_module()
    rows = example.read_text().splitlines()
    assert rows
    for line in rows:
        row = json.loads(line)
        metadata = {k: v for k, v in row.items() if k != "responses_create_params"}
        metadata["tools"] = row["responses_create_params"].get("tools") or []
        body = gym_bridge.build_run_body(
            "http://trainer:30000/sessions/abc", row["responses_create_params"]["input"], {}, metadata
        )
        validated = mod.EnvLoopAgentRunRequest.model_validate(body)
        assert isinstance(validated.model_extra["task_name"], str)
        assert isinstance(validated.model_extra["var_num"], int)
        assert validated.model_extra["max_episode_steps"] == 30
        tool_names = [t["name"] if isinstance(t, dict) else t.name for t in validated.responses_create_params.tools]
        assert tool_names == ["act"]


def _alfworld_reward(monkeypatch, **coefs):
    """recipes/alfworld/reward.py loaded by file path (recipe dirs must never be
    added to conftest sys.path twice: bare `reward` modules would shadow), with
    its coefficients read from the given env at import."""
    for name in ("ALFWORLD_FORMAT_COEF", "ALFWORLD_STEP_PENALTY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in coefs.items():
        monkeypatch.setenv(name, value)
    return _load_module("alfworld_reward", REPO / "recipes" / "alfworld" / "reward.py")


def test_alfworld_recipe_shaping_identity(monkeypatch):
    """recipes/alfworld's shaping plugin: identity with the coefficient env vars unset."""
    from types import SimpleNamespace

    mod = _alfworld_reward(monkeypatch)
    sample = SimpleNamespace(metadata={"env_metrics": {"step_count": 7.0, "action_format_errors": 2.0}}, response="")
    assert mod.shape(1.0, sample) == 1.0
    assert mod.shape(0.0, sample) == 0.0


def test_alfworld_format_term_reads_env_counters_not_text(monkeypatch):
    """The format bonus is the share of steps with a usable act call, from the
    env's counters. The merged response carries the env's observations too, so
    the retry hint after a format fault must not read as a well-formed call."""
    from types import SimpleNamespace

    mod = _alfworld_reward(monkeypatch, ALFWORLD_FORMAT_COEF="0.5")
    hint_in_response = '<tool_call>\n{"name": "act", "arguments": {"command": "x"}}\n</tool_call>'

    def sample(steps, errors, response=""):
        return SimpleNamespace(
            metadata={"env_metrics": {"step_count": steps, "action_format_errors": errors}}, response=response
        )

    assert mod.shape(0.0, sample(4.0, 0.0)) == 0.5
    assert mod.shape(0.0, sample(4.0, 1.0, hint_in_response)) == 0.375
    assert mod.shape(0.0, sample(4.0, 4.0, hint_in_response)) == 0.0
    assert mod.shape(1.0, SimpleNamespace(metadata={}, response=hint_in_response)) == 1.0  # never stepped


def test_dapo_filter_is_shared_not_copied():
    """The dynamic-sampling filter wrappers live once in miles_agent/dapo_filter.py
    (loaded by dotted path; run.py puts the repo root on the actors' PYTHONPATH).
    No recipe may carry a local copy that could diverge."""
    assert (REPO / "miles_agent" / "dapo_filter.py").is_file()
    for recipe in ("search_r1", "alfworld", "scienceworld", "webshop"):
        assert not (REPO / "recipes" / recipe / "dapo_filter.py").exists(), recipe


def test_miles_eval_only_branch_and_checkpoint_layout():
    """miles_agent.eval_ckpt relies on train.py's eval-only branch (--num-rollout 0 with an eval interval: one
    eval pass over the pushed weights, nothing trains) and on the Megatron layout its load view mimics: the
    tracker file naming iter_<7 digits>, which miles reads where it resolves a --load dir."""
    miles = REPO / "3rdparty" / "miles"
    if not (miles / "train.py").is_file():
        pytest.skip("3rdparty/miles not checked out")
    assert "if args.num_rollout == 0 and args.eval_interval is not None:" in (miles / "train.py").read_text()
    checkpoint = (miles / "miles" / "backends" / "megatron_utils" / "checkpoint.py").read_text()
    assert "latest_checkpointed_iteration.txt" in checkpoint and 'f"iter_{int(step):07d}"' in checkpoint


def test_miles_startup_folds_the_family_template_kwargs():
    """tool_call_markup.tito_render_chat folds the TITO family's registered kwargs into the launch's
    --apply-chat-template-kwargs the way miles' argument resolution does (the family's values win), so a render
    from the launcher's JSON string equals the engines' and the hook's already-folded args fold idempotently:
    pin the lines it mirrors."""
    arguments = REPO / "3rdparty" / "miles" / "miles" / "utils" / "arguments.py"
    if not arguments.is_file():
        pytest.skip("3rdparty/miles not checked out")
    src = arguments.read_text()
    assert "resolved_path, resolved_kwargs = resolve_fixed_chat_template(tito_model)" in src
    assert "user_kwargs = dict(args.apply_chat_template_kwargs or {})" in src
    assert "user_kwargs.update(resolved_kwargs)" in src


def test_miles_fixed_template_resolution_we_render_with():
    """tito_render_chat reads the family's template through miles' resolve_fixed_chat_template: the bundled
    file's path (None for a family that keeps the HF-native template) and the registered kwargs; pin that
    surface for the families MODEL_FAMILIES names."""
    pytest.importorskip("miles.utils.chat_template_utils", reason="miles not importable here")
    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, resolve_fixed_chat_template

    path, kwargs = resolve_fixed_chat_template("qwen3")
    assert Path(path) == TEMPLATE_DIR / "qwen3_fixed.jinja" and kwargs == {"clear_thinking": False}
    path, kwargs = resolve_fixed_chat_template("qwen35")
    assert Path(path) == TEMPLATE_DIR / "qwen3.5_fixed.jinja" and kwargs == {"preserve_thinking": True}
    path, kwargs = resolve_fixed_chat_template("nemotron3")
    assert path is None and kwargs == {"truncate_history_thinking": False}


def test_miles_opd_helpers_exist():
    """miles_agent.opd.teacher_scoring is glue over miles' own OPD scoring (rollout/on_policy_distillation.py): the
    request (_score_payload), the transport (_post_json), the reply trim the stock reward reads (_trim_input_field)
    and the --opd-teacher-urls map (parse_teacher_urls). Private helpers: pin them in the pin's source, and their
    call shapes where miles imports."""
    path = REPO / "3rdparty" / "miles" / "miles" / "rollout" / "on_policy_distillation.py"
    if not path.is_file():
        pytest.skip("3rdparty/miles not checked out")
    src = path.read_text()
    for line in (
        "def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, str]:",
        "def _score_payload(",
        "async def _post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None)",
        "def _trim_input_field(meta_info: dict[str, Any], field: str, response_length: int) -> list[Any]:",
        "return values[1:][-response_length:] if response_length > 0 else []",
        '"max_new_tokens": 0,',
        '"logprob_start_len": 0,',
    ):
        assert line in src, line
    mod = pytest.importorskip("miles.rollout.on_policy_distillation", reason="miles not importable here")
    assert mod.parse_teacher_urls(["default=http://t:13141/generate"]) == {"default": "http://t:13141/generate"}
    payload = mod._score_payload([1, 2, 3], 0)
    assert payload["input_ids"] == [1, 2, 3] and payload["return_logprob"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 0 and "top_logprobs_num" not in payload
    assert mod._score_payload([1], 16)["top_logprobs_num"] == 16
    inspect.signature(mod._post_json).bind("http://t:13141/generate", {}, timeout_secs=900.0)
    assert mod._trim_input_field({"f": [None, [-1.0, 2, ""], [-2.0, 3, ""]]}, "f", 1) == [[-2.0, 3, ""]]


def test_opd_generate_fn_signature():
    """launcher.OPD_GENERATE_FN is a --custom-generate-function-path: miles awaits it with one GenerateFnInput (state,
    sample, sampling_params, evaluation; args = state.args), takes a GenerateFnOutput(samples) back and registers its
    add_arguments while parsing (arguments.py) - pin the shapes the hook builds on, and the hook's own surface where
    miles imports."""
    base_types = REPO / "3rdparty" / "miles" / "miles" / "rollout" / "base_types.py"
    if not base_types.is_file():
        pytest.skip("3rdparty/miles not checked out")
    src = base_types.read_text()
    assert "class GenerateFnInput:" in src and "class GenerateFnOutput:" in src
    for line in (
        "state: GenerateState",
        "sample: Sample",
        "sampling_params: dict[str, Any]",
        "evaluation: bool",
        "return self.state.args",
        "samples: Sample | list[Sample]",
    ):
        assert line in src, line
    arguments = (REPO / "3rdparty" / "miles" / "miles" / "utils" / "arguments.py").read_text()
    assert "args_partial.custom_generate_function_path," in arguments
    assert 'if fn is not None and callable(getattr(fn, "add_arguments", None)):' in arguments
    pytest.importorskip("miles.rollout.base_types", reason="miles not importable here")
    from miles_agent import launcher
    from miles_agent.opd import generate as hook

    module_path, _, attr = launcher.OPD_GENERATE_FN.rpartition(".")
    assert (module_path, attr) == (hook.__name__, "generate")
    assert inspect.iscoroutinefunction(hook.generate)
    assert list(inspect.signature(hook.generate).parameters) == ["input"]
    assert callable(hook.generate.add_arguments)


# The tool-call markup belongs to the model (design rule 3): its chat template renders it, the engine parser reads it,
# tool_call_markup derives every token-space fact by rendering twice. These never appear in code: the markup strings,
# the Qwen3.5 / Qwen3 call-marker token ids and the parser name.
MARKUP_LITERALS = ("<tool_call>", "<|im_start|>", "248058", "248059", "151657", "151667", "qwen3_coder")


def test_no_literal_tool_call_markup_in_opd_and_r2s():
    """No hook, stage, recipe or job script under miles_agent/ or scripts/ spells out the tool-call
    markup, a call-marker token id or a parser name: the one exception is launcher.MODEL_FAMILIES, whose rows name
    the TITO family whose template renders and whose parser reads the markup (and the tests' fixtures, which pin those
    facts against the templates themselves)."""
    launcher = REPO / "miles_agent" / "launcher.py"
    offenders = []
    for root in ("miles_agent", "scripts"):
        for path in sorted((REPO / root).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if path == launcher:
                start = text.index("\nMODEL_FAMILIES = {\n")
                text = text[:start] + text[text.index("\n}\n", start) + 3 :]
            offenders += [
                f"{path.relative_to(REPO)}:{i}: {literal}"
                for i, line in enumerate(text.splitlines(), 1)
                for literal in MARKUP_LITERALS
                if literal in line
            ]
    assert offenders == []
