"""Behavior tests for miles_agent.launcher, the recipe run.py modules (alfworld, scienceworld, webshop,
search_r1, tmax) and the OPD presets recipes/<env>/opd.py layered on them.

miles (torch) is not importable on dev boxes, so miles' command_utils (U),
chat_template_utils, model_args_utils and typer are stubbed in sys.modules for
this module only (every miles module imported before is set aside meanwhile:
where the pin is importable, as in a container, the stubs still rule);
the stubs mirror the real surface the launcher touches
(ExecuteTrainConfig.create_backend() and the backend's execute_train /
convert_checkpoint, dataclass_cli, get_default_wandb_args, repo_base_dir, the
parsers bound per TITO family). tests/test_contracts.py checks that surface
against the real modules wherever miles imports (scripts/smoke_tests.sh).
"""

import dataclasses
import importlib
import importlib.util
import inspect
import json
import os
import re
import runpy
import shlex
import sys
import types
import typing
from pathlib import Path

import pytest
from conftest import MILES_TEMPLATES

REPO = Path(__file__).resolve().parent.parent
RECIPES = ("search_r1", "alfworld", "scienceworld", "webshop")
# The env recipes with an OPD preset (recipes/<env>/opd.py, the R2S paper's student arm over run.py).
OPD_RECIPES = ("alfworld", "scienceworld", "webshop")
CAPTURED: dict = {}
_STUB_KEYS = (
    "miles",
    "miles.utils",
    "miles.utils.external_utils",
    "miles.utils.external_utils.command_utils",
    "miles.utils.external_utils.model_args_utils",
    "miles.utils.chat_template_utils",
    "miles.utils.chat_template_utils.tito_tokenizer",
    "typer",
)
# What miles binds per TITO family (tito_tokenizer.py): the fixed template file under 3rdparty/miles, its extra
# kwargs, the (reasoning, tool-call) parsers; tests/test_contracts.py checks the surface against the pin.
_TITO_FAMILIES = {
    "default": (None, {}, None, None),
    "qwen3": ("qwen3_fixed.jinja", {"clear_thinking": False}, "qwen3", "qwen25"),
    "qwen35": ("qwen3.5_fixed.jinja", {"preserve_thinking": True}, "qwen3", "qwen3_coder"),
    "qwen36": ("qwen3.6_fixed.jinja", {"preserve_thinking": True}, "qwen3", "qwen3_coder"),
    "nemotron3": (None, {"truncate_history_thinking": False}, "nemotron_3", "qwen3_coder"),
}


def _make_stub_u():
    U = types.ModuleType("miles.utils.external_utils.command_utils")

    class Backend:
        # miles >= 0.1.2 BaseCommandBackend: the launch API lives on the backend
        # an ExecuteTrainConfig creates (signatures mirrored from base_backend.py).
        def __init__(self, config):
            self.config = config

        def execute_train(
            self,
            train_args,
            num_gpus_per_node,
            megatron_model_type,
            train_script="train.py",
            before_ray_job_submit=None,
            extra_env_vars=None,
            megatron_path="/root/Megatron-LM",
            job_lifetime="independent",
            prepare_cmd=None,
            extra_manifests=None,
            config=None,
        ):
            CAPTURED.clear()
            CAPTURED.update(
                train_args=train_args,
                num_gpus_per_node=num_gpus_per_node,
                megatron_model_type=megatron_model_type,
                train_script=train_script,
                extra_env_vars=dict(extra_env_vars or {}),
                config=config if config is not None else self.config,
                megatron_path=megatron_path,
            )

        def convert_checkpoint(
            self,
            model_name,
            megatron_model_type,
            num_gpus_per_node,
            multinode=False,
            num_nodes=None,
            extra_args="",
            dir_dst="/root",
            hf_checkpoint=None,
            megatron_path="/root/Megatron-LM",
        ):
            CAPTURED.update(
                convert=dict(
                    model_name=model_name,
                    megatron_model_type=megatron_model_type,
                    num_gpus_per_node=num_gpus_per_node,
                    dir_dst=dir_dst,
                    hf_checkpoint=hf_checkpoint,
                    megatron_path=megatron_path,
                )
            )

    @dataclasses.dataclass
    class ExecuteTrainConfig:
        cuda_core_dump: bool = False
        num_nodes: int = 1
        extra_env_vars: str = ""
        output_dir: str = "/root/shared_data"

        def create_backend(self):
            return Backend(self)

    def dataclass_cli(func=None, **kw):
        return func if func is not None else (lambda f: f)

    def get_default_wandb_args(test_file, run_name_prefix=None, run_id=None):
        # Deterministic mirror of miles' helper (create_run_id -> "RUNID").
        import os

        if not os.environ.get("WANDB_API_KEY"):
            return ""
        test_file = Path(test_file)
        test_name = test_file.stem
        if len(test_name) < 6:
            test_name = f"{test_file.parent.name}_{test_name}"
        name = run_id or "RUNID"
        if run_name_prefix is not None:
            name = f"{run_name_prefix}_{name}"
        return f"--use-wandb --wandb-project miles-{test_name} --wandb-group {name} --wandb-key 'k' "

    U.ExecuteTrainConfig = ExecuteTrainConfig
    U.dataclass_cli = dataclass_cli
    U.get_default_wandb_args = get_default_wandb_args
    U.repo_base_dir = Path("/opt/miles")
    return U


def _make_stub_chat_template_utils():
    ctu = types.ModuleType("miles.utils.chat_template_utils")

    def resolve_reasoning_and_tool_call_parser(tito_model, user_reasoning_parser=None, user_tool_call_parser=None):
        if tito_model not in _TITO_FAMILIES:
            raise ValueError(f"{tito_model!r} is not a valid TITOTokenizerType")
        return _TITO_FAMILIES[tito_model][2:]

    ctu.resolve_reasoning_and_tool_call_parser = resolve_reasoning_and_tool_call_parser
    ctu.tito_tokenizer = _make_stub_tito_tokenizer()
    return ctu


def _make_stub_tito_tokenizer():
    """The TITO class surface launcher.write_serve_template reads: TEMPLATE_DIR (the pin's real template files),
    TITOTokenizerType(name).get_tokenizer_class -> FIXED_TEMPLATE.template / .extra_kwargs and the two parsers."""
    tito_tokenizer = types.ModuleType("miles.utils.chat_template_utils.tito_tokenizer")
    tito_tokenizer.TEMPLATE_DIR = MILES_TEMPLATES

    @dataclasses.dataclass(frozen=True)
    class FixedTemplate:
        template: str | None = None
        extra_kwargs: dict = dataclasses.field(default_factory=dict)

    class TITOTokenizerType(str):
        def __new__(cls, value):
            if value not in _TITO_FAMILIES:
                raise ValueError(f"{value!r} is not a valid TITOTokenizerType")
            return str.__new__(cls, value)

        @classmethod
        def get_tokenizer_class(cls, t):
            template, extra_kwargs, reasoning, tool_call = _TITO_FAMILIES[str(t)]
            bound = {
                "FIXED_TEMPLATE": FixedTemplate(template, extra_kwargs),
                "reasoning_parser": reasoning,
                "tool_call_parser": tool_call,
            }
            return type(f"{t}TITOTokenizer", (), bound)

    tito_tokenizer.FixedTemplate = FixedTemplate
    tito_tokenizer.TITOTokenizerType = TITOTokenizerType
    return tito_tokenizer


@pytest.fixture(scope="module")
def stubs():
    # every miles module imported so far goes too: where the real pin is importable, an earlier import of
    # miles.utils.external_utils.model_args_utils would survive the stub packages and run _check_model_script
    # against the real pin (no scripts/models/custom.py there); all of them come back at teardown
    saved = {k: sys.modules.get(k) for k in _STUB_KEYS}
    saved.update({k: v for k, v in sys.modules.items() if k == "miles" or k.startswith("miles.")})
    for k in saved:
        sys.modules.pop(k, None)
    saved_agent = {k: sys.modules.get(k) for k in ("miles_agent.launcher",)}
    pkgs = {}
    for name in ("miles", "miles.utils", "miles.utils.external_utils"):
        m = types.ModuleType(name)
        m.__path__ = []
        pkgs[name] = m
    U = _make_stub_u()
    pkgs["miles"].utils = pkgs["miles.utils"]
    pkgs["miles.utils"].external_utils = pkgs["miles.utils.external_utils"]
    pkgs["miles.utils.external_utils"].command_utils = U
    pkgs["miles.utils"].chat_template_utils = _make_stub_chat_template_utils()
    typer = types.ModuleType("typer")
    typer.run = lambda f: None
    model_args_utils = types.ModuleType("miles.utils.external_utils.model_args_utils")
    model_args_utils._MODEL_SCRIPT_DIR = None  # _check_model_script is a no-op under the stubs (its own test sets a dir)
    sys.modules.update(pkgs)
    sys.modules["miles.utils.external_utils.command_utils"] = U
    sys.modules["miles.utils.external_utils.model_args_utils"] = model_args_utils
    ctu = pkgs["miles.utils"].chat_template_utils
    sys.modules["miles.utils.chat_template_utils"] = ctu
    sys.modules["miles.utils.chat_template_utils.tito_tokenizer"] = ctu.tito_tokenizer
    sys.modules["typer"] = typer
    sys.modules.pop("miles_agent.launcher", None)
    launcher = importlib.import_module("miles_agent.launcher")
    runs = {}
    for recipe in RECIPES:
        spec = importlib.util.spec_from_file_location(f"_run_{recipe}", REPO / "recipes" / recipe / "run.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        runs[recipe] = mod
    # A preset does `import run` for its sibling run.py (the recipe dir is sys.path[0] under the training entry, as under
    # eval_ckpt.load_recipe); here the sibling is bound by name for the load, since conftest puts search_r1's dir
    # on sys.path, and `run` is restored right after.
    saved_run = sys.modules.get("run")
    opds = {}
    for recipe in OPD_RECIPES:
        sys.modules["run"] = runs[recipe]
        spec = importlib.util.spec_from_file_location(f"_opd_{recipe}", REPO / "recipes" / recipe / "opd.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        opds[recipe] = mod
    sys.modules.pop("run", None)
    if saved_run is not None:
        sys.modules["run"] = saved_run
    yield types.SimpleNamespace(U=U, launcher=launcher, runs=runs, opds=opds)
    for recipe in RECIPES:
        sys.modules.pop(f"_run_{recipe}", None)
    for recipe in OPD_RECIPES:
        sys.modules.pop(f"_opd_{recipe}", None)
    for k, v in {**saved, **saved_agent}.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture(autouse=True)
def _no_wandb_key(monkeypatch):
    # wandb args (and the save dir's wandb_run_id file) appear only where a test sets the key itself: a
    # developer's key in the environment must not make every launch write under its save dir
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def _run(stubs, recipe, **kw):
    mod = stubs.runs[recipe]
    mod.execute(mod.ScriptArgs(**kw))
    return dict(CAPTURED)


def _tokens(cap):
    return cap["train_args"].split()


def _flag(cap, name):
    toks = _tokens(cap)
    return toks[toks.index(name) + 1]


# ---------------------------------------------------------------- recipe module surface


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_module_keeps_entry_points(stubs, recipe):
    """eval_full.py imports run and uses run.ScriptArgs / cleanup / prepare / execute(args)."""
    mod = stubs.runs[recipe]
    assert dataclasses.is_dataclass(mod.ScriptArgs)
    assert issubclass(mod.ScriptArgs, stubs.launcher.BaseScriptArgs)
    assert isinstance(mod.SPEC, stubs.launcher.RecipeSpec)
    assert mod.SPEC.name == recipe
    assert Path(mod.SPEC.run_file).resolve() == (REPO / "recipes" / recipe / "run.py").resolve()
    assert mod.cleanup is stubs.launcher.cleanup
    assert mod.prepare is stubs.launcher.prepare
    assert list(inspect.signature(mod.execute).parameters) == ["args"]
    assert callable(mod.main)


@pytest.mark.parametrize("recipe", RECIPES)
def test_dataclass_cli_surface_survives_inheritance(stubs, recipe):
    """Mirror what miles' dataclass_cli._wrap does: every __init__ parameter of the
    concrete class must be a dataclass field with resolvable type hints."""
    cls = stubs.runs[recipe].ScriptArgs
    params = [p for p in inspect.signature(cls.__init__).parameters.values() if p.name != "self"]
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    assert [p.name for p in params] == list(fields)
    for p in params:
        assert p.name in hints
        assert p.default is not inspect.Parameter.empty, f"{p.name} has no default"
    # base fields and ExecuteTrainConfig fields ride along
    for name in ("mode", "topology", "algo", "save_dir", "extra_train_args", "cuda_core_dump", "output_dir"):
        assert name in fields


def test_recipe_overrides_take_effect(stubs, monkeypatch):
    monkeypatch.delenv("MILES_AGENT_DATA", raising=False)
    monkeypatch.delenv("MILES_AGENT_RESULTS", raising=False)
    a = stubs.runs["alfworld"].ScriptArgs()
    c = stubs.runs["scienceworld"].ScriptArgs()
    s = stubs.runs["search_r1"].ScriptArgs()
    w = stubs.runs["webshop"].ScriptArgs()
    assert (a.rollout_batch_size, a.global_batch_size, a.rollout_max_response_len) == (64, 512, 512)
    assert (a.rollout_max_context_len, a.train_max_tokens_per_gpu) == (32768, 32768)
    # scienceworld keeps the alfworld campaign's episode shape
    for name in ("rollout_batch_size", "global_batch_size", "rollout_max_response_len"):
        assert getattr(c, name) == getattr(a, name), name
    assert (c.rollout_max_context_len, c.train_max_tokens_per_gpu) == (32768, 32768)
    assert (s.rollout_batch_size, s.global_batch_size, s.rollout_max_response_len) == (64, 512, 1024)
    assert (s.rollout_max_context_len, s.train_max_tokens_per_gpu) == (16384, 0)
    assert (w.rollout_batch_size, w.global_batch_size, w.rollout_max_response_len) == (16, 128, 512)
    assert (w.rollout_max_context_len, w.train_max_tokens_per_gpu) == (32768, 32768)
    # webshop pins the TCOD validation protocol; the others keep the base defaults
    assert (w.eval_temperature, w.eval_max_response_len, w.eval_interval, w.seed) == (0.4, 4096, 50, 42)
    for args in (a, c, s):
        assert (args.eval_temperature, args.eval_max_response_len, args.eval_interval, args.seed) == (1.0, 0, 50, 1234)
    # the save dirs follow the results root (launcher.results_dir): <repo>/results/<recipe>/ with it unset
    assert (a.save_dir, c.save_dir, s.save_dir, w.save_dir) == tuple(
        stubs.launcher.results_dir(recipe) for recipe in ("alfworld", "scienceworld", "search_r1", "webshop")
    )
    assert a.save_dir == f"{REPO}/results/alfworld/"
    # shared defaults come from the base
    for args in (a, c, s, w):
        assert (args.num_rollout, args.n_samples_per_prompt) == (500, 8)
        assert (args.algo, args.topology) == ("grpo", "colocated_1n")


def test_recipe_data_defaults_follow_the_miles_agent_roots(stubs, monkeypatch):
    """The recipes' --prompt-data / --eval-data / --save-dir defaults sit under MILES_AGENT_DATA and
    MILES_AGENT_RESULTS (launcher.data_path / results_dir, read when the args are built; <repo>/data and
    <repo>/results with the roots unset). The cluster entry passes all three explicitly, so these matter to a
    bare `python run.py` only; an explicit path always wins."""
    monkeypatch.setenv("MILES_AGENT_DATA", "/d")
    monkeypatch.setenv("MILES_AGENT_RESULTS", "/r")
    eval_file = {
        "alfworld": "alfworld_eval274.jsonl",
        "scienceworld": "scienceworld_eval274.jsonl",
        "search_r1": "search_r1_eval893.jsonl",
        "webshop": "webshop_eval.jsonl",
    }
    for recipe in RECIPES:
        args = stubs.runs[recipe].ScriptArgs()
        assert (args.prompt_data, args.eval_data, args.save_dir) == (
            f"/d/{recipe}_train.jsonl",
            f"/d/{eval_file[recipe]}",
            f"/r/{recipe}/",
        ), recipe
    assert stubs.launcher.data_path("x.jsonl") == "/d/x.jsonl"
    assert stubs.launcher.results_dir("alfworld") == "/r/alfworld/"
    # the platform recipes (ScriptArgs(LaunchArgs)) follow the same roots
    for recipe, train in (("tmax", "tmax_train.jsonl"),):
        spec = importlib.util.spec_from_file_location(f"_run_{recipe}", REPO / "recipes" / recipe / "run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = mod.ScriptArgs()
        assert (args.prompt_data, args.save_dir) == (f"/d/{train}", f"/r/{recipe}/"), recipe
    # an explicit path still wins and reaches the train args as given
    cap = _run(stubs, "alfworld", save_dir="/s/", prompt_data="/p/train.jsonl", eval_data="/p/eval.jsonl")
    assert (_flag(cap, "--save"), _flag(cap, "--prompt-data")) == ("/s/", "/p/train.jsonl")
    monkeypatch.delenv("MILES_AGENT_DATA")
    monkeypatch.delenv("MILES_AGENT_RESULTS")
    args = stubs.runs["webshop"].ScriptArgs()
    assert (args.prompt_data, args.save_dir) == (f"{REPO}/data/webshop_train.jsonl", f"{REPO}/results/webshop/")


def test_execute_refuses_a_recipe_without_eval_slices(stubs):
    """--eval-prompt-data takes one or more values: with no slices it would go out empty and miles' argparse would
    refuse it at train time, after the allocation."""
    alfworld = stubs.runs["alfworld"]
    spec = stubs.launcher.RecipeSpec(name="x", run_file=__file__)
    with pytest.raises(ValueError, match="eval_slices"):
        stubs.launcher.execute(alfworld.ScriptArgs(), spec)


def test_execute_refuses_empty_recipe_paths(stubs):
    args = stubs.launcher.BaseScriptArgs()
    spec = stubs.launcher.RecipeSpec(name="x", run_file=__file__, eval_slices=(("x", ""),))
    with pytest.raises(ValueError, match="save_dir"):
        stubs.launcher.execute(args, spec)


# ---------------------------------------------------------------- command content


@pytest.mark.parametrize(
    ("recipe", "algo", "expected"),
    [
        ("alfworld", "grpo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("alfworld", "gspo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("alfworld", "ppo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("alfworld", "dapo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure_nonzero_std"),
        ("scienceworld", "grpo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("scienceworld", "dapo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure_nonzero_std"),
        ("search_r1", "grpo", "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"),
        ("search_r1", "dapo", "miles_agent.dapo_filter.check_no_aborted_nonzero_std"),
        ("webshop", "grpo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("webshop", "gspo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("webshop", "ppo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"),
        ("webshop", "dapo", "miles_agent.dapo_filter.check_no_aborted_no_transport_failure_nonzero_std"),
    ],
)
def test_dynamic_sampling_filter_paths(stubs, recipe, algo, expected):
    cap = _run(stubs, recipe, algo=algo)
    assert _flag(cap, "--dynamic-sampling-filter-path") == expected
    # no recipe-local dapo_filter module may be referenced any more
    assert not any(t.startswith("dapo_filter.") for t in _tokens(cap))


def test_filter_paths_resolve_to_functions_in_dapo_filter_module():
    src = (REPO / "miles_agent" / "dapo_filter.py").read_text()
    for fn in (
        "check_no_aborted_nonzero_std",
        "check_no_aborted_no_transport_failure",
        "check_no_aborted_no_transport_failure_nonzero_std",
    ):
        assert f"def {fn}(" in src
    for recipe in RECIPES:
        assert not (REPO / "recipes" / recipe / "dapo_filter.py").exists()


def test_context_budget_and_packing(stubs):
    cap = _run(stubs, "webshop")
    assert _flag(cap, "--rollout-max-context-len") == "32768"
    assert _flag(cap, "--max-tokens-per-gpu") == "32768"
    cap = _run(stubs, "search_r1")
    assert _flag(cap, "--rollout-max-context-len") == "16384"
    assert _flag(cap, "--max-tokens-per-gpu") == "16384"
    assert _flag(cap, "--rollout-max-response-len") == "1024"
    # search_r1 keeps its coupled semantics: packing follows the context
    cap = _run(stubs, "search_r1", rollout_max_context_len=8192)
    assert _flag(cap, "--max-tokens-per-gpu") == "8192"
    # alfworld/webshop pin packing explicitly: it stays put when the context moves
    cap = _run(stubs, "alfworld", rollout_max_context_len=16384)
    assert _flag(cap, "--max-tokens-per-gpu") == "32768"


def test_batch_shape(stubs):
    cap = _run(stubs, "webshop")
    assert (_flag(cap, "--rollout-batch-size"), _flag(cap, "--global-batch-size")) == ("16", "128")
    cap = _run(stubs, "alfworld")
    assert (_flag(cap, "--rollout-batch-size"), _flag(cap, "--global-batch-size")) == ("64", "512")


def test_eval_slices(stubs):
    def eval_block(cap):
        toks = _tokens(cap)
        i = toks.index("--eval-prompt-data") + 1
        j = toks.index("--n-samples-per-eval-prompt")
        return toks[i:j]

    # webshop default = the canonical layout (WEBSHOP_PROTOCOL unset): fixed
    # held-out test idx 0..499 + the 256-goal pool sample
    assert eval_block(_run(stubs, "webshop", eval_data="/d/e.jsonl")) == [
        "webshop",
        "/d/e.jsonl@[0:500]",
        "webshop_pool",
        "/d/e.jsonl@[500:756]",
    ]
    assert eval_block(_run(stubs, "alfworld", eval_data="/d/e.jsonl")) == [
        "alfworld_seen",
        "/d/e.jsonl@[0:140]",
        "alfworld_unseen",
        "/d/e.jsonl@[140:274]",
    ]
    assert eval_block(_run(stubs, "scienceworld", eval_data="/d/e.jsonl")) == [
        "scienceworld_seen",
        "/d/e.jsonl@[0:140]",
        "scienceworld_unseen",
        "/d/e.jsonl@[140:274]",
    ]
    sr = eval_block(_run(stubs, "search_r1", eval_data="/d/e.jsonl"))
    assert sr[0:2] == ["nq", "/d/e.jsonl@[0:128]"] and sr[-2:] == ["bamboogle", "/d/e.jsonl@[768:893]"]
    assert len(sr) == 14


def test_webshop_eval_slices_follow_protocol(stubs, monkeypatch):
    # recipes/webshop/run.py reads WEBSHOP_PROTOCOL at import (the training
    # entry exports the same variable it prepared the data with).
    def load(protocol):
        monkeypatch.setenv("WEBSHOP_PROTOCOL", protocol)
        spec = importlib.util.spec_from_file_location(
            f"_run_webshop_{protocol}", REPO / "recipes" / "webshop" / "run.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    assert load("canonical").SPEC.eval_slices == (("webshop", "@[0:500]"), ("webshop_pool", "@[500:756]"))
    assert load("tcod").SPEC.eval_slices == (("webshop", "@[0:100]"), ("webshop_heldout", "@[100:500]"))
    with pytest.raises(SystemExit):
        load("bogus")


def test_eval_temperature_knob(stubs):
    # Base default validates at the training temperature; webshop pins the
    # verl-agent/TCOD validation temperature 0.4 without touching rollout.
    cap = _run(stubs, "alfworld")
    assert (_flag(cap, "--eval-temperature"), _flag(cap, "--rollout-temperature")) == ("1.0", "1.0")
    cap = _run(stubs, "webshop")
    assert (_flag(cap, "--eval-temperature"), _flag(cap, "--rollout-temperature")) == ("0.4", "1.0")
    cap = _run(stubs, "webshop", eval_temperature=1.0)
    assert (_flag(cap, "--eval-temperature"), _flag(cap, "--rollout-temperature")) == ("1.0", "1.0")


def test_seed_and_eval_response_cap_knobs(stubs):
    # TCOD protocol: seed 42, validation generates up to 4096 tokens/turn while
    # training keeps 512. Other recipes: miles' seed 1234 and no eval cap flag
    # (eval falls back to the rollout cap).
    cap = _run(stubs, "webshop")
    assert (_flag(cap, "--seed"), _flag(cap, "--eval-max-response-len")) == ("42", "4096")
    assert (_flag(cap, "--rollout-max-response-len"), _flag(cap, "--eval-interval")) == ("512", "50")
    cap = _run(stubs, "alfworld")
    assert _flag(cap, "--seed") == "1234" and "--eval-max-response-len" not in _tokens(cap)
    cap = _run(stubs, "webshop", eval_max_response_len=0, seed=7)
    assert _flag(cap, "--seed") == "7" and "--eval-max-response-len" not in _tokens(cap)


def test_topology_and_algo_blocks(stubs):
    cap = _run(stubs, "alfworld", topology="colocated_1n", algo="ppo")
    assert cap["train_script"] == "train.py"
    assert "--colocate" in _tokens(cap) and "--keep-old-actor" not in _tokens(cap)
    cap = _run(stubs, "alfworld", topology="async_3n", algo="ppo")
    assert cap["train_script"] == "train_async.py"
    assert "--keep-old-actor" in _tokens(cap) and "--pin-rollout-manager-to-head" in _tokens(cap)
    assert _flag(cap, "--update-weights-interval") == "2"
    cap = _run(stubs, "alfworld", topology="fully_async_3n", algo="grpo")
    assert "--fully-async" in _tokens(cap) and _flag(cap, "--max-weight-staleness") == "2"
    assert "--async-max-concurrent-samples" not in _tokens(cap)
    cap = _run(stubs, "alfworld", topology="fully_async_3n", algo="dapo")
    assert _flag(cap, "--async-max-concurrent-samples") == "128"
    assert _flag(cap, "--advantage-estimator") == "grpo"
    cap = _run(stubs, "alfworld", algo="gspo")
    assert _flag(cap, "--eps-clip") == "4e-4" and "--use-kl-loss" not in _tokens(cap)
    cap = _run(stubs, "alfworld", topology="async_4n_teacher", algo="opd", teacher_url="http://t:13141")
    assert cap["train_script"] == "train_async.py" and "--use-opd" in _tokens(cap)
    assert _flag(cap, "--rollout-num-gpus") == "8" and _flag(cap, "--update-weights-interval") == "2"
    cap = _run(stubs, "alfworld", topology="async_2n_teacher", algo="opd", teacher_url="http://t:13141")
    assert cap["train_script"] == "train_async.py" and _flag(cap, "--rollout-num-gpus") == "2"
    cap = _run(stubs, "alfworld", mode="debug_rollout_only", use_tis=True, extra_train_args="--foo 1")
    toks = _tokens(cap)
    assert "--debug-rollout-only" in toks and "--use-tis" in toks and toks[-2:] == ["--foo", "1"]


def test_resume_logic(stubs, tmp_path):
    save = tmp_path / "arm"
    save.mkdir()
    cap = _run(stubs, "alfworld", save_dir=f"{save}/")
    assert "--load" not in _tokens(cap)
    (save / "latest_checkpointed_iteration.txt").write_text("10")
    cap = _run(stubs, "alfworld", save_dir=f"{save}/")
    assert _flag(cap, "--load") == f"{save}/"
    # PPO with an actor tracker but no critic tracker starts fresh
    cap = _run(stubs, "alfworld", save_dir=f"{save}/", algo="ppo")
    assert "--load" not in _tokens(cap) and "--critic-load" not in _tokens(cap)
    critic = tmp_path / "arm_critic"
    critic.mkdir()
    (critic / "latest_checkpointed_iteration.txt").write_text("10")
    cap = _run(stubs, "alfworld", save_dir=f"{save}/", algo="ppo")
    assert _flag(cap, "--load") == f"{save}/" and _flag(cap, "--critic-load") == str(critic)


def test_wandb_run_id_persists_across_links(stubs, tmp_path, monkeypatch):
    """Every launch of one save dir (chain links, crash restarts) passes the same --wandb-run-id, drawn once
    and kept in <save_dir>/wandb_run_id, so wandb resumes one run; a fresh save dir gets a fresh id."""
    monkeypatch.setenv("WANDB_API_KEY", "k")
    save = tmp_path / "arm"
    cap = _run(stubs, "alfworld", save_dir=f"{save}/")
    run_id = _flag(cap, "--wandb-run-id")
    assert re.fullmatch(r"[a-z0-9]{8}", run_id) and _flag(cap, "--wandb-group") == "arm"
    assert (save / "wandb_run_id").read_text() == f"{run_id}\n"
    (save / "latest_checkpointed_iteration.txt").write_text("10")
    cap = _run(stubs, "alfworld", save_dir=f"{save}/")  # the next link: resume, same run
    assert _flag(cap, "--load") == f"{save}/" and _flag(cap, "--wandb-run-id") == run_id
    assert (save / "wandb_run_id").read_text() == f"{run_id}\n"
    # a hand-written id (a run to attach to) is kept as it is
    other = tmp_path / "arm2"
    other.mkdir()
    (other / "wandb_run_id").write_text("abc123xy\n")
    assert _flag(_run(stubs, "alfworld", save_dir=f"{other}/"), "--wandb-run-id") == "abc123xy"
    # a file that is no run id is replaced, with a warning
    (other / "wandb_run_id").write_text("not a/run id\n")
    drawn = _flag(_run(stubs, "alfworld", save_dir=f"{other}/"), "--wandb-run-id")
    assert re.fullmatch(r"[a-z0-9]{8}", drawn) and (other / "wandb_run_id").read_text() == f"{drawn}\n"
    # another save dir draws its own id
    assert _flag(_run(stubs, "alfworld", save_dir=f"{tmp_path}/arm3/"), "--wandb-run-id") != run_id
    # wandb off (no key): no wandb args and nothing written
    monkeypatch.delenv("WANDB_API_KEY")
    cap = _run(stubs, "alfworld", save_dir=f"{tmp_path}/arm4/")
    assert "--wandb-run-id" not in _tokens(cap) and "--use-wandb" not in _tokens(cap)
    assert not (tmp_path / "arm4").exists()


def test_preempt_sentinel_reaches_miles(stubs):
    assert "--preempt-sentinel" not in _tokens(_run(stubs, "alfworld"))
    cap = _run(stubs, "alfworld", preempt_sentinel="/sync/preempt")
    assert _flag(cap, "--preempt-sentinel") == "/sync/preempt" and "--save" in _tokens(cap)
    assert "--preempt-sentinel" not in _tokens(_run(stubs, "alfworld", preempt_sentinel="/sync/preempt", save_interval=0))  # miles asserts --save next to the sentinel


def test_cleanup_spares_the_external_ray_head(stubs, monkeypatch):
    # The v0.1.2 image's Python lives under /opt/sglang, so `pgrep -f sglang`
    # killed the job's own Ray head and `ray job submit` found no dashboard.
    cmds = []
    monkeypatch.setattr(stubs.launcher.subprocess, "run", lambda cmd, **kw: cmds.append(cmd))
    monkeypatch.setattr(stubs.launcher.time, "sleep", lambda s: None)
    stubs.launcher.cleanup()
    assert len(cmds) == len(stubs.launcher.CLEANUP_PGREP)
    for cmd in cmds:
        argv = shlex.split(cmd.split("|")[0])
        assert argv[0] == "pgrep"
        assert not ("-f" in argv and any("sglang" in a for a in argv)), cmd


def test_shared_science_flags_present(stubs):
    for recipe in RECIPES:
        cap = _run(stubs, recipe)
        toks = _tokens(cap)
        for flag, value in (
            ("--lr", "1e-6"),
            ("--lr-decay-iters", "10000"),
            ("--lr-warmup-iters", "142"),
            ("--adam-beta2", "0.98"),
            ("--eps-clip-high", "0.28"),
            ("--kl-loss-coef", "0.001"),
            ("--session-server-port", "7100"),
            ("--session-server-ip", "0.0.0.0"),
            ("--sglang-router-port", "7200"),
            ("--custom-agent-function-mode", "inline"),
            ("--sglang-tool-call-parser", "qwen3_coder"),
            ("--tito-model", "qwen35"),
            ("--num-gpus-per-node", "4"),
            ("--reward-key", "train"),
            ("--eval-reward-key", "env"),
            ("--custom-agent-function-path", "miles_agent.gym_bridge.run"),
            ("--custom-rm-path", "miles_agent.reward_routing.reward_func"),
        ):
            assert _flag(cap, flag) == value, (recipe, flag)
        assert "--observe-training-entropy" in toks and "--use-session-server" in toks
        assert cap["megatron_model_type"] == "qwen3.5-4B" and cap["num_gpus_per_node"] == 4
        argv = shlex.split(cap["train_args"])
        assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": false}'


def test_topology_strings_scale_with_gpus_and_nodes(stubs):
    """--num-gpus-per-node and --num-nodes (the run's ray nodes; 0 = the topology's own count) shape the
    ray-side split; at the defaults every string is today's byte for byte (the golden digests stay), and
    --num-nodes itself never reaches miles: train.py has no such flag."""
    launcher = stubs.launcher
    make = stubs.runs["alfworld"].ScriptArgs

    def value(text, name):
        toks = text.split()
        return toks[toks.index(name) + 1]

    assert launcher._topology_args(make(topology="colocated_1n", num_gpus_per_node=8)) == (
        "--colocate --actor-num-nodes 1 --actor-num-gpus-per-node 8 "
    )
    three = launcher._topology_args(make(topology="async_3n", num_gpus_per_node=8, num_nodes=3))
    assert three == (
        "--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rollout-num-gpus 16 --pin-rollout-manager-to-head "
        "--update-weights-interval 2 "
    )
    assert launcher._topology_args(make(topology="async_4n_teacher", num_gpus_per_node=4, num_nodes=0)) == (
        launcher._DISAGG_ARGS + "--update-weights-interval 2 "
    )
    assert launcher._DISAGG_ARGS == launcher._disagg_args(4, 3) == (
        "--actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus 8 --pin-rollout-manager-to-head "
    )
    two = launcher._topology_args(make(topology="async_2n_teacher", num_gpus_per_node=8))
    assert (value(two, "--actor-num-gpus-per-node"), value(two, "--rollout-num-gpus")) == ("4", "4")
    five = launcher._topology_args(make(topology="fully_async_3n", num_gpus_per_node=8, num_nodes=5))
    assert five.startswith(
        "--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rollout-num-gpus 32 --pin-rollout-manager-to-head "
        "--fully-async --max-weight-staleness "
    )
    with pytest.raises(ValueError, match="--num-nodes must be >= 2 for async_3n"):
        launcher._topology_args(make(topology="async_3n", num_nodes=1))
    with pytest.raises(ValueError, match="--num-nodes must be >= 2 for fully_async_3n"):
        launcher._topology_args(make(topology="fully_async_3n", num_nodes=1))
    with pytest.raises(ValueError, match="--num-gpus-per-node must be >= 2 for async_2n_teacher"):
        launcher._topology_args(make(topology="async_2n_teacher", num_gpus_per_node=1))
    # through the launch: the derived split and the explicit --num-gpus-per-node, never a --num-nodes token
    cap = _run(stubs, "alfworld", topology="async_3n", num_gpus_per_node=8, num_nodes=3)
    toks = _tokens(cap)
    assert "--num-nodes" not in toks
    assert (_flag(cap, "--rollout-num-gpus"), _flag(cap, "--num-gpus-per-node")) == ("16", "8")
    assert cap["num_gpus_per_node"] == 8 and cap["train_script"] == "train_async.py"
    assert "--num-nodes" not in _tokens(_run(stubs, "alfworld"))


def test_num_gpus_per_node_reaches_execute_train_unchanged(stubs):
    """--num-gpus-per-node is handed to miles' execute_train as given and drives the colocated split and
    misc_args' own flag; the default stays 4."""
    cap = _run(stubs, "alfworld", num_gpus_per_node=8)
    assert cap["num_gpus_per_node"] == 8
    assert (_flag(cap, "--num-gpus-per-node"), _flag(cap, "--actor-num-gpus-per-node")) == ("8", "8")
    cap = _run(stubs, "search_r1")
    assert cap["num_gpus_per_node"] == 4 and _flag(cap, "--actor-num-gpus-per-node") == "4"


def test_model_family_follows_the_model_name(stubs):
    """One knob: --model-name picks the megatron model type, the TITO family, the
    chat template kwargs and, through the TITO family, the engines' parser."""
    alfworld = stubs.runs["alfworld"]
    args = alfworld.ScriptArgs()
    assert (args.model_name, args.megatron_model_type, args.tito_model, args.chat_template_kwargs) == (
        "Qwen3.5-4B",
        "qwen3.5-4B",
        "qwen35",
        '{"enable_thinking": false}',
    )
    cap = _run(stubs, "alfworld", model_name="Qwen3-4B-Instruct-2507")
    assert cap["megatron_model_type"] == "qwen3-4B-Instruct-2507"
    assert _flag(cap, "--tito-model") == "qwen3" and _flag(cap, "--sglang-tool-call-parser") == "qwen25"
    assert "--apply-chat-template-kwargs" not in _tokens(cap)
    cap = _run(stubs, "alfworld", model_name="Qwen3.6-27B")
    assert cap["megatron_model_type"] == "qwen3.6-27B"
    assert _flag(cap, "--tito-model") == "qwen36" and _flag(cap, "--sglang-tool-call-parser") == "qwen3_coder"
    argv = shlex.split(cap["train_args"])
    assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    # explicit flags win over the family's
    cap = _run(stubs, "alfworld", model_name="Qwen3.5-9B", megatron_model_type="custom", chat_template_kwargs="")
    assert cap["megatron_model_type"] == "custom" and _flag(cap, "--tito-model") == "qwen35"
    assert "--apply-chat-template-kwargs" not in _tokens(cap)
    with pytest.raises(ValueError, match="MODEL_FAMILIES"):
        alfworld.ScriptArgs(model_name="Llama-3-8B", megatron_model_type="llama3-8B")
    with pytest.raises(ValueError, match="--megatron-model-type"):
        alfworld.ScriptArgs(model_name="Llama-3-8B", tito_model="default", chat_template_kwargs="")
    with pytest.raises(ValueError, match="binds no tool-call parser"):
        _run(
            stubs,
            "alfworld",
            model_name="Llama-3-8B",
            megatron_model_type="m",
            tito_model="default",
            chat_template_kwargs="",
        )


def test_model_type_must_have_a_script_in_the_pin(stubs, monkeypatch, tmp_path):
    """A derived type ("Qwen3.5-4B-sft" -> qwen3.5-4B-sft) is refused at the flags when the pin has no such
    scripts/models file, naming the override; the check needs the pin's module and is skipped without it."""
    alfworld = stubs.runs["alfworld"]
    (tmp_path / "qwen3.5-4B.py").write_text("")
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.external_utils.model_args_utils",
        types.SimpleNamespace(_MODEL_SCRIPT_DIR=tmp_path),
    )
    assert alfworld.ScriptArgs(model_name="Qwen3.5-4B").megatron_model_type == "qwen3.5-4B"
    with pytest.raises(ValueError, match="--megatron-model-type"):
        alfworld.ScriptArgs(model_name="Qwen3.5-4B-sft")
    assert alfworld.ScriptArgs(model_name="Qwen3.5-4B-sft", megatron_model_type="qwen3.5-4B").megatron_model_type
    monkeypatch.delitem(sys.modules, "miles.utils.external_utils.model_args_utils")
    assert alfworld.ScriptArgs(model_name="Qwen3.5-4B-sft").megatron_model_type == "qwen3.5-4B-sft"


def test_moe_family_reads_the_hf_checkpoint_through_the_bridge(stubs):
    """Nemotron-3.5-Lightning: Megatron-Bridge reads the HF checkpoint (no torch_dist twin, the HF dir is the
    reference), the experts spread over a node's GPUs, the engines' routing replayed; a dense family keeps EP 1."""
    alfworld = stubs.runs["alfworld"]
    name = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    args = alfworld.ScriptArgs(model_name=name, hf_checkpoint="/m/nemo")
    assert (args.megatron_model_type, args.tito_model, args.chat_template_kwargs) == (
        "nemotron-3-nano-30b-a3b",
        "nemotron3",
        '{"enable_thinking": false}',
    )
    assert (args.megatron_to_hf_mode, args.expert_parallel, args.megatron_attention_backend, args.moe) == (
        "bridge",
        4,
        "auto",
        True,
    )
    CAPTURED.clear()
    alfworld.prepare(args)
    assert "convert" not in CAPTURED
    cap = _run(stubs, "alfworld", model_name=name, hf_checkpoint="/m/nemo", ref_load="/b/twin")
    assert cap["megatron_model_type"] == "nemotron-3-nano-30b-a3b"
    assert (_flag(cap, "--megatron-to-hf-mode"), _flag(cap, "--ref-load")) == ("bridge", "/m/nemo")
    assert (_flag(cap, "--expert-model-parallel-size"), _flag(cap, "--moe-token-dispatcher-type")) == ("4", "alltoall")
    assert "--use-rollout-routing-replay" in _tokens(cap) and _flag(cap, "--attention-backend") == "auto"
    # in-flight requests keep their routing across a weight update (no retract and re-prefill)
    assert _flag(cap, "--pause-generation-mode") == "in_place"
    assert _flag(cap, "--tito-model") == "nemotron3" and _flag(cap, "--sglang-tool-call-parser") == "qwen3_coder"
    # the engines' MoE kernels route through sglang's top-k, which records the routing the trainer replays
    assert _flag(cap, "--sglang-moe-runner-backend") == "flashinfer_cutlass"
    # explicit flags win: one expert a GPU group of 2, flash attention, sglang's MoE kernels
    cap = _run(
        stubs,
        "alfworld",
        model_name=name,
        expert_parallel=2,
        megatron_attention_backend="flash",
        sglang_moe_runner_backend="auto",
    )
    assert (_flag(cap, "--expert-model-parallel-size"), _flag(cap, "--attention-backend")) == ("2", "flash")
    assert _flag(cap, "--sglang-moe-runner-backend") == "auto"
    # without expert parallelism no dispatcher; the routing is replayed all the same
    cap = _run(stubs, "alfworld", model_name=name, expert_parallel=1)
    assert "--moe-token-dispatcher-type" not in _tokens(cap) and "--use-rollout-routing-replay" in _tokens(cap)
    # a dense family: its torch_dist twin, EP 1, flash attention, no replay
    cap = _run(stubs, "alfworld", ref_load="/b/twin")
    tokens = _tokens(cap)
    assert _flag(cap, "--ref-load") == "/b/twin" and "--megatron-to-hf-mode" not in tokens
    assert _flag(cap, "--expert-model-parallel-size") == "1" and _flag(cap, "--attention-backend") == "flash"
    assert not {
        "--moe-token-dispatcher-type",
        "--use-rollout-routing-replay",
        "--pause-generation-mode",
        "--sglang-moe-runner-backend",
    } & set(tokens)
    with pytest.raises(ValueError, match="raw or bridge"):
        alfworld.ScriptArgs(megatron_to_hf_mode="hf")
    # TMax on Nemotron: thinking on (the recipe's), EP over the 8-GPU training node
    spec = importlib.util.spec_from_file_location("_run_tmax", REPO / "recipes" / "tmax" / "run.py")
    tmax = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tmax)
    tmax.execute(tmax.ScriptArgs(run_tag="t", num_gpus_per_node=8, model_name=name, hf_checkpoint="/m/nemo"))
    cap = dict(CAPTURED)
    assert (_flag(cap, "--megatron-to-hf-mode"), _flag(cap, "--expert-model-parallel-size")) == ("bridge", "8")
    assert "--use-rollout-routing-replay" in _tokens(cap) and _flag(cap, "--tito-model") == "nemotron3"
    assert _flag(cap, "--sglang-moe-runner-backend") == "flashinfer_cutlass"
    argv = shlex.split(cap["train_args"])
    assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": true}'


def test_qwen_moe_checkpoints_by_their_name(stubs):
    """Qwen names a mixture of experts by its active parameters (-A3B). The Qwen rows are dense, so such a name
    would otherwise train without expert parallelism or the engines' routing (a silent mismatch, miles
    docs/models/nemotron); is_moe reads the marker, and the TITO family stays the row's."""
    alfworld = stubs.runs["alfworld"]
    for name, tito in (
        ("Qwen3-30B-A3B-Instruct-2507", "qwen3"),
        ("Qwen3.5-35B-A3B", "qwen35"),
        ("Qwen3.6-35B-A3B", "qwen36"),
    ):
        args = alfworld.ScriptArgs(model_name=name)
        assert (args.moe, args.expert_parallel, args.tito_model, args.megatron_to_hf_mode) == (True, 4, tito, "raw")
        cap = _run(stubs, "alfworld", model_name=name)
        assert (_flag(cap, "--expert-model-parallel-size"), _flag(cap, "--moe-token-dispatcher-type")) == (
            "4",
            "alltoall",
        )
        assert "--use-rollout-routing-replay" in _tokens(cap) and _flag(cap, "--pause-generation-mode") == "in_place"
        assert "--sglang-moe-runner-backend" not in _tokens(cap)  # sglang's pick of the MoE kernels
    # the marker is the active-parameter count alone
    for name in ("Qwen3.5-4B", "Qwen3-4B-Instruct-2507", "Qwen3.5-9B-AWQ"):
        args = alfworld.ScriptArgs(model_name=name)
        assert (args.moe, args.expert_parallel) == (False, 1), name


def test_fine_tuned_teacher_dir_resolves_its_family(stubs):
    """The row of the longest key prefixing the name; else, for a fine-tuned checkpoint's dir (the paper's
    teachers, r2s-<env>-<teacher>-teacher-for-<student>), the row of the unique longest key found anywhere in it, whatever
    the case; two keys of one length raise."""
    launcher = stubs.launcher
    families = launcher.MODEL_FAMILIES
    assert launcher.model_family("r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b") is families["Qwen3.5-"]
    # the Qwen3 pairs' teacher: "Qwen3-" is the one key found in the name (the Instruct-2507 row needs its full name)
    assert launcher.model_family("r2s-webshop-qwen3-30b-a3b-teacher-for-qwen3-8b") is families["Qwen3-"]
    teacher = "r2s-scienceworld-Qwen3-30B-A3B-Instruct-2507-teacher"
    assert launcher.model_family(teacher) is families["Qwen3-30B-A3B-Instruct-2507"]
    # the longest prefix: the 2507 row over Qwen3-, which keeps every other Qwen3 checkpoint
    assert launcher.model_family("Qwen3-30B-A3B-Instruct-2507") is families["Qwen3-30B-A3B-Instruct-2507"]
    assert launcher.model_family("Qwen3-30B-A3B-Instruct-2507-sft") is families["Qwen3-30B-A3B-Instruct-2507"]
    assert launcher.model_family("Qwen3-30B-A3B") is families["Qwen3-"]
    assert launcher.model_family("Qwen3-8B") is families["Qwen3-"]
    assert launcher._listed_family("Llama-3-8B") is None
    with pytest.raises(ValueError, match="more than one model family"):
        launcher.model_family("merge-of-qwen3.5-4b-and-qwen3.6-27b")
    # the fine-tune's family flows into the args; its megatron type, were it trained, cannot be derived
    alfworld = stubs.runs["alfworld"]
    args = alfworld.ScriptArgs(model_name="r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b", megatron_model_type="qwen3.5-4B")
    assert (args.tito_model, args.chat_template_kwargs, args.moe) == ("qwen35", '{"enable_thinking": false}', False)
    with pytest.raises(ValueError, match="--megatron-model-type"):
        alfworld.ScriptArgs(model_name="r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b")


def test_qwen3_30b_a3b_family_is_moe(stubs):
    """The paper's teacher of the Qwen3 students: a Qwen3 MoE on the pin's qwen3-30B-A3B script (there is none of
    the checkpoint's name) with its rope theta 1e7 handed to that script through MODEL_ARGS_ROTARY_BASE."""
    launcher = stubs.launcher
    family = launcher.MODEL_FAMILIES["Qwen3-30B-A3B-Instruct-2507"]
    assert (family.tito_model, family.chat_template_kwargs, family.moe) == ("qwen3", "", True)
    assert family.megatron_model_type == "qwen3-30B-A3B" and family.env == {"MODEL_ARGS_ROTARY_BASE": "10000000"}
    assert not any(other.env for key, other in launcher.MODEL_FAMILIES.items() if key != "Qwen3-30B-A3B-Instruct-2507")
    args = stubs.runs["alfworld"].ScriptArgs(model_name="Qwen3-30B-A3B-Instruct-2507")
    assert (args.megatron_model_type, args.tito_model, args.chat_template_kwargs) == ("qwen3-30B-A3B", "qwen3", "")
    assert (args.moe, args.expert_parallel, args.megatron_to_hf_mode) == (True, 4, "raw")
    cap = _run(stubs, "alfworld", model_name="Qwen3-30B-A3B-Instruct-2507")
    assert cap["megatron_model_type"] == "qwen3-30B-A3B"
    assert _flag(cap, "--tito-model") == "qwen3" and "--apply-chat-template-kwargs" not in _tokens(cap)
    assert _flag(cap, "--expert-model-parallel-size") == "4" and "--use-rollout-routing-replay" in _tokens(cap)
    models = REPO / "3rdparty" / "miles" / "scripts" / "models"
    if models.is_dir():
        assert not (models / "qwen3-30B-A3B-Instruct-2507.py").exists()
        assert "MODEL_ARGS_ROTARY_BASE" in (models / "qwen3-30B-A3B.py").read_text()


def test_family_env_reaches_launch_env(stubs, monkeypatch):
    """ModelFamily.env lands in the driver's environment (miles builds the model args there) and in the actors'
    extra_env_vars; a value the job set wins in both; launch()'s caller still wins; a family without env sets
    nothing."""
    name = "Qwen3-30B-A3B-Instruct-2507"
    monkeypatch.setenv("MODEL_ARGS_ROTARY_BASE", "5000000")
    cap = _run(stubs, "alfworld", model_name=name)
    assert cap["extra_env_vars"]["MODEL_ARGS_ROTARY_BASE"] == "5000000"
    assert os.environ["MODEL_ARGS_ROTARY_BASE"] == "5000000"
    monkeypatch.delenv("MODEL_ARGS_ROTARY_BASE")
    cap = _run(stubs, "alfworld", model_name=name)
    assert cap["extra_env_vars"]["MODEL_ARGS_ROTARY_BASE"] == "10000000"
    assert os.environ["MODEL_ARGS_ROTARY_BASE"] == "10000000"
    monkeypatch.delenv("MODEL_ARGS_ROTARY_BASE")
    alfworld = stubs.runs["alfworld"]
    caller_env = {"MODEL_ARGS_ROTARY_BASE": "7"}
    stubs.launcher.launch(alfworld.ScriptArgs(model_name=name), alfworld.SPEC, "", extra_env_vars=caller_env)
    assert CAPTURED["extra_env_vars"]["MODEL_ARGS_ROTARY_BASE"] == "7"
    monkeypatch.delenv("MODEL_ARGS_ROTARY_BASE")
    assert "MODEL_ARGS_ROTARY_BASE" not in _run(stubs, "alfworld")["extra_env_vars"]
    assert "MODEL_ARGS_ROTARY_BASE" not in os.environ


def test_prepare_applies_family_env(stubs, monkeypatch):
    """prepare()'s conversion evaluates the family's scripts/models file in this process (base_backend.py
    convert_checkpoint -> shell_safe_model_args), long before launch(): ModelFamily.env is in os.environ by the time
    the backend converts, a value the job set wins there too, and a family without env sets nothing."""
    launcher = stubs.launcher
    name = "Qwen3-30B-A3B-Instruct-2507"
    seen = []

    def convert(self, **kwargs):  # the heavy step: torchrun convert_hf_to_torch_dist.py on the allocation
        seen.append(os.environ.get("MODEL_ARGS_ROTARY_BASE"))

    monkeypatch.setattr(type(stubs.U.ExecuteTrainConfig().create_backend()), "convert_checkpoint", convert)
    alfworld = stubs.runs["alfworld"]
    monkeypatch.setenv("MODEL_ARGS_ROTARY_BASE", "5000000")
    launcher.prepare(alfworld.ScriptArgs(model_name=name))
    assert seen == ["5000000"] and os.environ["MODEL_ARGS_ROTARY_BASE"] == "5000000"
    monkeypatch.delenv("MODEL_ARGS_ROTARY_BASE")
    launcher.prepare(alfworld.ScriptArgs(model_name=name))
    assert seen == ["5000000", "10000000"] and os.environ["MODEL_ARGS_ROTARY_BASE"] == "10000000"
    monkeypatch.delenv("MODEL_ARGS_ROTARY_BASE")
    launcher.prepare(alfworld.ScriptArgs())
    assert seen == ["5000000", "10000000", None] and "MODEL_ARGS_ROTARY_BASE" not in os.environ


@pytest.mark.parametrize(
    ("tito", "template_name"), [("qwen3", "qwen3_fixed.jinja"), ("qwen35", "qwen3.5_fixed.jinja")]
)
def test_write_serve_template_defaults_thinking_off(stubs, template_render, tmp_path, capsys, tito, template_name):
    """The served template (the R2S teacher and judge servers, any serve-template caller) renders thinking off when a
    request says nothing, as MODEL_FAMILIES' enable_thinking false does in training; an explicit request value
    still wins; "template" keeps the file as it is; the parser returned is the family's."""
    launcher = stubs.launcher
    path = MILES_TEMPLATES / template_name
    if not path.is_file():
        pytest.skip(f"{path} not present (3rdparty/miles not checked out)")
    out = tmp_path / "chat_template.jinja"
    assert launcher.write_serve_template(tito, out) == launcher.tool_call_parser(tito)
    assert f"SERVE_TEMPLATE {template_name} thinking=off" in capsys.readouterr().err
    original, served = template_render(path.read_text()), template_render(out.read_text())
    chat = dict(messages=[{"role": "user", "content": "hi"}], tools=[], add_generation_prompt=True)
    assert served(**chat) == original(**chat, enable_thinking=False)
    assert served(**chat) != original(**chat)
    assert served(**chat, enable_thinking=True) == original(**chat, enable_thinking=True)
    assert served(**chat, enable_thinking=False) == original(**chat, enable_thinking=False)
    assert launcher.write_serve_template(tito, out, thinking="template") == launcher.tool_call_parser(tito)
    assert out.read_text() == path.read_text()
    with pytest.raises(ValueError, match="off or template"):
        launcher.write_serve_template(tito, out, thinking="on")
    with pytest.raises(ValueError, match="no fixed template"):
        launcher.write_serve_template("nemotron3", out)
    # the CLI a serving script calls: --tito-model keeps SERVE_TITO_MODEL the knob, whatever the dir's name
    launcher.main(["serve-template", "/m/whatever", str(out), "--thinking", "off", "--tito-model", tito])
    assert capsys.readouterr().out.strip() == launcher.tool_call_parser(tito)
    assert served(**chat) == template_render(out.read_text())(**chat)
    with pytest.raises(SystemExit, match="off or template"):
        launcher.main(["serve-template", "/m/whatever", str(out), "--thinking", "on", "--tito-model", tito])


def test_launcher_main_prints_parsers(stubs, monkeypatch, capsys):
    """python3 -m miles_agent.launcher parser|reasoning-parser <hf_dir>: the dir's basename picks the family and
    the TITO binding gives the parser (the stub's table here, the pin's in the container); an unlisted dir is
    one line on stderr and exit 1."""
    launcher = stubs.launcher
    for command, expected in (
        ("parser", launcher.tool_call_parser("qwen35")),
        ("reasoning-parser", launcher.reasoning_parser("qwen35")),
    ):
        monkeypatch.setattr(sys, "argv", ["launcher.py", command, "/models/Qwen3.5-9B/"])
        runpy.run_path(str(REPO / "miles_agent" / "launcher.py"), run_name="__main__")
        assert capsys.readouterr().out.strip() == expected
    launcher.main(["parser", "/ckpt/r2s-webshop-qwen3-8b-teacher"])
    assert capsys.readouterr().out.strip() == launcher.tool_call_parser("qwen3")
    with pytest.raises(SystemExit, match="MODEL_FAMILIES"):
        launcher.main(["parser", "/ckpt/Llama-3-8B"])
    with pytest.raises(ValueError, match="binds no reasoning parser"):
        launcher.reasoning_parser("default")


def test_tokenizer_probe_detects_vocab_drift(stubs, monkeypatch, capsys):
    """tokenizer-probe <policy_hf> <teacher_hf> (the OPD teacher gate): the teacher scores the student's token
    ids, so the two tokenizers must agree in size and on the ids of ASCII, CJK and the student's own rendered
    tool call; dict-backed tokenizers behind a stand-in transformers."""

    class Tokenizer:
        def __init__(self, vocab):
            self.vocab = vocab

        def __len__(self):
            return len(self.vocab)

        def encode(self, text):
            return [self.vocab.get(char, ord(char)) for char in text]

        @staticmethod
        def apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=False):
            # a template that writes a call's markup inside the assistant turn, as the real ones do
            turns = []
            for message in messages:
                calls = [json.dumps(call["function"], sort_keys=True) for call in message.get("tool_calls", [])]
                markup = "".join(f"⟨call⟩{call}⟨/call⟩" for call in calls)
                turns.append(f"<{message['role']}>{message['content']}{markup}</{message['role']}>\n")
            return "".join(turns)

    vocab = {"你": 1, "好": 2, "<": 3, "⟨": 4}
    tokenizers = {
        "/m/student": Tokenizer(vocab),
        "/m/teacher": Tokenizer(dict(vocab)),
        "/m/bigger": Tokenizer({**vocab, "<extra>": 4}),
        "/m/shifted": Tokenizer({**vocab, "好": 9}),
        "/m/markup": Tokenizer({**vocab, "⟨": 9}),  # differs on the rendered call's markup alone
    }
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: tokenizers[path])
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    launcher = stubs.launcher
    assert launcher.tokenizer_probe("/m/student", "/m/teacher") is None
    assert "vocabulary sizes differ" in launcher.tokenizer_probe("/m/student", "/m/bigger")
    assert "tokenizes differently" in launcher.tokenizer_probe("/m/student", "/m/shifted")
    assert "tokenizes differently" in launcher.tokenizer_probe("/m/student", "/m/markup")
    launcher.main(["tokenizer-probe", "/m/student", "/m/teacher"])
    assert capsys.readouterr().out.strip() == "TOKENIZER_PROBE_OK"
    with pytest.raises(SystemExit, match="TOKENIZER_PROBE_FAIL .*tokenizes differently"):
        launcher.main(["tokenizer-probe", "/m/student", "/m/shifted"])


# ---------------------------------------------------------------- OPD arm (algo opd, teacher topologies, pilot)

TEACHER = "http://10.0.0.7:13141"


def _opd(stubs, recipe="alfworld", **kw):
    """An OPD arm through the recipe's execute: the teacher URL the job's gate passes, the default Qwen3.5 family
    (thinking off at the template)."""
    return _run(stubs, recipe, algo="opd", teacher_url=TEACHER, **kw)


def _opd_block(coef="1.0", teacher=TEACHER):
    block = (
        "--advantage-estimator grpo --entropy-coef 0.0 --eps-clip 0.2 --use-opd --opd-type sglang "
        f"--opd-kl-coef {coef} --opd-log-prob-top-k 0"
    )
    return block.split() + (["--opd-teacher-urls", f"default={teacher}/generate"] if teacher else [])


def test_opd_algo_block_exact_tokens(stubs):
    """The opd block is miles' native sampled OPD and nothing else: the grpo estimator over a zeroed train
    reward (advantage 0), the teacher KL as the per-token advantage, clip 0.2, the teacher's /generate as the
    default route of --opd-teacher-urls (the base URL's trailing slash does not double up)."""
    launcher = stubs.launcher
    args = stubs.runs["alfworld"].ScriptArgs(algo="opd", teacher_url=TEACHER + "/")
    assert launcher._algo_args(args).split() == _opd_block()
    args = stubs.runs["alfworld"].ScriptArgs(algo="opd", opd_kl_coef=0.5)  # eval-only: no teacher
    assert launcher._algo_args(args).split() == _opd_block(coef="0.5", teacher="")
    toks = _tokens(_opd(stubs))
    start = toks.index("--advantage-estimator")
    assert toks[start : start + len(_opd_block())] == _opd_block()


def test_opd_block_has_no_grpo_kl_or_clip_high(stubs):
    """Nothing of the GRPO/PPO blocks nor of the paper branch's patched flags leaks into the opd arm on any
    topology; the reward slot stays reward_routing's and the clip is symmetric."""
    for topology in ("async_4n_teacher", "async_2n_teacher", "colocated_1n", "async_3n"):
        cap = _opd(stubs, topology=topology)
        toks = _tokens(cap)
        for flag in (
            "--use-kl-loss",
            "--kl-loss-coef",
            "--kl-loss-type",
            "--eps-clip-high",
            "--keep-old-actor",
            "--rm-url",
            "--opd-kl-type",
            "--opd-topk",
            "--opd-teacher-alpha",
            "--opd-teacher-load",
        ):
            assert flag not in toks, (topology, flag)
        assert toks.count("--eps-clip") == 1 and _flag(cap, "--eps-clip") == "0.2"
        assert toks.count("--advantage-estimator") == 1 and _flag(cap, "--advantage-estimator") == "grpo"
        assert _flag(cap, "--custom-rm-path") == "miles_agent.reward_routing.reward_func"
        assert (_flag(cap, "--reward-key"), _flag(cap, "--eval-reward-key")) == ("train", "env")


def test_optimizer_args_defaults_match_previous_constant(stubs):
    """_optimizer_args at the defaults is the constant it replaces, byte for byte (1e-6 spelled as before), in
    every recipe's GRPO launch; the scheduler-resume flag belongs to the opd arm alone."""
    previous = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --lr-decay-iters 10000 --lr-warmup-iters 142 "
        "--lr-warmup-init 0.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    assert stubs.launcher._optimizer_args(stubs.launcher.BaseScriptArgs()) == previous
    for recipe in RECIPES:
        cap = _run(stubs, recipe)
        assert previous in cap["train_args"] and "--use-checkpoint-opt-param-scheduler" not in _tokens(cap)


def test_optimizer_knobs_reach_train_args(stubs):
    """lr / warmup / weight decay are fields (the OPD presets pin 1e-6 / 0 / 0.1), spelled without a zero-padded
    exponent; the opd arm adds --use-checkpoint-opt-param-scheduler right after the optimizer block."""
    cap = _run(stubs, "alfworld", lr=2e-6, lr_warmup_iters=0, weight_decay=0.1)
    assert (_flag(cap, "--lr"), _flag(cap, "--lr-warmup-iters"), _flag(cap, "--weight-decay")) == ("2e-6", "0", "0.1")
    assert _flag(_run(stubs, "alfworld", lr=1e-5), "--lr") == "1e-5"
    assert _flag(_run(stubs, "webshop", lr=0.0001, weight_decay=0.0), "--lr") == "0.0001"
    assert _flag(_run(stubs, "webshop", weight_decay=0.0), "--weight-decay") == "0"
    cap = _opd(stubs, lr_warmup_iters=0, weight_decay=0.1)
    toks = _tokens(cap)
    assert (_flag(cap, "--lr"), _flag(cap, "--lr-warmup-iters"), _flag(cap, "--weight-decay")) == ("1e-6", "0", "0.1")
    assert toks.index("--use-checkpoint-opt-param-scheduler") == toks.index("--adam-beta2") + 2


def test_teacher_topologies_emit_ref_strings(stubs):
    """async_4n_teacher is async_3n's ray-side split (the teacher node never joins ray); async_2n_teacher is the
    paper branch's one-node split, 2 actor + 2 rollout GPUs; both run train_async.py with the update interval
    (the node counts against the entry: test_teacher_topology_node_counts_match_the_entry_table)."""
    launcher = stubs.launcher
    make = stubs.runs["alfworld"].ScriptArgs
    cap = _opd(stubs, topology="async_4n_teacher")
    assert cap["train_script"] == "train_async.py" and _flag(cap, "--rollout-num-gpus") == "8"
    four = make(algo="opd", teacher_url=TEACHER, topology="async_4n_teacher")
    assert launcher._topology_args(four) == launcher._DISAGG_ARGS + "--update-weights-interval 2 "
    assert launcher._topology_args(four) == launcher._topology_args(make(topology="async_3n"))
    cap = _opd(stubs, topology="async_2n_teacher", update_weights_interval=1)
    assert cap["train_script"] == "train_async.py"
    two = make(algo="opd", teacher_url=TEACHER, topology="async_2n_teacher", update_weights_interval=1)
    assert launcher._topology_args(two) == (
        "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --rollout-num-gpus 2 --pin-rollout-manager-to-head "
        "--update-weights-interval 1 "
    )
    toks = _tokens(cap)
    assert "--fully-async" not in toks and "--colocate" not in toks and "--async-max-concurrent-samples" not in toks
    assert (_flag(cap, "--actor-num-gpus-per-node"), _flag(cap, "--rollout-num-gpus")) == ("2", "2")


def test_teacher_topology_node_counts_match_the_entry_table(stubs):
    """The ray-side split of a *_teacher topology is one and the same with the job's own teacher node, with an
    already serving teacher (the gate passes --teacher-url either way, so the launcher cannot tell them apart)
    and for an eval-only decode without one: its GPUs fill the entry's node count less one, which is the
    --num-nodes TEACHER_URL takes (docs section 2.3, --num-nodes 3 / 1); the extra node serves the teacher."""
    launcher = stubs.launcher
    make = stubs.runs["alfworld"].ScriptArgs
    entry = (REPO / "scripts/run_local.sh").read_text()
    table = {topo: int(n) for topo, n in re.findall(r"(\w+)\) NEED_NODES=(\d+) ;;", entry)}  # the entry's TOPOLOGY table
    gpus_per_node = make().num_gpus_per_node
    assert (table["async_4n_teacher"], table["async_2n_teacher"], gpus_per_node) == (4, 2, 4)
    for topology in ("async_4n_teacher", "async_2n_teacher"):
        own = launcher._topology_args(make(algo="opd", teacher_url=TEACHER, topology=topology))
        shared = launcher._topology_args(make(algo="opd", teacher_url="http://shared:13141", topology=topology))
        decode = launcher._topology_args(make(algo="opd", num_rollout=0, topology=topology))
        assert own == shared == decode
        toks = own.split()
        actor = int(toks[toks.index("--actor-num-nodes") + 1]) * int(toks[toks.index("--actor-num-gpus-per-node") + 1])
        rollout = int(toks[toks.index("--rollout-num-gpus") + 1])
        ray_nodes = -(-(actor + rollout) // gpus_per_node)
        assert table[topology] == ray_nodes + 1, topology


def test_opd_generate_fn_switch(stubs):
    """The custom generate function: miles' own agentic episode by default, the OPD wrapper around it for the
    opd arm and for the pilot dump under any algo; the agent, reward and filter hooks do not move."""
    launcher = stubs.launcher
    stock, opd = launcher.STOCK_AGENTIC_GENERATE_FN, launcher.OPD_GENERATE_FN
    assert stock == "miles.rollout.generate_hub.agentic_tool_call.generate"
    assert opd == "miles_agent.opd.generate.generate"
    assert _flag(_run(stubs, "alfworld"), "--custom-generate-function-path") == stock
    assert _flag(_run(stubs, "search_r1", algo="dapo"), "--custom-generate-function-path") == stock
    cap = _opd(stubs)
    assert _flag(cap, "--custom-generate-function-path") == opd
    assert _flag(cap, "--custom-agent-function-path") == "miles_agent.gym_bridge.run"
    assert _flag(cap, "--dynamic-sampling-filter-path") == stubs.runs["alfworld"].SPEC.filter_path
    assert _flag(_run(stubs, "webshop", opd_dump=True), "--custom-generate-function-path") == opd
    # session_args alone keeps the stock function: tmax calls it without one
    session = launcher.session_args(stubs.runs["alfworld"].ScriptArgs(), filter_path="f")
    assert f"--custom-generate-function-path {stock} " in session and opd not in session


def test_opd_env_vars(stubs):
    """The opd arm's actors read zero_shape as the shaping plugin (the task reward trains as 0 while the env score
    stays in eval/* and rollout/episode_raw_reward) and the dump dir only under --opd-dump."""
    env = _opd(stubs)["extra_env_vars"]
    assert env["REWARD_SHAPING_PATH"] == "miles_agent.opd.zero_shape" and "MILES_OPD_DUMP_DIR" not in env
    env = _opd(stubs, "webshop", opd_dump=True, save_dir="/r/wsopd1/")["extra_env_vars"]
    assert env["REWARD_SHAPING_PATH"] == "miles_agent.opd.zero_shape"
    assert env["MILES_OPD_DUMP_DIR"] == "/r/wsopd1/opd_dump"
    for recipe in RECIPES:
        env = _run(stubs, recipe)["extra_env_vars"]
        assert env["REWARD_SHAPING_PATH"] == "reward.shape" and "MILES_OPD_DUMP_DIR" not in env


def test_pilot_dump_env_is_set_under_algo_grpo(stubs):
    """The R2S pilot is a plain debug_rollout_only run with the dump on: no teacher, the recipe's own shaping
    plugin, the OPD generate hook installed, nothing of the opd block."""
    pilot = dict(algo="grpo", opd_dump=True, mode="debug_rollout_only", rollout_temperature=1.0, save_dir="/r/p1/")
    cap = _run(stubs, "alfworld", **pilot)
    env = cap["extra_env_vars"]
    assert env["MILES_OPD_DUMP_DIR"] == "/r/p1/opd_dump" and env["REWARD_SHAPING_PATH"] == "reward.shape"
    toks = _tokens(cap)
    assert _flag(cap, "--custom-generate-function-path") == stubs.launcher.OPD_GENERATE_FN
    assert "--use-opd" not in toks and "--opd-teacher-urls" not in toks and "--debug-rollout-only" in toks
    assert _flag(cap, "--rollout-temperature") == "1.0" and "--use-checkpoint-opt-param-scheduler" not in toks


def test_pilot_dump_disables_eval_and_save(stubs):
    """The pilot evaluates and saves nowhere within its steps whatever the preset's intervals (the campaign
    branch's submit script) and skips the eval before training (--skip-eval-before-train; the eval after the
    last step is miles' own), with one --eval-interval / --save-interval each; the dump under a training mode,
    the opd arm and a plain debug_rollout_only run keep their intervals and their step-0 eval."""
    cap = _run(stubs, "webshop", opd_dump=True, mode="debug_rollout_only", eval_interval=25, save_interval=25)
    toks = _tokens(cap)
    assert (_flag(cap, "--eval-interval"), _flag(cap, "--save-interval")) == ("1000000", "1000000")
    assert toks.count("--eval-interval") == 1 and toks.count("--save-interval") == 1 and "--save" in toks
    assert toks.count("--skip-eval-before-train") == 1
    cap = _run(stubs, "webshop", opd_dump=True, eval_interval=25, save_interval=25)
    assert (_flag(cap, "--eval-interval"), _flag(cap, "--save-interval")) == ("25", "25")
    assert "--skip-eval-before-train" not in _tokens(cap)
    cap = _opd(stubs, eval_interval=25, save_interval=25)
    assert (_flag(cap, "--eval-interval"), _flag(cap, "--save-interval")) == ("25", "25")
    assert "--skip-eval-before-train" not in _tokens(cap)
    cap = _run(stubs, "alfworld", mode="debug_rollout_only")
    assert (_flag(cap, "--eval-interval"), _flag(cap, "--save-interval")) == ("50", "50")
    assert "--skip-eval-before-train" not in _tokens(cap)


def test_validate_opd_rules(stubs):
    """execute() refuses what the protocol excludes before the allocation: an opd arm without a teacher (unless
    eval-only), a teacher URL or a teacher topology outside opd, a second reward path, off-policy corrections."""
    with pytest.raises(ValueError, match="--teacher-url"):
        _run(stubs, "alfworld", algo="opd")
    _run(stubs, "alfworld", algo="opd", num_rollout=0)  # eval-only (eval_ckpt): nothing is scored
    with pytest.raises(ValueError, match="--algo opd knob"):
        _run(stubs, "alfworld", teacher_url=TEACHER)
    for topology in ("async_4n_teacher", "async_2n_teacher"):
        with pytest.raises(ValueError, match="needs --algo opd"):
            _run(stubs, "alfworld", topology=topology)
    for extra in ("--rm-url http://t/generate", "--custom-rm-path=x.y", "--foo 1 --rm-url=http://t"):
        with pytest.raises(ValueError, match="--extra-train-args"):
            _opd(stubs, extra_train_args=extra)
    assert _tokens(_opd(stubs, extra_train_args="--no-offload-train"))[-1] == "--no-offload-train"
    for knob in ("use_tis", "use_rollout_logprobs"):
        with pytest.raises(ValueError, match="OPD protocol"):
            _opd(stubs, **{knob: True})
    # an external or shared teacher serves the other topologies too
    for topology in ("colocated_1n", "async_3n"):
        assert "--use-opd" in _tokens(_opd(stubs, topology=topology))


def test_validate_opd_refuses_teacher_url_with_path(stubs):
    """--teacher-url is the server's base URL: _algo_args appends /generate, so a URL that already ends in /generate
    (or carries any path, a query or a fragment: http://h:13141?x=1 would become ...?x=1/generate) is refused with
    the fix named instead of becoming .../generate/generate; a trailing slash is not a path."""
    for url in (TEACHER + "/generate", TEACHER + "/generate/", TEACHER + "/v1", TEACHER + "?x=1", TEACHER + "#f"):
        with pytest.raises(ValueError, match=r"base URL .*no path, query or fragment.*appends /generate"):
            _run(stubs, "alfworld", algo="opd", teacher_url=url)
    cap = _run(stubs, "alfworld", algo="opd", teacher_url=TEACHER + "/")
    assert _flag(cap, "--opd-teacher-urls") == f"default={TEACHER}/generate"


def test_validate_opd_reports_bad_chat_template_kwargs(stubs):
    """A malformed --chat-template-kwargs under --algo opd is one ValueError naming the flag (the thinking check
    parses it), not a JSONDecodeError traceback, and a JSON value that is not an object is refused the same way;
    the GRPO arms pass the string on to miles unparsed."""
    for bad in ("{enable_thinking: false}", "'{\"enable_thinking\": false}'", "[1]"):
        with pytest.raises(ValueError, match="--chat-template-kwargs"):
            _opd(stubs, chat_template_kwargs=bad)
    _opd(stubs, chat_template_kwargs='{"enable_thinking": false}')
    assert _flag(_run(stubs, "alfworld", chat_template_kwargs="{bad"), "--apply-chat-template-kwargs") == "'{bad'"


def test_validate_opd_warns_when_thinking_is_not_disabled(stubs, capsys):
    """The paper trained every family with thinking off; a Qwen3 hybrid's template default (think) is a warning,
    not a refusal; the compact JSON the jobs pass counts; GRPO arms are not the protocol's business."""
    _opd(stubs, model_name="Qwen3-8B")
    assert "WARN: --algo opd with thinking not disabled" in capsys.readouterr().err
    _opd(stubs, model_name="Qwen3-8B", chat_template_kwargs='{"enable_thinking":false}')
    assert "WARN" not in capsys.readouterr().err
    _opd(stubs)  # Qwen3.5: the family row disables thinking
    assert "WARN" not in capsys.readouterr().err
    _run(stubs, "alfworld", model_name="Qwen3-8B")
    assert "WARN" not in capsys.readouterr().err


# ---------------------------------------------------------------- OPD presets (recipes/<env>/opd.py)


def _preset(stubs, recipe, **kw):
    """A launch through the preset's own path: run.execute (launcher.execute with run's SPEC) on its OpdArgs."""
    stubs.runs[recipe].execute(stubs.opds[recipe].OpdArgs(**kw))
    return dict(CAPTURED)


def _eval_block(cap):
    toks = _tokens(cap)
    return toks[toks.index("--eval-prompt-data") + 1 : toks.index("--n-samples-per-eval-prompt")]


@pytest.mark.parametrize("recipe", OPD_RECIPES)
def test_opd_preset_flags(stubs, recipe, capsys):
    """recipes/<env>/opd.py (design §2.5): the paper's protocol flag by flag - 150 steps of 16 x 1 (WebShop 16 x 8
    = 128), T 0.4 for rollout and eval, 512 / 4096 tokens a turn, context and packing 32768, lr 1e-6 constant with
    no warmup and weight decay 0.1, eval and save every 25, weights every step, seed 42, thinking off for every
    family - over run's environment (eval slices, data, filter, /run timeout), the opd block and the scheduler flag
    present, nothing of the GRPO block, only the job's --teacher-url added."""
    launcher = stubs.launcher
    run = stubs.runs[recipe]
    cap = _preset(stubs, recipe, teacher_url=TEACHER)
    toks = _tokens(cap)
    samples, batch = ("8", "128") if recipe == "webshop" else ("1", "16")
    for flag, value in (
        ("--num-rollout", "150"),
        ("--rollout-batch-size", "16"),
        ("--n-samples-per-prompt", samples),
        ("--global-batch-size", batch),
        ("--rollout-temperature", "0.4"),
        ("--eval-temperature", "0.4"),
        ("--rollout-top-p", "1.0"),
        ("--eval-top-p", "1.0"),
        ("--n-samples-per-eval-prompt", "1"),
        ("--rollout-max-response-len", "512"),
        ("--eval-max-response-len", "4096"),
        ("--rollout-max-context-len", "32768"),
        ("--max-tokens-per-gpu", "32768"),
        ("--lr", "1e-6"),
        ("--lr-decay-style", "constant"),
        ("--lr-warmup-iters", "0"),
        ("--weight-decay", "0.1"),
        ("--adam-beta1", "0.9"),
        ("--adam-beta2", "0.98"),
        ("--eval-interval", "25"),
        ("--save-interval", "25"),
        ("--update-weights-interval", "1"),
        ("--seed", "42"),
        ("--actor-num-nodes", "1"),
        ("--actor-num-gpus-per-node", "4"),
        ("--rollout-num-gpus", "8"),
        ("--custom-generate-function-path", launcher.OPD_GENERATE_FN),
        ("--custom-rm-path", "miles_agent.reward_routing.reward_func"),
        ("--dynamic-sampling-filter-path", run.SPEC.filter_path),
        ("--tito-model", "qwen35"),
    ):
        assert _flag(cap, flag) == value, (recipe, flag)
    # megatron's clip-grad 1.0 is not restated; the opd block is contiguous and alone; the scheduler flag follows
    # the optimizer block; nothing of the GRPO block, of the other drivers or of the pilot
    assert "--clip-grad" not in toks and toks.count("--max-tokens-per-gpu") == 1
    start = toks.index("--advantage-estimator")
    assert toks[start : start + len(_opd_block())] == _opd_block()
    assert toks.index("--use-checkpoint-opt-param-scheduler") == toks.index("--adam-beta2") + 2
    for flag in (
        "--use-kl-loss",
        "--kl-loss-coef",
        "--eps-clip-high",
        "--keep-old-actor",
        "--fully-async",
        "--skip-eval-before-train",
    ):
        assert flag not in toks, (recipe, flag)
    assert "--colocate" not in toks and "--debug-rollout-only" not in toks and "--pin-rollout-manager-to-head" in toks
    assert cap["train_script"] == "train_async.py" and cap["megatron_model_type"] == "qwen3.5-4B"
    argv = shlex.split(cap["train_args"])
    assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    # run's environment: the GRPO recipe's eval slices, data and save paths, actors' PYTHONPATH and /run timeout;
    # the recipe's shaping terms give way to zero_shape; no dump dir (the pilot passes --opd-dump)
    grpo = _run(stubs, recipe)
    assert _eval_block(cap) == _eval_block(grpo)
    for flag in ("--prompt-data", "--save", "--hf-checkpoint", "--ref-load"):
        assert _flag(cap, flag) == _flag(grpo, flag), flag
    env = cap["extra_env_vars"]
    assert env["REWARD_SHAPING_PATH"] == "miles_agent.opd.zero_shape" and "MILES_OPD_DUMP_DIR" not in env
    assert env["NEMO_GYM_RUN_TIMEOUT"] == "1200" and env["PYTHONPATH"] == grpo["extra_env_vars"]["PYTHONPATH"]
    # thinking off is the preset's, whatever the family's template default: a Qwen3 hybrid raises no WARN
    capsys.readouterr()
    cap = _preset(stubs, recipe, teacher_url=TEACHER, model_name="Qwen3-8B")
    assert "WARN" not in capsys.readouterr().err
    assert _flag(cap, "--tito-model") == "qwen3" and _flag(cap, "--sglang-tool-call-parser") == "qwen25"
    argv = shlex.split(cap["train_args"])
    assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": false}'


@pytest.mark.parametrize("recipe", OPD_RECIPES)
def test_opd_preset_module_surface(stubs, recipe, monkeypatch):
    """The preset is run.py's sibling: OpdArgs is the one BaseScriptArgs dataclass it defines (eval_ckpt's
    --recipe-script opd.py finds it by definition site), over run.ScriptArgs with run's paths; SPEC is run's; the
    CLI surface survives the inheritance; main mirrors run's main (cleanup, prepare unless --skip-prepare, the
    launch through run.execute, where an eval_full-style rewrite of launcher.execute_train still lands)."""
    launcher = stubs.launcher
    mod, run = stubs.opds[recipe], stubs.runs[recipe]
    assert mod.run is run and not hasattr(mod, "SPEC")
    own = [
        obj
        for obj in vars(mod).values()
        if isinstance(obj, type) and issubclass(obj, launcher.BaseScriptArgs) and obj.__module__ == mod.__name__
    ]
    assert own == [mod.OpdArgs] and issubclass(mod.OpdArgs, run.ScriptArgs) and dataclasses.is_dataclass(mod.OpdArgs)
    # what U.dataclass_cli reads: every __init__ parameter a field with a resolvable hint and a default, the
    # topology and algo choices the launcher's
    params = [p for p in inspect.signature(mod.OpdArgs.__init__).parameters.values() if p.name != "self"]
    hints = typing.get_type_hints(mod.OpdArgs)
    assert [p.name for p in params] == [f.name for f in dataclasses.fields(mod.OpdArgs)]
    assert all(p.name in hints and p.default is not inspect.Parameter.empty for p in params)
    base_hints = typing.get_type_hints(launcher.BaseScriptArgs)
    for name in ("topology", "algo"):
        assert typing.get_args(hints[name]) == typing.get_args(base_hints[name]), name
    args, base = mod.OpdArgs(), run.ScriptArgs()
    assert (args.algo, args.topology, args.num_rollout, args.seed) == ("opd", "async_4n_teacher", 150, 42)
    assert (args.save_dir, args.prompt_data, args.eval_data) == (base.save_dir, base.prompt_data, base.eval_data)
    assert (args.opd_kl_coef, args.opd_dump, args.teacher_url) == (1.0, False, "")
    calls = []
    monkeypatch.setattr(run, "cleanup", lambda: calls.append("cleanup"))
    monkeypatch.setattr(run, "prepare", lambda args: calls.append("prepare"))
    CAPTURED.clear()
    mod.main(mod.OpdArgs(teacher_url=TEACHER, skip_prepare=True))
    assert calls == ["cleanup"] and "--use-opd" in _tokens(CAPTURED)
    mod.main(mod.OpdArgs(teacher_url=TEACHER))
    assert calls == ["cleanup", "cleanup", "prepare"]
    monkeypatch.setattr(launcher, "execute_train", lambda args, **kw: calls.append(type(args).__name__))
    mod.main(mod.OpdArgs(teacher_url=TEACHER, skip_prepare=True))
    assert calls[-1] == "OpdArgs"


@pytest.mark.parametrize("recipe", OPD_RECIPES)
def test_opd_preset_pilot_and_eval_lines(stubs, recipe):
    """The preset's other job lines (design D14, §2.7): the pilot dump rolls the arm's episodes (caps, seed,
    thinking off, 16 x n a step for 150 steps) at T 1.0 under --algo grpo on one colocated node, no teacher, the
    dump dir on, nothing evaluated or saved within the run and no eval before it - the preset's teacher topology
    must give way, since it needs --algo opd; an eval-only decode (num_rollout 0) needs no teacher and keeps its
    one eval; the arm itself refuses to start without a teacher."""
    pilot = dict(
        algo="grpo",
        opd_dump=True,
        mode="debug_rollout_only",
        rollout_temperature=1.0,
        topology="colocated_1n",
        save_dir="/r/p1/",
    )
    cap = _preset(stubs, recipe, **pilot)
    toks = _tokens(cap)
    assert cap["train_script"] == "train.py" and "--colocate" in toks and "--debug-rollout-only" in toks
    assert not {"--use-opd", "--opd-teacher-urls", "--use-checkpoint-opt-param-scheduler"} & set(toks)
    assert "--skip-eval-before-train" in toks
    assert _flag(cap, "--custom-generate-function-path") == stubs.launcher.OPD_GENERATE_FN
    env = cap["extra_env_vars"]
    assert env["MILES_OPD_DUMP_DIR"] == "/r/p1/opd_dump" and env["REWARD_SHAPING_PATH"] == "reward.shape"
    for flag, value in (
        ("--rollout-temperature", "1.0"),
        ("--num-rollout", "150"),
        ("--rollout-batch-size", "16"),
        ("--n-samples-per-prompt", "8" if recipe == "webshop" else "1"),
        ("--rollout-max-response-len", "512"),
        ("--rollout-max-context-len", "32768"),
        ("--seed", "42"),
        ("--eval-interval", "1000000"),
        ("--save-interval", "1000000"),
    ):
        assert _flag(cap, flag) == value, (recipe, flag)
    argv = shlex.split(cap["train_args"])
    assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    with pytest.raises(ValueError, match="needs --algo opd"):
        _preset(stubs, recipe, **{**pilot, "topology": "async_4n_teacher"})
    # eval_ckpt's decodes: num_rollout 0 on one node under the arm's eval protocol, no teacher
    cap = _preset(stubs, recipe, num_rollout=0, topology="colocated_1n")
    toks = _tokens(cap)
    assert "--use-opd" in toks and "--opd-teacher-urls" not in toks and "--colocate" in toks
    assert (_flag(cap, "--eval-temperature"), _flag(cap, "--eval-max-response-len")) == ("0.4", "4096")
    assert "--skip-eval-before-train" not in toks
    with pytest.raises(ValueError, match="--teacher-url"):
        _preset(stubs, recipe)


# ---------------------------------------------------------------- env plumbing


def test_extra_env_vars(stubs, monkeypatch):
    monkeypatch.delenv("MEGATRON_PATH", raising=False)
    for recipe in RECIPES:
        cap = _run(stubs, recipe)
        env = cap["extra_env_vars"]
        assert env["PYTHONPATH"] == f"/root/Megatron-LM:{REPO / 'recipes' / recipe}:{REPO}:/opt/miles"
        assert env["REWARD_SHAPING_PATH"] == "reward.shape"
        assert env["WANDB_CONSOLE"] == "off"
        assert env["NEMO_GYM_URL"] == stubs.runs[recipe].ScriptArgs().nemo_gym_url
        assert "MILES_ROUTER_EXTERNAL_HOST" not in env
    assert _run(stubs, "alfworld")["extra_env_vars"]["NEMO_GYM_RUN_TIMEOUT"] == "1200"
    assert _run(stubs, "scienceworld")["extra_env_vars"]["NEMO_GYM_RUN_TIMEOUT"] == "1200"
    assert _run(stubs, "webshop")["extra_env_vars"]["NEMO_GYM_RUN_TIMEOUT"] == "1200"
    assert "NEMO_GYM_RUN_TIMEOUT" not in _run(stubs, "search_r1")["extra_env_vars"]


def test_nemo_gym_url_defaults_to_the_local_agent_port(stubs, monkeypatch):
    """NEMO_GYM_URL unset: the launcher and the bridge agree on the local agent server, http://127.0.0.1:8200
    (the port start_gym.sh serves the agent on); set, both honour it (the bridge splits a comma list)."""
    monkeypatch.delenv("NEMO_GYM_URL", raising=False)
    from miles_agent import gym_bridge

    assert stubs.runs["alfworld"].ScriptArgs().nemo_gym_url == "http://127.0.0.1:8200"
    assert gym_bridge.agent_urls() == ["http://127.0.0.1:8200"]
    assert _run(stubs, "alfworld")["extra_env_vars"]["NEMO_GYM_URL"] == "http://127.0.0.1:8200"
    monkeypatch.setenv("NEMO_GYM_URL", "http://192.0.2.10:8200, http://192.0.2.11:8200/")
    assert stubs.runs["alfworld"].ScriptArgs().nemo_gym_url == "http://192.0.2.10:8200, http://192.0.2.11:8200/"
    assert gym_bridge.agent_urls() == ["http://192.0.2.10:8200", "http://192.0.2.11:8200"]
    cap = _run(stubs, "alfworld", nemo_gym_url="http://192.0.2.12:8200")
    assert cap["extra_env_vars"]["NEMO_GYM_URL"] == "http://192.0.2.12:8200"


def test_megatron_path_comes_from_the_environment(stubs, monkeypatch):
    """MEGATRON_PATH overrides the image path for the actors' PYTHONPATH head and the conversion (read when the
    args are built, so a job exports it before `python run.py`); an explicit --megatron-path still wins."""
    monkeypatch.setenv("MEGATRON_PATH", "/m")
    cap = _run(stubs, "alfworld")
    assert cap["extra_env_vars"]["PYTHONPATH"].startswith("/m:") and cap["megatron_path"] == "/m"
    assert stubs.runs["alfworld"].ScriptArgs(megatron_path="/x").megatron_path == "/x"
    monkeypatch.delenv("MEGATRON_PATH")
    assert stubs.runs["alfworld"].ScriptArgs().megatron_path == "/root/Megatron-LM"


def test_session_server_external_host(stubs):
    """miles >= 0.1.2 raises on MILES_ROUTER_EXTERNAL_HOST; the host travels as a flag."""
    cap = _run(stubs, "alfworld", session_server_external_host="10.0.0.1")
    assert _flag(cap, "--session-server-external-host") == "10.0.0.1"
    assert "MILES_ROUTER_EXTERNAL_HOST" not in cap["extra_env_vars"]
    assert "--session-server-external-host" not in _tokens(_run(stubs, "alfworld"))


def test_webshop_forwards_webshop_env_vars_only(stubs, monkeypatch):
    monkeypatch.setenv("WEBSHOP_DENSE_REWARD", "1")
    monkeypatch.setenv("WEBSHOP_STEP_PENALTY", "0.01")
    monkeypatch.setenv("WEBSHOP_HOME", "/p/webshop")
    monkeypatch.setenv("ALFWORLD_STEP_PENALTY", "0.5")
    monkeypatch.setenv("NOT_A_KNOB", "x")
    env = _run(stubs, "webshop")["extra_env_vars"]
    assert env["WEBSHOP_DENSE_REWARD"] == "1"
    assert env["WEBSHOP_STEP_PENALTY"] == "0.01"
    assert env["WEBSHOP_HOME"] == "/p/webshop"
    assert "ALFWORLD_STEP_PENALTY" not in env and "NOT_A_KNOB" not in env
    # other recipes forward nothing
    for recipe in ("alfworld", "scienceworld", "search_r1"):
        env = _run(stubs, recipe)["extra_env_vars"]
        assert not [k for k in env if k.startswith("WEBSHOP_")]
    # passthrough can never clobber the launcher's own keys
    monkeypatch.setenv("WEBSHOP_X", "y")
    spec = dataclasses.replace(stubs.runs["webshop"].SPEC, env_passthrough_prefixes=("WANDB_",))
    monkeypatch.setenv("WANDB_CONSOLE", "on")
    env = stubs.launcher.build_extra_env_vars(stubs.runs["webshop"].ScriptArgs(), spec)
    assert env["WANDB_CONSOLE"] == "off"


def test_wandb_naming(stubs, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    cap = _run(stubs, "alfworld", save_dir=f"{tmp_path}/alftrain1/")  # writable: the launch keeps its run id there
    assert _flag(cap, "--wandb-project") == "miles-alfworld_run"
    assert _flag(cap, "--wandb-group") == "alftrain1"  # stable group = arm
    cap = _run(stubs, "webshop", save_dir=f"{tmp_path}/ws1/")
    assert _flag(cap, "--wandb-project") == "miles-webshop_run"
    assert _flag(cap, "--wandb-group") == "ws1"
    cap = _run(stubs, "search_r1", save_dir=f"{tmp_path}/sr1/")
    assert _flag(cap, "--wandb-project") == "miles-search_r1_run"
    assert _flag(cap, "--wandb-group") == "sr1"  # stable group for every recipe
    # no arm name in the save dir: the recipe and topology name the group (the run id is not the file's here)
    monkeypatch.setattr(stubs.launcher, "wandb_run_id", lambda save_dir: "fixedid1")
    cap = _run(stubs, "alfworld", save_dir="/", topology="async_3n")
    assert _flag(cap, "--wandb-group") == "alfworld_async_3n" and _flag(cap, "--wandb-run-id") == "fixedid1"
    monkeypatch.delenv("WANDB_API_KEY")
    assert "--use-wandb" not in _tokens(_run(stubs, "alfworld"))


def test_execute_train_resolved_at_call_time(stubs, monkeypatch):
    """eval_full.py wraps launcher.execute_train after importing run; execute must honor it."""
    seen = {}
    monkeypatch.setattr(stubs.launcher, "execute_train", lambda args, **kw: seen.update(kw, args=args))
    mod = stubs.runs["search_r1"]
    mod.execute(mod.ScriptArgs())
    assert seen["train_script"] == "train.py" and "--prompt-data" in seen["train_args"]
    assert seen["args"].__class__ is mod.ScriptArgs


def test_eval_full_rewrites_the_launch(stubs, monkeypatch, tmp_path):
    """recipes/search_r1/eval_full.py: full-test slices and --load on the real launch path."""
    monkeypatch.setitem(sys.modules, "run", stubs.runs["search_r1"])
    monkeypatch.setattr(stubs.launcher, "execute_train", stubs.launcher.execute_train)
    spec = importlib.util.spec_from_file_location("_eval_full", REPO / "recipes" / "search_r1" / "eval_full.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.EvalArgs(save_dir=f"{tmp_path}/view/", eval_data="/d/test.jsonl")
    mod.patch_execute_train(args)
    stubs.runs["search_r1"].execute(args)
    toks = _tokens(CAPTURED)
    assert toks[-4:] == ["--load", f"{tmp_path}/view/", "--no-load-optim", "--no-load-rng"]
    assert "bamboogle" in toks and "/d/test.jsonl@[51588:51713]" in toks


def test_prepare_forwards_checkpoint_conversion(stubs):
    mod = stubs.runs["alfworld"]
    CAPTURED.clear()
    mod.prepare(mod.ScriptArgs(base_dir="/b"))
    assert CAPTURED["convert"]["dir_dst"] == "/b"
    assert CAPTURED["convert"]["megatron_model_type"] == "qwen3.5-4B"


def test_tmax_recipe_is_tmaxs_science_on_the_launch_platform(stubs):
    """recipes/tmax/run.py: TMax-4B's batch, budgets, DPPO loss and thinking, fully async on the job's 8-GPU nodes
    with the trainer at TP 1."""
    spec = importlib.util.spec_from_file_location("_run_tmax", REPO / "recipes" / "tmax" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.prepare is stubs.launcher.prepare and mod.cleanup is stubs.launcher.cleanup
    mod.execute(mod.ScriptArgs(run_tag="t", num_gpus_per_node=8))
    cap = dict(CAPTURED)
    tokens = _tokens(cap)
    assert cap["train_script"] == "train_async.py" and "--fully-async" in tokens
    assert (_flag(cap, "--actor-num-nodes"), _flag(cap, "--actor-num-gpus-per-node")) == ("1", "8")
    assert _flag(cap, "--rollout-num-gpus") == "24" and _flag(cap, "--rollout-num-gpus-per-engine") == "1"
    # no staleness cap unless max_weight_staleness > 0
    assert "--max-weight-staleness" not in tokens and _flag(cap, "--async-max-concurrent-samples") == "512"
    # 8 prompts x 32 samples, one optimizer step a batch, 500 steps
    batch = int(_flag(cap, "--rollout-batch-size")) * int(_flag(cap, "--n-samples-per-prompt"))
    assert (batch, _flag(cap, "--global-batch-size"), _flag(cap, "--num-rollout")) == (256, "256", "500")
    # DPPO against the engine's log-probs, PPO's clip off, no TIS, no error mask, no KL
    assert _flag(cap, "--custom-tis-function-path") == "miles_agent.dppo.dppo_binary_tv"
    assert {"--use-rollout-logprobs", "--get-mismatch-metrics", "--skip-actor-forward-only"} <= set(tokens)
    assert float(_flag(cap, "--eps-clip")) >= 1e3 and float(_flag(cap, "--eps-clip-high")) >= 1e3
    unwanted = {"--use-tis", "--eps-clip-c", "--rollout-sample-filter-path", "--use-kl-loss", "--kl-coef"}
    assert not unwanted & set(tokens)
    assert {"--disable-grpo-std-normalization", "--calculate-per-token-loss"} <= set(tokens)
    assert "def dppo_binary_tv(" in (REPO / "miles_agent" / "dppo.py").read_text()
    # --skip-actor-forward-only refuses dropout
    assert _flag(cap, "--hidden-dropout") == "0.0" and _flag(cap, "--attention-dropout") == "0.0"
    assert (_flag(cap, "--lr"), _flag(cap, "--weight-decay"), _flag(cap, "--clip-grad")) == ("1e-6", "0.0", "1.0")
    assert _flag(cap, "--lr-decay-style") == "constant"
    assert (_flag(cap, "--rollout-temperature"), _flag(cap, "--rollout-top-p"), _flag(cap, "--rollout-top-k")) == (
        "1.0",
        "1.0",
        "-1",
    )
    # 16,384 a turn; the agent's 67,584 plus the margin for its estimate, on the engines and the trainer
    assert _flag(cap, "--rollout-max-response-len") == "16384"
    context = str(mod.AGENT_CONTEXT + mod.CONTEXT_MARGIN)
    for name in ("--max-seq-len", "--sglang-context-length", "--max-tokens-per-gpu"):
        assert _flag(cap, name) == context
    assert _flag(cap, "--tensor-model-parallel-size") == "1" and _flag(cap, "--context-parallel-size") == "1"
    assert int(_flag(cap, "--log-probs-chunk-size")) > 0
    # a full micro-batch's logits gradient in one piece: the trainer's allocator grows one mapping
    argv = shlex.split(cap["train_args"])
    assert json.loads(argv[argv.index("--train-env-vars") + 1]) == mod.TRAIN_ENV_VARS
    assert mod.TRAIN_ENV_VARS["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    # thinking on, the Qwen3.5 parser; the engines' FP32 LM head and the trainer's, its entropy observed
    assert argv[argv.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": true}'
    assert _flag(cap, "--sglang-tool-call-parser") == "qwen3_coder"
    assert {"--sglang-enable-fp32-lm-head", "--fp32-lm-head", "--observe-training-entropy"} <= set(tokens)
    mod.execute(mod.ScriptArgs(run_tag="t", fp32_lm_head=False, max_weight_staleness=4))
    assert not {"--fp32-lm-head", "--observe-training-entropy"} & set(_tokens(dict(CAPTURED)))
    assert _flag(CAPTURED, "--max-weight-staleness") == "4" and "--sglang-enable-fp32-lm-head" in _tokens(CAPTURED)
    # the engines' attention: trtllm_mha in 64-token pages; "" leaves sglang's pick
    assert (_flag(cap, "--sglang-attention-backend"), _flag(cap, "--sglang-page-size")) == ("trtllm_mha", "64")
    mod.execute(mod.ScriptArgs(run_tag="t", attention_backend=""))
    assert not {"--sglang-attention-backend", "--sglang-page-size"} & set(_tokens(dict(CAPTURED)))
    # MTP: Qwen3.5's own layer as a top-1 draft of mtp_steps steps; off by default
    assert "--sglang-speculative-algorithm" not in tokens
    mod.execute(mod.ScriptArgs(run_tag="t", mtp_steps=3))
    spec = dict(CAPTURED)
    names = ("algorithm", "eagle-topk", "num-steps", "num-draft-tokens")
    assert [_flag(spec, f"--sglang-speculative-{n}") for n in names] == ["NEXTN", "1", "3", "4"]
    assert _flag(cap, "--dynamic-sampling-filter-path") == stubs.launcher.NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER
    assert (_flag(cap, "--seed"), _flag(cap, "--rollout-seed")) == ("42", "42") and "--rollout-shuffle" in tokens
    assert _flag(cap, "--save-interval") == "20" and "--eval-interval" not in tokens
    assert cap["extra_env_vars"]["NEMO_GYM_RUN_TIMEOUT"] == "9000"
    mod.execute(mod.ScriptArgs(run_tag="t", num_gpus_per_node=8, train_nodes=2, rollout_nodes=2))
    assert (_flag(CAPTURED, "--actor-num-nodes"), _flag(CAPTURED, "--rollout-num-gpus")) == ("2", "16")
    # rows in an order of their own (prepare.py rows --order): miles' shuffle off
    mod.execute(mod.ScriptArgs(run_tag="t", rollout_shuffle=False))
    assert "--rollout-shuffle" not in _tokens(dict(CAPTURED))
    with pytest.raises(ValueError, match="eval-prompt-data"):
        mod.execute(mod.ScriptArgs(run_tag="t", eval_interval=50))
    mod.execute(mod.ScriptArgs(run_tag="t", eval_interval=50, eval_prompt_data="/d/e.jsonl"))
    assert _flag(CAPTURED, "--eval-temperature") == "0.7" and _flag(CAPTURED, "--eval-top-p") == "0.95"
    # eval-only without an eval interval has nothing to evaluate: refused before the launch
    with pytest.raises(ValueError, match="eval-only"):
        mod.execute(mod.ScriptArgs(eval_only=True, eval_interval=0))
