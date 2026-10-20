"""recipes/search_r1/eval_full.py without miles: where the full wiki_search test set comes from.

prepare_full_test_data converts the 51,713-row test.jsonl next to the eval file (prepare_data.py, a job-unique
temp, an atomic rename). Its source is --full-test-jsonl, else $WIKI_SEARCH_DATA/test.jsonl (the rows dir the
wrapper exports), else <MILES_AGENT_DATA>/wiki_search/test.jsonl, else a ValueError
naming the flag; a file converted earlier needs no source at all.
miles' command_utils, typer and the sibling run.py are stubbed in sys.modules for the load (tests/test_launcher.py's
idiom, reduced to the surface eval_full.py touches), and the conversion is a recorder that writes the rows the
row-count gate expects.
"""

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EVAL_FULL = REPO / "recipes" / "search_r1" / "eval_full.py"


def _load_eval_full(monkeypatch):
    pkgs = {}
    for name in ("miles", "miles.utils", "miles.utils.external_utils"):
        pkgs[name] = types.ModuleType(name)
        pkgs[name].__path__ = []
    U = types.ModuleType("miles.utils.external_utils.command_utils")
    U.dataclass_cli = lambda func=None, **kw: func if func is not None else (lambda f: f)
    pkgs["miles"].utils = pkgs["miles.utils"]
    pkgs["miles.utils"].external_utils = pkgs["miles.utils.external_utils"]
    pkgs["miles.utils.external_utils"].command_utils = U

    @dataclasses.dataclass
    class ScriptArgs:
        # the fields of run.ScriptArgs that eval_full.py reads
        topology: str = "colocated_1n"
        skip_prepare: bool = False
        num_rollout: int = 500
        save_dir: str = ""
        eval_data: str = ""

    run = types.ModuleType("run")
    run.ScriptArgs = ScriptArgs
    run.launcher = types.SimpleNamespace(execute_train=lambda *a, **k: None)
    run.cleanup = lambda: None
    run.prepare = lambda args: None
    run.execute = lambda args: None
    typer = types.ModuleType("typer")
    typer.run = lambda f: None
    modules = {**pkgs, "miles.utils.external_utils.command_utils": U, "run": run, "typer": typer}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_eval_full_sources", EVAL_FULL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_full_source_prefers_the_flag_then_the_data_root(monkeypatch, tmp_path):
    mod = _load_eval_full(monkeypatch)
    calls = []
    replaced = []

    def fake_run(argv, check=False, **kwargs):
        # prepare_data.py --input <src> --output <tmp>: record the argv, write the rows the gate expects
        calls.append(list(argv))
        Path(argv[argv.index("--output") + 1]).write_text("{}\n" * mod.FULL_TEST_ROWS)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.os, "replace", lambda src, dst: replaced.append((str(src), str(dst))))
    for key in ("WIKI_SEARCH_DATA", "MILES_AGENT_DATA"):
        monkeypatch.delenv(key, raising=False)

    def convert(case, **kwargs):
        here = tmp_path / case
        here.mkdir()
        args = mod.EvalArgs(eval_data=str(here / "search_r1_eval893.jsonl"), **kwargs)
        mod.prepare_full_test_data(args)
        argv = calls[-1]
        dst = here / "search_r1_test51713.jsonl"
        assert argv[0] == sys.executable and argv[1] == str(mod.SCRIPT_DIR / "prepare_data.py")
        tmp = Path(argv[argv.index("--output") + 1])
        assert tmp.parent == here and tmp.name.startswith("search_r1_test51713.tmp.")
        assert replaced[-1] == (str(tmp), str(dst)) and args.eval_data == str(dst)
        return argv[argv.index("--input") + 1]

    monkeypatch.setenv("WIKI_SEARCH_DATA", "/w")
    monkeypatch.setenv("MILES_AGENT_DATA", "/d")
    assert convert("flag", full_test_jsonl="/flag/test.jsonl") == "/flag/test.jsonl"
    assert convert("wiki") == "/w/test.jsonl"  # the wrapper's export wins over the data root (the ladder's order)
    monkeypatch.delenv("WIKI_SEARCH_DATA")
    assert convert("root") == "/d/wiki_search/test.jsonl"
    monkeypatch.delenv("MILES_AGENT_DATA")
    (tmp_path / "none").mkdir()
    before = len(calls)
    with pytest.raises(ValueError, match="--full-test-jsonl.*WIKI_SEARCH_DATA"):
        mod.prepare_full_test_data(mod.EvalArgs(eval_data=str(tmp_path / "none" / "search_r1_eval893.jsonl")))
    assert len(calls) == before
    assert mod.full_test_source(mod.EvalArgs(full_test_jsonl="/f/test.jsonl")) == Path("/f/test.jsonl")


def test_eval_full_flag_is_a_field_and_a_converted_file_needs_no_source(monkeypatch, tmp_path):
    """--full-test-jsonl is a dataclass field of EvalArgs (dataclass_cli turns every field into a flag), empty by
    default; a test file converted by an earlier job is reused without any source."""
    mod = _load_eval_full(monkeypatch)
    fields = {f.name: f for f in dataclasses.fields(mod.EvalArgs)}
    assert fields["full_test_jsonl"].default == "" and fields["eval_load_tag"].default == ""
    assert issubclass(mod.EvalArgs, sys.modules["run"].ScriptArgs)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: pytest.fail("the conversion ran"))
    for key in ("WIKI_SEARCH_DATA", "MILES_AGENT_DATA"):
        monkeypatch.delenv(key, raising=False)
    dst = tmp_path / "search_r1_test51713.jsonl"
    dst.write_text("{}\n" * mod.FULL_TEST_ROWS)
    args = mod.EvalArgs(eval_data=str(tmp_path / "search_r1_eval893.jsonl"))
    mod.prepare_full_test_data(args)
    assert args.eval_data == str(dst)
