"""Behavior tests for miles_agent.eval_ckpt (eval-only re-decoding of a saved checkpoint) and the recipe shims.

miles' command_utils and typer are the stubs of tests/test_launcher.py: its module-scoped `stubs` fixture is
imported here and runs again for this module. The recipe scripts are fakes written to tmp_path and loaded by path,
the way the tool loads them under the training entry.
"""

import importlib
import runpy
import shlex
import sys
import typing
from pathlib import Path

import pytest
from test_launcher import CAPTURED, _flag, _tokens, stubs  # noqa: F401  (the fixture registers itself here)

REPO = Path(__file__).resolve().parent.parent
RECIPES = ("alfworld", "webshop", "scienceworld")
TRACKER = "latest_checkpointed_iteration.txt"
# launcher.cleanup / launcher.prepare calls made by evaluate(); the fixture below records them
CALLS: list[str] = []

# A recipe the way recipes/<env>/run.py is laid out (ScriptArgs, SPEC, the launcher's entry points) ...
FAKE_RUN = '''\
from dataclasses import dataclass

from miles_agent import launcher
from miles_agent.launcher import BaseScriptArgs, RecipeSpec


@dataclass
class ScriptArgs(BaseScriptArgs):
    save_dir: str = "/root/fake/"
    prompt_data: str = "/root/fake_train.jsonl"
    eval_data: str = "/root/fake_eval.jsonl"


SPEC = RecipeSpec(name="fake", run_file=__file__, eval_slices=(("fake_seen", "@[0:10]"),))
cleanup = launcher.cleanup
prepare = launcher.prepare


def execute(args):
    launcher.execute(args, SPEC)
'''
# ... and a preset the way recipes/<env>/opd.py layers on it (design §2.5): the paper's eval protocol (T 0.4,
# 4096 per turn, thinking off), a multi-node training topology, no SPEC of its own.
FAKE_OPD = '''\
from dataclasses import dataclass

import run
from run import ScriptArgs


@dataclass
class OpdArgs(ScriptArgs):
    topology: str = "async_3n"
    num_rollout: int = 150
    rollout_temperature: float = 0.4
    eval_temperature: float = 0.4
    eval_max_response_len: int = 4096
    chat_template_kwargs: str = '{"enable_thinking": false}'
'''


@pytest.fixture(scope="module")
def eval_ckpt(request):
    """miles_agent.eval_ckpt imported under the stubs (requested by name: the imported fixture as a parameter
    would read as a redefinition to the linter); its `launcher` is the stubbed one."""
    request.getfixturevalue("stubs")
    sys.modules.pop("miles_agent.eval_ckpt", None)
    module = importlib.import_module("miles_agent.eval_ckpt")
    yield module
    sys.modules.pop("miles_agent.eval_ckpt", None)


@pytest.fixture
def recipe_dir(tmp_path, eval_ckpt, monkeypatch):
    """A fake recipe dir (run.py + opd.py). cleanup() would pgrep/kill and sleep 5 s and prepare() is the stub
    conversion, so both are recorded instead; the execute_train patch must not outlive the test."""
    recipe = tmp_path / "recipes" / "fake"
    recipe.mkdir(parents=True)
    (recipe / "run.py").write_text(FAKE_RUN)
    (recipe / "opd.py").write_text(FAKE_OPD)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(eval_ckpt.launcher, "cleanup", lambda: CALLS.append("cleanup"))
    monkeypatch.setattr(eval_ckpt.launcher, "prepare", lambda args: CALLS.append("prepare"))
    monkeypatch.setattr(eval_ckpt.launcher, "execute_train", eval_ckpt.launcher.execute_train)
    for name in ("run", "opd"):
        sys.modules.pop(name, None)
    CALLS.clear()
    CAPTURED.clear()
    yield recipe
    for name in ("run", "opd"):
        sys.modules.pop(name, None)


def _arm(results: Path, tag: str = "arm", saved: int = 149) -> Path:
    """An arm's save dir with its final checkpoint complete (the tracker names it)."""
    arm = results / tag
    (arm / f"iter_{saved:07d}").mkdir(parents=True)
    (arm / TRACKER).write_text(str(saved))
    return arm


def _eval_args(eval_ckpt, recipe):
    return eval_ckpt.eval_args_class(eval_ckpt.script_args_class(recipe))


def test_prepare_load_view_writes_latest_iteration_and_waits_for_a_complete_tracker(eval_ckpt, tmp_path, monkeypatch):
    """The iter_ dir of a save in progress exists before Megatron writes the tracker: the view waits for the
    tracker (polling), then pins the iteration with its own tracker and a symlink; the arm's tracker stays."""
    arm = tmp_path / "arm"
    (arm / "iter_0000124").mkdir(parents=True)
    (arm / "iter_0000149").mkdir()
    (arm / TRACKER).write_text("124")
    polls = []

    def tracker_catches_up(seconds):
        polls.append(seconds)
        (arm / TRACKER).write_text("149")

    monkeypatch.setattr(eval_ckpt.time, "sleep", tracker_catches_up)
    view = eval_ckpt.prepare_load_view(arm, 149, num_rollout=150)
    assert view == arm / "eval_view_149" and polls == [eval_ckpt.TRACKER_POLL_S]
    assert (view / TRACKER).read_text() == "149"
    link = view / "iter_0000149"
    assert link.is_symlink() and link.resolve() == (arm / "iter_0000149").resolve()
    assert (arm / TRACKER).read_text() == "149" and not (arm / "eval_view_124").exists()
    # idempotent; -1 = the tracker's value; neither waits
    assert eval_ckpt.prepare_load_view(arm, 149, num_rollout=150) == view
    assert eval_ckpt.prepare_load_view(arm, -1, num_rollout=150) == view
    assert polls == [eval_ckpt.TRACKER_POLL_S]
    # an earlier complete save pins without waiting, in its own view
    earlier = eval_ckpt.prepare_load_view(arm, 124, num_rollout=150)
    assert earlier == arm / "eval_view_124"
    assert (earlier / TRACKER).read_text() == "124" and (earlier / "iter_0000124").is_symlink()
    # a tracker caught while rewritten in place reads as "not yet"
    (arm / TRACKER).write_text("")
    assert eval_ckpt.read_tracker(arm) is None


def test_prepare_load_view_rejects_unreachable_iter(eval_ckpt, tmp_path, monkeypatch):
    """F10: a 150-step arm ends at iter_0000149 (0-based save ids); past that nothing waits, it fails fast."""
    monkeypatch.setattr(eval_ckpt.time, "sleep", lambda seconds: pytest.fail("waited for an unreachable iteration"))
    arm = _arm(tmp_path)
    with pytest.raises(ValueError, match="iter_0000149"):
        eval_ckpt.prepare_load_view(arm, 150, num_rollout=150)
    assert not (arm / "eval_view_150").exists()
    # no save yet: -1 names nothing; a complete tracker whose iter_ dir is gone is an error, not a wait
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(FileNotFoundError):
        eval_ckpt.prepare_load_view(fresh, -1, num_rollout=150)
    with pytest.raises(FileNotFoundError, match="iter_0000124"):
        eval_ckpt.prepare_load_view(arm, 124, num_rollout=150)


def test_prepare_load_view_tolerates_a_concurrent_decode(eval_ckpt, tmp_path, monkeypatch):
    """Decodes 2 and 3 of a cell (seeds 43/44) build the same <arm>/eval_view_<it>/ at the same time: the loser
    of the symlink race keeps the winner's link to the same iter_, and a link that names another iter_ is refused."""
    arm = _arm(tmp_path)
    (arm / "iter_0000124").mkdir()
    real_symlink_to = Path.symlink_to

    def raced(self, target, *args, **kwargs):
        real_symlink_to(self, target, *args, **kwargs)  # the other decode won between the checks and this call
        raise FileExistsError(str(self))

    monkeypatch.setattr(Path, "symlink_to", raced)
    view = eval_ckpt.prepare_load_view(arm, 149, num_rollout=150)
    link = view / "iter_0000149"
    assert link.is_symlink() and link.resolve() == (arm / "iter_0000149").resolve()
    assert (view / TRACKER).read_text() == "149"
    # the later decode finds the earlier one's link, already correct
    monkeypatch.setattr(Path, "symlink_to", real_symlink_to)
    assert eval_ckpt.prepare_load_view(arm, 149, num_rollout=150) == view
    assert link.resolve() == (arm / "iter_0000149").resolve() and (view / TRACKER).read_text() == "149"
    # a link under the iter_'s name that resolves elsewhere is an error naming both, never a silent decode
    stale = arm / "eval_view_124"
    stale.mkdir()
    (stale / "iter_0000124").symlink_to(arm / "iter_0000149")
    with pytest.raises(ValueError, match=r"iter_0000124 resolves to .*iter_0000149, not to .*iter_0000124"):
        eval_ckpt.prepare_load_view(arm, 124, num_rollout=150)
    assert not (stale / TRACKER).exists()


def test_read_tracker_rejects_non_numeric_text(eval_ckpt, tmp_path, monkeypatch):
    """Megatron's ``release`` tracker (a converted base, never an arm) names no save id: one ValueError with the file
    and its text instead of an int() traceback, and prepare_load_view surfaces it rather than waiting."""
    base = tmp_path / "Qwen3.5-4B_torch_dist"
    base.mkdir()
    (base / TRACKER).write_text("release\n")
    with pytest.raises(ValueError, match=f"{TRACKER}: 'release' ") as info:
        eval_ckpt.read_tracker(base)
    assert str(base / TRACKER) in str(info.value) and "\n" not in str(info.value)
    monkeypatch.setattr(eval_ckpt.time, "sleep", lambda seconds: pytest.fail("waited on a tracker naming no save id"))
    with pytest.raises(ValueError, match="'release'"):
        eval_ckpt.prepare_load_view(base, -1, num_rollout=150)
    with pytest.raises(ValueError, match="'release'"):
        eval_ckpt.prepare_load_view(base, 149, num_rollout=150)


def test_recipe_script_surface(eval_ckpt, recipe_dir):
    """The args class is the script's own BaseScriptArgs subclass (an imported one does not count); SPEC is the
    script's or, for a preset, its run's."""
    run = eval_ckpt.load_recipe(recipe_dir, "run.py")
    opd = eval_ckpt.load_recipe(recipe_dir, "opd.py")
    assert eval_ckpt.script_args_class(run) is run.ScriptArgs
    assert eval_ckpt.script_args_class(opd) is opd.OpdArgs and opd.ScriptArgs is run.ScriptArgs
    assert eval_ckpt.recipe_spec(run) is run.SPEC and eval_ckpt.recipe_spec(opd) is run.SPEC
    with pytest.raises(FileNotFoundError, match="nope.py"):
        eval_ckpt.load_recipe(recipe_dir, "nope.py")


def test_eval_ckpt_uses_the_recipe_script_args(eval_ckpt, recipe_dir, tmp_path, monkeypatch):
    """`--recipe-script opd.py`: the CLI carries the preset's flags (T 0.4, 4096 per turn, thinking off) with
    num_rollout forced to 0 under colocated_1n, and the launch loads the view with --no-load-optim --no-load-rng."""
    results = tmp_path / "results"
    arm = _arm(results)
    seen = {}

    def typer_run(cli):
        # the real typer builds the CLI from U.dataclass_cli's signature; the stub leaves the function as is
        seen["argv"] = list(sys.argv)
        seen["cls"] = cls = typing.get_type_hints(cli)["args"]
        cli(cls(save_dir=f"{results}/ev_arm_s1001/", eval_load_tag="arm", seed=1001))

    monkeypatch.setattr(eval_ckpt.typer, "run", typer_run)
    argv = ["eval_ckpt.py", "--recipe-script", "opd.py", "--eval-load-tag", "arm", "--seed", "1001"]
    monkeypatch.setattr(sys, "argv", argv)
    eval_ckpt.main(recipe_dir)
    assert seen["argv"] == ["eval_ckpt.py", "--eval-load-tag", "arm", "--seed", "1001"]
    cls = seen["cls"]
    assert issubclass(cls, sys.modules["opd"].OpdArgs) and issubclass(cls, eval_ckpt.EvalArgs)
    # the eval default replaces the preset's training topology; the arm's step count stays for the save-id bound
    assert cls().topology == "colocated_1n" and cls().num_rollout == 150 and cls().eval_load_iter == -1
    toks = _tokens(CAPTURED)
    assert CAPTURED["train_script"] == "train.py" and "--colocate" in toks and "--debug-rollout-only" not in toks
    assert _flag(CAPTURED, "--num-rollout") == "0" and _flag(CAPTURED, "--seed") == "1001"
    assert _flag(CAPTURED, "--eval-temperature") == "0.4" and _flag(CAPTURED, "--eval-max-response-len") == "4096"
    assert _flag(CAPTURED, "--rollout-temperature") == "0.4" and "--eval-interval" in toks
    quoted = shlex.split(CAPTURED["train_args"])
    assert quoted[quoted.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    view = arm / "eval_view_149"
    assert toks[-4:] == ["--load", str(view), "--no-load-optim", "--no-load-rng"] and toks.count("--load") == 1
    assert (view / "iter_0000149").is_symlink() and (view / TRACKER).read_text() == "149"
    # the recipe's entry points, as its main runs them, and run.SPEC's eval slices
    assert CALLS == ["cleanup", "prepare"] and "fake_seen" in toks


def test_eval_base_skips_load(eval_ckpt, recipe_dir, tmp_path):
    """--eval-base decodes the HF weights of --hf-checkpoint: no load view, no --load, no conversion."""
    recipe = eval_ckpt.load_recipe(recipe_dir, "run.py")
    cls = _eval_args(eval_ckpt, recipe)
    args = cls(save_dir=f"{tmp_path}/results/base_s42/", eval_base=True, seed=42, hf_checkpoint="/models/Qwen3.5-9B")
    eval_ckpt.evaluate(args, recipe)
    toks = _tokens(CAPTURED)
    assert "--load" not in toks and "--no-load-optim" not in toks and "--no-load-rng" not in toks
    assert _flag(CAPTURED, "--num-rollout") == "0" and _flag(CAPTURED, "--hf-checkpoint") == "/models/Qwen3.5-9B"
    assert args.skip_prepare is True and CALLS == ["cleanup"]
    assert not list((tmp_path / "results").rglob("eval_view_*"))
    with pytest.raises(ValueError, match="--eval-load-tag"):
        eval_ckpt.evaluate(cls(save_dir=f"{tmp_path}/results/x/", eval_base=True, eval_load_tag="arm"), recipe)


def test_eval_base_runs_rollout_only(eval_ckpt, recipe_dir, tmp_path):
    """F2(3): no Megatron actor at all (--debug-rollout-only; miles drops colocate/offload itself), the engines keep
    the HF weights; train.py's eval-only branch needs --eval-interval, which the launcher always emits."""
    recipe = eval_ckpt.load_recipe(recipe_dir, "opd.py")
    cls = _eval_args(eval_ckpt, recipe)
    args = cls(save_dir=f"{tmp_path}/results/base_s1001/", eval_base=True, seed=1001)
    eval_ckpt.evaluate(args, recipe)
    toks = _tokens(CAPTURED)
    assert args.mode == "debug_rollout_only" and "--debug-rollout-only" in toks
    assert _flag(CAPTURED, "--num-rollout") == "0" and "--eval-interval" in toks
    assert CAPTURED["train_script"] == "train.py" and _flag(CAPTURED, "--seed") == "1001"
    assert _flag(CAPTURED, "--eval-temperature") == "0.4" and _flag(CAPTURED, "--eval-max-response-len") == "4096"


def test_evaluate_refuses_other_topologies(eval_ckpt, recipe_dir, tmp_path):
    """An explicit --topology other than colocated_1n is an error, not a silent rewrite: train_async.py has no
    eval-only branch, and the job's TOPOLOGY sized the allocation."""
    recipe = eval_ckpt.load_recipe(recipe_dir, "opd.py")
    cls = _eval_args(eval_ckpt, recipe)
    with pytest.raises(ValueError, match="colocated_1n"):
        eval_ckpt.evaluate(cls(save_dir=f"{tmp_path}/r/x/", eval_load_tag="arm", topology="async_3n"), recipe)
    assert CALLS == []


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_shims_call_eval_ckpt_main(eval_ckpt, monkeypatch, recipe):
    """recipes/<env>/eval_ckpt.py is a shim: it hands its own dir to miles_agent.eval_ckpt.main."""
    seen = []
    monkeypatch.setattr(eval_ckpt, "main", lambda recipe_dir: seen.append(Path(recipe_dir)))
    monkeypatch.setattr(sys, "path", list(sys.path))
    runpy.run_path(str(REPO / "recipes" / recipe / "eval_ckpt.py"), run_name="__main__")
    assert seen == [(REPO / "recipes" / recipe).resolve()]
