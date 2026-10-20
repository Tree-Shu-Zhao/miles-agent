# pyright: reportMissingImports=false
"""Eval-only re-decoding of a saved checkpoint under the recipe's own evaluation protocol.

The paper reports a cell from its final checkpoint under three decodes: the in-training eval after
step 150 (seed 42, T 0.4, 4096 tokens per turn) and two eval-only passes over the same weights with
fresh seeds (43, 44); the number is mean +- std of ``eval/<slice>/env/won`` (WebShop: won, never
purchased; ScienceWorld: ``env/won``, its top-level eval key is the shaped score). The anchors
(teacher alone, student alone) go through the same path. Glue over the recipe script (``run.py``, or
``opd.py`` for the OPD arms), which keeps every protocol flag:

1. ``--num-rollout 0`` under ``--topology colocated_1n``: train.py's eval-only branch
   (3rdparty/miles/train.py:115) pushes the loaded weights to the engines, runs one eval pass and
   returns; train_async.py has no such branch.
2. A load view: ``<arm>/eval_view_<it>/`` holds a symlink to the arm's ``iter_<it>`` plus its own
   ``latest_checkpointed_iteration.txt``, so the ``--load`` that ``patch_execute_train`` appends (with
   ``--no-load-optim --no-load-rng``) picks exactly that iteration and the arm's own tracker is never
   touched. The view is built only once the arm's tracker has reached ``it``: the iter_ dir appears
   before the save completes (the partial-checkpoint pitfall) and the tracker is Megatron's completion
   marker. Save ids are 0-based: the final checkpoint of a 150-step arm is ``iter_0000149``.
3. ``--eval-base``: no ``--load``; ``--mode debug_rollout_only`` with ``--skip-prepare`` builds no
   Megatron actor and the engines serve the HF dir of POLICY_MODEL (``--hf-checkpoint``). The paper ran
   every anchor this way: the Qwen3-30B-A3B MoE weight push dies in sglang's flattened-bucket update
   (expert w13 shape mismatch), and the rollout-only numbers matched the pushed ones (Qwen3.5-9B
   ALFWorld 52.1/53.0 vs 51.4/53.0, 2026-09-12).

The recipe script exposes its args dataclass (``ScriptArgs`` in run.py, ``OpdArgs`` in opd.py: the one
BaseScriptArgs subclass it defines) and ``SPEC`` (opd.py's through its ``run``); cleanup / prepare /
execute are the launcher's, as in every recipe's ``main``.

Entry lines (scripts/run_local.sh, or scripts/eval_ckpt.sh; one decode per run, RUN_TAG = the decode's own tag):
    RECIPE=<env> RUN_SCRIPT=eval_ckpt.py TOPOLOGY=colocated_1n \\
        EXTRA_ARGS="--recipe-script opd.py --eval-load-tag <arm RUN_TAG> --seed 43"   # then --seed 44
    RECIPE=<env> RUN_SCRIPT=eval_ckpt.py TOPOLOGY=colocated_1n POLICY_MODEL=<HF dir name> \\
        EXTRA_ARGS="--recipe-script opd.py --eval-base --seed 42"                        # anchors: 42, 43, 44
``--eval-load-iter N`` waits for the arm's tracker to reach N; the default -1 takes what it names now.
An arm trained with a non-default ``--num-rollout`` needs the same flag on its eval job line: the save-id
bound (``num_rollout - 1``) is read from this invocation, not from the arm.
"""

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, fields, make_dataclass
from pathlib import Path
from types import ModuleType

import miles.utils.external_utils.command_utils as U
import typer

from miles_agent import launcher

TRACKER = "latest_checkpointed_iteration.txt"
# Megatron writes the tracker after the iter_ dir is complete; an arm still saving is polled this often.
TRACKER_POLL_S = 60.0


@dataclass
class EvalArgs(launcher.BaseScriptArgs):
    """The eval flags over the recipe script's own (eval_args_class mixes them into its dataclass)."""

    # eval-only exists in train.py only: a preset's training topology default gives way (an explicit
    # --topology must agree, evaluate() checks)
    topology: str = "colocated_1n"
    eval_load_tag: str = ""  # the arm's RUN_TAG under <results>/ (a sibling of this decode's --save-dir)
    eval_load_iter: int = -1  # the arm's save id to decode; -1 = what its tracker names now
    eval_base: bool = False  # the HF weights of --hf-checkpoint instead, rollout-only (the anchors)


def read_tracker(save_dir: Path) -> int | None:
    """The arm's last completed save id; None before the first save (or while the tracker is rewritten). A tracker
    that names no save id (Megatron's ``release``, a converted base) is an error, not a wait."""
    path = save_dir / TRACKER
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{path}: {text!r} is not a save id (Megatron's 'release' marks a converted base)") from None


def prepare_load_view(arm_save_dir: Path, it: int, *, num_rollout: int) -> Path:
    """A Megatron --load dir pinned at the arm's ``iter_<it>`` (``it`` < 0: the save its tracker names now).

    Waits while the arm's tracker is behind a reachable ``it``; an ``it`` past the arm's last save id
    (``num_rollout - 1``: a 150-step arm ends at iter_0000149) fails fast instead. Decodes 2 and 3 of a cell
    build the same view at the same time: an existing link to the same ``iter_<it>`` is theirs, any other is an error.
    """
    if it > num_rollout - 1:
        raise ValueError(
            f"--eval-load-iter {it} is past the arm's last save id {num_rollout - 1} "
            f"(num_rollout {num_rollout}: the final checkpoint is iter_{num_rollout - 1:07d})"
        )
    saved = read_tracker(arm_save_dir)
    if it < 0:
        if saved is None:
            raise FileNotFoundError(f"{arm_save_dir / TRACKER}: the arm has not completed a save yet")
        it = saved
    while saved is None or saved < it:
        print(f"EVAL_CKPT waiting for {arm_save_dir / TRACKER} to reach {it} (now {saved})", flush=True)
        time.sleep(TRACKER_POLL_S)
        saved = read_tracker(arm_save_dir)
    src_iter = arm_save_dir / f"iter_{it:07d}"
    if not src_iter.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {src_iter}")
    view = arm_save_dir / f"eval_view_{it}"
    view.mkdir(exist_ok=True)
    link = view / src_iter.name
    target = src_iter.resolve()
    try:
        link.symlink_to(target)
    except FileExistsError:
        pass  # the other decode won the race: its link is checked below like any pre-existing one
    if link.resolve() != target:
        raise ValueError(f"{link} resolves to {link.resolve()}, not to {target}")
    (view / TRACKER).write_text(str(it))
    return view


def patch_execute_train(load_view: Path) -> None:
    """Append the load view to the launcher's command: launcher.execute resolves ``execute_train`` at call
    time (launcher.py "Import contract"), the way recipes/search_r1/eval_full.py rewrites it."""
    real = launcher.execute_train

    def patched(script_args, train_args: str, **kwargs):
        # weights only: optimizer and rng state are dead weight for an eval pass (and most of the bytes)
        load = f"--load {load_view} --no-load-optim --no-load-rng "
        return real(script_args, train_args=f"{train_args.rstrip()} {load}", **kwargs)

    launcher.execute_train = patched


def load_recipe(recipe_dir: Path, script: str) -> ModuleType:
    """Import recipes/<env>/<script> by path, the recipe dir on sys.path for its ``import run`` and bare modules."""
    path = recipe_dir / script
    if not path.is_file():
        raise FileNotFoundError(f"{recipe_dir} has no {script}")
    if str(recipe_dir) not in sys.path:
        sys.path.insert(0, str(recipe_dir))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def script_args_class(recipe: ModuleType) -> type[launcher.BaseScriptArgs]:
    """The script's own args dataclass (ScriptArgs in run.py, OpdArgs in opd.py), found by definition site: a
    preset imports run.py's class as its base."""
    own = [
        obj
        for obj in vars(recipe).values()
        if isinstance(obj, type) and issubclass(obj, launcher.BaseScriptArgs) and obj.__module__ == recipe.__name__
    ]
    if len(own) != 1:
        raise ValueError(f"{recipe.__file__} defines {len(own)} BaseScriptArgs dataclasses, expected exactly one")
    return own[0]


def recipe_spec(recipe: ModuleType) -> launcher.RecipeSpec:
    """The script's RecipeSpec: its own ``SPEC`` or, for a preset layered on run.py, run's."""
    spec = getattr(recipe, "SPEC", None) or getattr(getattr(recipe, "run", None), "SPEC", None)
    if not isinstance(spec, launcher.RecipeSpec):
        raise ValueError(f"{recipe.__file__} has no RecipeSpec (SPEC, or run.SPEC)")
    return spec


def eval_args_class(script_args: type[launcher.BaseScriptArgs]) -> type[EvalArgs]:
    """EvalArgs over the script's dataclass: its whole flag surface and defaults plus EvalArgs' own fields. A
    dataclass base contributes its whole field table (inherited defaults included), so EvalArgs' own fields are
    re-declared on the subclass: they win over the script's, and the script's win over BaseScriptArgs'."""
    own = [(f.name, f.type, f.default) for f in fields(EvalArgs) if f.name in EvalArgs.__annotations__]
    return make_dataclass("EvalArgs", own, bases=(script_args, EvalArgs))


def evaluate(args: EvalArgs, recipe: ModuleType) -> None:
    """One eval pass of the recipe protocol over the pinned weights; mirrors the recipe's ``main``."""
    if args.topology != "colocated_1n":
        raise ValueError("eval-only (--num-rollout 0) exists only in train.py: use --topology colocated_1n")
    # The arm's step count (its preset's default, or the --num-rollout it ran with) bounds the save ids.
    arm_num_rollout = args.num_rollout
    args.num_rollout = 0
    if args.eval_base:
        if args.eval_load_tag:
            raise ValueError("--eval-base decodes the HF weights of --hf-checkpoint: drop --eval-load-tag")
        # no Megatron actor, no conversion: the engines keep the HF weights (module docstring, 3)
        args.mode = "debug_rollout_only"
        args.skip_prepare = True
        print(f"EVAL_CKPT base={args.hf_checkpoint} seed={args.seed} (rollout-only, the HF weights)", flush=True)
    else:
        if not (args.save_dir and args.eval_load_tag):
            raise ValueError("--save-dir <results>/<RUN_TAG>/ and --eval-load-tag <arm RUN_TAG> (or --eval-base)")
        arm_save_dir = Path(args.save_dir.rstrip("/")).parent / args.eval_load_tag
        view = prepare_load_view(arm_save_dir, args.eval_load_iter, num_rollout=arm_num_rollout)
        patch_execute_train(view)
        print(f"EVAL_CKPT load={view} seed={args.seed}", flush=True)
    launcher.cleanup()
    if not args.skip_prepare:
        launcher.prepare(args)
    launcher.execute(args, recipe_spec(recipe))


def main(recipe_dir: Path) -> None:
    """Entry point of recipes/<env>/eval_ckpt.py: ``--recipe-script`` (default run.py; opd.py for the OPD arms)
    picks the script whose flags the CLI then carries, plus EvalArgs'."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe-script", default="run.py")
    picked, rest = pre.parse_known_args(sys.argv[1:])
    recipe = load_recipe(Path(recipe_dir), picked.recipe_script)
    eval_args = eval_args_class(script_args_class(recipe))

    @U.dataclass_cli
    def cli(args: eval_args) -> None:
        evaluate(args, recipe)

    sys.argv = [sys.argv[0], *rest]  # typer parses it; --recipe-script is no dataclass field
    typer.run(cli)
