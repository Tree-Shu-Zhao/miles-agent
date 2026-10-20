# pyright: reportMissingImports=false
"""Eval-only launcher: run the FULL wiki_search test set (51,713 rows) against
one saved checkpoint, reusing run.py's science config verbatim.

Glue only — imports run.py and changes exactly three things at launch time:
1. --num-rollout 0 (train.py's eval-only special case; train_async.py has no
   such branch, so this only works with --topology colocated_1n),
2. the --eval-prompt-data slices: 893-row validation offsets -> full test.jsonl
   source-block offsets (same block order, verified),
3. appends --load <save_dir>, where save_dir is prepared as a "load view":
   a symlink to the chosen iter_NNNNNNN of the source run plus a
   latest_checkpointed_iteration.txt pinning that iteration (Megatron only
   loads whatever the tracker in --load names — this is how you load a
   non-latest checkpoint without touching the source run's tracker).

train.py pushes weights to the engines BEFORE the eval-only branch ("always
update weight first so that sglang has the loaded weights from training"), so
the eval genuinely scores the --load'ed checkpoint.

Usage (RUN_SCRIPT=eval_full.py on the training entry):
    python3 eval_full.py --topology colocated_1n \
        --save-dir $MILES_AGENT_RESULTS/evalfull_colo399/ ... \
        --eval-load-tag colo1n_r3 --eval-load-iter 399
    # the full test set comes from --full-test-jsonl, else $WIKI_SEARCH_DATA/test.jsonl (the entry's export),
    # else <MILES_AGENT_DATA>/wiki_search/test.jsonl
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import miles.utils.external_utils.command_utils as U
import run
import typer

SCRIPT_DIR = Path(__file__).resolve().parent

# Full test.jsonl source-block offsets (row counts verified against
# $MILES_AGENT_DATA/wiki_search/test.jsonl, same block order as the 893 set):
# nq 3610, triviaqa 11313, popqa 14267, hotpotqa 7405, 2wikimultihopqa 12576,
# musique 2417, bamboogle 125 -> total 51713.
FULL_TEST_ROWS = 51713
FULL_SLICES = [
    ("nq", 0, 3610),
    ("triviaqa", 3610, 14923),
    ("popqa", 14923, 29190),
    ("hotpotqa", 29190, 36595),
    ("2wikimultihopqa", 36595, 49171),
    ("musique", 49171, 51588),
    ("bamboogle", 51588, 51713),
]


@dataclass
class EvalArgs(run.ScriptArgs):
    # Source run tag under <results>/ (sibling of this job's save_dir) and the
    # checkpoint iteration to evaluate.
    eval_load_tag: str = ""
    eval_load_iter: int = -1
    # The full wiki_search test.jsonl to convert; "" = the data root's copy (full_test_source).
    full_test_jsonl: str = ""


def full_test_source(args: EvalArgs) -> Path:
    """Where the full test.jsonl comes from: --full-test-jsonl, else $WIKI_SEARCH_DATA/test.jsonl (the rows dir the
    wrapper exports; the ladder reads the same name), else <MILES_AGENT_DATA>/wiki_search/test.jsonl; otherwise the
    flag is required."""
    if args.full_test_jsonl:
        return Path(args.full_test_jsonl)
    if os.environ.get("WIKI_SEARCH_DATA"):
        return Path(os.environ["WIKI_SEARCH_DATA"]) / "test.jsonl"
    if os.environ.get("MILES_AGENT_DATA"):
        return Path(os.environ["MILES_AGENT_DATA"]) / "wiki_search" / "test.jsonl"
    raise ValueError(
        "no source for the full wiki_search test set: pass --full-test-jsonl <test.jsonl> "
        "or set WIKI_SEARCH_DATA (its test.jsonl is used) or MILES_AGENT_DATA (its wiki_search/test.jsonl)"
    )


def prepare_full_test_data(args: EvalArgs) -> None:
    """Convert the gym test.jsonl next to the existing eval file (idempotent)."""
    dst = Path(args.eval_data).parent / "search_r1_test51713.jsonl"

    def row_count(p: Path) -> int:
        return sum(1 for _ in open(p)) if p.exists() else -1

    # Concurrent eval jobs race on this shared file: never trust a partial
    # write — convert to a job-unique temp and atomically rename into place.
    if row_count(dst) != FULL_TEST_ROWS:
        src = full_test_source(args)
        tmp = dst.with_suffix(f".tmp.{os.getpid()}")
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_data.py"),
                "--input", str(src),
                "--output", str(tmp),
            ],
            check=True,
        )
        assert row_count(tmp) == FULL_TEST_ROWS, f"conversion produced {row_count(tmp)} rows"
        os.replace(tmp, dst)
    args.eval_data = str(dst)


def prepare_load_view(args: EvalArgs) -> None:
    """Turn save_dir into a Megatron --load dir pinned at eval_load_iter."""
    results_root = Path(args.save_dir.rstrip("/")).parent
    src_iter = results_root / args.eval_load_tag / f"iter_{args.eval_load_iter:07d}"
    assert src_iter.is_dir(), f"checkpoint not found: {src_iter}"
    view = Path(args.save_dir)
    view.mkdir(parents=True, exist_ok=True)
    link = view / src_iter.name
    if not link.is_symlink():
        link.symlink_to(src_iter)
    (view / "latest_checkpointed_iteration.txt").write_text(str(args.eval_load_iter))


def patch_execute_train(args: EvalArgs) -> None:
    """Swap the 893-row eval slices for the full-set ones and add --load."""
    e = args.eval_data
    full_eval = "--eval-prompt-data " + "".join(
        f"{name} {e}@[{lo}:{hi}] " for name, lo, hi in FULL_SLICES
    )
    # run.py imported the launcher after putting the repo root on sys.path.
    launcher = run.launcher
    real_execute_train = launcher.execute_train

    def patched(script_args, train_args: str, **kwargs):
        new_args, n = re.subn(
            r"--eval-prompt-data .*?(?=--n-samples-per-eval-prompt)",
            full_eval,
            train_args,
            flags=re.S,
        )
        assert n == 1, f"eval-prompt-data block not found in: {train_args[:500]}"
        # Weights only: optimizer/rng state is dead weight for eval (and the
        # bulk of the checkpoint bytes).
        new_args += f"--load {args.save_dir} --no-load-optim --no-load-rng "
        return real_execute_train(script_args, train_args=new_args, **kwargs)

    launcher.execute_train = patched  # launcher.execute resolves it at call time


@U.dataclass_cli
def main(args: EvalArgs):
    assert args.eval_load_tag and args.eval_load_iter >= 0, (
        "--eval-load-tag and --eval-load-iter are required"
    )
    assert args.topology == "colocated_1n", (
        "eval-only (--num-rollout 0) exists only in train.py: use colocated_1n"
    )
    args.num_rollout = 0
    run.cleanup()
    if not args.skip_prepare:
        run.prepare(args)
    prepare_full_test_data(args)
    prepare_load_view(args)
    patch_execute_train(args)
    run.execute(args)


if __name__ == "__main__":
    typer.run(main)
