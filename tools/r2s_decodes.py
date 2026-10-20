#!/usr/bin/env python3
"""Mean +- std of eval/<slice>/env/won over the decodes of an R2S cell (docs/r2s.md section 9).

A cell is reported from its final checkpoint under three decodes, one run each: the arm's last
in-training eval (its training seed, 42) and two eval_ckpt.py passes over the same weights (seeds 43
and 44; the paper's own cells ran 42 / 1001 / 1002, the seeds only naming independent decodes). miles
prints every eval metrics dict it logs as one line of the job log
(3rdparty/miles/miles/ray/rollout/metrics.py:55, ``eval <rollout_id>: {...}``), so the decodes are
read from the logs whether or not wandb was on; a run's LAST eval record is its decode, the links of
a resumed arm oldest first. Per slice: the mean and std over the decodes and the values themselves;
a recipe with seen / unseen splits (ALFWorld, ScienceWorld) adds their mean per decode, the paper's
table entry (WebShop's entry is the ``webshop`` slice alone; the pool slice is diagnostic).

The estimator: --ddof 1 (default) is the sample std (n - 1, statistics.stdev), the summary the paper
tables were filled from (the text says "mean +- std over the three decodes" and names no estimator);
--ddof 0 is the population std (statistics.pstdev). A ``won`` miles printed as ``nan`` (no sample
scored) is shown as ``nan`` for that decode and left out of the mean and std, with a WARN line.

Three checks guard the entry. An arm's decode must be the eval of its final step: with --num-rollout N
(default 150, the presets') a run holding training-step records whose last eval is not rollout N - 1
gets ``WARN <run>: final eval is rollout <r>, expected <N - 1>`` (a cut-short chain); an eval-only pass
(eval_ckpt.py, --eval-base) holds no training record and miles dispatches its one eval as rollout 0
(train.py:115-117). A run's logs must hold the wrapper's JOB_PASS line (Contract D, docs/cluster.md: the wrapper
echoes ``JOB_PASS_miles_run_<RUN_TAG>`` once the entry returned 0, itself and so without a task label; a
chain's requeued link carries it): without one the run died, and an arm that died inside its first
rollout looks like an eval-only pass (one eval record, no training record), hence ``WARN <run>: no
JOB_PASS in its logs (the run did not finish)``. And an eval line is one logger write of several KB, past
the pipe's atomic size, so a task launcher that prefixes task labels (``0: ``) splits it in pieces and whole records of the same task land between them
(ray's forwarded actor lines, the driver's own timer lines, xtrace). The pieces are joined back around
those records, and a joined dict is the eval only when every key
is ``eval/...`` (a foreign tail piece can close it early); an eval record that never closes is counted in
its run's header line, ``unparsed=<n>``, and a nonzero count means the decode printed may be an earlier
eval than the run's last.

    python3 tools/r2s_decodes.py --runs r2s_alf1 r2s_alf1_d43 r2s_alf1_d44       # one RUN_TAG per decode
    python3 tools/r2s_decodes.py --logs run_11.log run_12.log run_13.log         # one job log per decode

--runs resolves a RUN_TAG to its job logs under --runs-dir (default $RUNS_DIR, else $MILES_AGENT_RESULTS/runs): the
entry writes <tag>_gym_<jobid>.log per link of the chain and the job log is run_<jobid>.log, or
miles_run_<jobid>.log when the wrapper set its own log name.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import statistics
import sys
from collections.abc import Iterator
from typing import NamedTuple

_EVAL = re.compile(r"metrics\.py:\d+ - eval (\d+): (\{.*)$")  # metrics.py:55, "file:line - eval N: {"
# a training step's records: metrics.py:93 "perf N:" per training
# rollout, log_utils.py:125 "rollout N:" and :554 "step N:" from the trainer; an eval-only pass writes none
_TRAIN = re.compile(r"\b(?:log_utils|perf|metrics)\.py:\d+ - (?:rollout|step|perf) \d+: \{")
_WON = re.compile(r"^eval/(.+)/env/won$")
_LABEL = re.compile(r"^(\d+): ?(.*)$")  # the launcher's per-rank label; the driver is rank 0
# a fresh log record of task 0 (ray's actor prefix, the driver logger's timestamp, xtrace), not the next
# piece of a line the job log wrapped
_NEW_RECORD = re.compile(r"^(?:\x1b\[|\[\d{4}-|\(\w+ pid=|\+ )")
_WRAPPED_VALUE = re.compile(r"\b(?:np\.\w+|tensor)\(([^()]*)\)")
_NON_FINITE = re.compile(r"(?<=: )-?(?:nan|inf)(?=[,}])")
_MAX_PIECES = 64  # pieces glued onto one eval line before it is given up as unparsed


class Decode(NamedTuple):
    """A run's decode: its last eval record; unparsed counts the eval records of its logs that never closed,
    train_records its training-step records (0: an eval-only pass), job_pass its JOB_PASS lines (0: the run did not
    finish). A NamedTuple, not a dataclass: the tests load the tool by path without registering it, and a dataclass
    resolves postponed annotations through sys.modules."""

    name: str
    rollout_id: int
    metrics: dict
    unparsed: int
    train_records: int
    job_pass: int


def _driver_lines(path: str) -> Iterator[str]:
    with open(path, errors="replace") as f:
        for raw in f:
            m = _LABEL.match(raw.rstrip("\n"))
            if m is None:
                yield raw.rstrip("\n")
            elif m.group(1) == "0":
                yield m.group(2)


def _parse(text: str) -> dict | None:
    try:
        value = ast.literal_eval(_NON_FINITE.sub("None", _WRAPPED_VALUE.sub(r"\1", text)))
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _is_eval_dict(d: dict) -> bool:
    """Every key eval/...: the tail piece of another wrapped line can close the glue early with its own keys."""
    return all(isinstance(k, str) and k.startswith("eval/") for k in d)


def eval_records(path: str) -> tuple[list[tuple[int, dict]], int]:
    """(rollout_id, metrics) per eval line of a job log, wrapped lines joined back, and how many did not parse."""
    out: list[tuple[int, dict]] = []
    bad = 0
    pending: list | None = None  # [rollout_id, text, pieces] of a dict still being read
    for line in _driver_lines(path):
        if m := _EVAL.search(line):
            if pending is not None:
                bad += 1  # the previous eval line never closed
            pending = [int(m.group(1)), m.group(2), 0]
        elif pending is None or _NEW_RECORD.match(line):
            continue  # a whole record between the pieces is skipped, not glued on
        else:
            pending[1] += line
            pending[2] += 1
        if pending[1].rstrip().endswith("}") and (parsed := _parse(pending[1])) and _is_eval_dict(parsed):
            out.append((pending[0], parsed))
            pending = None
        elif pending[2] >= _MAX_PIECES:
            bad += 1
            pending = None
    if pending is not None:
        bad += 1
    return out, bad


def train_records(path: str) -> int:
    """How many training-step records a job log holds (0: an eval-only pass, whose one eval is rollout 0)."""
    return sum(1 for line in _driver_lines(path) if _TRAIN.search(line))


def job_pass_lines(path: str) -> int:
    """How many JOB_PASS lines a job log holds: the wrapper echoes JOB_PASS_miles_run_<RUN_TAG> once the entry returned
    0 (its own echo: no task label), so a run without one over all its links died."""
    return sum(1 for line in _driver_lines(path) if line.startswith("JOB_PASS"))


def run_logs(runs_dir: str, tag: str) -> list[str]:
    """The job logs of a RUN_TAG, oldest link first, through the entry's <tag>_gym_<jobid>.log: run_<jobid>.log,
    else miles_run_<jobid>.log (a wrapper that set its own log name)."""
    paths = glob.glob(os.path.join(runs_dir, f"{tag}_gym_*.log"))
    jobs = sorted(int(m.group(1)) for p in paths if (m := re.search(r"_gym_(\d+)\.log$", p)))
    if not jobs:
        sys.exit(f"{tag}: no {tag}_gym_<jobid>.log under {runs_dir}")
    logs = []
    for job in jobs:
        names = [os.path.join(runs_dir, f"{prefix}{job}.log") for prefix in ("run_", "miles_run_")]
        logs.append(next((name for name in names if os.path.exists(name)), names[0]))
    return logs


def decode(name: str, logs: list[str]) -> Decode:
    """A run's decode: the last eval record over its logs, with the counts the checks read."""
    records: list[tuple[int, dict]] = []
    unparsed = trained = passed = 0
    for path in logs:
        if os.path.isfile(path):
            found, bad = eval_records(path)
            records += found
            unparsed += bad
            trained += train_records(path)
            passed += job_pass_lines(path)
    if not records:
        sys.exit(f"{name}: no eval record in {' '.join(logs)} (unparsed={unparsed})")
    rollout_id, metrics = records[-1]
    return Decode(
        name=name, rollout_id=rollout_id, metrics=metrics, unparsed=unparsed, train_records=trained, job_pass=passed
    )


def _won(d: Decode, slice_name: str) -> float | None:
    """The decode's won on a slice; None when miles printed it as nan (no sample scored)."""
    key = f"eval/{slice_name}/env/won"
    if key not in d.metrics:
        sys.exit(f"{d.name}: eval {d.rollout_id} has no {key}")
    value = d.metrics[key]
    return None if value is None else float(value)


def _stat(values: list[float | None], ddof: int) -> str:
    """mean +- std over the finite values (a nan is shown and left out); ddof 1 = sample std, 0 = population std."""
    finite = [v for v in values if v is not None]
    if len(finite) > 1:
        std = statistics.stdev(finite) if ddof else statistics.pstdev(finite)
    else:
        std = 0.0
    mean = statistics.fmean(finite) if finite else float("nan")
    return f"{mean:.4f} +- {std:.4f}  [{' '.join('nan' if v is None else f'{v:.4f}' for v in values)}]"


def summarize(decodes: list[Decode], *, num_rollout: int = 150, ddof: int = 1) -> list[str]:
    slices = sorted({m.group(1) for d in decodes for k in d.metrics if (m := _WON.match(k))})
    lines = [f"{d.name}: eval {d.rollout_id}  unparsed={d.unparsed}" for d in decodes]
    for d in decodes:
        # an arm's decode is its final step's eval; an eval-only pass (no training record) is rollout 0 by construction
        if d.train_records and d.rollout_id != num_rollout - 1:
            lines.append(
                f"WARN {d.name}: final eval is rollout {d.rollout_id}, expected {num_rollout - 1} "
                f"(--num-rollout {num_rollout}; a cut-short arm?)"
            )
        # an arm that died inside its first rollout looks like an eval-only pass (one eval record, no training record)
        if not d.job_pass:
            lines.append(f"WARN {d.name}: no JOB_PASS in its logs (the run did not finish)")
    won = {(i, s): _won(d, s) for i, d in enumerate(decodes) for s in slices}
    for (i, s), value in won.items():
        if value is None:
            lines.append(f"WARN {decodes[i].name}: eval/{s}/env/won is nan (left out of the mean +- std)")
    for s in slices:
        lines.append(f"{s:24s} {_stat([won[i, s] for i in range(len(decodes))], ddof)}")
    splits = [s for s in slices if s.endswith(("_seen", "_unseen"))]
    if len(splits) == 2:
        per_decode = [[won[i, s] for s in splits] for i in range(len(decodes))]
        means = [None if None in pair else statistics.fmean(pair) for pair in per_decode]
        lines.append(f"{'seen+unseen mean':24s} {_stat(means, ddof)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", metavar="RUN_TAG", help="one RUN_TAG per decode")
    source.add_argument("--logs", nargs="+", metavar="LOG", help="one job log per decode")
    default_dir = os.environ.get("RUNS_DIR") or (
        os.path.join(os.environ["MILES_AGENT_RESULTS"], "runs") if "MILES_AGENT_RESULTS" in os.environ else None
    )
    ap.add_argument(
        "--runs-dir",
        default=default_dir,
        help="the job logs' directory (default $RUNS_DIR, else $MILES_AGENT_RESULTS/runs)",
    )
    ap.add_argument(
        "--num-rollout",
        type=int,
        default=150,
        help="the arm's --num-rollout: its decode must be the eval of rollout N - 1 (default 150, the presets')",
    )
    ap.add_argument(
        "--ddof", type=int, choices=(0, 1), default=1, help="the std: 1 = sample std (default), 0 = population std"
    )
    args = ap.parse_args(argv)
    if args.runs:
        if not args.runs_dir:
            ap.error("--runs needs --runs-dir (or RUNS_DIR / MILES_AGENT_RESULTS in the environment)")
        decodes = [decode(tag, run_logs(args.runs_dir, tag)) for tag in args.runs]
    else:
        decodes = [decode(os.path.basename(path), [path]) for path in args.logs]
    print("\n".join(summarize(decodes, num_rollout=args.num_rollout, ddof=args.ddof)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
