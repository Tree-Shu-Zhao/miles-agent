"""Prompt rows of Harbor tasks for miles (the tmax recipe's --prompt-data and
--eval-prompt-data): one row a task, its prompt the task's instruction, its metadata what gym_bridge posts to the
trial server's /run (``instance_id``: the task's name as the server knows it; ``max_episode_steps``: the agent's
turns, its harness's step limit).

    python -m miles_harbor.rows --dataset swebench-verified@1.0 --download-dir <dir> --out rows.jsonl \\
        [--tasks <name or glob> ...] [--n-tasks N [--seed S]] [--max-episode-steps 100]

The tasks are the trial server's (trial_server.load_tasks), so every row names one it serves; --n-tasks takes a
seeded sample of them.
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from harbor.models.task.task import Task

from miles_harbor.trial_server import load_tasks


def pick(names: list[str], n_tasks: int | None, seed: int) -> list[str]:
    """``n_tasks`` of ``names`` (all: None), a sample by ``seed``, in name order."""
    names = sorted(names)
    if n_tasks is None or n_tasks >= len(names):
        return names
    return sorted(random.Random(seed).sample(names, n_tasks))


def rows(tasks: dict[str, Path], max_episode_steps: int | None) -> list[dict]:
    out = []
    for name in sorted(tasks):
        metadata: dict = {"instance_id": name}
        if max_episode_steps:
            metadata["max_episode_steps"] = max_episode_steps
        out.append({"prompt": Task(task_dir=tasks[name]).instruction, "metadata": metadata})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="<name>@<version> (registry) or <org>/<name>[@<ref>] (Hub)")
    ap.add_argument("--download-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", help="only these tasks (names or globs)")
    ap.add_argument("--n-tasks", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episode-steps", type=int)
    args = ap.parse_args(argv)
    tasks = asyncio.run(load_tasks(args.dataset, args.download_dir, args.tasks or None))
    chosen = pick(list(tasks), args.n_tasks, args.seed)
    out = rows({name: tasks[name] for name in chosen}, args.max_episode_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in out))
    print(f"ROWS {len(out)} of {len(tasks)} tasks of {args.dataset} -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
