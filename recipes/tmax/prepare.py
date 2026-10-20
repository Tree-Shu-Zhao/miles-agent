"""TMax-15K as Harbor tasks (the trial server's HARBOR_DATASET), and the recipe's rows of those whose image is in
``--images`` (one image a line).

    # allenai/tmax-15k-open-instruct: data/train-00000-of-00001.parquet and task-data.tar.gz, extracted
    python recipes/tmax/prepare.py tasks --parquet train.parquet --task-data <task-data dir> --out <dataset dir>
    python recipes/tmax/prepare.py rows --dataset <dataset dir> --images <image list> --out train.jsonl \
        [--order <image list>]

``tasks``: one Harbor task a row (its dir named by the row's task_id): instruction.md and tests/test.sh as TMax's
task data has them, and a task.toml of the row's prebuilt image (env_config.image; its setup.sh is baked in, so it
is not run) with TMax's sandbox limits: 4 GB of memory, no CPU limit (the provider's block of pinned CPUs), the tests
600 s. Each row's user message is checked to be vanillux's (miles_harbor.vanillux.render_instance) of the
instruction, so the harness gives the model the prompt TMax trained on.

``rows``: miles_harbor.rows' rows (prompt: the instruction; metadata: instance_id, max_episode_steps 64, TMax's tool
calls) of the tasks whose image is in ``--images`` (one image reference a line, ``docker://org/img:tag`` or
``org/img:tag``, an untagged one being ``:latest``: the images the sandboxes' store holds); the recipe shuffles
them (--rollout-shuffle, seed 42). With ``--order`` (one image a
line: the list the store was imported from, a seeded shuffle of the images) the rows come in that order instead, an
image's tasks by name, for the recipe's --no-rollout-shuffle: as the import goes on, a store's rows extend the rows
of the store before it, so a run resumed on the longer rows (miles keeps its offset into them) meets no task twice.
A store that holds none of the tasks' images is a FATAL: no rows file of 0 rows is written.

Run it with Harbor and miles_harbor importable (gym_root/miles_harbor's venv,
PYTHONPATH=gym_root:gym_root/miles_harbor).
"""

import argparse
import json
import sys
from pathlib import Path

TASK_TOML = """\
version = "1.0"

[metadata]
author_name = "allenai/tmax"
category = "terminal"
tags = ["tmax-15k", "{source}"]

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 3600.0

[environment]
build_timeout_sec = 600.0
docker_image = "{image}"
cpus = 1
memory_mb = 4096
storage_mb = 10240
"""


def write_tasks(parquet: Path, task_data: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq
    from miles_harbor.vanillux import render_instance

    counts = {"tasks": 0, "prompt_mismatch": 0}
    seen: set[str] = set()
    for row in pq.read_table(parquet).to_pylist():
        task_id = row["env_config"]["task_id"]
        if task_id != row["ground_truth"] or task_id in seen:
            raise ValueError(f"row of {task_id}: ground_truth {row['ground_truth']!r}, or a second row of it")
        seen.add(task_id)
        src = task_data / task_id
        instruction = (src / "instruction.md").read_text()
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        if render_instance(instruction.strip()) != user:
            counts["prompt_mismatch"] += 1
            print(f"PROMPT_MISMATCH {task_id}", file=sys.stderr)
        dst = out / task_id
        (dst / "tests").mkdir(parents=True, exist_ok=True)
        # Harbor wants the directory; left empty, it is not uploaded into the sandbox (a file in it, even a
        # .keep, would be: an exec and a tar per trial, and a stray file in the agent's working directory)
        (dst / "environment").mkdir(exist_ok=True)
        (dst / "instruction.md").write_text(instruction)
        test_sh = dst / "tests" / "test.sh"
        test_sh.write_bytes((src / "tests" / "test.sh").read_bytes())
        test_sh.chmod(0o755)
        (dst / "task.toml").write_text(TASK_TOML.format(image=row["env_config"]["image"], source=row["source"]))
        counts["tasks"] += 1
    return counts


def image_ref(image: str) -> str:
    """An image reference as the store's import spells it: no ``docker://``, ``:latest`` when its last path part
    carries no tag."""
    ref = image.strip().removeprefix("docker://")
    return ref if ":" in ref.rsplit("/", 1)[-1] else f"{ref}:latest"


def write_rows(
    dataset: Path,
    images: Path | None,
    out: Path,
    max_episode_steps: int,
    order: Path | None = None,
) -> dict:
    from harbor.models.task.task import Task
    from miles_harbor.rows import rows

    if images is None:
        raise ValueError("rows needs the store's image list (--images)")
    image_lines = images.read_text().splitlines()
    have_images = {image_ref(line) for line in image_lines if line.strip()}
    tasks, image_of, missing = {}, {}, 0
    for task_dir in sorted(p for p in dataset.iterdir() if (p / "task.toml").is_file()):
        task = Task(task_dir=task_dir)
        image = task.config.environment.docker_image
        if image_ref(image) in have_images:
            tasks[task.short_name] = task_dir
            image_of[task.short_name] = image
        else:
            missing += 1
    if not tasks:  # a rows file of 0 rows would pass here and fail at serve time: refuse it where the cause is
        what = f"({missing} tasks without an image)" if missing else "(no task dir with a task.toml)"
        raise SystemExit(
            f"FATAL: no task of {dataset} has its image in {images} {what}: nothing to write"
        )
    out_rows = rows(tasks, max_episode_steps)
    if order is not None:
        listed = [line.strip() for line in order.read_text().splitlines() if line.strip()]
        rank = {image: i for i, image in enumerate(listed)}
        # stable: an image's tasks keep their name order; an image the list lacks goes last
        out_rows.sort(key=lambda row: rank.get(image_of[row["metadata"]["instance_id"]], len(rank)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in out_rows))
    return {"rows": len(out_rows), "no_image": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--parquet", type=Path, required=True)
    t.add_argument("--task-data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("rows")
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument(
        "--images", type=Path, help="the store's images, one reference a line (docker://org/img:tag or org/img:tag)"
    )
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--max-episode-steps", type=int, default=64)
    r.add_argument("--order", type=Path, help="the rows in this image list's order (one image a line)")
    args = ap.parse_args(argv)
    if args.cmd == "tasks":
        counts = write_tasks(args.parquet, args.task_data, args.out)
    else:
        if args.images is None:
            r.error("--images <the store's image list> is required")
        counts = write_rows(args.dataset, args.images, args.out, args.max_episode_steps, args.order)
    print(f"TMAX_PREPARE {args.cmd} {json.dumps(counts)}", file=sys.stderr)
    return 1 if counts.get("prompt_mismatch") else 0


if __name__ == "__main__":
    sys.exit(main())
