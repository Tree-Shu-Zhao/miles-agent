"""ScienceWorld task splits, TCOD-aligned (arXiv 2604.24005), ported from
the author's earlier environment code (env_servers/scienceworld/task_splits.py).

Source of truth: TCOD_examples/scienceworld/get_sciworld_data.py in the pinned
TCOD checkout. Their construction:

    train = 17 TRAIN_TASK_NAMES x first int(0.5 * variations) variations
    test  = remaining 13 task types x first int(0.5 * variations) variations
    (both lists then random.seed(42)-shuffled)

We reproduce the SETS exactly but not their shuffle orders:
  - "train" / "train_heldout" are emitted in sorted-name order (deterministic;
    miles shuffles training prompts itself via --rollout-shuffle);
  - "test" is the frozen taskset of the TCOD evaluation (1308 rows of
    {"task_name", "var_num"}; the exact eval set the TCOD numbers were
    measured on), read VERBATIM in file order from the list that
    scripts/fetch_data.sh scienceworld writes (SCIENCEWORLD_TEST_LIST;
    recipes/scienceworld/data/README.md). The list is not committed; its
    SET is synthesized_test() below, its ORDER was one shuffle of the
    original campaign (the set difference at get_sciworld_data.py:131
    iterates in hash order, so the order is not reproducible).

"train_heldout" (the second half of each train type's variations) never
overlaps train and only feeds the valid_seen eval subset — TCOD has no
seen-task eval; this is our addition, disclosed in the docs.

Unlike the earlier code, dataset rows address a task by (task_name, var_num), not
by a split index, so there is no index invariant to keep between prepare_data
and the server; the split name on a row is validated for membership only.

TASK_VARIATIONS is copied verbatim from get_sciworld_data.py.
"""

import json
from pathlib import Path

TASK_VARIATIONS = {
    "boil": 30,
    "melt": 30,
    "freeze": 30,
    "change-the-state-of-matter-of": 30,
    "use-thermometer": 540,
    "measure-melting-point-known-substance": 436,
    "measure-melting-point-unknown-substance": 300,
    "power-component": 20,
    "power-component-renewable-vs-nonrenewable-energy": 20,
    "test-conductivity": 900,
    "test-conductivity-of-unknown-substances": 600,
    "find-living-thing": 300,
    "find-non-living-thing": 300,
    "find-plant": 300,
    "find-animal": 300,
    "grow-plant": 126,
    "grow-fruit": 126,
    "chemistry-mix": 32,
    "chemistry-mix-paint-secondary-color": 36,
    "chemistry-mix-paint-tertiary-color": 36,
    "lifespan-longest-lived": 125,
    "lifespan-shortest-lived": 125,
    "lifespan-longest-lived-then-shortest-lived": 125,
    "identify-life-stages-1": 14,
    "identify-life-stages-2": 10,
    "inclined-plane-determine-angle": 168,
    "inclined-plane-friction-named-surfaces": 1386,
    "inclined-plane-friction-unnamed-surfaces": 162,
    "mendelian-genetics-known-plant": 120,
    "mendelian-genetics-unknown-plant": 480,
}

TRAIN_TASK_NAMES = [
    "boil",
    "melt",
    "change-the-state-of-matter-of",
    "use-thermometer",
    "measure-melting-point-known-substance",
    "power-component",
    "test-conductivity",
    "find-living-thing",
    "find-plant",
    "grow-plant",
    "chemistry-mix",
    "chemistry-mix-paint-secondary-color",
    "lifespan-shortest-lived",
    "identify-life-stages-2",
    "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces",
    "mendelian-genetics-known-plant",
]

PERCENTAGE = 0.5
EXPECTED_TRAIN = 2294
EXPECTED_TEST = 1308
SPLITS = ("train", "train_heldout", "test")


def build_train() -> list[tuple[str, int]]:
    return [(name, var) for name in sorted(TRAIN_TASK_NAMES) for var in range(int(TASK_VARIATIONS[name] * PERCENTAGE))]


def build_train_heldout() -> list[tuple[str, int]]:
    return [
        (name, var)
        for name in sorted(TRAIN_TASK_NAMES)
        for var in range(int(TASK_VARIATIONS[name] * PERCENTAGE), TASK_VARIATIONS[name])
    ]


def load_test(test_jsonl: str | Path) -> list[tuple[str, int]]:
    """Frozen TCOD eval taskset, order-preserving. Accepts both the recipe's
    compact rows ({"task_name", "var_num"}) and the campaign's raw rows
    ({"task_desc": "<json with task_name/var_num>"})."""
    tasks: list[tuple[str, int]] = []
    with Path(test_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_desc" in row and "task_name" not in row:
                row = json.loads(row["task_desc"])
            tasks.append((str(row["task_name"]), int(row["var_num"])))
    if len(tasks) != EXPECTED_TEST:
        raise RuntimeError(f"{test_jsonl}: {len(tasks)} rows != {EXPECTED_TEST}")
    if len(set(tasks)) != len(tasks):
        raise RuntimeError(f"{test_jsonl}: duplicate (task, variation) pairs")
    train = set(build_train())
    overlap = train & set(tasks)
    if overlap:
        raise RuntimeError(f"train/test overlap (should be disjoint by task type): {sorted(overlap)[:3]}")
    return tasks


def build_splits(test_jsonl: str | Path) -> dict[str, list[tuple[str, int]]]:
    train = build_train()
    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(f"train split {len(train)} tasks != {EXPECTED_TRAIN}")
    return {
        "train": train,
        "train_heldout": build_train_heldout(),
        "test": load_test(test_jsonl),
    }


def synthesized_test() -> list[tuple[str, int]]:
    """The TCOD eval SET in sorted (task_name, var_num) order: the same 1308 tasks as the frozen
    list, not its order (recipes/scienceworld/data/README.md). load_test accepts its serialisation."""
    return [
        (name, var)
        for name in sorted(set(TASK_VARIATIONS) - set(TRAIN_TASK_NAMES))
        for var in range(int(TASK_VARIATIONS[name] * PERCENTAGE))
    ]


def write_test_list(tasks: list[tuple[str, int]], out_path: str | Path) -> None:
    """The recipe's compact rows, one per line, json.dumps default separators (the frozen list's bytes)."""
    with Path(out_path).open("w", encoding="utf-8") as fh:
        for name, var in tasks:
            fh.write(json.dumps({"task_name": name, "var_num": int(var)}) + "\n")


def validate_task(task_name: str, var_num: int, split: str | None = None) -> str | None:
    """Row-level check the env server runs at /reset (no frozen file needed):
    returns an error string or None. ``split`` (optional) must be one of
    SPLITS; train / train_heldout membership is exact, test rows only need a
    non-train task type (the frozen file is the recipe's business)."""
    if task_name not in TASK_VARIATIONS:
        return f"unknown ScienceWorld task {task_name!r}"
    if not isinstance(var_num, int) or isinstance(var_num, bool):
        return f"var_num must be an int, got {var_num!r}"
    n_var = TASK_VARIATIONS[task_name]
    if not 0 <= var_num < n_var:
        return f"var_num {var_num} out of range for {task_name} ({n_var} variations)"
    if split is None:
        return None
    if split not in SPLITS:
        return f"unknown split {split!r}; expected one of {SPLITS}"
    cut = int(n_var * PERCENTAGE)
    if split == "train" and not (task_name in TRAIN_TASK_NAMES and var_num < cut):
        return f"{task_name}/{var_num} is not in the train split"
    if split == "train_heldout" and not (task_name in TRAIN_TASK_NAMES and var_num >= cut):
        return f"{task_name}/{var_num} is not in the train_heldout split"
    if split == "test" and task_name in TRAIN_TASK_NAMES:
        return f"{task_name} is a train task type; it cannot be in the test split"
    return None
