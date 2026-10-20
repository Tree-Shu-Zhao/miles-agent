"""recipes/tmax/prepare.py rows: the rows of the tasks whose image the list has (--images, the store's image list),
in name order or in the order of the image list the store was imported from (--order). A store that holds none of
the tasks' images is a FATAL, not a rows file of 0 rows."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# prepare.py's own imports are the stdlib's (harbor and miles_harbor are imported inside write_tasks and write_rows),
# so the module loads anywhere; the tests that call write_rows take the `harbor` fixture
spec = importlib.util.spec_from_file_location("_tmax_prepare", REPO / "recipes" / "tmax" / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def harbor():
    """write_rows imports harbor and miles_harbor.rows: skipped where harbor is not importable (a machine without
    the Harbor venv); the cluster's CPU job runs these tests."""
    pytest.importorskip("harbor")
    miles_harbor = str(REPO / "gym_root" / "miles_harbor")
    if miles_harbor not in sys.path:
        sys.path.insert(0, miles_harbor)


def write_task(root: Path, name: str, image: str) -> None:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (task / "instruction.md").write_text(f"Do {name}.\n")
    (task / "task.toml").write_text(f'[environment]\ndocker_image = "{image}"\n')


def ids(path: Path) -> list[str]:
    return [json.loads(line)["metadata"]["instance_id"] for line in path.read_text().splitlines()]


def test_rows_follow_the_import_order_and_grow_as_the_store_does(tmp_path, harbor):
    dataset = tmp_path / "tasks"
    for name, image in (("t1", "org/i:c"), ("t2", "org/i:a"), ("t3", "org/i:b"), ("t4", "org/i:a"), ("t5", "org/i:d")):
        write_task(dataset, name, image)
    order = tmp_path / "images.txt"
    order.write_text("org/i:a\norg/i:b\norg/i:c\norg/i:d\n")
    store = tmp_path / "store_images.txt"  # the images the store holds
    store.write_text("org/i:c\norg/i:a\n")
    counts = prepare.write_rows(dataset, store, tmp_path / "plain.jsonl", 64)
    assert counts == {"rows": 3, "no_image": 2} and ids(tmp_path / "plain.jsonl") == ["t1", "t2", "t4"]
    row = json.loads((tmp_path / "plain.jsonl").read_text().splitlines()[0])
    assert row["metadata"] == {"instance_id": "t1", "max_episode_steps": 64} and row["prompt"] == "Do t1.\n"
    # in the list's order (an image's tasks by name); the store a step further on: the same rows, then its new ones
    prepare.write_rows(dataset, store, tmp_path / "first.jsonl", 64, order)
    store.write_text("org/i:c\norg/i:a\norg/i:d\n")
    prepare.write_rows(dataset, store, tmp_path / "later.jsonl", 64, order)
    assert ids(tmp_path / "first.jsonl") == ["t2", "t4", "t1"]
    assert ids(tmp_path / "later.jsonl") == ["t2", "t4", "t1", "t5"]
    # an image the list lacks: last
    order.write_text("org/i:c\n")
    prepare.write_rows(dataset, store, tmp_path / "unlisted.jsonl", 64, order)
    assert ids(tmp_path / "unlisted.jsonl") == ["t1", "t2", "t4", "t5"]


def test_rows_fatal_on_a_store_without_a_task(tmp_path, harbor):
    """A store that holds none of the tasks' images has nothing to serve: a FATAL of the prepare step, before any file
    is written, never a rows file of 0 rows (exit 0) that the trial server refuses later."""
    dataset = tmp_path / "tasks"
    for name, image in (("t1", "org/i:c"), ("t2", "org/i:a")):
        write_task(dataset, name, image)
    # a store with none of the tasks' images
    other = tmp_path / "other_images.txt"
    other.write_text("org/j:z\n")
    with pytest.raises(SystemExit, match="nothing to write") as e:
        prepare.write_rows(dataset, other, tmp_path / "none.jsonl", 64)
    assert str(e.value).startswith("FATAL: no task of ") and "(2 tasks without an image)" in str(e.value)
    assert not (tmp_path / "none.jsonl").exists()
    # a dataset without a task
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no task dir with a task.toml"):
        prepare.write_rows(tmp_path / "empty", other, tmp_path / "none.jsonl", 64)
