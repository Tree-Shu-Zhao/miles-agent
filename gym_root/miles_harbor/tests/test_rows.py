"""miles_harbor.rows: the prompt rows of Harbor tasks gym_bridge posts to the trial server (the dataset's tasks
from a stand-in for trial_server.load_tasks)."""

import json

import pytest

pytest.importorskip("harbor")  # the Harbor package is in gym_root/miles_harbor's venv, not in every checkout

import miles_harbor.rows as rows_module  # noqa: E402
from miles_harbor.rows import main, pick, rows  # noqa: E402


@pytest.fixture
def tasks(tmp_path):
    """Two task dirs as Harbor lays them out, by name."""
    out = {}
    for name, instruction in (("django__django-15098", "Fix the locale.\n"), ("astropy__astropy-12907", "Fix it.\n")):
        root = tmp_path / "tasks" / name
        (root / "environment").mkdir(parents=True)
        (root / "tests").mkdir()
        (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
        (root / "instruction.md").write_text(instruction)
        (root / "task.toml").write_text('[environment]\ndocker_image = "alpine:3"\n')
        out[name] = root
    return out


def test_a_row_is_the_tasks_instruction_and_its_name(tasks):
    assert rows(tasks, 100) == [
        {"prompt": "Fix it.\n", "metadata": {"instance_id": "astropy__astropy-12907", "max_episode_steps": 100}},
        {"prompt": "Fix the locale.\n", "metadata": {"instance_id": "django__django-15098", "max_episode_steps": 100}},
    ]
    assert rows(tasks, None)[0]["metadata"] == {"instance_id": "astropy__astropy-12907"}


def test_a_sample_is_seeded_and_in_name_order():
    names = [f"t{i:02d}" for i in range(20)]
    assert pick(names, None, 0) == names and pick(names, 50, 0) == names
    sample = pick(list(reversed(names)), 5, 7)
    assert sample == pick(names, 5, 7) == sorted(sample) and len(set(sample)) == 5
    assert pick(names, 5, 8) != sample


def test_the_cli_writes_one_row_a_task(tasks, tmp_path, monkeypatch):
    asked = []

    async def fake_load_tasks(dataset, download_dir, task_names):
        asked.append((dataset, download_dir, task_names))
        return tasks

    monkeypatch.setattr(rows_module, "load_tasks", fake_load_tasks)
    out = tmp_path / "data" / "rows.jsonl"
    argv = ["--dataset", "swebench-verified@1.0", "--download-dir", str(tmp_path / "dl"), "--out", str(out)]
    assert main([*argv, "--n-tasks", "1", "--seed", "3", "--max-episode-steps", "100", "--tasks", "a*"]) == 0
    assert asked == [("swebench-verified@1.0", tmp_path / "dl", ["a*"])]
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["metadata"]["instance_id"] in tasks and row["metadata"]["max_episode_steps"] == 100
