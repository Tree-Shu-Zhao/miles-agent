"""miles_harbor.history_audit: SCRIPT run by bash in toy repositories (one whose HEAD has a future: a branch, a tag,
a reflog and a dangling commit past it; one without), its output read, and the report over a Harbor job dir."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")  # the Harbor package is in gym_root/miles_harbor's venv, not in every checkout

from miles_harbor.history_audit import OUTPUT, SCRIPT, TIMEOUT_S, HistoryAudit, audits, read, report  # noqa: E402

# committer times (git reads a bare number as a timestamp from 100000000 on)
BASE, HEAD, FIX, DANGLING = 1_600_000_000, 1_600_001_000, 1_600_002_000, 1_600_003_000


def git(repo, *args, when=None):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    if when is not None:
        env |= {"GIT_AUTHOR_DATE": f"{when} +0000", "GIT_COMMITTER_DATE": f"{when} +0000"}
    argv = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(argv, cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, name, when):
    (repo / name).write_text(name)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name, when=when)
    return git(repo, "rev-parse", "HEAD")


def make_repo(path, future):
    """base <- head (HEAD, main); with a future: fix (branch fix, tag v2) after head, and a commit no ref or reflog
    reaches (dangling), each committed at its time above. Each commit adds a file of its name, head's named HEAD (a
    revision the audit's git commands must not read as a path)."""
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    commit(path, "base", BASE)
    repo = SimpleNamespace(path=path, head=commit(path, "HEAD", HEAD))
    if future:
        git(path, "checkout", "-q", "-b", "fix")
        repo.fix = commit(path, "fix", FIX)
        git(path, "tag", "v2")
        git(path, "checkout", "-q", "main")
        repo.dangling = git(path, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "dangling", when=DANGLING)
    return repo


def audit_output(cwd):
    out = subprocess.run(["bash", "-c", SCRIPT], cwd=cwd, capture_output=True, text=True, timeout=60)
    return out.stdout + out.stderr


@pytest.fixture
def leaky(tmp_path):
    return make_repo(tmp_path / "leaky", future=True)


@pytest.fixture
def clean(tmp_path):
    return make_repo(tmp_path / "clean", future=False)


def test_a_future_is_seen_through_refs_reflogs_and_unreachable_commits(leaky):
    a = read(audit_output(leaky.path))
    assert a["done"] and (a["head"], a["head_ct"]) == (leaky.head, HEAD)
    assert (a["shallow"], a["alternates"], a["dirty"]) == (0, 0, 0)
    assert (a["head_commits"], a["all_commits"], a["extra"], a["future"]) == (2, 4, 2, 2)
    # oldest first: "<committer time> <sha> <subject>"
    future = [line.split()[:2] for line in a["future_commit"]]
    assert future == [[str(FIX), leaky.fix], [str(DANGLING), leaky.dangling]]
    assert (a["refs"], a["refs_ahead"], a["ref_ahead"]) == (3, 2, [f"{FIX} refs/heads/fix", f"{FIX} refs/tags/v2"])
    assert (a["reflog_extra"], a["stash"], a["unreachable"]) == (1, 0, {"commit": 1})


def test_a_head_without_a_future_has_none(clean):
    a = read(audit_output(clean.path))
    assert a["done"] and a["head"] == clean.head and (a["extra"], a["future"], a["refs_ahead"]) == (0, 0, 0)
    assert (a["future_commit"], a["ref_ahead"], a["reflog_extra"], a["unreachable"]) == ([], [], 0, {})


def test_no_repository_is_an_error(tmp_path):
    a = read(audit_output(tmp_path))
    assert a["error"] == "no-repository" and "done" not in a


def trial(job, name, task, output):
    """A trial's dir as Harbor leaves it: result.json, and the agent's output unless None."""
    (job / name / "agent").mkdir(parents=True)
    (job / name / "result.json").write_text(json.dumps({"task_name": task}))
    if output is not None:
        (job / name / "agent" / OUTPUT).write_text(output)


def test_the_report_counts_the_audits_and_names_each_leak(leaky, clean, tmp_path):
    job = tmp_path / "job"
    trial(job, "a__x", "a", audit_output(leaky.path))
    trial(job, "b__x", "b", audit_output(clean.path))
    trial(job, "c__x", "c", None)
    trial(job, "d__x", "d", f"AUDIT head={clean.head} head_ct={HEAD}\n")
    assert report(audits(job)) == [
        "HISTORY_AUDIT tasks=4 audited=2 errors=2 with_extra=1 with_future=1 with_ref_ahead=1 with_reflog_extra=1 "
        "with_unreachable_commits=1 shallow=0 dirty=0",
        f"HISTORY_FUTURE a future=2 extra=2 first={FIX} {leaky.fix} fix",
        f"HISTORY_REF_AHEAD a {FIX} refs/heads/fix",
        f"HISTORY_REF_AHEAD a {FIX} refs/tags/v2",
        "HISTORY_ERROR c no output",
        "HISTORY_ERROR d unfinished",
    ]


async def test_the_agent_runs_the_script_and_keeps_its_output(tmp_path):
    ran = []

    class Environment:
        async def exec(self, command, timeout_sec=None):
            ran.append((command, timeout_sec))
            return SimpleNamespace(stdout="AUDIT done\n", stderr="warning: x\n")

    await HistoryAudit(logs_dir=tmp_path).run("Fix it.", Environment(), None)
    assert ran == [(SCRIPT, TIMEOUT_S)] and (tmp_path / OUTPUT).read_text() == "AUDIT done\nwarning: x\n"
