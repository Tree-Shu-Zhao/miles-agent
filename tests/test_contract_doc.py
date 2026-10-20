"""docs/cluster.md and the job-wrapper repository's `runbooks/contract.md` hold the same Contract A table, Contract C
prose and Contract D table (one source for both repos; the wrapper repo is private, this page is public). The first
test diffs the three blocks against a checkout of that repo (`CLUSTER_JOBS_DIR`, default ~/Working/cluster-jobs) and
skips when it is absent; the second checks, with no other checkout, that every Contract A name of
tests/contract_vars.txt is a backticked token of the page's A table (and that the names file equals the wrapper
repo's when that checkout is present)."""
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "cluster.md"
VARS = REPO / "tests" / "contract_vars.txt"


def _cluster_jobs():
    return Path(os.environ.get("CLUSTER_JOBS_DIR") or Path.home() / "Working" / "cluster-jobs")


def _section(text, heading):
    """The lines after the first heading line that starts with `heading`, up to the next `## ` heading."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(heading)]
    assert starts, f"no heading starting with {heading!r}"
    body = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return body


def _table(lines):
    """The `|` lines, runs of whitespace collapsed."""
    return [re.sub(r"\s+", " ", line.strip()) for line in lines if line.lstrip().startswith("|")]


def _prose(lines):
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _blocks(text):
    return {"A": _table(_section(text, "## A.")), "C": _prose(_section(text, "## C.")),
            "D": _table(_section(text, "## D."))}


def _names():
    return [line.split()[0] for line in VARS.read_text().splitlines() if line.strip() and not line.startswith("#")]


def test_cluster_doc_matches_cluster_jobs_contract():
    contract = _cluster_jobs() / "runbooks" / "contract.md"
    if not contract.is_file():
        pytest.skip(f"no wrapper-repo checkout at {contract.parent.parent} (CLUSTER_JOBS_DIR)")
    ours, theirs = _blocks(DOC.read_text()), _blocks(contract.read_text())
    assert ours["A"] == theirs["A"]
    assert ours["C"] == theirs["C"]
    assert ours["D"] == theirs["D"]


def test_cluster_doc_lists_every_contract_a_variable():
    names = _names()
    assert len(names) == 43 and len(set(names)) == 43
    table = _table(_section(DOC.read_text(), "## A."))
    assert len(table) == 21  # the header, the separator and the 19 rows
    tokens = set()
    for line in table:
        for span in re.findall(r"`([^`]*)`", line):
            for word in span.split():
                tokens.add(word.split("=", 1)[0])
    missing = [name for name in names if name not in tokens]
    assert missing == []
    d_table = _table(_section(DOC.read_text(), "## D."))
    assert len(d_table) == 4 and d_table[2].startswith("| wrapper (job) |") and d_table[3].startswith("| entry (project) |")
    assert "`JOB_PASS_<PROJECT_JOB_NAME>_<RUN_TAG>`" in d_table[2] and "`RUN_INNER_PASS`" in d_table[3]
    theirs = _cluster_jobs() / "tests" / "contract_vars.txt"
    if theirs.is_file():
        assert VARS.read_text() == theirs.read_text()
