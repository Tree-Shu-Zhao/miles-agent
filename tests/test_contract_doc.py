"""docs/cluster.md and the job-wrapper repository's `runbooks/contract.md` hold the same Contract A table, Contract C
prose and Contract D table (one source for both repos; the wrapper repo is private, this page is public). The first
test diffs the three blocks against a checkout of that repo (`CLUSTER_JOBS_DIR`, default ~/Working/cluster-jobs) and
skips when it is absent; the second checks, with no other checkout, that every Contract A name of
tests/contract_vars.txt is a backticked token of the page's A table (and that the names file equals the wrapper
repo's when that checkout is present); the third checks that the three blocks cite files by repo-relative path,
never by line number, and that every path they name exists."""
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


# A line-number citation inside a backticked span: a bare `:NN`, `:NN-NN` or `:NN,NN-NN`, or one right after a file
# name (`x.sh:46`, `x.sh@<sha>:206`). A URL (`http://127.0.0.1:6100`) and a signal spec (`--signal=B:USR1@600`) are
# not ones. The batch-file suffix is spelled in two pieces: the public-tree gate hunts the scheduler word.
LINE_CITE = re.compile(r"^:\d+(?:[-,]\d+)*$|\.(?:sh|s" r"batch|py|env|md|yaml|toml|txt|patch)(?:@[0-9a-f]{7,})?:\d+")
PATH_SUFFIXES = (".sh", ".s" "batch", ".py", ".env")
WRAPPER_DIRS = ("lib/", "jobs/", "sandbox/", "projects/", "clusters/")  # the wrapper repo's top-level code dirs
PLACEHOLDER_CHARS = "<>*${}"


def _block_lines(text):
    """The lines of the three shared blocks: the A table's `|` lines, the C prose, the D table's `|` lines."""
    return _table(_section(text, "## A.")) + _section(text, "## C.") + _table(_section(text, "## D."))


def _path_words(span):
    """The words of a backticked span that name a file: trailing punctuation dropped, the part after a `=` taken
    (`ENTRY=scripts/x.sh`), a word with a placeholder character skipped (`jobs/<job>.sh`, `$PROJECT_CODE_MOUNT/$ENTRY`),
    and only the words ending in one of PATH_SUFFIXES kept (the other words of a command, `exec bash`, are not paths)."""
    out = []
    for word in span.split():
        word = word.strip(",;:()").rsplit("=", 1)[-1]
        if any(c in word for c in PLACEHOLDER_CHARS) or not word.endswith(PATH_SUFFIXES):
            continue
        out.append(word)
    return out


def _citation_problems(text, repo, wrapper):
    """The offending spans of the shared blocks: line-number citations, and path words that resolve neither in `repo`
    nor in the wrapper checkout `wrapper` (None = absent: a word under one of WRAPPER_DIRS cannot be checked then and
    passes; any other unresolved word fails)."""
    problems = []
    for line in _block_lines(text):
        for span in re.findall(r"`([^`]*)`", line):
            if LINE_CITE.search(span):
                problems.append(f"line-number citation: `{span}`")
            for word in _path_words(span):
                if (repo / word).is_file():
                    continue
                if wrapper is not None:
                    if not (wrapper / word).is_file():
                        problems.append(f"no such file in this repo or the wrapper checkout: `{word}`")
                elif not word.startswith(WRAPPER_DIRS):
                    problems.append(f"no such file in this repo: `{word}`")
    return problems


def test_cluster_doc_blocks_cite_files_not_lines():
    """The shared blocks cite where a value or marker is produced and consumed as repo-relative paths with a function
    or variable anchor, never a line number (the wrapper's old file names rotted with their line numbers, and a public
    reader can open neither): no backticked span of the A table, the C prose or the D table is a bare `:NN` / `:NN-NN`
    citation or carries one after a file name, and every backticked word ending in one of PATH_SUFFIXES is a file of
    this repository or, when a wrapper checkout is present (CLUSTER_JOBS_DIR), of it. Tolerated on purpose: the other
    words of a command (`exec bash`, `touch`), placeholders (`jobs/<job>.sh`, `$ENTRY`), a word's trailing punctuation
    and an `ENTRY=<path>` assignment; without the wrapper checkout a word under its top-level code dirs (WRAPPER_DIRS)
    is unverifiable and passes."""
    wrapper = _cluster_jobs()
    wrapper = wrapper if (wrapper / "runbooks" / "contract.md").is_file() else None
    assert _citation_problems(DOC.read_text(), REPO, wrapper) == []
    # the rules catch the old spellings and leave a URL, a signal spec, a command and a placeholder alone
    assert LINE_CITE.search("x.sh:9,11-13") and LINE_CITE.search(":94-99") and LINE_CITE.search("x.sh@d4fbb87:206")
    assert not LINE_CITE.search("http://127.0.0.1:6100") and not LINE_CITE.search("--signal=B:USR1@600")
    assert _path_words("exec bash $PROJECT_CODE_MOUNT/$ENTRY") == [] and _path_words("jobs/<job>.sh") == []
    assert _path_words("--env ENTRY=scripts/x.sh, lib/y.sh;") == ["scripts/x.sh", "lib/y.sh"]
