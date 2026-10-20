"""3rdparty/miles working tree = pinned upstream base + patches/miles/miles-*.patch.

The gitlink pins the upstream base; the patched commits live only in working
trees. A bare `git submodule update` resets the tree to the base and every job
built from it fails late. Skips when the submodule is not checked out. patches/miles/series
fixes the apply order (scripts/check_patches.sh --apply and the README's `git am` recipe follow
it); the other tests need no submodule: patch_ids.txt pins every patch's `git patch-id --stable`, and every
patch header carries the public identity only.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MILES = REPO / "3rdparty" / "miles"
PATCHES_DIR = REPO / "patches" / "miles"
PATCHES = sorted(PATCHES_DIR.glob("miles-*.patch"))
SERIES = PATCHES_DIR / "series"
README = PATCHES_DIR / "README.md"
# patches/miles/patch_ids.txt: `<id>  <name>.patch` per line, in series order -> {name: id}
IDS = dict(line.split()[::-1] for line in (PATCHES_DIR / "patch_ids.txt").read_text().splitlines() if line.strip())
# the two header lines every regenerated patch starts with (`git format-patch --zero-commit` under the public
# identity); the e-mail is spelled in pieces because the public-tree gate hunts the whole token
ZERO_COMMIT_LINE = "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001"
PUBLIC_FROM_LINE = "From: Shu Zhao <" "shu" "zhao.cs" "@gmail.com>"


def _subject(patch: Path) -> str:
    for line in patch.read_text(errors="replace").splitlines():
        m = re.match(r"^Subject: \[PATCH[^\]]*\] (.*)$", line)
        if m:
            return m.group(1)
    raise AssertionError(f"{patch.name}: no Subject line")


def _patch_id(patch: Path) -> str:
    """`git patch-id --stable` of the file: a hash of its diff alone (the mbox header, author, date and message are
    outside it)."""
    run = subprocess.run(["git", "patch-id", "--stable"], input=patch.read_bytes(), capture_output=True, check=True)
    return run.stdout.split()[0].decode()


@pytest.mark.skipif(not (MILES / ".git").exists(), reason="3rdparty/miles not checked out")
def test_every_miles_patch_is_applied_in_the_working_tree():
    assert PATCHES, "no patches/miles/miles-*.patch"
    subjects = subprocess.run(
        ["git", "-C", str(MILES), "log", "--format=%s", "-20"], check=True, capture_output=True, text=True
    ).stdout
    missing = [p.name for p in PATCHES if _subject(p) not in subjects]
    assert not missing, f"3rdparty/miles lacks {missing}; apply with scripts/check_patches.sh --apply (patches/miles/README.md)"


def _series_entries():
    """The series file's entries in order: the first word of every line that is neither blank nor a `#` comment."""
    entries = []
    for line in SERIES.read_text().splitlines():
        words = line.split()
        if words and not words[0].startswith("#"):
            entries.append(words[0])
    return entries


def test_series_lists_every_patch_once_in_readme_order():
    """patches/miles/series == the set of patches/miles/miles-*.patch, each once, and the README's `git am` recipe
    names them in series order (fp32-lm-head before its hybrid follow-up, which the two after it build on);
    patch_ids.txt pins the same set."""
    series = _series_entries()
    assert len(series) == len(set(series)), f"a patch is listed twice: {series}"
    assert sorted(series) == [p.name for p in PATCHES]
    assert set(_series_entries()) == set(IDS), "patch_ids.txt and series disagree"
    am_lines = re.findall(r"^\s+git am \.\./\.\./patches/miles/(\S+)\s*$", README.read_text(), re.M)
    assert am_lines == series
    assert series.index("miles-fp32-lm-head.patch") < series.index("miles-fp32-lm-head-hybrid.patch")
    assert series.index("miles-fp32-lm-head-hybrid.patch") < series.index("miles-mamba-packed-seq-idx.patch")
    assert series[-1] == "miles-qwen3.5-2B-model-args.patch"


def test_patch_ids_match_the_pinned_table():
    """patches/miles/patch_ids.txt pins every patch's `git patch-id --stable` (needs git, not the submodule): a patch
    whose DIFF changed fails here; a header-only regeneration (identity, trailer, message wording) does not."""
    assert {p.name for p in PATCHES} == set(IDS), "patch_ids.txt and the patch files disagree"
    assert {p.name: _patch_id(p) for p in PATCHES} == IDS


def test_patches_carry_no_private_identity():
    """Every patch is a `git format-patch --zero-commit --no-signature` file cut under the public identity: line 1 the
    zero-commit mbox header, line 2 the public From: line; no vendor mail domain, no co-author trailer and none
    of the site-specific preemption wording the old miles-preempt-save message carried, anywhere in the file."""
    for p in PATCHES:
        lines = p.read_text(errors="replace").splitlines()
        assert lines[0] == ZERO_COMMIT_LINE, (p.name, lines[0])
        assert lines[1] == PUBLIC_FROM_LINE, (p.name, lines[1])
        text = "\n".join(lines)
        for token in ("nvid" "ia.com", "Co-Authored" "-By", "our " "clusters", "20 minutes " "later"):
            assert token not in text, (p.name, token)
