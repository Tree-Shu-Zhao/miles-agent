"""The repository's public-facing metadata: LICENSE (Apache-2.0 text, appendix-free, no copyright line), CITATION.cff,
README's opener and links, .gitattributes export-ignore, .gitignore globs, pyproject wording, and no copyright claim for
this work in NOTICE or README (user decisions 1, 2, 6, 7 of 2026-10-18)."""
import hashlib
import io
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LICENSE_SHA256 = "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"  # apache.org LICENSE-2.0.txt lines 1-177


def test_license_is_the_appendix_free_apache_2_text():
    data = (REPO / "LICENSE").read_bytes()
    assert hashlib.sha256(data).hexdigest() == LICENSE_SHA256
    text = data.decode()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert lines[0] == "Apache License" and lines[1] == "Version 2.0, January 2004"
    for n, title in enumerate(["Definitions.", "Grant of Copyright License.", "Grant of Patent License.", "Redistribution.",
                               "Submission of Contributions.", "Trademarks.", "Disclaimer of Warranty.", "Limitation of Liability.",
                               "Accepting Warranty or Additional Liability."], 1):
        assert f"{n}. {title}" in text
    assert lines[-1] == "END OF TERMS AND CONDITIONS" and "APPENDIX" not in text
    # no notice line of our own; the licence's section 4(c) spells "(c)", so that is not an arm of this regex
    assert not re.search(r"^\s*Copyright\b|©", text, re.M)


def test_citation_names_an_author_without_affiliation_or_email():
    cff = (REPO / "CITATION.cff").read_text()
    for key in ("cff-version: 1.2.0", 'title: "miles-agent"', "license: Apache-2.0", "family-names:", "given-names:", "repository-code:"):
        assert key in cff, key
    assert not re.search(r"affiliation|email|orcid|@", cff)


def test_readme_opens_as_personal_research_code_and_links_what_exists():
    text = (REPO / "README.md").read_text()
    head = text[:600].lower()
    assert "personal research code" in head and "heavily deleted or refactored" in head
    for rel in re.findall(r"`(scripts/[a-z_]+\.sh)`", text) + re.findall(r"`(docs/[a-z_0-9]+\.md)`", text):
        assert (REPO / rel).is_file(), rel
    assert "| `r2s` |" in text and "docs/r2s.md" in text  # decision 6: tag + docs/<paper>.md
    # the deleted paths and the cluster root, each spelled in two pieces: the gate scans this file too
    for bad in ("inf" "ra/", "$" "P", "sample_" "filters", "harness_" "agent", "swebench_" "env", "swe_pr" "orl"):
        assert bad not in text, bad


def test_git_archive_omits_the_private_paths():
    if not (REPO / ".git").exists():
        pytest.skip("no .git here (a source tar): git archive needs the repository")
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD"], check=True, capture_output=True).stdout
    names = tarfile.open(fileobj=io.BytesIO(tar)).getnames()
    assert names and not [n for n in names if n == "reports" or n.startswith("reports/")]
    assert "docs/agentic_rl_metrics.zh.md" not in names
    attrs = (REPO / ".gitattributes").read_text()
    assert "reports/ export-ignore" in attrs and "docs/agentic_rl_metrics.zh.md export-ignore" in attrs


def test_gitignore_keeps_runs_and_secrets_out():
    text = (REPO / ".gitignore").read_text().splitlines()
    for glob in ("outputs/", "reports/", "runs/", "wandb/", ".claude/", ".codex/", "*.log", "*.sq" "sh", "*.tar", "*.ckpt",
                 "*.safetensors", ".env", "/local.env", "/.gym_venvs/", "/models/", ".pids/", "/data/", "results/"):
        assert glob in text, glob
    assert not any("cluster" in line for line in text)


def test_pyproject_carries_no_site_wording():
    text = (REPO / "pyproject.toml").read_text()
    assert "on the cluster" not in text and "wiki_search first" not in text


def test_notice_and_readme_claim_no_copyright():
    """No copyright claim for this work (decision 2): NOTICE carries the upstream projects' own notices only — none of
    them names the author or the vendor — and README has no Copyright line at all."""
    author = ("Shu" " Zhao", "Zhao, Shu")
    vendor = "NV" "IDIA"
    notice_lines = [line for line in (REPO / "NOTICE").read_text().splitlines() if "Copyright" in line]
    assert notice_lines  # the third-party notices stay (they are the upstreams', not ours)
    for line in notice_lines:
        assert vendor not in line and not any(name in line for name in author), line
    readme = (REPO / "README.md").read_text()
    assert not [line for line in readme.splitlines() if "copyright" in line.lower()]
