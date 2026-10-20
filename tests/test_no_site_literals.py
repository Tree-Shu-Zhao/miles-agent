"""Public-tree gate (DESIGN D22, scan_history sec.7, user decisions 2 and 12 of 2026-10-18): no file that
`git archive` would publish carries a personal or site identity, the vendor word, a scheduler/container word outside
the sanctioned places, a hardware model name, a bare cluster name or job id, or a path of the deleted private wrapper
tree. Runs on the private repo too: the file set is exactly the export (export-ignored paths such as reports/ are
outside it), and a checkout without .git (a source tar) is walked under the same rules. Every hunted literal is spelled
in two pieces here so this file does not trip itself; tests/site_literal_allow.txt (`path:regex # reason`) exempts
lines for the USABILITY, HW and CLU classes only — IDENTITY and DANGLING have no allowlist."""

import os
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO / "tests" / "site_literal_allow.txt"

IDENTITY = [  # (pattern, why); no allowlist
    ("/lus" "tre/fsw", "a personal site root"),
    ("/scr" "atch/fsw", "a personal site root"),
    ("/ra" "id/scratch", "a site scratch root"),
    ("ai-at-" "nvi" "dia", "a site account"),
    ("it-nv-" "on-nv", "a site account"),
    ("nvi" r"dia\.com", "a corporate mail domain"),
    ("gitlab-" "master", "an internal host"),
    (r"pdx\.s8" r"k\.io", "an internal host"),
    ("nvl" "72", "a node host name"),
    # the public identity's own e-mail; every other "shu" "zhao"@ is a site literal
    ("shu" "zhao" r"(?!\.cs@gmail\.com)", "the owner's username: write $USER"),
    ("Treezz" "Z", "a personal hub namespace"),
    ("smz" "5505", "a personal account id"),
    (r"forge\.core" "weave", "an internal host"),
    (r"\.secre" "ts/", "the private secrets layout"),
    ("oci-" "ag" "a|ns" "c-" "lh" "r|ns" "c-" "svg|oci-" "hs" "g|cs-oci-" "or" "d|aws-" "cm" "h", "a cluster name"),
    (r"(?<![0-9.])100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?!\.?[0-9])",
     "a 100.x address: tests use RFC 5737 (192.0.2.x, 198.51.100.x, 203.0.113.x)"),
    (r"hf_[A-Za-z0-9]{20,}", "a Hugging Face token"),
    ("glp" r"at-[A-Za-z0-9_-]{8,}", "a GitLab token"),
    ("gh" r"p_[A-Za-z0-9]{8,}|github_" r"pat_[A-Za-z0-9_]{8,}", "a GitHub token"),
    ("AK" r"IA[0-9A-Z]{16}", "an AWS key id"),
    ("tsk" "ey-" r"[A-Za-z0-9_-]{4,}", "a tailscale key"),
    (r"BEGIN [A-Z ]*PRIV" "ATE KEY", "a private key block"),
    ("WANDB_API" r"_KEY\s*=\s*[0-9a-fA-F]{40}", "a wandb key value"),
    # -Nemotron: the one Hub model id MODEL_FAMILIES keys by name
    (r"\bNVI" r"DIA\b(?!-NeMo)(?!-Nemotron)|\bnvi" r"dia\b(?!-smi)",
     "the vendor word (the dependency home ...-NeMo/, the model id ...-Nemotron-... and the tool ...-smi stay)"),
]
USABILITY = [r"\bsba" r"tch\b", r"\bsr" r"un\b", r"\bscon" r"trol\b", "SLU" "RM_", r"\benr" r"oot\b", r"\bpyx" r"is\b",
             r"\.sq" r"sh\b", r"\btail" r"net\b", r"\$" r"P\b", r"\bNR" r"LP\b", "/opt/miles" "-agent"]
HW = [  # every piece with a backslash is raw
    r"\b(GB" "300|GB" "200|B" "200|B" "300|H" "100|H" "200|A" "100|L4" "0S|GH" "200|D" "GX|H" r"GX)\b",
    r"\b(aar" r"ch64|ar" r"m64)\b",  # fix B8: the CPU architecture words are hardware wording too (decision 12)
]
CLU = [r"\b(o" "ga|n" "sc|c" r"mh)\b", r"\bc" r"lu\b",
       # fix N12: a job id of the private scheduler (a run tag such as alftrain1 or tmax4b2 is not one)
       r"\(job [0-9]{6,7}\)", r"\bjob [0-9]{6,7}\b", r"\b(o" "ga|n" "sc|c" r"mh) [0-9]{6,7}\b"]
# the deleted private wrapper tree's root; the broader by-name residue grep of the split's rules is the reviewers'
# manual check
DANGLING = [(r"(^|[^A-Za-z0-9_./-])inf" "ra/", "a path of the deleted private tree")]
INFRA_ARM = 0  # DANGLING's index of that root arm
# mirrors cluster-jobs' contract (tests/test_contract_doc.py): the wrapper's script names are not paths of this tree
DANGLING_INFRA_ONLY = ("docs/cluster.md",)
CLASSES = {
    "IDENTITY": [(re.compile(p), why) for p, why in IDENTITY],
    "USABILITY": [(re.compile(p), "a scheduler/container word outside docs/cluster.md and examples/slurm/ (D22)")
                  for p in USABILITY],
    "HW": [(re.compile(p), "hardware wording (user decision 12)") for p in HW],
    "CLU": [(re.compile(p), "a bare cluster name, a job id or the private CLI") for p in CLU],
    "DANGLING": [(re.compile(p), why) for p, why in DANGLING],
}
SCANNED_CLASSES = tuple(c for c in CLASSES if c != "IDENTITY")  # the parametrized scan; IDENTITY has its own test
# the classes allowed() consults the allowlist for (R34, fix B13): DANGLING, like IDENTITY, takes none
ALLOWLISTED_CLASSES = ("USABILITY", "HW", "CLU")
ALLOWLIST_SELF_EXEMPT = ("USABILITY", "HW")  # the allowlist file names what it allows: outside these two scans only
# the tar-mode walk never enters these (plus any .venv*)
SKIP_DIR_NAMES = (".git", "3rdparty", "__pycache__", ".pytest_cache", "node_modules")


def _git(*args, input=None):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, input=input).stdout


def _git_toplevel():
    try:
        out = _git("rev-parse", "--show-toplevel")
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out.decode("utf-8", "surrogateescape").strip()).resolve()


def _under(rel_posix, prefix):
    prefix = prefix.rstrip("/")
    return rel_posix == prefix or rel_posix.startswith(prefix + "/")


def _parents(rel_posix):
    """The directory prefixes of a path as `git archive` queries them: `a/`, `a/b/` (a `dir/` attribute pattern
    marks the directory entry, not the files inside, and the archive prunes the whole subtree there)."""
    parts = rel_posix.split("/")[:-1]
    return ["/".join(parts[: i + 1]) + "/" for i in range(len(parts))]


def _git_exported():
    """The checkout: tracked + untracked-unignored, minus the export-ignore hits (on the file or on one of its
    directories), the gitlinks and uv.lock."""
    raw = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8", "surrogateescape")
    listed = [p for p in raw.split("\0") if p]
    queried = sorted(set(listed) | {d for p in listed for d in _parents(p)})
    attrs = _git("check-attr", "--stdin", "-z", "export-ignore", input="\0".join(queried).encode()) if queried else b""
    ignored = set()
    fields = attrs.decode("utf-8", "surrogateescape").split("\0")
    for i in range(0, len(fields) - 2, 3):  # path, attr, value
        if fields[i + 2] not in ("unspecified", "unset"):
            ignored.add(fields[i])
    out = []
    for p in sorted(set(listed)):
        if p in ignored or any(d in ignored for d in _parents(p)) or p == "uv.lock" or not (REPO / p).is_file():
            continue
        out.append(p)
    return out


def _export_ignore_patterns():
    """The `<pattern> export-ignore` lines of .gitattributes: a `dir/` pattern is a prefix, a file pattern is exact."""
    path = REPO / ".gitattributes"
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        words = raw.split()
        if len(words) >= 2 and not words[0].startswith("#") and "export-ignore" in words[1:]:
            out.append(words[0])
    return out


def _gitignore_rules():
    """.gitignore's lines as (pattern, anchored, dir_only): a leading or inner `/` anchors the pattern at the root
    (matched over the relative path), a bare name matches at every depth, a trailing `/` matches directories only.
    Read from the file, never hardcoded; negations are not supported (the file carries none)."""
    path = REPO / ".gitignore"
    if not path.is_file():
        return []
    rules = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        pattern = line.strip("/")
        rules.append((pattern, line.startswith("/") or "/" in pattern, line.endswith("/")))
    return rules


def _excluded(rel_posix, is_dir, export_ignored, gitignore):
    name = rel_posix.rsplit("/", 1)[-1]
    if is_dir and (name in SKIP_DIR_NAMES or name.startswith(".venv")):
        return True
    for pattern in export_ignored:
        if (pattern.endswith("/") and _under(rel_posix, pattern)) or rel_posix == pattern:
            return True
    for pattern, anchored, dir_only in gitignore:
        if dir_only and not is_dir:
            continue
        if fnmatchcase(rel_posix if anchored else name, pattern):
            return True
    return False


def _walked_exported():
    """A source tar (no .git): the tree walked with the exclusions git would apply (R18)."""
    export_ignored, gitignore = _export_ignore_patterns(), _gitignore_rules()
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        rel_dir = Path(dirpath).relative_to(REPO).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = sorted(d for d in dirnames if not _excluded(prefix + d, True, export_ignored, gitignore))
        for name in sorted(filenames):
            rel = prefix + name
            if rel in ("uv.lock", ".git"):  # (.git is a file in a linked worktree)
                continue
            if _excluded(rel, False, export_ignored, gitignore) or not (REPO / rel).is_file():
                continue
            out.append(rel)
    return out


def exported_files():
    """Paths `git archive` would publish, relative to REPO: read through git when REPO is a checkout, else (a source
    tar, which has no .git) the tree walked under the same rules."""
    if (REPO / ".git").exists() and _git_toplevel() == REPO:
        rels = _git_exported()
    else:
        rels = _walked_exported()
    return [Path(p) for p in rels]


def read_text(rel):
    data = (REPO / rel).read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def allowlist():
    entries = []
    for n, raw in enumerate(ALLOWLIST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert " # " in line, f"{ALLOWLIST.name}:{n}: an entry needs ' # <reason>'"
        rule, reason = line.rsplit(" # ", 1)
        path, regex = rule.split(":", 1)
        assert reason.strip(), f"{ALLOWLIST.name}:{n}: empty reason"
        entries.append((path.strip(), re.compile(regex), reason.strip(), n))
    return entries


def allowed(rel_posix, line, entries):
    return any(_under(rel_posix, p) and rx.search(line) for p, rx, _, _ in entries)


def patterns(cls, rel_posix):
    """The (regex, why) pairs `cls` applies to the file: a DANGLING_INFRA_ONLY file is checked for the deleted tree's
    root arm only (its other words are the mirrored contract's script names)."""
    pats = CLASSES[cls]
    if cls == "DANGLING" and any(_under(rel_posix, p) for p in DANGLING_INFRA_ONLY):
        return [pats[INFRA_ARM]]
    return pats


def _caught(cls, rel_posix, line):
    return [why for rx, why in patterns(cls, rel_posix) if rx.search(line)]


@pytest.fixture(scope="module")
def corpus():
    files = []
    for rel in exported_files():
        text = read_text(rel)
        if text is not None:
            files.append((rel.as_posix(), text))
    assert files, "no file found: not run inside the repository or its source tar?"
    return files


def test_no_identity_literal_anywhere(corpus):
    hits = []
    for rel, text in corpus:
        for n, line in enumerate(text.splitlines(), 1):
            for rx, why in CLASSES["IDENTITY"]:
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {why}: {line.strip()[:100]}")
                    break
    assert not hits, "identity literals (no allowlist; use $USER, placeholders, RFC 5737 addresses):\n" + "\n".join(
        hits)


@pytest.mark.parametrize("cls", SCANNED_CLASSES)
def test_no_site_word_outside_the_sanctioned_places(corpus, cls):
    entries = allowlist() if cls in ALLOWLISTED_CLASSES else []
    allowlist_rel = ALLOWLIST.relative_to(REPO).as_posix()
    hits = []
    for rel, text in corpus:
        if cls in ALLOWLIST_SELF_EXEMPT and rel == allowlist_rel:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for rx, why in patterns(cls, rel):
                if rx.search(line) and not allowed(rel, line, entries):
                    hits.append(f"{rel}:{n}: {why}: {line.strip()[:100]}")
                    break
    assert not hits, (f"{cls}: scrub these lines (an allowlist entry needs a DESIGN reason; DANGLING takes none):\n"
                      + "\n".join(hits))


def test_allowlist_entries_are_used(corpus):
    stale = []
    for path, rx, reason, n in allowlist():
        if not any(_under(rel, path) and any(rx.search(line) for line in text.splitlines()) for rel, text in corpus):
            stale.append(f"site_literal_allow.txt:{n}: {path}:{rx.pattern} ({reason}) matches nothing")
    assert not stale, "\n".join(stale)


def test_patterns_behave():
    """Assembled at run time so this file carries none of the literals."""
    cgnat = ".".join(["100", "64", "0", "1"])
    owner = "shu" + "zhao"
    positives = ["/users/x/" + owner, owner + "@nvi" + "dia.com", cgnat + ":8100", "at " + cgnat + ".",
                 "hf_" + "A" * 24, "WANDB_API" + "_KEY=" + "a" * 40, "-----BEGIN RSA " + "PRIV" + "ATE KEY-----",
                 "sb" + "atch -N 1", "on a GB" + "300 node", "aar" + "ch64 wheels", "an ar" + "m64 build",
                 "the " + "og" + "a login node", "see " + "inf" + "ra/jobs/x.sh", "NVI" + "DIA hardware",
                 "on " + "oci" + "-aga", "the " + "tail" + "net address", "$" + "P/models", "(job " + "1234567)",
                 "job " + "654321 failed", "og" + "a 1234567"]
    negatives = ["192.0.2.1:8100", "198.51.100.7", "203.0.113.9", "hf_token", "HF_HUB_OFFLINE=1",
                 "#WANDB_API" + "_KEY=", "github.com/NVI" + "DIA-NeMo/Gym",
                 "NVI" + "DIA-Nemotron-3.5-Lightning-30B-A3B", "nvi" + "dia-smi -L", "NVI" + "DIA_SMI=1",
                 owner + ".cs@gmail.com", "/root/miles", "include the clue",
                 "the infrastructure/ team", "x86_64 wheels", "1" + cgnat, "unset $(env | grep -E '^(PMI|SLURM)')",
                 "(job 12345)", "job 12345678", "alftrain1", "tmax4b2", "og" + "a_1234567"]
    every = [rx for cls in CLASSES.values() for rx, _ in cls]
    for s in positives:
        assert any(rx.search(s) for rx in every), f"should be caught: {s}"
    for s in negatives:
        assert not any(rx.search(s) for rx in every), f"should pass: {s}"
    # the job-id shapes on their own (fix N12): the three positives match one of them, a run tag or a short number none
    jobid = [rx for rx, _ in CLASSES["CLU"][2:]]
    for s in ["(job " + "1234567)", "job " + "654321 failed", "og" + "a 1234567"]:
        assert any(rx.search(s) for rx in jobid), f"should be a job id: {s}"
    for s in ["(job 12345)", "job 12345678", "alftrain1", "tmax4b2", "og" + "a_1234567"]:
        assert not any(rx.search(s) for rx in jobid), f"not a job id: {s}"
    # Q51: docs/cluster.md mirrors the private wrapper's contract — its script names pass there, a deleted path does
    # not
    mirrored_row = "| `GPUS_PER_NODE` | measured when unset | `miles_run_" + "inner.sh:46` |"
    assert not _caught("DANGLING", "docs/cluster.md", mirrored_row)
    assert _caught("DANGLING", "docs/cluster.md", "see `" + "inf" + "ra/jobs/x.sh`")
