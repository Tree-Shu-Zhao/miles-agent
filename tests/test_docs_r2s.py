"""docs/r2s.md's entry lines: every one uses the knobs the scripts' headers document, with the node count the entry's
topology table demands, and runs as written under --dry-run."""

import re
import shlex
import subprocess
from pathlib import Path

import pytest

from miles_agent.r2s import common, divergence, judge, label, select, takeover, teacher_ft
from test_scripts_sh import _fakes, _env, BASH

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "r2s.md"
STAGES = {"select": select, "divergence": divergence, "takeover": takeover, "judge": judge, "label": label}
MODULE = re.compile(r"-m miles_agent\.r2s\.(\w+)")
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```$", re.M | re.S)
# a knob the doc's prose documents: `NAME=value` in backticks (not a long flag, not a $VAR)
KNOB = re.compile(r"`([A-Z][A-Z0-9_]*)=[^`]*`")
# the headers that document an entry's knobs (MODE=r2s_pipeline's live in the script the entry sources, a *_teacher
# topology's in the teacher node's)
JOB_HEADERS = {
    "run_local.sh": ("scripts/run_local.sh", "scripts/r2s_pipeline.sh", "scripts/teacher_node.sh"),
    "r2s_teacher_ft.sh": ("scripts/r2s_teacher_ft.sh",),
    "eval_ckpt.sh": ("scripts/eval_ckpt.sh",),
}
ENTRY = re.compile(r"(?:^|\s)scripts/(run_local|eval_ckpt|r2s_teacher_ft)\.sh(?:\s|$)")
# the entry's topology table (scripts/run_local.sh): `async_4n_teacher) NEED_NODES=4 ;;`, an `a|b)` arm naming two
NODES = re.compile(r"([\w|]+)\) NEED_NODES=(\d+) ;;")
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$", re.S)  # a prefix assignment of the statement
PLACEHOLDER = re.compile(r"<(\w+)>")  # <tag>, <arm>, <pilot>, <host>: redirections to bash, filled in
RECIPES_WITH_EVAL_CKPT = ("alfworld", "scienceworld", "webshop")


def _code(raw: str) -> str:
    """A physical line without its comment and trailing blanks."""
    return raw.split("#", 1)[0].rstrip()


def fenced_statements(text: str) -> list[list[str]]:
    """The statements of the doc's fenced blocks as written, each its physical lines (a line ending in a backslash
    continues on the next); comment-only and blank lines out."""
    statements: list[list[str]] = []
    for block in FENCE.findall(text):
        lines: list[str] = []
        for raw in block.splitlines():
            if not lines and not _code(raw).strip():
                continue
            lines.append(raw)
            if not _code(raw).endswith("\\"):
                statements.append(lines)
                lines = []
        assert not lines, f"a fenced block ends inside a continuation: {lines}"
    return statements


def joined(statement: list[str]) -> str:
    """The statement as bash reads it: the backslash-newline removed and nothing put in its place (a continuation
    line keeps its indentation), comments dropped, whitespace runs collapsed."""
    return " ".join("".join(_code(raw).removesuffix("\\") for raw in statement).split())


def fenced_commands(text: str) -> list[str]:
    return [joined(statement) for statement in fenced_statements(text)]


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def stage_parser(module, monkeypatch):
    """The parser main() builds, taken at the one call every stage makes after its add_argument calls."""

    def capture(ap, argv):
        raise _Captured(ap)

    monkeypatch.setattr(common, "parse_stage_args", capture)
    with pytest.raises(_Captured) as info:
        module.main([])
    return info.value.parser


def doc_flags(line: str) -> list[str]:
    """The --flags after the module token (`[--tau x | --judge-file y]` brackets and `--flag=value` stripped)."""
    tokens = line.split()
    start = next(i for i, tok in enumerate(tokens) if tok.startswith("miles_agent.r2s."))
    flags = []
    for tok in tokens[start + 1 :]:
        tok = tok.strip("[]|").split("=", 1)[0]
        if tok.startswith("--"):
            flags.append(tok)
    return flags


def header_words(path: str) -> set[str]:
    """The upper-case identifiers of a script's leading comment block (its header)."""
    words: set[str] = set()
    for line in (REPO / path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        words.update(re.findall(r"[A-Z][A-Z0-9_]*", line))
    return words


def doc_knobs(text: str) -> set[str]:
    """The knobs the doc documents itself, outside its fenced blocks: the backticked names of its knob tables
    (section 4.4) and `NAME=value` spans in its prose (PROMPT, the prompt files' knob, has no entry header)."""
    prose = FENCE.sub("", text)
    knobs = set(KNOB.findall(prose))
    for row in prose.splitlines():
        if row.startswith("| `"):
            knobs.update(re.findall(r"`([A-Z][A-Z0-9_]*)`", row.split("|")[1]))
    return knobs


def entry_statements(text: str) -> list[tuple[str, dict[str, str], list[str], str]]:
    """The doc's entry statements as bash reads them: (the entry's name, its prefix knobs NAME -> value, the words
    after the script path, the whole command)."""
    found = []
    for command in fenced_commands(text):
        entry = ENTRY.search(command)
        if not entry:
            continue
        words = shlex.split(command)
        knobs: dict[str, str] = {}
        while words and ASSIGNMENT.match(words[0]):
            name, value = ASSIGNMENT.match(words.pop(0)).groups()
            knobs[name] = value
        assert words and words[0] == f"scripts/{entry.group(1)}.sh", f"{command!r}: the script path must follow the knobs"
        found.append((f"{entry.group(1)}.sh", knobs, words[1:], command))
    assert found, "docs/r2s.md shows no entry line"
    return found


def flags_of(words: list[str]) -> tuple[dict[str, str], list[str]]:
    """The `--flag value` / `--flag=value` pairs after the script path (a flag followed by another flag, or by
    nothing, is a bare switch) and the positional words."""
    flags: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if word.startswith("--") and "=" in word:
            name, value = word.split("=", 1)
            flags[name] = value
        elif word.startswith("--"):
            if i + 1 < len(words) and not words[i + 1].startswith("--"):
                flags[word] = words[i + 1]
                i += 1
            else:
                flags[word] = ""
        else:
            positional.append(word)
        i += 1
    return flags, positional


def entry_nodes() -> dict[str, int]:
    """The entry's topology table, scripts/run_local.sh's `case "$TOPOLOGY"` (fully_async, tmax's, is computed there and
    has no row)."""
    text = (REPO / "scripts" / "run_local.sh").read_text(encoding="utf-8")
    return {topo: int(n) for group, n in NODES.findall(text) for topo in group.split("|")}


def want_nodes(nodes: dict[str, int], topology: str, knobs: dict[str, str]) -> int:
    """The entry's node count for a topology: its table's count, one fewer for a *_teacher topology when TEACHER_URL
    names an already serving teacher (the entry's *_teacher arm; docs section 2.3)."""
    return nodes[topology] - (1 if topology.endswith("_teacher") and "TEACHER_URL" in knobs else 0)


# the CLI lines: every `python -m miles_agent.r2s.<stage>` line of the doc's fenced blocks names only flags that
# stage's parser has (the five stage parsers are built inside main() and captured at the parse call; teacher_ft's
# comes from build_parser(), since its main imports torch first)
def test_r2s_doc_cli_lines_match_help(monkeypatch):
    parsers = {}
    for line in fenced_commands(DOC.read_text(encoding="utf-8")):
        found = MODULE.search(line)
        if not found:
            continue
        stage = found.group(1)
        if stage not in parsers:
            parsers[stage] = (
                teacher_ft.build_parser() if stage == "teacher_ft" else stage_parser(STAGES[stage], monkeypatch)
            )
        known = set(parsers[stage]._option_string_actions)
        unknown = [flag for flag in doc_flags(line) if flag not in known]
        assert not unknown, f"docs/r2s.md names {unknown} for miles_agent.r2s.{stage}; its --help has {sorted(known)}"
    missing = {*STAGES, "teacher_ft"} - set(parsers)
    assert not missing, f"docs/r2s.md shows no CLI line for {sorted(missing)}"
    # the shared surface (common.stage_parser) is on every stage, and the doc's lines use it
    shared = {"--env", "--env-config", "--out", "--summary", "--force", "--limit", "--seed", "--stop-file"}
    for stage, parser in parsers.items():
        if stage != "teacher_ft":
            assert shared <= set(parser._option_string_actions), stage


def test_r2s_doc_run_lines_use_documented_knobs():
    """Every entry statement, parsed as bash would: its prefix knobs are names the entry's headers (or the doc's own
    knob prose and tables) document; a run_local.sh line names --recipe and --run-tag and carries the --num-nodes
    its topology demands (the entry's table, one fewer under TEACHER_URL); an eval_ckpt.sh
    line names a recipe with an eval_ckpt.py and exactly one of --eval-load-tag / --eval-base; the FT entry takes
    no argument."""
    text = DOC.read_text(encoding="utf-8")
    nodes = entry_nodes()
    assert nodes["colocated_1n"] == 1 and nodes["async_4n_teacher"] == 4 and nodes["async_2n_teacher"] == 2, nodes
    documented = doc_knobs(text)
    headers = {job: set().union(*(header_words(p) for p in paths)) | documented for job, paths in JOB_HEADERS.items()}
    checked = 0
    for job, knobs, words, command in entry_statements(text):
        unknown = [k for k in knobs if k not in headers[job] and not k.startswith("R2S_ARGS_")]
        assert not unknown, f"{command!r}: {unknown} are in no header of {JOB_HEADERS[job]} and not in the doc's knob prose"
        flags, positional = flags_of(words)
        if job == "run_local.sh":
            assert flags.get("--recipe") and flags.get("--run-tag"), f"{command!r}: --recipe and --run-tag are required"
            assert not positional, f"{command!r}: {positional} are not flags of the entry"
            topology = flags.get("--topology", "colocated_1n")
            want = want_nodes(nodes, topology, knobs)
            assert int(flags.get("--num-nodes", 1)) == want, f"{command!r}: {topology} needs --num-nodes {want}"
        elif job == "eval_ckpt.sh":
            assert positional and positional[0] in RECIPES_WITH_EVAL_CKPT, f"{command!r}: the recipe comes first"
            assert ("--eval-load-tag" in flags) != ("--eval-base" in flags), f"{command!r}: one of --eval-load-tag / --eval-base"
        else:
            assert not words, f"{command!r}: the FT entry's knobs are environment variables; it takes no argument"
        checked += 1
    assert checked >= 8, f"only {checked} entry lines found in docs/r2s.md"


def test_r2s_doc_run_lines_dry_run_as_written(tmp_path):
    """Every entry statement pasted as written, `--dry-run` right after the script path: the prefix knobs reach the
    entry, the `<name>` placeholders filled in, the roots under tmp_path (local.env off; test_scripts_sh's fakes
    stand in for the commands the entry probes), and the entry prints the dry-run line of what it would start:
    `DRY_RUN driver:` with the statement's topology (run_local.sh, and eval_ckpt.sh through it on colocated_1n),
    `DRY_RUN r2s_pipeline:` under MODE=r2s_pipeline, `DRY_RUN torchrun:` from the FT entry."""
    _fakes(tmp_path, state={"gpus": 4})
    roots = {}
    for name in ("models", "data", "results"):
        (tmp_path / name).mkdir()
        roots[f"MILES_AGENT_{name.upper()}"] = str(tmp_path / name)
    env = _env(tmp_path, LOCAL_ENV="/dev/null", **roots)
    checked = 0
    for job, knobs, words, command in entry_statements(DOC.read_text(encoding="utf-8")):
        path = f"scripts/{job}"
        pasted = PLACEHOLDER.sub(r"\1", command).replace(path, f"{path} --dry-run", 1)
        run = subprocess.run([BASH, "-c", pasted], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, f"{pasted!r}: bash exited {run.returncode}:\n{run.stdout}\n{run.stderr}"
        assert "FATAL" not in run.stdout + run.stderr, f"{pasted!r}:\n{run.stdout}\n{run.stderr}"
        lines = run.stdout.splitlines()
        flags = flags_of(words)[0]
        if job == "r2s_teacher_ft.sh":
            marker = [l for l in lines if l.startswith("DRY_RUN torchrun: torchrun --standalone --nproc-per-node ")]
        elif knobs.get("MODE", flags.get("--mode")) == "r2s_pipeline":
            marker = [l for l in lines if l == "DRY_RUN r2s_pipeline: r2s_pipeline_main"]
        else:
            topology = flags.get("--topology", "colocated_1n") if job == "run_local.sh" else "colocated_1n"
            script = knobs.get("RUN_SCRIPT", "run.py") if job == "run_local.sh" else "eval_ckpt.py"
            marker = [l for l in lines if l.startswith("DRY_RUN driver: ") and f" {script} --topology {topology} " in l]
        assert len(marker) == 1, f"{pasted!r}: no dry-run line of the {job} entry in:\n{run.stdout}"
        checked += 1
    assert checked >= 8, f"only {checked} entry lines found in docs/r2s.md"
