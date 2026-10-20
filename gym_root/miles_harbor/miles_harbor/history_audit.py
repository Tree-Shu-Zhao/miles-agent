"""What a task's repository tells an agent about its future: git objects past the commit the task starts at (SWE-bench
issue #465: tags, remote branches, reflogs and packed objects that hold the fix).

    scripts/harbor_run.sh with HARBOR_AGENT_IMPORT_PATH=miles_harbor.history_audit:HistoryAudit HARBOR_VERIFY=0 \\
        HARBOR_REPORT=miles_harbor.history_audit
    python -m miles_harbor.history_audit <harbor job dir>   # the report, from the trials' outputs

HistoryAudit is a Harbor agent that solves nothing: in the task's sandbox (the agent's view: the task's image, its
Dockerfile's build), it runs SCRIPT where an agent starts (the task's workdir: /testbed for SWE-bench) and keeps what
SCRIPT prints (OUTPUT, in the trial's agent logs). Per repository, as ``AUDIT`` lines:
- ``head``: the commit HEAD names, its committer time, whether the clone is shallow or borrows objects (alternates),
  and how many tracked files differ from it;
- ``commits``: HEAD's history, every commit object of the repository (reachable or not), and the rest (``extra``);
- ``future``: the extra commits committed after HEAD (a later state of the repository: the leak), the first few as
  ``future_commit`` lines;
- ``ref_ahead``: each ref (branch, tag, remote, stash, note) whose commit HEAD's history lacks, with its commit's time;
- ``reflog``: the reflogs' entries and the commits they name outside HEAD's history; ``stash`` entries;
- ``unreachable``: the objects nothing reaches, reflogs aside (``git fsck --unreachable --no-reflogs``), by type.

The report (main) prints one HISTORY_AUDIT line over the tasks, then each task with a future commit (HISTORY_FUTURE),
a ref ahead of HEAD (HISTORY_REF_AHEAD) or no audit (HISTORY_ERROR).
"""

import argparse
import json
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

OUTPUT = "history_audit.txt"
# fsck reads every object of a large repository (django: ~500k)
TIMEOUT_S = 1800
FUTURE_LISTED = 5

SCRIPT = r"""
export GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0 GIT_PAGER=cat LC_ALL=C
# (each revision ends at --: a file named HEAD at the top of the work tree would make it ambiguous)
g() { git -c safe.directory='*' "$@"; }
n() { awk 'END { print NR }' "$@"; }
if ! gd=$(g rev-parse --absolute-git-dir 2> /dev/null); then
  echo "AUDIT error=no-repository pwd=$PWD"
  exit 0
fi
t=$(mktemp -d)
head=$(g rev-parse HEAD)
head_ct=$(g log -1 --format=%ct HEAD --)
echo "AUDIT head=$head head_ct=$head_ct shallow=$([ -f "$gd/shallow" ] && echo 1 || echo 0)" \
  "alternates=$(cat "$gd/objects/info/alternates" 2> /dev/null | n)" \
  "dirty=$(g status --porcelain --untracked-files=no 2> /dev/null | n)"
g rev-list HEAD -- | sort > "$t/head"
g cat-file --batch-all-objects --batch-check='%(objecttype) %(objectname)' | awk '$1 == "commit" { print $2 }' \
  | sort > "$t/all"
comm -23 "$t/all" "$t/head" > "$t/extra"
echo "AUDIT head_commits=$(n "$t/head") all_commits=$(n "$t/all") extra=$(n "$t/extra")"
: > "$t/future"
# (git log --stdin with no input would log HEAD)
[ ! -s "$t/extra" ] || g log --no-walk=unsorted --format='%ct %H %s' --stdin < "$t/extra" \
  | awk -v h="$head_ct" '$1 > h' | sort -n > "$t/future"
echo "AUDIT future=$(n "$t/future")"
head -n FUTURE_LISTED "$t/future" | sed 's/^/AUDIT future_commit /'
g for-each-ref --format='%(objectname) %(refname)' > "$t/refs"
while read -r sha ref; do
  c=$(g rev-parse -q --verify "$sha^{commit}" 2> /dev/null) || continue
  g merge-base --is-ancestor "$c" HEAD || echo "AUDIT ref_ahead $(g log -1 --format=%ct "$c" --) $ref"
done < "$t/refs" > "$t/ahead"
echo "AUDIT refs=$(n "$t/refs") refs_ahead=$(n "$t/ahead")"
cat "$t/ahead"
find "$gd/logs" -type f -exec cat {} + 2> /dev/null > "$t/reflog"
awk '{ print $2 }' "$t/reflog" | sort -u | comm -23 - "$t/head" > "$t/reflog_extra"
echo "AUDIT reflog=$(n "$t/reflog") reflog_extra=$(n "$t/reflog_extra") stash=$(g stash list 2> /dev/null | n)"
g fsck --connectivity-only --unreachable --no-reflogs --no-progress 2> /dev/null \
  | awk '{ c[$2]++ } END { s = "AUDIT unreachable"; for (k in c) s = s " " k "=" c[k]; print s }'
echo "AUDIT done"
""".replace("FUTURE_LISTED", str(FUTURE_LISTED))


class HistoryAudit(BaseAgent):
    """A Harbor agent that runs SCRIPT in the task's sandbox and keeps its output (the module docstring)."""

    @staticmethod
    def name() -> str:
        return "history-audit"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        result = await environment.exec(SCRIPT, timeout_sec=TIMEOUT_S)
        (self.logs_dir / OUTPUT).write_text((result.stdout or "") + (result.stderr or ""))


def read(text: str) -> dict[str, Any]:
    """SCRIPT's output: its ``key=value`` fields (numbers as ints), its future commits and its refs ahead of HEAD."""
    audit: dict[str, Any] = {"future_commit": [], "ref_ahead": []}
    for line in text.splitlines():
        if not line.startswith("AUDIT "):
            continue
        rest = line[len("AUDIT ") :]
        kind, _, detail = rest.partition(" ")
        if kind in ("future_commit", "ref_ahead"):
            audit[kind].append(detail)
        elif kind == "unreachable":
            audit["unreachable"] = {k: int(v) for k, _, v in (word.partition("=") for word in detail.split())}
        elif kind == "done":
            audit["done"] = True
        else:
            for word in rest.split():
                key, eq, value = word.partition("=")
                if eq:
                    audit[key] = int(value) if value.isdigit() else value
    return audit


def audits(job_dir: Path) -> dict[str, dict[str, Any]]:
    """Each trial's audit by its task (``error``: the trial left none, or SCRIPT did not finish)."""
    found: dict[str, dict[str, Any]] = {}
    for result in sorted(job_dir.glob("*/result.json")):
        task = json.loads(result.read_text()).get("task_name") or result.parent.name
        output = result.parent / "agent" / OUTPUT
        audit = read(output.read_text()) if output.is_file() else {"error": "no output"}
        if "error" not in audit and not audit.get("done"):
            audit["error"] = "unfinished"
        found[task] = audit
    return found


def report(found: dict[str, dict[str, Any]]) -> list[str]:
    """The report's lines (the module docstring)."""
    ok = {task: a for task, a in found.items() if "error" not in a}

    def count(predicate) -> int:
        return sum(1 for a in ok.values() if predicate(a))

    counts = {
        "tasks": len(found),
        "audited": len(ok),
        "errors": len(found) - len(ok),
        "with_extra": count(lambda a: a.get("extra", 0) > 0),
        "with_future": count(lambda a: a.get("future", 0) > 0),
        "with_ref_ahead": count(lambda a: a.get("refs_ahead", 0) > 0),
        "with_reflog_extra": count(lambda a: a.get("reflog_extra", 0) > 0),
        "with_unreachable_commits": count(lambda a: a.get("unreachable", {}).get("commit", 0) > 0),
        "shallow": count(lambda a: a.get("shallow") == 1),
        "dirty": count(lambda a: a.get("dirty", 0) > 0),
    }
    lines = ["HISTORY_AUDIT " + " ".join(f"{key}={value}" for key, value in counts.items())]
    for task, a in sorted(found.items()):
        if "error" in a:
            lines.append(f"HISTORY_ERROR {task} {a['error']}")
            continue
        if a.get("future", 0) > 0:
            first = a["future_commit"][0] if a["future_commit"] else ""
            lines.append(f"HISTORY_FUTURE {task} future={a['future']} extra={a.get('extra')} first={first}")
        for ref in a["ref_ahead"]:
            lines.append(f"HISTORY_REF_AHEAD {task} {ref}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("job_dir", type=Path, help="a Harbor job dir: its trials' dirs, each with result.json")
    args = parser.parse_args()
    for line in report(audits(args.job_dir)):
        print(line, flush=True)


if __name__ == "__main__":
    main()
