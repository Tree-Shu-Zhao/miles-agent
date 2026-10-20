#!/bin/bash
# scripts/harbor_run.sh — one Harbor evaluation (`harbor run`) of a dataset with a harness of miles_harbor.harnesses
# (or one of Harbor's own agents), and its summary: the job config built from the knobs below, the run, the job dir
# copied to --out with a summary, and these lines on stdout:
#   HARBOR_JOB_CONFIG <json>                 the job config (the agent's env values masked)
#   HARBOR_RUN_EXIT rc=<n>                   `harbor run`'s exit code
#   HARBOR_SUMMARY trials= scored= passed= mean_reward= errors=
#   HARBOR_AGENT_STATS <key>_median= <key>_max= ...      HARBOR_EXCEPTION <type> <count>
#   HARBOR_NOT_PASSED <task> reward= exception=          one per trial that did not pass (or raised)
#   HARBOR_RESULTS <out dir>                 last; a `harbor run` that failed then ends in FATAL with its code
# A USR1 or TERM interrupts Harbor (which stops its sandboxes); the summary still runs.
#
#   scripts/harbor_run.sh [--dry-run] --out <dir> [--provider <yaml>] [<data files...>]
#   <data files>   rows whose metadata.instance_id name the tasks to run, joined with HARBOR_TASKS (default: every
#                  task of the dataset)
#   --provider     a sandbox backend's provider yaml (HARBOR_PROVIDER; not for Harbor's docker environment)
#   --dry-run      print HARBOR_JOB_CONFIG and `DRY_RUN harbor_run: <argv>`, exit 0: nothing run, nothing written
#   -h / --help    this text
# Knobs (scripts/lib/common.sh loads local.env first; the environment wins)
#   HARBOR_DATASET          required: <name>@<version> (the registry), <org>/<name>[@<ref>] (the Hub) or /a/dir of
#                           task dirs
#   HARBOR_TASKS            comma-separated task names or globs to keep; HARBOR_N_TASKS a cap
#   HARBOR_AGENT            the agent (default oracle): one of Harbor's own by name, or with HARBOR_HARNESS_LOCAL=1 a
#                           harness of miles_harbor.harnesses by its class (installed from its pack); a harness that
#                           runs beside the sandbox (no pack) is always taken by its class; HARBOR_AGENT_IMPORT_PATH
#                           any class instead
#   HARBOR_ENDPOINT_FILE    a JSON {"base_url": ..., "model": ...} of the model server the harness calls (through its
#                           ENDPOINT binding); HARBOR_MODEL the model name, HARBOR_AGENT_ENV env over the binding's
#                           (a JSON object; its values land in the results' job.yaml: no real secret)
#   HARBOR_AGENT_KWARGS     the agent's kwargs (a JSON object)
#   HARBOR_N_ATTEMPTS       1; HARBOR_N_CONCURRENT 32 trials at once
#   HARBOR_VERIFY           1; 0: no verifier, the agent's run alone
#   HARBOR_REPORT           a module run on the job dir after the summary (`python -m <module> <job dir>`, into
#                           report.txt)
#   HARBOR_ENVIRONMENT_IMPORT_PATH   the sandbox backend, Harbor's environment.import_path: empty = Harbor's docker
#                           environment; a sandbox backend's <module>:<Class> is passed through as it is; with
#                           --provider its kwargs are {sandbox_provider, layer_cache} (HARBOR_LAYER_CACHE, default 1)
#   HARBOR_ENVIRONMENT_KWARGS   the backend's kwargs without a provider, a JSON object
#   HARBOR_WORK_DIR         Harbor's working dir: home/ (its caches), jobs/, tasks/ (downloads); default <out>/work
#   HARBOR_PYTHON           the interpreter with harbor and miles_harbor installed (default $PYTHON); HARBOR_BIN its
#                           `harbor` command (default: harbor on PATH)
#   HARBOR_PACK_ROOT        where the harness packs are (read by miles_harbor.harnesses in Harbor's process); default
#                           $HOME/.cache/miles-harness of the account's HOME, resolved and exported here before Harbor
#                           gets <work dir>/home as its HOME
#   RUN_TAG                 the job's name (default: the out dir's name), Harbor's RUN_TAG too (a sandbox backend's
#                           label of the job's sandboxes: miles_harbor.opensandbox_env's)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/harbor_run.sh [--dry-run] --out <dir> [--provider <yaml>] [<data files...>]"
need_value() { [ $# -ge 2 ] || fatal "$1 needs a value: $USAGE_LINE"; }
OUT=""
PROVIDER=${HARBOR_PROVIDER:-}
DATA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --out) need_value "$@"; OUT=$2; shift ;;
    --provider) need_value "$@"; PROVIDER=$2; shift ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    *) DATA+=("$1") ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN
[ -n "$OUT" ] || fatal "--out <dir> is required: $USAGE_LINE"
[ -n "${HARBOR_DATASET:-}" ] \
  || fatal "harbor_run needs HARBOR_DATASET (<name>@<version>, <org>/<name>[@<ref>] or /a/dir of task dirs)"
OUT=${OUT%/}
HR_PYTHON=${HARBOR_PYTHON:-$PYTHON}
HARBOR_BIN=${HARBOR_BIN:-harbor}
TAG=${RUN_TAG:-$(basename "$OUT")}
WORK=${HARBOR_WORK_DIR:-$OUT/work}
JOB=$WORK/job.yaml
IMPORT_PATH=${HARBOR_ENVIRONMENT_IMPORT_PATH:-}
if [ -n "$PROVIDER" ]; then
  if [ -z "$IMPORT_PATH" ]; then
    echo "WARN: --provider $PROVIDER is a sandbox backend's: not used by Harbor's docker environment (no HARBOR_ENVIRONMENT_IMPORT_PATH)"
  elif [ ! -f "$PROVIDER" ]; then
    is_dry_run || fatal "no provider file $PROVIDER"
    echo "WARN: no provider file $PROVIDER"
  fi
fi
for f in ${DATA[@]+"${DATA[@]}"}; do
  if [ ! -f "$f" ]; then
    is_dry_run || fatal "no data file $f"
    echo "WARN: no data file $f"
  fi
done
is_dry_run || mkdir -p "$WORK/home" "$WORK/jobs" "$WORK/tasks" "$OUT" || fatal "cannot create the work dir $WORK or the out dir $OUT"

# ---- the job config (the old job's builder: the dataset spelled three ways, the agent from its harness, the
# backend from HARBOR_ENVIRONMENT_IMPORT_PATH); under --dry-run printed, not written ----
WORK=$WORK TAG=$TAG "$HR_PYTHON" - "$PROVIDER" "$JOB" ${DATA[@]+"${DATA[@]}"} <<'PY'
import json, os, sys
provider, job_out, *data = sys.argv[1:]
env = os.environ
dry = env.get("DRY_RUN") == "1"
dataset = env["HARBOR_DATASET"]
if dataset.startswith("/"):  # a directory of task dirs on this machine (Harbor's local dataset)
    ds = {"path": dataset}
elif "/" in dataset:  # a Hub package: <org>/<name>[@<ref>]
    name, _, ref = dataset.partition("@")
    ds = {"name": name, **({"ref": ref} if ref else {})}
else:  # the registry: <name>@<version>
    name, _, version = dataset.partition("@")
    ds = {"name": name, **({"version": version} if version else {})}
if "path" not in ds:
    ds["download_dir"] = env["WORK"] + "/tasks"
names = [t for t in env.get("HARBOR_TASKS", "").split(",") if t]
for path in data:  # the rows' tasks (metadata.instance_id), after HARBOR_TASKS
    if dry and not os.path.isfile(path):
        continue  # the shell said WARN
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                name = (json.loads(line).get("metadata") or {}).get("instance_id")
                if not name:
                    sys.exit(f"FATAL: {path}:{n} has no metadata.instance_id (the task's name)")
                if name not in names:
                    names.append(name)
if names:
    ds["task_names"] = names
if env.get("HARBOR_N_TASKS"):
    ds["n_tasks"] = int(env["HARBOR_N_TASKS"])
try:
    from miles_harbor.harnesses import HARNESSES
except ImportError as e:
    HARNESSES = {}
    if env.get("HARBOR_ENDPOINT_FILE") and not env.get("HARBOR_AGENT_IMPORT_PATH"):
        sys.exit(f"FATAL: HARBOR_ENDPOINT_FILE needs the harness's ENDPOINT binding, and miles_harbor.harnesses is not"
                 f" importable by {sys.executable} ({e}): install gym_root/miles_harbor into HARBOR_PYTHON's venv")
    print(f"WARN: miles_harbor.harnesses is not importable by {sys.executable} ({e}): the agent is one of Harbor's own,"
          " by name (install gym_root/miles_harbor into HARBOR_PYTHON's venv for a harness)")
agent_name = env.get("HARBOR_AGENT", "oracle")
harness = HARNESSES.get(agent_name)
if env.get("HARBOR_AGENT_IMPORT_PATH"):
    agent = {"import_path": env["HARBOR_AGENT_IMPORT_PATH"]}
# installed from its pack, or one that runs beside the sandbox (no pack, nothing of Harbor's by its name)
elif harness is not None and (env.get("HARBOR_HARNESS_LOCAL") or not hasattr(harness, "PACK")):
    agent = {"import_path": f"{harness.__module__}:{harness.__name__}"}
else:
    agent = {"name": agent_name}
if env.get("HARBOR_ENDPOINT_FILE"):  # a served model's endpoint, as the harness takes one (its ENDPOINT)
    with open(env["HARBOR_ENDPOINT_FILE"], encoding="utf-8") as f:
        endpoint = json.load(f)
    binding = getattr(HARNESSES.get(agent_name), "ENDPOINT", None)
    if binding is None:
        sys.exit(f"FATAL: HARBOR_ENDPOINT_FILE: miles_harbor.harnesses has no ENDPOINT for the agent {agent_name!r}")
    # the model server checks no key; the harnesses want one
    agent.update(binding.agent_config(endpoint["base_url"], "EMPTY", endpoint["model"]))
if env.get("HARBOR_MODEL"):
    agent["model_name"] = env["HARBOR_MODEL"]
if env.get("HARBOR_AGENT_KWARGS"):
    agent["kwargs"] = json.loads(env["HARBOR_AGENT_KWARGS"])
if env.get("HARBOR_AGENT_ENV"):  # over the endpoint's, e.g. {"OPENAI_BASE_URL": ..., "MSWEA_API_KEY": ...}
    agent["env"] = {**agent.get("env", {}), **json.loads(env["HARBOR_AGENT_ENV"])}
import_path = env.get("HARBOR_ENVIRONMENT_IMPORT_PATH") or ""
environment = {}
if import_path and provider:  # a sandbox backend's kwargs: its provider
    # HARBOR_LAYER_CACHE=0: a Dockerfile replayed in every trial's sandbox instead of built once per machine
    environment = {"import_path": import_path,
                   "kwargs": {"sandbox_provider": provider, "layer_cache": env.get("HARBOR_LAYER_CACHE", "1") != "0"}}
elif import_path or env.get("HARBOR_ENVIRONMENT_KWARGS"):
    environment = {**({"import_path": import_path} if import_path else {}),
                   "kwargs": json.loads(env.get("HARBOR_ENVIRONMENT_KWARGS") or "{}")}
job = {
    "job_name": env["TAG"],
    "jobs_dir": env["WORK"] + "/jobs",
    "n_attempts": int(env.get("HARBOR_N_ATTEMPTS", "1")),
    "n_concurrent_trials": int(env.get("HARBOR_N_CONCURRENT", "32")),
    "quiet": True,
    **({"environment": environment} if environment else {}),  # none: Harbor's own default environment
    "agents": [agent],
    "datasets": [ds],
}
if env.get("HARBOR_VERIFY", "1") == "0":  # the agent's run alone: an agent that grades nothing (history_audit)
    job["verifier"] = {"disable": True}
if not dry:
    import yaml
    with open(job_out, "w") as f:
        f.write(yaml.safe_dump(job, sort_keys=False))
masked = [{**a, **({"env": dict.fromkeys(a["env"], "<set>")} if a.get("env") else {})} for a in job["agents"]]
print("HARBOR_JOB_CONFIG", json.dumps({**job, "agents": masked}))
PY

# the packs' root miles_harbor.harnesses reads in Harbor's process: its default is resolved from the account's HOME
# here, before the launch gives Harbor $WORK/home as its HOME (the default would otherwise move under the work dir)
if [ -z "${HARBOR_PACK_ROOT:-}" ]; then
  [ -n "${HOME:-}" ] || fatal "HARBOR_PACK_ROOT is unset and so is HOME: export HARBOR_PACK_ROOT=<the harness packs' root>"
  HARBOR_PACK_ROOT=$HOME/.cache/miles-harness
fi
export HARBOR_PACK_ROOT
if is_dry_run; then
  echo "DRY_RUN env: HOME=$WORK/home"
  echo "DRY_RUN env: RUN_TAG=$TAG"
  echo "DRY_RUN env: HARBOR_PACK_ROOT=$HARBOR_PACK_ROOT"
  dry_run_print harbor_run "$HARBOR_BIN" run -c "$JOB" -y
  exit 0
fi

# ---- the run; a USR1 / TERM interrupts Harbor, which stops its trials, and the summary follows ----
HARBOR_PID=""
trap 'echo "HARBOR_RUN_SIGNAL $(date "+%F %T")"; [ -z "$HARBOR_PID" ] || kill -INT "$HARBOR_PID" 2>/dev/null' USR1 TERM
set +e
# Harbor's caches (task downloads, its config) under the work dir, not in the account's home
( export HOME=$WORK/home RUN_TAG=$TAG; cd "$WORK" && exec "$HARBOR_BIN" run -c "$JOB" -y ) &
HARBOR_PID=$!
wait "$HARBOR_PID"
RC=$?
# a USR1 interrupts the first wait; Harbor is still stopping its trials
while kill -0 "$HARBOR_PID" 2>/dev/null; do wait "$HARBOR_PID"; RC=$?; done
set -e
HARBOR_PID=""
echo "HARBOR_RUN_EXIT rc=$RC"

# ---- results: the job dir to --out, the summary of the old job ----
JOB_DIR=$WORK/jobs/$TAG
mkdir -p "$OUT" || fatal "cannot create the out dir $OUT"
# HARBOR_WORK_DIR=<the out dir>: job.yaml is there already (cp refuses to copy a file onto itself)
[ "$JOB" -ef "$OUT/$(basename "$JOB")" ] || cp "$JOB" "$OUT/"
[ -z "$PROVIDER" ] || [ ! -f "$PROVIDER" ] || cp "$PROVIDER" "$OUT/"
[ ! -f "$JOB_DIR/result.json" ] || cp "$JOB_DIR/result.json" "$OUT/"
if [ -d "$JOB_DIR" ]; then
  tar czf "$OUT/jobs.tar.gz" -C "$WORK/jobs" "$TAG"
else
  echo "WARN: no job dir $JOB_DIR (harbor run wrote nothing)"
fi
summary_rc=0
"$HR_PYTHON" - "$JOB_DIR" "${HARBOR_VERIFY:-1}" <<'PY' | tee "$OUT/summary.txt" || summary_rc=$?
import collections, datetime, json, pathlib, statistics, sys
job = pathlib.Path(sys.argv[1])
verified = sys.argv[2] != "0"
rows = []
agent = collections.defaultdict(list)  # what the agent spent per trial, where Harbor recorded it
for path in sorted(job.glob("*/result.json")):
    r = json.loads(path.read_text())
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    exc = (r.get("exception_info") or {}).get("exception_type")
    rows.append((r.get("task_name"), rewards.get("reward"), exc))
    for key in ("n_input_tokens", "n_output_tokens"):
        if (value := (r.get("agent_result") or {}).get(key)) is not None:
            agent[key].append(value)
    timing = r.get("agent_execution") or {}
    if timing.get("started_at") and timing.get("finished_at"):
        span = [datetime.datetime.fromisoformat(timing[k].replace("Z", "+00:00")) for k in ("started_at", "finished_at")]
        agent["agent_s"].append((span[1] - span[0]).total_seconds())
n = len(rows)
scored = [x for _, x, _ in rows if x is not None]
passed = sum(1 for x in scored if x >= 1)
print("HARBOR_SUMMARY trials=%d scored=%d passed=%d mean_reward=%.4f errors=%d" % (
    n, len(scored), passed, sum(scored) / n if n else 0.0, sum(1 for *_, e in rows if e)))
print("HARBOR_AGENT_STATS " + " ".join(
    "%s_median=%.0f %s_max=%.0f" % (key, statistics.median(values), key, max(values))
    for key, values in sorted(agent.items()) if values))
for exc, k in collections.Counter(e for *_, e in rows if e).most_common():
    print("HARBOR_EXCEPTION %s %d" % (exc, k))
for task, reward, exc in rows:
    if (verified and (reward is None or reward < 1)) or exc:
        print("HARBOR_NOT_PASSED %s reward=%s exception=%s" % (task, reward, exc))
PY
[ "$summary_rc" -eq 0 ] || echo "WARN: the summary failed (rc=$summary_rc)"
# HARBOR_REPORT: a module whose main reads the job dir (miles_harbor.history_audit's report)
if [ -n "${HARBOR_REPORT:-}" ]; then
  "$HR_PYTHON" -m "$HARBOR_REPORT" "$JOB_DIR" > "$OUT/report.txt" 2>&1 || echo "HARBOR_REPORT_FAILED $HARBOR_REPORT rc=$?"
  cat "$OUT/report.txt"
fi
echo "HARBOR_RESULTS $OUT"
[ "$RC" -eq 0 ] || { echo "FATAL: harbor run rc=$RC (see HARBOR_NOT_PASSED / HARBOR_EXCEPTION above)"; exit "$RC"; }
