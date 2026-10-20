#!/bin/bash
# scripts/harbor_trial_server.sh — the Harbor trial server of a run's rollouts (miles_harbor.trial_server): each POST
# /run is one Harbor trial of the row's task, its agent pointed at the rollout's session server, its reward the answer;
# what NEMO_GYM_URL points at for the coding / terminal recipes (miles_agent.gym_bridge is the client). It announces
# itself on stdout and stays in the foreground:
#   HARBOR_TRIAL_SERVER_CONFIG <json>      the config it serves (task_names counted; not with --config)
#   HARBOR_TRIAL_SERVER_UP url=http://<host>:<port> <the /health answer> <the server's TRIAL_SERVER_TASKS line>
# then TERM / INT stop the server and exit 0; a server that exits by itself ends the script with its code (a FATAL
# when it is not 0).
#
#   scripts/harbor_trial_server.sh [--dry-run] --host <ip> --port <port> --config <server.yaml> [--selftest <n>]
#       serve the config as it is (a job launcher's call: TRIAL_SERVER_VENV=<venv> and TRIAL_SERVER_MODULE=<module>
#       in its environment pick the interpreter and the module; the flags come in any order)
#   scripts/harbor_trial_server.sh [--dry-run] --host <ip> --port <port> [--provider <yaml>] <data files...>
#       build the config from the knobs below and serve it: the tasks are the data rows' metadata.instance_id (JSONL
#       rows; a line that is no JSON object — an old-format list of names, one per line — is a FATAL, as is a file
#       without rows)
#   --selftest <n>   after the up line, <n> /run requests at once (their tasks the config's task_names in turn), their
#                    model endpoint a closed port of <host>: the sandboxes, the harness's install, the endpoint's
#                    route, the verifier and the teardown, without a GPU. A HARBOR_SELFTEST_ANSWER line per answer,
#                    then HARBOR_SELFTEST_PASS (every answer a reward; exit 0) or HARBOR_SELFTEST_FAIL (a FATAL, exit
#                    1); the server is stopped either way
#   --dry-run        print HARBOR_TRIAL_SERVER_CONFIG (the config it would write) and `DRY_RUN trial_server: <argv>`,
#                    exit 0: nothing started, nothing written (a missing file is a WARN)
#   -h / --help      this text
# Knobs (scripts/lib/common.sh loads local.env first; the environment wins)
#   HARBOR_DATASET                  required without --config: a Harbor dataset, <name>@<version> (the registry),
#                                   <org>/<name>[@<ref>] (the Hub) or /a/dir of task dirs
#   HARBOR_ENVIRONMENT_IMPORT_PATH  the sandbox backend, Harbor's environment.import_path: empty = Harbor's docker
#                                   environment; a sandbox backend's <module>:<Class> is passed through as it is
#   HARBOR_PROVIDER                 a sandbox backend's provider yaml (--provider <yaml> wins): with it the backend's
#                                   kwargs are {sandbox_provider, layer_cache} (HARBOR_LAYER_CACHE, default 1); under
#                                   Harbor's docker environment a provider is a WARN, not used
#   HARBOR_ENVIRONMENT_KWARGS       the backend's kwargs without a provider, a JSON object ({})
#   HARBOR_HARNESS                  mini-swe-agent (HARBOR_AGENT: the old name); HARBOR_AGENT_KWARGS its kwargs (JSON)
#   HARBOR_MAX_CONCURRENT           128 trials at once
#   HARBOR_AGENT_NETWORK            endpoint: the agent reaches its model endpoint alone; task: the task's own network
#   HARBOR_AGENT_TIMEOUT_S          the agent's run (default the task's own); HARBOR_TRIAL_TIMEOUT_S a trial's (7200)
#   HARBOR_KEEP_TRIALS              1 = keep every trial's directory
#   HARBOR_REWARD_REQUIRES_SUBMIT   1 = a run that did not submit scores 0, its tests not run
#   HARBOR_RECORDS                  each trial's record, a JSON line ($MILES_AGENT_RESULTS/<RUN_TAG>/harbor_trials.jsonl)
#   HARBOR_WORK_DIR                 the server's dir: home/ (Harbor's caches), tasks/ (downloads), trials/, its config
#                                   and log ($MILES_AGENT_RESULTS/<RUN_TAG>/trial_server; RUN_TAG defaults to harbor)
#   HARBOR_PYTHON                   the interpreter with harbor and miles_harbor installed (default $PYTHON);
#                                   TRIAL_SERVER_VENV=<venv> wins (its bin/python)
#   TRIAL_SERVER_MODULE             miles_harbor.trial_server
#   HARBOR_PACK_ROOT                where the harness packs are (read by miles_harbor.harnesses in the server itself);
#                                   default $HOME/.cache/miles-harness of the account's HOME, resolved and exported here
#                                   before the server gets $HARBOR_WORK_DIR/home as its HOME
#   CURL                            the tool override (scripts/lib/common.sh)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/harbor_trial_server.sh [--dry-run] --host <ip> --port <port> (--config <yaml> | [--provider <yaml>] <data files...>) [--selftest <n>]"
need_value() { [ $# -ge 2 ] || fatal "$1 needs a value: $USAGE_LINE"; }
HOST=""
PORT=""
CONFIG=""
PROVIDER=${HARBOR_PROVIDER:-}
SELFTEST=0
DATA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --host) need_value "$@"; HOST=$2; shift ;;
    --port) need_value "$@"; PORT=$2; shift ;;
    --config) need_value "$@"; CONFIG=$2; shift ;;
    --provider) need_value "$@"; PROVIDER=$2; shift ;;
    --selftest) need_value "$@"; SELFTEST=$2; shift ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    *) DATA+=("$1") ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN
{ [ -n "$HOST" ] && [ -n "$PORT" ]; } || fatal "--host and --port are required: $USAGE_LINE"
case "$PORT" in ''|*[!0-9]*) fatal "--port '$PORT' is not a number" ;; esac
case "$SELFTEST" in ''|*[!0-9]*) fatal "--selftest '$SELFTEST' is not a number" ;; esac
if [ -n "$CONFIG" ] && [ ${#DATA[@]} -gt 0 ]; then
  fatal "--config serves a file as it is: no data files with it: $USAGE_LINE"
fi
if [ -z "$CONFIG" ] && [ ${#DATA[@]} -eq 0 ]; then
  fatal "a --config <yaml> or data files are required: $USAGE_LINE"
fi

# ---- the interpreter and the module: a launcher's TRIAL_SERVER_VENV / TRIAL_SERVER_MODULE first ----
TS_PYTHON=${TRIAL_SERVER_VENV:+$TRIAL_SERVER_VENV/bin/python}
TS_PYTHON=${TS_PYTHON:-${HARBOR_PYTHON:-$PYTHON}}
TS_MODULE=${TRIAL_SERVER_MODULE:-miles_harbor.trial_server}
TAG=${RUN_TAG:-harbor}
WORK=${HARBOR_WORK_DIR:-$MILES_AGENT_RESULTS/$TAG/trial_server}
LOG=$WORK/trial_server.log
IMPORT_PATH=${HARBOR_ENVIRONMENT_IMPORT_PATH:-}
URL=http://$HOST:$PORT

# ---- the config: a launcher's file as it is, else built from the knobs and the data rows' tasks ----
if [ -n "$CONFIG" ]; then
  if [ ! -f "$CONFIG" ]; then
    is_dry_run || fatal "no config file $CONFIG"
    echo "WARN: no config file $CONFIG"
  fi
else
  CONFIG=$WORK/trial_server.yaml
  [ -n "${HARBOR_DATASET:-}" ] \
    || fatal "the trial server needs HARBOR_DATASET (<name>@<version>, <org>/<name>[@<ref>] or /a/dir of task dirs)"
  if [ -n "$PROVIDER" ]; then
    if [ -z "$IMPORT_PATH" ]; then
      echo "WARN: --provider $PROVIDER is a sandbox backend's: not used by Harbor's docker environment (no HARBOR_ENVIRONMENT_IMPORT_PATH)"
    elif [ ! -f "$PROVIDER" ]; then
      is_dry_run || fatal "no provider file $PROVIDER"
      echo "WARN: no provider file $PROVIDER"
    fi
  fi
  for f in "${DATA[@]}"; do
    if [ ! -f "$f" ]; then
      is_dry_run || fatal "no data file $f"
      echo "WARN: no data file $f"
    fi
  done
  RECORDS=${HARBOR_RECORDS:-$MILES_AGENT_RESULTS/$TAG/harbor_trials.jsonl}
  if is_dry_run; then
    OUT=-
  else
    mkdir -p "$WORK/home" "$WORK/tasks" "$WORK/trials" || fatal "cannot create the work dir $WORK (HARBOR_WORK_DIR)"
    OUT=$CONFIG
  fi
  # the server config, with the backend from HARBOR_ENVIRONMENT_IMPORT_PATH and its kwargs (a sandbox backend's provider
  # with --provider, HARBOR_ENVIRONMENT_KWARGS otherwise); `-`: printed, not written (--dry-run)
  HARBOR_RECORDS=$RECORDS WORK=$WORK IMPORT_PATH=$IMPORT_PATH PROVIDER=$PROVIDER \
    "$TS_PYTHON" - "$OUT" "${DATA[@]}" <<'PY'
import json, os, sys
out, *data = sys.argv[1:]
env = os.environ
dry = env.get("DRY_RUN") == "1"
names = set()
for path in data:
    if dry and not os.path.isfile(path):
        continue  # the shell said WARN
    rows = 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            if not isinstance(row, dict):  # an old-format list (names or images, one per line; a JSON array) is no rows file
                sys.exit(f"FATAL: {path}:{n} is not a JSON object row (the data files are JSONL rows with metadata.instance_id;"
                         " an old-format list is not one)")
            name = (row.get("metadata") or {}).get("instance_id")
            if not name:
                sys.exit(f"FATAL: {path}:{n} has no metadata.instance_id (the task's name)")
            names.add(name)
            rows += 1
    if rows == 0:
        sys.exit(f"FATAL: {path} has no rows (an empty data file serves no task)")
import_path = env.get("IMPORT_PATH") or ""
if import_path and env.get("PROVIDER"):  # a sandbox backend's kwargs: its provider; HARBOR_LAYER_CACHE=0 builds no layers
    environment_kwargs = {"sandbox_provider": env["PROVIDER"], "layer_cache": env.get("HARBOR_LAYER_CACHE", "1") != "0"}
else:
    environment_kwargs = json.loads(env.get("HARBOR_ENVIRONMENT_KWARGS") or "{}")
config = {
    "dataset": env["HARBOR_DATASET"],
    "download_dir": env["WORK"] + "/tasks",
    "task_names": sorted(names),
    "harness": env.get("HARBOR_HARNESS") or env.get("HARBOR_AGENT") or "mini-swe-agent",
    "agent_kwargs": json.loads(env.get("HARBOR_AGENT_KWARGS") or "{}"),
    **({"environment_import_path": import_path} if import_path else {}),
    "environment_kwargs": environment_kwargs,
    "trials_dir": env["WORK"] + "/trials",
    "records": env["HARBOR_RECORDS"],
    "max_concurrent": int(env.get("HARBOR_MAX_CONCURRENT") or 128),
    "agent_network": env.get("HARBOR_AGENT_NETWORK") or "endpoint",
    "keep_trials": env.get("HARBOR_KEEP_TRIALS", "0") == "1",
    "reward_requires_submit": env.get("HARBOR_REWARD_REQUIRES_SUBMIT", "0") == "1",
}
for key in ("agent_timeout_s", "trial_timeout_s"):
    if env.get("HARBOR_" + key.upper()):
        config[key] = float(env["HARBOR_" + key.upper()])
if out != "-":
    import yaml
    with open(out, "w") as f:
        f.write(yaml.safe_dump(config, sort_keys=False))
print("HARBOR_TRIAL_SERVER_CONFIG", json.dumps({**config, "task_names": "%d tasks" % len(names)}))
PY
fi

if is_dry_run; then
  dry_run_print trial_server "$TS_PYTHON" -m "$TS_MODULE" --config "$CONFIG" --host "$HOST" --port "$PORT"
  [ "$SELFTEST" = 0 ] || dry_run_print selftest "$TS_PYTHON" - "$URL" "$SELFTEST" "$CONFIG"
  exit 0
fi

# ---- the server, in the background; TERM / INT stop it (up to 30 s, then KILL) and exit 0 ----
TS_PID=""
stop_server() {
  set +e
  [ -n "$TS_PID" ] || return 0
  kill "$TS_PID" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$TS_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$TS_PID" 2>/dev/null
  TS_PID=""
  return 0
}
mkdir -p "$WORK/home" || fatal "cannot create the work dir $WORK (HARBOR_WORK_DIR)"
# the packs' root miles_harbor.harnesses reads in the server: its default is resolved from the account's HOME here,
# before the launch below gives the server $WORK/home as its HOME (the default would otherwise move under the work dir)
if [ -z "${HARBOR_PACK_ROOT:-}" ]; then
  [ -n "${HOME:-}" ] || fatal "HARBOR_PACK_ROOT is unset and so is HOME: export HARBOR_PACK_ROOT=<the harness packs' root>"
  HARBOR_PACK_ROOT=$HOME/.cache/miles-harness
fi
export HARBOR_PACK_ROOT
# Harbor's caches (task downloads, its config) under the work dir, not in the account's home
( export HOME=$WORK/home; cd "$WORK" && exec "$TS_PYTHON" -m "$TS_MODULE" --config "$CONFIG" --host "$HOST" --port "$PORT" > "$LOG" 2>&1 ) &
TS_PID=$!
trap 'stop_server; cleanup' EXIT
trap 'stop_server; exit 0' TERM INT
echo "harbor_trial_server: $TS_MODULE pid $TS_PID on $HOST:$PORT (config $CONFIG, log $LOG)"
# the dataset's tasks are downloaded before it serves: up to 20 min
ok=0
for _ in $(seq 240); do
  if "$CURL" -sf -m 10 --noproxy '*' -o /dev/null "$URL/health"; then ok=1; break; fi
  kill -0 "$TS_PID" 2>/dev/null || break
  sleep 5
done
if [ "$ok" != 1 ]; then
  echo "FATAL: the trial server did not serve on $HOST:$PORT (it exited, or 20 min passed; log $LOG):"
  tail -30 "$LOG"
  exit 1
fi
export NEMO_GYM_URL=$URL
echo "HARBOR_TRIAL_SERVER_UP url=$NEMO_GYM_URL $("$CURL" -s -m 10 --noproxy '*' "$URL/health") $(grep -m1 '^TRIAL_SERVER_TASKS' "$LOG" || true)"

# ---- --selftest <n>: the probe of the old sandbox job, its tasks the config's; its verdict is the exit status ----
if [ "$SELFTEST" != 0 ]; then
  rc=0
  "$TS_PYTHON" - "$NEMO_GYM_URL" "$SELFTEST" "$CONFIG" <<'PY' || rc=$?
import asyncio, json, sys, time
import httpx, yaml
url, n, config = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tasks = list(dict.fromkeys(yaml.safe_load(open(config, encoding="utf-8")).get("task_names") or []))
if not tasks:
    sys.exit("FATAL: the selftest needs the config's task_names (a server of a whole dataset lists none)")
# port 9 of the server's host: nothing listens, so the agent's route to it is refused at once
endpoint = f"http://{httpx.URL(url).host}:9/v1"

async def run(client, i, task):
    body = {"instance_id": task, "max_episode_steps": 5, "policy_base_url": endpoint,
            "responses_create_params": {"input": [], "max_output_tokens": 256}}
    t0 = time.monotonic()
    try:
        r = await client.post(url + "/run", json=body)
        status, answer = r.status_code, r.text
    except httpx.HTTPError as e:
        status, answer = 0, repr(e)
    print(f"HARBOR_SELFTEST_ANSWER {i} {task} status={status} s={time.monotonic() - t0:.0f} {answer[-1500:]}", flush=True)
    return status

async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        return await asyncio.gather(*(run(client, i, tasks[i % len(tasks)]) for i in range(n)))

t0 = time.monotonic()
statuses = asyncio.run(main())
ok = statuses.count(200)
print(f"HARBOR_SELFTEST_{'PASS' if ok == n else 'FAIL'} {ok} of {n} answered a reward in {time.monotonic() - t0:.0f} s"
      f" (statuses {sorted(set(statuses))}, {len(set(tasks[i % len(tasks)] for i in range(n)))} tasks)", flush=True)
sys.exit(0 if ok == n else 1)
PY
  [ "$rc" -eq 0 ] || { echo "FATAL: the selftest failed (HARBOR_SELFTEST_FAIL above, rc=$rc)"; exit 1; }
  exit 0
fi

# ---- the foreground wait: the server runs until a TERM / INT (exit 0) or its own exit ----
rc=0
wait "$TS_PID" || rc=$?
TS_PID=""
[ "$rc" -eq 0 ] || { echo "FATAL: the trial server exited with rc=$rc (log $LOG):"; tail -30 "$LOG"; exit "$rc"; }
echo "HARBOR_TRIAL_SERVER_STOP rc=0 (the server exited by itself)"
