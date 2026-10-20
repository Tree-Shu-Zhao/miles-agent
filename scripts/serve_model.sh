#!/bin/bash
# scripts/serve_model.sh — one model's OpenAI-compatible endpoint on this machine: sglang serving SERVE_MODEL_PATH on
# its GPUs, rendering and parsing as the training engines do (the fixed chat template and tool-call parser of miles'
# TITO family SERVE_TITO_MODEL, written by `python -m miles_agent.launcher serve-template`), checked with one tool
# call, and announced on stdout, in this order:
#   SERVE_PARSER <parser>
#   SERVE_PREFLIGHT PASS|NO_TOOL_CALL finish_reason= tool_calls= content= usage=
#   SERVE_UP url=http://<ip>:<port>/v1 model=<served name> parser=<parser> dp=<n>
# then it stays in the foreground as long as the server runs. A stop requested through --stop, scripts/stop_all.sh
# or a TERM / INT to this script stops the server (TERM to its process group, up to 30 s, then KILL) and ends with
# `SERVE_STOPPED rc=<the server's exit code>` and exit 0 (the TERM / INT trap is armed before the readiness wait and
# the preflight, which run as background jobs under `wait`, so a stop during them acts at once); a server that dies
# on its own is `FATAL: sglang exited with rc=<rc>` with its log's tail and that exit code. A job wrapper runs it in
# its own process group and waits for the SERVE_UP line (docs/cluster.md, PROJECT_SERVE_CMD); a plain machine runs
# it in a terminal or under nohup.
#
#   scripts/serve_model.sh [--dry-run] [<model dir>]     serve (the positional wins over SERVE_MODEL_PATH)
#   scripts/serve_model.sh [--dry-run] --stop            stop the server of SERVE_PORT through its pid file, the same
#                                                        TERM / 30 s / KILL sequence on its process group, after
#                                                        writing the request to <pid file>.stop (the foreground script
#                                                        then ends with SERVE_STOPPED, not a FATAL); exit 0
#                                                        (no pid file, no such process: a WARN, exit 0)
#   --dry-run     print `DRY_RUN serve_template:` and `DRY_RUN sglang:` with the two argv and exit 0: nothing started,
#                 nothing written (a missing model dir or GPU is a WARN here, a FATAL otherwise)
#   -h / --help   this text
# Knobs (scripts/lib/common.sh loads local.env first; the environment wins)
#   SERVE_MODEL_PATH         the HF checkpoint dir (its config.json); required unless given as the positional
#   SERVE_MODEL_NAME         the name it is served by (default: the dir's name)
#   SERVE_PORT               7000
#   SERVE_TITO_MODEL         the miles TITO family whose fixed template and tool-call parser the engines take (qwen35)
#   SERVE_THINKING           off (default): the template renders thinking off unless a request turns it on, as the
#                            training engines do; template: the template's own default
#   SERVE_CONTEXT_LENGTH     the engines' context window (default: the model's)
#   SERVE_DP                 data-parallel engines, one GPU each (default: the measured GPUs — GPUS_PER_NODE, else the
#                            CUDA_VISIBLE_DEVICES count, else `nvidia-smi -L`)
#   SERVE_EXTRA_ARGS         more sglang flags (word-split)
#   SERVE_REQUIRE_TOOL_CALL  1 = a preflight that gets no tool call is `FATAL: the served model returned no tool call`,
#                            the server is stopped and the exit code is 3; default 0 = the verdict is reported
#                            (SERVE_PREFLIGHT NO_TOOL_CALL) and serving goes on
#   SERVE_WORK_DIR           where the rendered template goes (default $RUNS_DIR/serve_<port>)
#   SERVE_PID_FILE           the server's pid file (default ${TMPDIR:-/tmp}/miles-agent-${USER:-user}/serve_<port>.pid,
#                            node-local — a pid names a process of this machine only — its dir created with mkdir -p):
#                            the setsid leader that --stop and scripts/stop_all.sh signal as a process group; their
#                            request is <pid file>.stop (the pid), emptied when a server starts
#   NODE_IP                  the address announced in SERVE_UP (default: getent hosts, then hostname -I)
#   RUN_TAG JOB_ID           name sglang's log, $RUNS_DIR/<RUN_TAG, default serve>_sglang_<JOB_ID>.log
#   SRC_MOUNT                the checkout miles_agent.launcher is imported from (default: this one)
#   PYTHON CURL              tool overrides; MILES_AGENT_RESULTS RUNS_DIR the roots (scripts/lib/common.sh)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/serve_model.sh [--dry-run] [--stop] [<model dir>]"
STOP_ONLY=0
MODEL_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --stop) STOP_ONLY=1 ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    *) [ -z "$MODEL_ARG" ] || fatal "one model dir at most, got '$MODEL_ARG' and '$1': $USAGE_LINE"
       MODEL_ARG=$1 ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

PORT=${SERVE_PORT:-7000}
# node-local by default: a pid names a process of this machine only, and a results dir shared by several machines
# must not collect every machine's pid files
PID_FILE=${SERVE_PID_FILE:-${TMPDIR:-/tmp}/miles-agent-${USER:-user}/serve_$PORT.pid}
STOP_FILE=$PID_FILE.stop   # --stop and scripts/stop_all.sh write the pid here: the foreground script's stop request
SGL_PID=""
BG_PID=""   # the background sleep or preflight the foreground is waiting on (the TERM / INT trap ends it)

# The stop sequence of the old serving job: TERM to the server's process group (setsid made the server its leader:
# the stop takes its schedulers and detokenizer with it), up to 30 s for the leader to go, then KILL to the group.
# `set +e` first: the closing KILL of an already empty group returns ESRCH, and this script's errexit must neither end
# the sequence there nor flip its exit code. Empty SGL_PID (nothing started yet): nothing to do.
stop_server() {
  set +e
  [ -n "$SGL_PID" ] || return 0
  kill -TERM -- "-$SGL_PID" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$SGL_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SGL_PID" 2>/dev/null
  return 0
}
# a sleep the TERM / INT trap can interrupt: a foreground `sleep` defers the trap until it ends, a `wait` does not
bg_sleep() { sleep "$1" & BG_PID=$!; wait "$BG_PID" 2>/dev/null || true; BG_PID=""; }
# TERM / INT: a stop request. The server group goes, its exit code is collected, and the exit is 0 with SERVE_STOPPED.
on_stop_signal() {
  local rc=0
  [ -z "$BG_PID" ] || kill "$BG_PID" 2>/dev/null
  stop_server
  wait "$SGL_PID" 2>/dev/null || rc=$?
  echo "SERVE_STOPPED rc=$rc (TERM / INT to serve_model.sh)"
  exit 0
}

# ---- --stop: the same sequence on the pid the file names ----
if [ "$STOP_ONLY" = 1 ]; then
  [ -f "$PID_FILE" ] || { echo "WARN: no pid file $PID_FILE: nothing to stop"; exit 0; }
  pid=$(tr -d '[:space:]' < "$PID_FILE")
  case "$pid" in ''|*[!0-9]*) echo "WARN: $PID_FILE holds no pid ('$pid'): nothing to stop"; exit 0 ;; esac
  kill -0 "$pid" 2>/dev/null || { echo "WARN: pid $pid of $PID_FILE is not running: nothing to stop"; exit 0; }
  if is_dry_run; then dry_run_print stop kill -TERM -- "-$pid"; exit 0; fi
  echo "serve_model: stopping the server group of pid $pid ($PID_FILE)"
  # the request first: the foreground serve_model.sh reads it when its wait returns and ends with SERVE_STOPPED
  printf '%s\n' "$pid" > "$STOP_FILE"
  SGL_PID=$pid
  stop_server
  echo "SERVE_STOPPED pid=$pid"
  exit 0
fi

# ---- the knobs (the old serving job's, with the GPUs measured and the node IP resolved by the lib) ----
SERVE_MODEL_PATH=${MODEL_ARG:-${SERVE_MODEL_PATH:-}}
[ -n "$SERVE_MODEL_PATH" ] || fatal "a model dir is required (the positional, or SERVE_MODEL_PATH=): $USAGE_LINE"
SERVE_MODEL_PATH=${SERVE_MODEL_PATH%/}
if [ ! -f "$SERVE_MODEL_PATH/config.json" ]; then
  is_dry_run || fatal "no HF checkpoint at $SERVE_MODEL_PATH (config.json missing)"
  echo "WARN: no HF checkpoint at $SERVE_MODEL_PATH (config.json missing)"
fi
TITO=${SERVE_TITO_MODEL:-qwen35}
THINKING=${SERVE_THINKING:-off}
NAME=${SERVE_MODEL_NAME:-$(basename "$SERVE_MODEL_PATH")}
NGPU=$(measure_gpus)
DP=${SERVE_DP:-$NGPU}
case "$DP" in ''|*[!0-9]*) fatal "SERVE_DP=$DP is not a number" ;; esac
if [ "$DP" -lt 1 ]; then
  is_dry_run || fatal "no GPU measured (GPUS_PER_NODE, CUDA_VISIBLE_DEVICES, nvidia-smi -L are all empty): set SERVE_DP"
  echo "WARN: no GPU measured: --dp-size $DP"
fi
if [ -n "${NODE_IP:-}" ]; then
  MY_IP=$NODE_IP
elif ! MY_IP=$(node_ip 2>/dev/null); then
  is_dry_run || fatal "could not determine this node's IP (getent hosts, hostname -I and NODE_IP are all empty)"
  MY_IP=127.0.0.1
  echo "WARN: node IP unknown: 127.0.0.1 stands in under --dry-run"
fi
WORK=${SERVE_WORK_DIR:-$RUNS_DIR/serve_$PORT}
TEMPLATE=$WORK/chat_template.jinja
SGL_LOG=$RUNS_DIR/${RUN_TAG:-serve}_sglang_${JOB_ID}.log
export SERVE_MODEL_PATH

# ---- the TITO family's fixed template (thinking off by default, SERVE_THINKING) and its tool-call parser ----
# (launcher.write_serve_template: the one copy of the code for every server that renders as the engines do)
if is_dry_run; then
  PARSER='<serve-template>'
  dry_run_print serve_template env "PYTHONPATH=${SRC_MOUNT:-$REPO}${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m miles_agent.launcher serve-template \
    "$SERVE_MODEL_PATH" "$TEMPLATE" --thinking "$THINKING" --tito-model "$TITO"
else
  mkdir -p "$WORK" "$RUNS_DIR" "$(dirname "$PID_FILE")"
  PARSER=$(PYTHONPATH=${SRC_MOUNT:-$REPO}${PYTHONPATH:+:$PYTHONPATH} "$PYTHON" -m miles_agent.launcher serve-template \
           "$SERVE_MODEL_PATH" "$TEMPLATE" --thinking "$THINKING" --tito-model "$TITO") \
    || fatal "serve-template failed for $SERVE_MODEL_PATH (SERVE_TITO_MODEL=$TITO)"
  [ -n "$PARSER" ] || fatal "serve-template printed no tool-call parser for SERVE_TITO_MODEL=$TITO"
  echo "SERVE_PARSER $PARSER"
fi

# ---- sglang ----
ARGS=(--model-path "$SERVE_MODEL_PATH" --served-model-name "$NAME" --tool-call-parser "$PARSER"
      --chat-template "$TEMPLATE" --host 0.0.0.0 --port "$PORT" --dp-size "$DP" --mem-fraction-static 0.85)
[ -z "${SERVE_CONTEXT_LENGTH:-}" ] || ARGS+=(--context-length "$SERVE_CONTEXT_LENGTH")
if is_dry_run; then
  # shellcheck disable=SC2086
  dry_run_print sglang setsid "$PYTHON" -m sglang.launch_server "${ARGS[@]}" ${SERVE_EXTRA_ARGS:-}
  exit 0
fi
# its own process group: the stop takes its schedulers and detokenizer with it
# shellcheck disable=SC2086
setsid "$PYTHON" -m sglang.launch_server "${ARGS[@]}" ${SERVE_EXTRA_ARGS:-} > "$SGL_LOG" 2>&1 &
SGL_PID=$!
trap stop_server EXIT
# armed before the readiness wait and the preflight: a stop during either acts at once
trap on_stop_signal TERM INT
echo "$SGL_PID" > "$PID_FILE"
: > "$STOP_FILE"   # no request yet (emptied, never removed: the scripts delete nothing)
echo "serve_model: sglang pid $SGL_PID (group leader) on :$PORT, dp=$DP, log $SGL_LOG, pid file $PID_FILE"
up=0
for _ in $(seq 120); do
  if "$CURL" -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$NAME\""; then up=1; break; fi
  kill -0 "$SGL_PID" 2>/dev/null || break
  bg_sleep 10
done
[ "$up" = 1 ] || { echo "FATAL: sglang did not serve $NAME"; tail -80 "$SGL_LOG"; exit 1; }
"$CURL" -sf "http://$MY_IP:$PORT/v1/models" > /dev/null || { echo "FATAL: the server does not answer at $MY_IP:$PORT"; exit 1; }

# ---- one tool call through the parser (a model that answers in prose is reported, not refused, unless
# SERVE_REQUIRE_TOOL_CALL=1 asks for the refusal); a background job under `wait`, so that a TERM / INT during the
# request (up to 300 s) acts at once ----
PREFLIGHT_OUT=$WORK/preflight.out
"$PYTHON" - "http://127.0.0.1:$PORT/v1/chat/completions" "$NAME" > "$PREFLIGHT_OUT" <<'PY' &
import json, sys, urllib.request

url, name = sys.argv[1:3]
tool = {"type": "function", "function": {"name": "bash", "description": "Run a bash command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
body = {"model": name, "temperature": 0, "max_tokens": 512, "tools": [tool],
        "messages": [{"role": "user", "content": "Use the bash tool to list the files in /tmp."}]}
request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(request, timeout=300))
choice = out["choices"][0]
calls = [(c["function"]["name"], c["function"]["arguments"]) for c in choice["message"].get("tool_calls") or []]
verdict = "PASS" if calls and calls[0][0] == "bash" else "NO_TOOL_CALL"
print(f"SERVE_PREFLIGHT {verdict} finish_reason={choice['finish_reason']} tool_calls={calls} "
      f"content={(choice['message'].get('content') or '')[:300]!r} usage={out.get('usage')}")
PY
BG_PID=$!
rc=0
wait "$BG_PID" || rc=$?
BG_PID=""
PREFLIGHT=$(cat "$PREFLIGHT_OUT")
[ "$rc" -eq 0 ] || { [ -z "$PREFLIGHT" ] || echo "$PREFLIGHT"; echo "FATAL: the preflight request to $NAME at :$PORT failed"; exit 1; }
echo "$PREFLIGHT"
case "$PREFLIGHT" in
  "SERVE_PREFLIGHT PASS"*) ;;
  *) if [ "${SERVE_REQUIRE_TOOL_CALL:-0}" = 1 ]; then
       echo "FATAL: the served model returned no tool call (SERVE_REQUIRE_TOOL_CALL=1)"
       stop_server
       exit 3
     fi ;;
esac

# ---- up: the line a wrapper waits for; then the foreground wait (TERM / INT: on_stop_signal, exit 0) ----
echo "SERVE_UP url=http://$MY_IP:$PORT/v1 model=$NAME parser=$PARSER dp=$DP"
rc=0
wait "$SGL_PID" || rc=$?
if [ -s "$STOP_FILE" ]; then
  # --stop or scripts/stop_all.sh asked for it: a requested stop is not a failure, whatever the server's exit code
  echo "SERVE_STOPPED rc=$rc (stop requested through $STOP_FILE)"
  exit 0
fi
[ "$rc" -eq 0 ] || { echo "FATAL: sglang exited with rc=$rc"; tail -80 "$SGL_LOG"; exit "$rc"; }
echo "SERVE_STOP rc=0 (the server exited by itself)"
