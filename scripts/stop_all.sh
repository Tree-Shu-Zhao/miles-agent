#!/bin/bash
# scripts/stop_all.sh — stop what a run, or a crashed one, left on this machine, in the launcher's own cleanup order
# (miles_agent/launcher.py cleanup(): the sglang engines — a process NAME match, a command-line match would also hit an
# external ray head whose python lives under an sglang venv — then the trainer: train.py, train_async.py,
# MegatronTrain), then the gym servers (`gym env start`, nemo_gym), the WebShop engine (launch_engine.sh,
# webshop/server), the servers scripts/serve_model.sh left (their pid files under ${TMPDIR:-/tmp}/miles-agent-${USER:-user}/,
# serve_model.sh's node-local default, plus SERVE_PID_FILE when set; each a setsid leader signalled as a process
# group, after <pid file>.stop is written so that the foreground serve_model.sh ends with SERVE_STOPPED instead of a
# FATAL), and last the ray cluster. Every signal goes through an external
# `xargs kill`: TERM to each hit, a 10 s grace, KILL to the survivors; this script's own shell and its parent are
# never signalled. Nothing is deleted: a pid file whose process is gone is skipped.
#
#   scripts/stop_all.sh [--dry-run] [--keep-ray]
#     --keep-ray   leave the ray cluster up (a head other runs share: scripts/ray_cluster.sh head)
#     --dry-run    print `DRY_RUN pgrep: <args>` per pattern and `DRY_RUN kill: -TERM <pids>` per hit (the groups as
#                  `-- -<pid>`), `DRY_RUN ray_stop:`; signal nothing
#   Ends with STOP_ALL_OK killed=<n> (the distinct pids TERMed). Knobs: TMPDIR USER SERVE_PID_FILE (the pid files),
#   RAY_BIN.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

KEEP_RAY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --keep-ray) KEEP_RAY=1 ;;
    *) echo "FATAL: unknown argument '$1': scripts/stop_all.sh [--dry-run] [--keep-ray]"; exit 1 ;;
  esac
  shift
done

. "$HERE/lib/common.sh"
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

TERMED=""         # the pids TERMed (pgrep hits), space-separated
TERMED_GROUPS=""  # the setsid leaders TERMed as process groups (the serve pid files)
KILLED=0          # distinct pids TERMed

note_termed() {  # <pid>
  case " $TERMED $TERMED_GROUPS " in *" $1 "*) return 0 ;; esac
  KILLED=$((KILLED + 1))
}

# <label> <pgrep args...>: TERM every match but this shell and its parent (the launcher's idiom: the signal goes
# through an external kill under xargs; `pgrep` exits 1 on no match, hence the `|| true`)
term_matching() {
  local label=$1 pids p
  shift
  if is_dry_run; then dry_run_print pgrep "$@"; fi
  pids=$(pgrep "$@" 2>/dev/null | grep -v "^$$\$" | grep -v "^$PPID\$" || true)
  [ -n "$pids" ] || return 0
  # shellcheck disable=SC2086
  if is_dry_run; then
    dry_run_print kill -TERM $pids
  else
    echo "stop_all: TERM $label: $(printf '%s ' $pids)"
    printf '%s\n' $pids | xargs -r kill -TERM 2>/dev/null || true
  fi
  for p in $pids; do note_termed "$p"; TERMED="$TERMED $p"; done
}

# <pid file>: TERM the process group of the setsid leader the file names (a stale file is skipped, never removed)
term_group_of() {
  local file=$1 pid
  pid=$(tr -d '[:space:]' < "$file")
  case "$pid" in ''|*[!0-9]*) echo "WARN: $file holds no pid ('$pid'): skipped"; return 0 ;; esac
  kill -0 "$pid" 2>/dev/null || { echo "stop_all: $file names pid $pid, which is gone: skipped"; return 0; }
  if is_dry_run; then
    dry_run_print kill -TERM -- "-$pid"
  else
    echo "stop_all: TERM group of $file: $pid"
    # the request first: the foreground serve_model.sh reads <pid file>.stop when its wait returns (SERVE_STOPPED)
    printf '%s\n' "$pid" > "$file.stop" 2>/dev/null || true
    printf -- '-%s\n' "$pid" | xargs -r kill -TERM -- 2>/dev/null || true
  fi
  note_termed "$pid"
  TERMED_GROUPS="$TERMED_GROUPS $pid"
}

# ---- 1. the engines and the trainer (launcher.CLEANUP_PGREP order), then a 5 s pause as cleanup() takes ----
term_matching sglang sglang
term_matching train.py -f train.py
term_matching train_async.py -f train_async.py
term_matching MegatronTrain -f MegatronTrain
is_dry_run || sleep 5
# ---- 2. the gym servers, the WebShop engine ----
term_matching "gym env start" -f "gym env start"
term_matching nemo_gym -f nemo_gym
term_matching launch_engine.sh -f launch_engine.sh
term_matching webshop/server -f webshop/server
# ---- 3. the servers scripts/serve_model.sh left: their pid files (serve_model.sh's node-local default dir, plus
# SERVE_PID_FILE when set), each a process group ----
PID_DIR=${TMPDIR:-/tmp}/miles-agent-${USER:-user}
for f in "$PID_DIR"/*.pid ${SERVE_PID_FILE:+"$SERVE_PID_FILE"}; do
  [ -f "$f" ] || continue
  term_group_of "$f"
done
# ---- 4. the survivors of a 10 s grace get KILL (a group leader's whole group) ----
if ! is_dry_run; then
  sleep 10
  survivors=""
  group_survivors=""
  for p in $TERMED; do if kill -0 "$p" 2>/dev/null; then survivors="$survivors $p"; fi; done
  for p in $TERMED_GROUPS; do if kill -0 "$p" 2>/dev/null; then group_survivors="$group_survivors $p"; fi; done
  if [ -n "$survivors" ]; then
    echo "stop_all: KILL the survivors:$survivors"
    # shellcheck disable=SC2086
    printf '%s\n' $survivors | xargs -r kill -KILL 2>/dev/null || true
  fi
  for p in $group_survivors; do
    echo "stop_all: KILL the surviving group of $p"
    printf -- '-%s\n' "$p" | xargs -r kill -KILL -- 2>/dev/null || true
  done
fi
# ---- 5. ray, last (unless --keep-ray) ----
if [ "$KEEP_RAY" = 0 ]; then
  if is_dry_run; then
    dry_run_print ray_stop "$RAY_BIN" stop --force
  elif command -v "$RAY_BIN" >/dev/null 2>&1; then
    "$RAY_BIN" stop --force || true
  else
    echo "WARN: no $RAY_BIN on PATH: ray not stopped"
  fi
fi
echo "STOP_ALL_OK killed=$KILLED"
