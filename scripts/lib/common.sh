#!/bin/bash
# scripts/lib/common.sh — the shared base of every public script (sourced, never run): the repo root, local.env,
# the tool variables, the model / data / result roots, the per-run dirs, the GPU count, the dry-run helpers, the
# EXIT-trap cleanup accumulator, the preemption trap, the HTTP boot gate, the fixed-port pre-check and the node IP.
# No scheduler call and no site path: every default hangs under the repo ($REPO/{models,data,results}), and every
# knob is read from the environment first, then from local.env, then defaulted. A job wrapper that drives the
# entries exports MODELS_DIR DATA_DIR RESULTS_DIR RUNS_DIR GPUS_PER_NODE JOB_ID SYNC_DIR PREEMPT_SENTINEL
# CONTRACT_VERSION (and the new names win over those aliases); a plain machine exports none of them.
#
# Usage, from a script under scripts/:   . "$REPO/scripts/lib/common.sh"
#   (the lib derives REPO from its own location; sourcing it runs load_local_env and sets the defaults below,
#   and arms `trap cleanup EXIT` — a script that installs its own EXIT trap replaces it and calls cleanup itself)
#
# Variables set here (each only when unset)
#   REPO LIB                          the checkout and this dir
#   PYTHON RAY_BIN GYM_BIN UV_BIN CURL NVIDIA_SMI SSH        tool overrides, default = the bare name on PATH; exported
#                                     on purpose: children such as the legacy scripts/start_gym.sh, the drivers and
#                                     the helpers a script runs inherit the same tools
#   MILES_AGENT_MODELS MILES_AGENT_DATA MILES_AGENT_RESULTS   the roots (then MODELS_DIR / DATA_DIR / RESULTS_DIR,
#                                     then $REPO/{models,data,results}); RUNS_DIR ($MILES_AGENT_RESULTS/runs)
#   MILES_TORCH_DIST_DIR              $MILES_AGENT_MODELS/torch_dist_v5
#   MILES_ROOT                        the miles checkout: /root/miles when that dir exists (the image), else the
#                                     submodule $REPO/3rdparty/miles
#   JOB_ID                            the wrapper's job id, else this shell's pid (log suffixes)
#   NEMO_GYM_PORT                     8200, the agent server port of the gym a run starts itself
#   PROJECT_PORTS                     the fixed ports a run pre-checks ("6100 6200 6379 7100-7131 7200 8100 8200
#                                     8265 13141-13143"); WEBSHOP_SERVER_PORT stays env_servers/webshop/defaults.sh's
#   LOCAL_ENV                         the knob file, default $REPO/local.env (LOCAL_ENV=/dev/null reads none)
# Functions
#   load_local_env [file]             KEY=value lines (^[A-Z_][A-Z0-9_]*=), comments and blanks skipped, the
#                                     value = what stands between its quotes or up to a trailing ` # note`,
#                                     `$REPO` / `${REPO}` / `$HOME` / `${HOME}` expanded at a word boundary only
#                                     (`$REPOSITORY`, `$HOMEBREW_PREFIX` stay as written); a variable already set
#                                     in the environment (even empty) wins; of two lines with one key the LAST
#                                     wins (source semantics); exported; never sourced, never eval'd
#   run_dirs <tag>                    RUN_DATA_DIR (MILES_DATA_DIR, else $MILES_AGENT_DATA/runs/<tag>), SYNC_DIR
#                                     ($MILES_AGENT_RESULTS/<tag>/sync), PREEMPT_SENTINEL ($MILES_AGENT_RESULTS/
#                                     <tag>/stop; its .done sits beside it); exported, nothing is created
#   measure_gpus                      prints GPUS_PER_NODE when set, else the count of CUDA_VISIBLE_DEVICES
#                                     entries when set and non-empty, else `nvidia-smi -L | grep -c '^GPU'`; 0 is
#                                     printed, never a FATAL (the caller decides)
#   contract_check                    CONTRACT_VERSION unset = local mode (silent); a major other than 1 -> FATAL
#   scienceworld_java_home            prints the JDK the ScienceWorld worker JVMs run on, the ONE order every caller
#                                     (scripts/lib/gym_ladder.sh, scripts/fetch_data.sh, scripts/build_gym_venvs.sh)
#                                     shares: SCIENCEWORLD_JAVA_HOME, SCIW_JAVA_HOME, JAVA_HOME, then $TOOLS_DIR/jdk-17
#                                     when TOOLS_DIR is set and that dir exists, else "" = the java on PATH
#   fatal <text>                      prints `FATAL: <text>` and exits 1
#   dry_run_print <label> <argv...>   prints `DRY_RUN <label>: <shell-quoted argv>`
#   is_dry_run                        true for DRY_RUN=1 (the argv-only depth: no side effect at all)
#   dry_run_depth                     prints argv (DRY_RUN=1), driver (DRY_RUN=driver: services up, no driver)
#   add_cleanup <fragment>            registers a fragment with the EXIT trap; cleanup runs them newest first
#                                     under set +e, so one failing fragment skips nothing and flips no exit code
#   preempt_term / arm_preempt_trap   TERM / USR1 raise $PREEMPT_SENTINEL instead of killing the run (the driver
#                                     finishes the step, saves and stops); armed only by the entry that wants it
#   wait_http_ready <url> <pid> <logfile> <tries> <sleep_s> [<predicate>]
#                                     boot gate for a background server: polls every <sleep_s>, up to <tries>
#                                     times, succeeds once <predicate> (a function name) returns 0 AND `curl -sf
#                                     <url>` gets a 2xx; bails out with the log's tail as soon as <pid> is gone
#   port_precheck <ports...> [-- <urls...>]
#                                     a-b ranges expanded; the port of every URL after `--` is skipped (those
#                                     holders are ours: RETRIEVER_URL NEMO_GYM_URL WEBSHOP_ENGINE_BASE_URL
#                                     TEACHER_URL); one `ss -tln`; a holder -> FATAL naming it (never a kill);
#                                     no ss on PATH -> a WARN, return 0
#   port_warn_check <ports...>        the session ports (7100-7131) right before the driver: `ss -tan` per port,
#                                     a holder -> `WARN: session port <p> occupied before launch:` + the evidence
#                                     (`ss -tanp`); always returns 0; silent without ss
#   node_ip                           prints the node's IPv4: getent hosts, then hostname -I, then NODE_IP; when
#                                     all are empty a FATAL goes to stderr (stdout is the answer) and it returns 1

fatal() { echo "FATAL: $*"; exit 1; }

dry_run_print() {  # <label> <argv...>
  local label=$1
  shift
  printf 'DRY_RUN %s:' "$label"
  [ $# -eq 0 ] || printf ' %q' "$@"
  printf '\n'
}
is_dry_run() { [ "${DRY_RUN:-}" = 1 ]; }
dry_run_depth() {
  case "${DRY_RUN:-}" in
    1) printf 'argv\n' ;;
    driver) printf 'driver\n' ;;
  esac
  return 0
}

# expand_var <text> <name> <value>: `${name}` and `$name` replaced by <value>, the bare form only at a word boundary
# (`$REPO/x` yes, `$REPOSITORY` no); plain parameter expansion does the cutting, so the value stays data, never code.
# The result lands in EXPANDED (not printed: a $(...) would cost a fork per line).
expand_var() {  # <text> <name> <value>
  local text=$1 name=$2 value=$3 out="" rest
  while case "$text" in *"\${$name}"*) true ;; *) false ;; esac; do
    out="$out${text%%"\${$name}"*}$value"
    text=${text#*"\${$name}"}
  done
  text="$out$text"
  out=""
  while case "$text" in *"\$$name"*) true ;; *) false ;; esac; do
    out="$out${text%%"\$$name"*}"
    rest=${text#*"\$$name"}
    case "$rest" in
      [ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_]*) out="$out\$$name" ;;
      *) out="$out$value" ;;
    esac
    text=$rest
  done
  EXPANDED="$out$text"
}

# shellcheck disable=SC2120  # [file] is the documented optional argument; the lib's own call below passes none ($LOCAL_ENV)
load_local_env() {  # [file]
  local file=${1:-$LOCAL_ENV} line key val seen=" "
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    case "$line" in
      ''|'#'*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key=${line%%=*}
    val=${line#*=}
    # the classes are spelled out: a locale's collation can fold lowercase into a [A-Z] range
    case "$key" in
      ''|[0123456789]*|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_]*) continue ;;
    esac
    # a quoted value is what stands between its quotes (whatever follows the closing one is a comment); an
    # unquoted one ends at the first ` #` and loses its trailing blanks: `#KEY=value   # note` lines of
    # local.env.example load as `value` once uncommented
    case "$val" in
      \"*) val=${val#\"}; val=${val%%\"*} ;;
      \'*) val=${val#\'}; val=${val%%\'*} ;;
      *) case "$val" in *[[:space:]]\#*) val=${val%%[[:space:]]#*} ;; esac
         val=${val%"${val##*[![:space:]]}"} ;;
    esac
    expand_var "$val" REPO "$REPO"
    expand_var "$EXPANDED" HOME "$HOME"
    val=$EXPANDED
    # the environment wins over the file; within the file the LAST line of a key wins (source semantics): a key this
    # load exported from an earlier line is overwritten by a later one
    case "$seen" in
      *" $key "*) export "$key=$val" ;;
      *) [ -n "${!key+x}" ] || { export "$key=$val"; seen="$seen$key "; } ;;
    esac
  done < "$file"
}

run_dirs() {  # <tag>
  [ -n "${1:-}" ] || fatal "run_dirs: a run tag is required"
  RUN_DATA_DIR=${MILES_DATA_DIR:-$MILES_AGENT_DATA/runs/$1}
  SYNC_DIR=${SYNC_DIR:-$MILES_AGENT_RESULTS/$1/sync}
  PREEMPT_SENTINEL=${PREEMPT_SENTINEL:-$MILES_AGENT_RESULTS/$1/stop}
  export RUN_DATA_DIR SYNC_DIR PREEMPT_SENTINEL
}

measure_gpus() {
  local n
  if [ -n "${GPUS_PER_NODE:-}" ]; then
    printf '%s\n' "$GPUS_PER_NODE"; return 0
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    n=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || true)
    printf '%s\n' "${n:-0}"; return 0
  fi
  # `|| true`: grep -c exits 1 on a zero count (its "0" is still printed), and under a caller's set -e that status
  # would end the caller without a word; a missing nvidia-smi counts as no GPU the same way
  n=$("$NVIDIA_SMI" -L 2>/dev/null | grep -c '^GPU' || true)
  printf '%s\n' "${n:-0}"
}

contract_check() {
  local v=${CONTRACT_VERSION:-}
  [ -n "$v" ] || return 0
  [ "${v%%.*}" = 1 ] || fatal "unsupported CONTRACT_VERSION $v (this entry speaks 1)"
}

# The JDK the ScienceWorld worker JVMs run on: one order for the ladder (recipe_vars scienceworld), fetch_data.sh's
# check and build_gym_venvs.sh's real-JVM check. "" = the java on PATH (the env server's own default).
scienceworld_java_home() {
  if [ -n "${SCIENCEWORLD_JAVA_HOME:-}" ]; then printf '%s\n' "$SCIENCEWORLD_JAVA_HOME"
  elif [ -n "${SCIW_JAVA_HOME:-}" ]; then printf '%s\n' "$SCIW_JAVA_HOME"
  elif [ -n "${JAVA_HOME:-}" ]; then printf '%s\n' "$JAVA_HOME"
  elif [ -n "${TOOLS_DIR:-}" ] && [ -d "$TOOLS_DIR/jdk-17" ]; then printf '%s\n' "$TOOLS_DIR/jdk-17"
  fi
  return 0
}

# single EXIT trap accumulating cleanups; set +e inside so one failing
# fragment cannot skip the rest or flip a successful run's exit code
G_CLEANUP=":"
cleanup() { set +e; eval "$G_CLEANUP"; }
add_cleanup() { G_CLEANUP="$1; $G_CLEANUP"; }

# A preemption on a machine without a user signal SIGTERMs the step's tasks
# (this script on every node; their children are not signalled). Dying here
# would take the whole step down before the save: raise the sentinel instead
# and carry on; the driver saves and stops, and the caller requeues or exits.
preempt_term() {
  echo "PREEMPT_TERM on node ${NODE_RANK:-0}: raising the sentinel"
  [ -z "${PREEMPT_SENTINEL:-}" ] || touch "$PREEMPT_SENTINEL" 2>/dev/null || true
}
arm_preempt_trap() { trap preempt_term TERM USR1; }

# wait_http_ready <url> <pid> <logfile> <tries> <sleep_s> [<predicate>]
# Boot gate for a background server: polls every <sleep_s>, up to <tries>
# times, and succeeds once the optional <predicate> (a function name) returns 0
# AND `curl -sf <url>` gets a 2xx. Bails out as soon as <pid> is gone, so a
# server that dies at boot fails the gate immediately with its real cause (the
# tail of <logfile>) instead of burning the whole budget.
wait_http_ready() {
  local url=$1 pid=$2 logfile=$3 tries=$4 sleep_s=$5 pred=${6:-:}
  for _ in $(seq "$tries"); do
    if "$pred" && "$CURL" -sf -o /dev/null "$url"; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "wait_http_ready: pid $pid exited before $url answered — last 100 lines of $logfile:"
      tail -100 "$logfile"; return 1
    fi
    sleep "$sleep_s"
  done
  echo "wait_http_ready: $url not ready after $tries x ${sleep_s}s — last 100 lines of $logfile:"
  tail -100 "$logfile"; return 1
}

port_expand() {  # "6100 7100-7131 8200" -> one port per line
  local p
  for p in "$@"; do
    case "$p" in
      *-*) seq "${p%-*}" "${p#*-}" ;;
      *) printf '%s\n' "$p" ;;
    esac
  done
}

# A fixed port held by someone else on a shared machine: say who, before anything starts. Nothing is ever
# killed here (a stale-process sweep is a job wrapper's business, with the wrapper's knowledge of ownership).
port_precheck() {  # <ports...> [-- <urls...>] (a URL may be a comma list: NEMO_GYM_URL's agent servers)
  local ports='' skip='' p u list host_port listing line
  local -a urls
  while [ $# -gt 0 ]; do
    [ "$1" != -- ] || { shift; break; }
    ports="$ports $1"
    shift
  done
  for list in "$@"; do
    IFS=, read -r -a urls <<< "$list"
    for u in ${urls[@]+"${urls[@]}"}; do
      host_port=${u#*://}
      host_port=${host_port%%/*}
      case "$host_port" in
        *:*) skip="$skip ${host_port##*:}" ;;
      esac
    done
  done
  if ! command -v ss >/dev/null 2>&1; then
    echo "WARN: no ss on PATH: port pre-check skipped"
    return 0
  fi
  listing=$(ss -tln 2>/dev/null || true)
  # shellcheck disable=SC2086
  for p in $(port_expand $ports); do
    case " $skip " in *" $p "*) continue ;; esac
    line=$(printf '%s\n' "$listing" | awk -v m=":$p" '$4 ~ m"$"' || true)
    [ -z "$line" ] || fatal "port :$p is already held: $line"
  done
  return 0
}

# The session ports before the driver starts: miles pins 7100-7131 (sub-9000: an ephemeral range starting at
# 9000 can squat anything above it between check and bind). If anything holds them, dump the evidence before
# miles dies on its own check — a WARN, never a failure.
port_warn_check() {  # <ports...>
  local sp
  command -v ss >/dev/null 2>&1 || return 0
  # shellcheck disable=SC2068
  for sp in $(port_expand $@); do
    if ss -tan "( sport = :$sp )" 2>/dev/null | grep -q ":$sp"; then
      echo "WARN: session port $sp occupied before launch:"
      ss -tanp "( sport = :$sp )" 2>/dev/null || true
    fi
  done
  return 0
}

node_ip() {
  local ip
  ip=$(getent hosts "$(hostname)" 2>/dev/null | awk '{print $1; exit}' || true)
  [ -n "$ip" ] || ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
  [ -n "$ip" ] || ip=${NODE_IP:-}
  [ -n "$ip" ] || { echo "FATAL: could not determine node IP (getent hosts, hostname -I and NODE_IP are all empty)" >&2; return 1; }
  printf '%s\n' "$ip"
}

# ---- the defaults: environment first, then local.env, then the repo ----
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck disable=SC2034  # read by the sourcing scripts (scripts/run_local.sh, ray_cluster.sh, start_gym.sh: . "$LIB/<lib>.sh")
LIB=$REPO/scripts/lib
LOCAL_ENV=${LOCAL_ENV:-$REPO/local.env}
load_local_env

PYTHON=${PYTHON:-python3}
RAY_BIN=${RAY_BIN:-ray}
GYM_BIN=${GYM_BIN:-gym}
UV_BIN=${UV_BIN:-uv}
CURL=${CURL:-curl}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
SSH=${SSH:-ssh}
# exported on purpose: the children (the legacy scripts/start_gym.sh, the drivers, the helpers a script runs) must
# call the same tools the entry resolved
export PYTHON RAY_BIN GYM_BIN UV_BIN CURL NVIDIA_SMI SSH

MILES_AGENT_MODELS=${MILES_AGENT_MODELS:-${MODELS_DIR:-$REPO/models}}
MILES_AGENT_DATA=${MILES_AGENT_DATA:-${DATA_DIR:-$REPO/data}}
MILES_AGENT_RESULTS=${MILES_AGENT_RESULTS:-${RESULTS_DIR:-$REPO/results}}
RUNS_DIR=${RUNS_DIR:-$MILES_AGENT_RESULTS/runs}
MILES_TORCH_DIST_DIR=${MILES_TORCH_DIST_DIR:-$MILES_AGENT_MODELS/torch_dist_v5}
if [ -z "${MILES_ROOT:-}" ]; then
  if [ -d /root/miles ]; then MILES_ROOT=/root/miles; else MILES_ROOT=$REPO/3rdparty/miles; fi
fi
JOB_ID=${JOB_ID:-$$}
NEMO_GYM_PORT=${NEMO_GYM_PORT:-8200}
PROJECT_PORTS=${PROJECT_PORTS:-"6100 6200 6379 7100-7131 7200 8100 8200 8265 13141-13143"}
export MILES_AGENT_MODELS MILES_AGENT_DATA MILES_AGENT_RESULTS RUNS_DIR MILES_TORCH_DIST_DIR MILES_ROOT
export JOB_ID NEMO_GYM_PORT PROJECT_PORTS

trap cleanup EXIT
