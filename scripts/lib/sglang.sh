# Shared sglang server start / wait / stop for the jobs that serve a model next to training or a pipeline stage
# (scripts/teacher_node.sh: the OPD teacher; the R2S pipeline stages) and for a plain
# machine with two servers: sourced, no scheduler call. Contract with the sourcing script: `add_cleanup <fragment>`
# (the training entry's EXIT-trap accumulator) takes each server's stop; RUN_TAG names the logs, which go under
# ${RUNS_DIR:-$MILES_AGENT_RESULTS/runs} suffixed by ${JOB_ID:-$$}; ${SRC_MOUNT:-$REPO} is the repo root
# (python3 -m miles_agent.launcher).
#   start_sglang <name> <model_dir> <port> <cuda_ids|all> <dp> [sglang args...]
#       python3 -m sglang.launch_server on 0.0.0.0:<port>, TP 1 x DP <dp>, in its own process group (setsid: the
#       stop takes its schedulers and detokenizer with it). <cuda_ids> is the server's CUDA_VISIBLE_DEVICES, `all`
#       the node's GPUs; a DP wider than those (a GPU-less node included) is refused before the model loads: return
#       1 after a one-line FATAL, never a wordless exit of a set -e caller. SGLANG_PID_<name> / SGLANG_LOG_<name>
#       hold the pid and the log (sglang_pid / sglang_log).
#   wait_sglang <name> <url> <tries> <sleep_s> [stop_file]
#       `curl -sf <url>` every <sleep_s>, up to <tries> times; fails at once, with the log's tail, when the server is
#       gone (the training entry's wait_http_ready over the registry: the teacher node runs before that helper exists,
#       a plain machine never has it). With <stop_file> (the preempt sentinel) every poll checks it first: present,
#       the wait ends with return 2 and no tail, the caller's stop (the R2S pipeline requeues), not a failure.
#   stop_sglang <name>              TERM the process group, 30 s, then KILL
#   free_port <preferred> [span=20]  prints the first port from <preferred> that binds on 0.0.0.0 (a probe socket,
#       closed at once), return 1 when none of the <span> does: a port a leftover server holds (an earlier job's
#       process outside this job's cleanup) moves a server up instead of failing its boot with "address already in use"
#   serve_template <model_dir> <out>
#       writes the model family's fixed chat template with thinking defaulted off and prints its tool-call parser
#       (launcher.write_serve_template): the servers that render and parse as the training engines do.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

sglang_pid() { local v=SGLANG_PID_$1; printf '%s' "${!v:-}"; }
sglang_log() { local v=SGLANG_LOG_$1; printf '%s' "${!v:-}"; }

start_sglang() {
  local name=$1 model_dir=$2 port=$3 cuda_ids=$4 dp=$5 gpus runs log cuda_env=
  shift 5
  [ -f "$model_dir/config.json" ] || { echo "FATAL: start_sglang $name: no HF model dir $model_dir"; return 1; }
  # `|| true`: grep -c exits 1 on a zero count (its "0" is still printed), and under a caller's set -e that status
  # would end the caller without a word; the check below returns the reason instead
  if [ "$cuda_ids" = all ]; then
    gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  else
    gpus=$(printf '%s' "$cuda_ids" | tr ',' '\n' | grep -c . || true)
    cuda_env="CUDA_VISIBLE_DEVICES=$cuda_ids"
  fi
  [ "$gpus" -ge "$dp" ] || { echo "FATAL: start_sglang $name: --dp-size $dp over $gpus GPU(s) ($cuda_ids)"; return 1; }
  runs=${RUNS_DIR:-${MILES_AGENT_RESULTS:?RUNS_DIR or MILES_AGENT_RESULTS must be set}/runs}
  mkdir -p "$runs"
  log=$runs/${RUN_TAG:?}_${name}_${JOB_ID:-$$}.log
  # $cuda_env unquoted on purpose: empty for `all`, one assignment otherwise
  env $cuda_env setsid python3 -m sglang.launch_server --model-path "$model_dir" --host 0.0.0.0 --port "$port" \
    --tp 1 --dp-size "$dp" "$@" > "$log" 2>&1 &
  printf -v "SGLANG_PID_$name" '%s' "$!"
  printf -v "SGLANG_LOG_$name" '%s' "$log"
  add_cleanup "stop_sglang $name"
  echo "SGLANG_START $name pid=$! port=$port gpus=$cuda_ids dp=$dp model=$model_dir log=$log"
}

wait_sglang() {
  local name=$1 url=$2 tries=$3 sleep_s=$4 stop_file=${5:-} pid log i
  pid=$(sglang_pid "$name"); log=$(sglang_log "$name")
  for i in $(seq "$tries"); do
    if [ -n "$stop_file" ] && [ -e "$stop_file" ]; then
      echo "wait_sglang: $name: stop file $stop_file raised while waiting"; return 2
    fi
    if curl -sf -o /dev/null "$url"; then echo "SGLANG_UP $name url=$url after $i x ${sleep_s}s"; return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "wait_sglang: $name (pid $pid) exited before $url answered — last 100 lines of $log:"
      tail -100 "$log"; return 1
    fi
    sleep "$sleep_s"
  done
  echo "wait_sglang: $url not ready after $tries x ${sleep_s}s — last 100 lines of $log:"
  tail -100 "$log"; return 1
}

free_port() {
  local preferred=$1 span=${2:-20} i
  for (( i = 0; i < span; i++ )); do
    if python3 -c 'import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)
s.close()' "$((preferred + i))" 2>/dev/null; then printf '%s' "$((preferred + i))"; return 0; fi
  done
  return 1
}

stop_sglang() {
  local pid
  pid=$(sglang_pid "$1")
  [ -n "$pid" ] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || return 0
  for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$pid" 2>/dev/null || true
}

serve_template() {
  PYTHONPATH=${SRC_MOUNT:-$REPO}${PYTHONPATH:+:$PYTHONPATH} \
    python3 -m miles_agent.launcher serve-template "$1" "$2"
}
