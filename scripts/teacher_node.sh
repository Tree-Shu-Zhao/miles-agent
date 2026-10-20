#!/bin/bash
# The OPD teacher of a *_teacher topology: the allocation's LAST node serves it with sglang, outside ray, for
# miles_agent.opd.teacher_scoring (the student's whole token stream goes in as input_ids, its sampled log-probs come
# out: scoring only, so no chat template, no parser, no cuda graphs). Sourced by the training entry after
# scripts/lib/sglang.sh; no scheduler call, so a plain machine runs the teacher side with SYNC_DIR on a filesystem
# the head sees.
#   teacher_node_main   the teacher node: start, wait for /health_generate, publish $SYNC_DIR/TEACHER_URL
#                       (http://<node ip>:13141, written whole: .tmp then mv) or touch $SYNC_DIR/TEACHER_FAILED;
#                       serve until $SYNC_DIR/ENDED (the head's cleanup drops it) or the server dies (TEACHER_FAILED,
#                       exit 1); TEACHER_ENDED. TERM / USR1 are logged and served through: a preemption's save on
#                       the head must never be cut short by this task dying first (the wrapper's kill-on-bad-exit). Exits.
#   teacher_gate        the head, once gym is up: TEACHER_URL (an already serving teacher) wins, else wait for
#                       TEACHER_URL / TEACHER_FAILED, checking PREEMPT_SENTINEL every poll (a preemption during the
#                       teacher's boot: TEACHER_GATE_PREEMPT and the entry's "PREEMPT before launch" exit 0,
#                       so the caller's preempt handling requeues at once, not after the wait); curl
#                       /health_generate; the tokenizer probe (python3 -m miles_agent.launcher tokenizer-probe
#                       <student> <teacher>, its stderr kept in ${RUNS_DIR:-$MILES_AGENT_RESULTS/runs}/<RUN_TAG>_tokenizer_probe_<JOB_ID>.log:
#                       TOKENIZER_PROBE_FAIL is the drift FATAL, anything else "the probe could not run";
#                       TEACHER_TOKENIZER_CHECK=0 skips it, as does an external URL with no teacher dir named);
#                       exports TEACHER_URL_ARG="--teacher-url <url>" for the recipe's --algo opd; TEACHER_GATE_OK.
# Knobs: TEACHER_MODEL (a name under $MILES_AGENT_MODELS) or TEACHER_HF (the dir); TEACHER_DP 4 (TP 1),
# TEACHER_MEM_FRAC 0.75, TEACHER_CONTEXT_LEN 33792 (the arm's 32768 context + 1024: an episode that fills the
# student's context still scores, sglang refuses an input at its limit), TEACHER_WAIT_TRIES 480 (x 5 s: the
# campaign's value, a MoE teacher loads slowly); SYNC_DIR, MY_IP, POLICY_HF, RUN_TAG are the caller's.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TEACHER_PORT=13141

teacher_hf_dir() {  # the teacher's HF dir, or nothing when the job names none (an external TEACHER_URL)
  if [ -n "${TEACHER_HF:-}" ]; then
    printf '%s' "$TEACHER_HF"
  elif [ -n "${TEACHER_MODEL:-}" ]; then
    printf '%s' "${MILES_AGENT_MODELS:?MILES_AGENT_MODELS must be set}/$TEACHER_MODEL"
  fi
}

teacher_node_main() {
  : "${SYNC_DIR:?the head and the teacher node share SYNC_DIR}"
  local hf ip
  hf=$(teacher_hf_dir)
  [ -n "$hf" ] || { echo "FATAL: a *_teacher topology needs TEACHER_MODEL (a name under $MILES_AGENT_MODELS) or TEACHER_HF"; touch "$SYNC_DIR/TEACHER_FAILED"; exit 1; }
  ip=${MY_IP:-$(getent hosts "$(hostname)" | awk '{print $1; exit}')}
  [ -n "$ip" ] || { echo "FATAL: could not determine the teacher node's IP"; touch "$SYNC_DIR/TEACHER_FAILED"; exit 1; }
  trap 'echo "TEACHER_SIGNAL: serving until $SYNC_DIR/ENDED (the head saves first)"' TERM USR1
  start_sglang teacher "$hf" "$TEACHER_PORT" all "${TEACHER_DP:-4}" \
      --mem-fraction-static "${TEACHER_MEM_FRAC:-0.75}" --chunked-prefill-size 4096 \
      --context-length "${TEACHER_CONTEXT_LEN:-33792}" --disable-cuda-graph --log-level warning \
    || { touch "$SYNC_DIR/TEACHER_FAILED"; exit 1; }
  wait_sglang teacher "http://127.0.0.1:$TEACHER_PORT/health_generate" "${TEACHER_WAIT_TRIES:-480}" 5 \
    || { echo "FATAL: teacher $hf failed to boot"; touch "$SYNC_DIR/TEACHER_FAILED"; exit 1; }
  echo "http://$ip:$TEACHER_PORT" > "$SYNC_DIR/TEACHER_URL.tmp" && mv "$SYNC_DIR/TEACHER_URL.tmp" "$SYNC_DIR/TEACHER_URL"
  echo "TEACHER_UP model=$hf url=http://$ip:$TEACHER_PORT dp=${TEACHER_DP:-4}"
  until [ -f "$SYNC_DIR/ENDED" ]; do
    kill -0 "$(sglang_pid teacher)" 2>/dev/null \
      || { echo "FATAL: the teacher server died mid-run (see $(sglang_log teacher))"; touch "$SYNC_DIR/TEACHER_FAILED"; exit 1; }
    sleep 10
  done
  echo "TEACHER_ENDED"
  exit 0
}

teacher_gate() {
  local url hf tries=${TEACHER_WAIT_TRIES:-480}
  if [ -n "${TEACHER_URL:-}" ]; then
    url=$TEACHER_URL
    echo "TEACHER_GATE: TEACHER_URL=$url (an already serving teacher; no node of this job serves one)"
  else
    : "${SYNC_DIR:?}"
    for _ in $(seq "$tries"); do
      # the entry's launch gate ("PREEMPT before launch", exit 0, and the wrapper
      # requeues on the sentinel) sits after this wait; a sentinel raised during a slow teacher boot
      # takes that path now instead of holding the head for the remaining polls
      if [ -n "${PREEMPT_SENTINEL:-}" ] && [ -e "$PREEMPT_SENTINEL" ]; then
        echo "TEACHER_GATE_PREEMPT: the sentinel is up while waiting for the teacher"
        echo "PREEMPT before launch: not starting the driver"; exit 0
      fi
      [ -f "$SYNC_DIR/TEACHER_FAILED" ] && { echo "FATAL: the teacher node reported a failure (see the ${RUN_TAG:-}_teacher_* log)"; exit 1; }
      [ -f "$SYNC_DIR/TEACHER_URL" ] && break
      sleep 5
    done
    [ -f "$SYNC_DIR/TEACHER_URL" ] || { echo "FATAL: no teacher URL after $tries x 5 s (TEACHER_WAIT_TRIES)"; exit 1; }
    url=$(cat "$SYNC_DIR/TEACHER_URL")
  fi
  curl -sf -o /dev/null "$url/health_generate" || { echo "FATAL: teacher $url does not answer /health_generate from the head"; exit 1; }
  hf=$(teacher_hf_dir)
  if [ "${TEACHER_TOKENIZER_CHECK:-1}" != 1 ]; then
    echo "TEACHER_GATE: tokenizer probe skipped (TEACHER_TOKENIZER_CHECK=$TEACHER_TOKENIZER_CHECK)"
  elif [ -z "$hf" ]; then
    echo "TEACHER_GATE: tokenizer probe skipped (TEACHER_URL names no TEACHER_MODEL / TEACHER_HF to read)"
  else
    # the probe's stderr goes to a log: TOKENIZER_PROBE_FAIL <reason> is a drift (launcher.main's verdict, matched
    # anywhere: a warning without its trailing newline may precede it on the same line), anything else (a missing
    # dir, an import error) means the probe never compared the two tokenizers
    local runs log
    runs=${RUNS_DIR:-${MILES_AGENT_RESULTS:?RUNS_DIR or MILES_AGENT_RESULTS must be set}/runs}
    mkdir -p "$runs"
    log=$runs/${RUN_TAG:?}_tokenizer_probe_${JOB_ID:-$$}.log
    if ! PYTHONPATH=${SRC_MOUNT:-$REPO}${PYTHONPATH:+:$PYTHONPATH} \
        python3 -m miles_agent.launcher tokenizer-probe "${POLICY_HF:?}" "$hf" 2> "$log"; then
      cat "$log" >&2
      if grep -q 'TOKENIZER_PROBE_FAIL' "$log"; then
        echo "FATAL: teacher tokenizer differs from the student's ($hf vs $POLICY_HF; TEACHER_TOKENIZER_CHECK=0 skips the probe)"
      else
        echo "FATAL: the tokenizer probe could not run (its stderr is above and in $log): TEACHER_MODEL=${TEACHER_MODEL:-} TEACHER_HF=${TEACHER_HF:-} -> $hf, POLICY_HF=$POLICY_HF"
      fi
      exit 1
    fi
  fi
  export TEACHER_URL_ARG="--teacher-url $url"
  echo "TEACHER_GATE_OK url=$url"
}
