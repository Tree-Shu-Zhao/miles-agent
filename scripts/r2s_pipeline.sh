# The R2S repair pipeline on one GPU machine: stages A -> B [-> J] -> C -> D of miles_agent.r2s over the pilot dump
# of an opd arm (docs/r2s.md section 4), with the servers the stages dial started here over scripts/lib/sglang.sh.
# The base teacher (:13141, GPUs 0..TEACHER_DP-1; the family's fixed template with thinking defaulted off, its
# tool-call and reasoning parsers, --served-model-name teacher) serves from the first stage through D, stage D's
# labels being that teacher's own privileged pass (design F1); the student's base checkpoint prefills stage A
# (:13142, the next GPU, stopped after A); when JUDGE_MODEL is not the teacher's dir a judge serves stage J (:13143,
# the GPUs after the teacher's, stopped after J), else the teacher's server answers the judge's requests (their
# enable_thinking true wins over the template's default, D6). Sourced: the training entry runs r2s_pipeline_main
# right after GYM_UP under MODE=r2s_pipeline, so the recipe's gym stack, data prep, preempt
# sentinel and idle watchdog are inherited (D7); no scheduler call, so a plain machine with the recipe's
# env_loop_agent on :8200 sources this file under scripts/lib/sglang.sh's contract (add_cleanup, RUN_TAG, the
# MILES_AGENT_* dirs) and calls r2s_pipeline_main.
#   TAG           names the outputs: $MILES_AGENT_DATA/r2s/<TAG>/{states,fold,judge,tau,labels}.jsonl,
#                 each with <file>.summary.json (the stage's marker closes it)
#   RECIPE        alfworld | scienceworld | webshop: the stages' --env (select.ENV_DEFAULTS), the gym the job booted
#   DUMP_GLOB     the pilot dump files, stage A's input ($MILES_AGENT_RESULTS/<pilot RUN_TAG>/opd_dump/dump_*.jsonl);
#                 STATES_IN / FOLD_IN start from a finished stage's output instead
#   STAGES        A,B,C,D (default; B,C,D under STATES_IN, C,D under FOLD_IN), any subset in that order; J goes
#                 before C when the env's check is judge (WebShop) and C runs
#   TEACHER_MODEL (a name under $MILES_AGENT_MODELS) or TEACHER_HF  the base teacher being repaired
#   JUDGE_MODEL   Qwen3.5-9B (the paper's judge for both teachers, D15) or JUDGE_HF
#   POLICY_HF     the student's base checkpoint (the entry's POLICY_MODEL, default Qwen3.5-4B): the
#                 tokenizer of the rows and stage A's student server
#   R2S_ARGS_A|B|J|C|D  more flags for one stage, verbatim and last (--force reruns a finished stage, --limit N
#                 a smoke, --env-config a table laid over ENV_DEFAULTS): a knob change never edits this script.
#                 The value is word-split without quote re-parsing (a JSON value is one word without spaces); the
#                 one case that needs it: a Qwen3 hybrid student whose campaign v1 rows were rendered thinking off
#                 (launcher.py MODEL_FAMILIES["Qwen3-"] keeps the template's default, thinking on, so stage D
#                 would drop every such state as prompt_retok_mismatch) passes
#                 R2S_ARGS_A='--chat-template-kwargs {"enable_thinking":false}' and the same R2S_ARGS_D
#   TOP_N 2400 MAX_PER_GOAL 3 (A); SAMPLES 8 TAKEOVER_TEMPERATURE 1.0 MIN_REPLAY_OK 0.7 (B); K 16 (D); CONCURRENCY
#                 32 (A's and B's requests in flight); MIN_STATES 100 (the states B needs); STOP_AFTER_FOLD 0 (1:
#                 the job ends after B with R2S_PIPELINE_PASS fold_only; B must be in STAGES); TEACHER_DP 2;
#                 TEACHER_WAIT_TRIES 480 (x 5 s per server): the campaign's values (the campaign branch's
#                 submit script)
# Gates and markers: DIVERGENCE_DONE and >= MIN_STATES states -> R2S_DIVERGENCE_PASS; TAKEOVER_DONE ->
# R2S_TAKEOVER_PASS; JUDGE_DONE -> R2S_JUDGE_PASS; SELECT_DONE -> R2S_SELECT_PASS; LABEL_DONE -> R2S_LABEL_PASS;
# R2S_PIPELINE_PASS. A stage's gate refusal (TAKEOVER_FAIL under MIN_REPLAY_OK, LABEL_FAIL under the env's labeled
# fraction: exit 2) and a missing marker are FATAL. A stage whose summary already carries its marker says so and
# does nothing (the stages' own resume; R2S_ARGS_<S>=--force reruns it). With PREEMPT_SENTINEL set every stage gets
# --stop-file and every server wait polls the same file: a raised sentinel ends the stage between rows, or the wait
# while a server boots, the driver writes <sentinel>.done ({"stage", "rows"}; "boot", 0 for a wait), prints
# R2S_PIPELINE_PREEMPT_STOP and returns 0, so the wrapper requeues and the requeued job
# resumes per stage and row; without it the stages and the waits run to completion.
R2S_TEACHER_PORT=13141
R2S_STUDENT_PORT=13142
R2S_JUDGE_PORT=13143
R2S_CONTEXT_LEN=32768  # the stages' --max-ctx (select.MAX_CTX): no server is handed a longer input
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
. "${SRC_MOUNT:-$REPO}/scripts/lib/sglang.sh"

stage_on() { case ",$R2S_STAGES," in *,"$1",*) ;; *) return 1 ;; esac; }

gpu_list() {  # <first> <last>: "first,...,last"; nothing when last < first
  local i out=
  for (( i = $1; i <= $2; i++ )); do out=$out${out:+,}$i; done
  printf '%s' "$out"
}

wait_child() {  # <pid>: bash runs the TERM / USR1 trap that raises the sentinel only once a foreground child exits
  local xtrace=
  case $- in *x*) xtrace=1 ;; esac
  { set +x; } 2>/dev/null
  while kill -0 "$1" 2>/dev/null; do sleep 5 & wait $! || true; done
  [ -z "$xtrace" ] || set -x
  wait "$1"
}

preempt_stop() {  # <stage> <rc> <rows>: the sentinel is up; <sentinel>.done and the marker for the requeue
  printf '{"stage": "%s", "rows": %s}\n' "$1" "$3" > "$PREEMPT_SENTINEL.done"
  echo "R2S_PIPELINE_PREEMPT_STOP stage=$1 rc=$2 rows=$3: the sentinel is up; the requeued job resumes here"
}

wait_server() {  # <name> <url>: wait_sglang with the sentinel as its stop file; raised while waiting (rc 2) it
  local rc=0       # is the job's stop, not a failure, and the caller's `|| return 0` ends the pipeline; else FATAL
  wait_sglang "$1" "$2" "$TEACHER_WAIT_TRIES" 5 "${PREEMPT_SENTINEL:-}" || rc=$?
  case $rc in
    0) ;;
    2) preempt_stop boot "$rc" 0; return 1 ;;
    *) echo "FATAL: the $1 server did not come up"; exit 1 ;;
  esac
}

run_stage() {  # <letter> <marker> <out jsonl> <module> [flags...]: then --stop-file, then R2S_ARGS_<letter>
  local letter=$1 marker=$2 out=$3 module=$4 extra=R2S_ARGS_$1 pid rc=0 rows=0
  shift 4
  echo "R2S_STAGE_START $letter $module out=$out"
  # ${!extra} unquoted on purpose: the job's extra flags, word-split, after the driver's so that theirs win
  python3 -m "$module" "$@" ${PREEMPT_SENTINEL:+--stop-file "$PREEMPT_SENTINEL"} ${!extra:-} &
  pid=$!
  wait_child "$pid" || rc=$?
  if [ -n "${PREEMPT_SENTINEL:-}" ] && [ -e "$PREEMPT_SENTINEL" ]; then
    [ ! -f "$out" ] || rows=$(( $(wc -l < "$out") ))
    preempt_stop "$letter" "$rc" "$rows"
    return 1
  fi
  [ "$rc" -eq 0 ] \
    || { echo "FATAL: stage $letter ($module) exited $rc (2: its gate refused the run; see $out.summary.json)"; exit 1; }
  grep -qx "$marker" "$out.summary.json" 2>/dev/null \
    || { echo "FATAL: stage $letter ended without $marker closing $out.summary.json"; exit 1; }
}

r2s_pipeline_main() {
  local models out check s states fold judge tau labels judge_file ngpu teacher_gpus rest_gpus rest_dp
  local teacher_on teacher_url judge_url judge_hf judge_model template parser reasoning n_states
  : "${TAG:?TAG=<pipeline tag> names the outputs}" "${RUN_TAG:?}" "${RECIPE:?RECIPE=alfworld | scienceworld | webshop}"
  models=${MILES_AGENT_MODELS:?MILES_AGENT_MODELS must be set}
  TEACHER_HF=${TEACHER_HF:-$models/${TEACHER_MODEL:?TEACHER_MODEL=<name under the models dir> or TEACHER_HF=<dir>: the base teacher}}
  POLICY_HF=${POLICY_HF:-$models/${POLICY_MODEL:-Qwen3.5-4B}}
  judge_hf=${JUDGE_HF:-$models/${JUDGE_MODEL:-Qwen3.5-9B}}
  out=${MILES_AGENT_DATA:?MILES_AGENT_DATA must be set}/r2s/$TAG
  TOP_N=${TOP_N:-2400} MAX_PER_GOAL=${MAX_PER_GOAL:-3} CONCURRENCY=${CONCURRENCY:-32} MIN_STATES=${MIN_STATES:-100}
  SAMPLES=${SAMPLES:-8} TAKEOVER_TEMPERATURE=${TAKEOVER_TEMPERATURE:-1.0} MIN_REPLAY_OK=${MIN_REPLAY_OK:-0.7}
  K=${K:-16} STOP_AFTER_FOLD=${STOP_AFTER_FOLD:-0} TEACHER_DP=${TEACHER_DP:-2} TEACHER_WAIT_TRIES=${TEACHER_WAIT_TRIES:-480}
  export PYTHONPATH=${SRC_MOUNT:-$REPO}${PYTHONPATH:+:$PYTHONPATH}
  # the env's check from the one home of the per-env table (select.ENV_DEFAULTS, D8): judge puts stage J before C
  check=$(python3 -c 'import sys; from miles_agent.r2s import select; print(select.load_env_config(sys.argv[1]).check)' "$RECIPE") \
    || { echo "FATAL: RECIPE=$RECIPE is not an R2S environment (miles_agent/r2s/select.py ENV_DEFAULTS: alfworld | scienceworld | webshop)"; exit 1; }
  R2S_STAGES=${STAGES:-A,B,C,D}
  if [ -z "${STAGES:-}" ]; then
    [ -z "${STATES_IN:-}" ] || R2S_STAGES=B,C,D
    [ -z "${FOLD_IN:-}" ] || R2S_STAGES=C,D
  fi
  if [ "$check" = judge ] && stage_on C && ! stage_on J; then R2S_STAGES=${R2S_STAGES/C/J,C}; fi
  for s in $(printf '%s' "$R2S_STAGES" | tr ',' ' '); do
    case $s in A|B|J|C|D) ;; *) echo "FATAL: STAGES=${STAGES:-}: the letters A B J C D, comma-separated"; exit 1 ;; esac
  done
  states=${STATES_IN:-$out/states.jsonl} fold=${FOLD_IN:-$out/fold.jsonl}
  judge=$out/judge.jsonl tau=$out/tau.jsonl labels=$out/labels.jsonl
  judge_file=
  [ "$check" != judge ] || judge_file=$judge
  # a port a leftover server holds (an earlier job's process, outside this job's cleanup) moves that server up: the
  # three are probed in order, each from above the one before, so a shifted server never takes the next one's port
  local port_name port_var port next_port=0
  for port_name in TEACHER STUDENT JUDGE; do
    port_var=R2S_${port_name}_PORT
    port=${!port_var}; [ "$port" -ge "$next_port" ] || port=$next_port
    port=$(free_port "$port" 20) || { echo "FATAL: no free port in ${!port_var}..$(( ${!port_var} + 19 )) for the $port_name server"; exit 1; }
    [ "$port" = "${!port_var}" ] || echo "R2S_PORT_SHIFT $port_name ${!port_var} -> $port (the port was in use)"
    printf -v "$port_var" '%s' "$port"; next_port=$((port + 1))
  done
  teacher_url=http://127.0.0.1:$R2S_TEACHER_PORT
  judge_url=$teacher_url judge_model=teacher
  [ "${judge_hf%/}" = "${TEACHER_HF%/}" ] || { judge_url=http://127.0.0.1:$R2S_JUDGE_PORT; judge_model=judge; }
  teacher_on=1
  [ "$R2S_STAGES" != C ] || teacher_on=0  # a selection alone (a config change, D8) needs no server
  # `|| true`: grep -c exits 1 on a zero count. The teacher takes the first TEACHER_DP GPUs, the student and the
  # judge the rest (a 4-GPU node: teacher 0,1, student 2, judge 2,3 DP2)
  ngpu=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  teacher_gpus=$(gpu_list 0 $(( TEACHER_DP - 1 )))
  rest_gpus=$(gpu_list "$TEACHER_DP" $(( ngpu - 1 )))
  rest_dp=$(( ngpu - TEACHER_DP ))

  # ---- fail before any server boots: inputs, outputs and GPUs every requested stage needs ----
  if [ "$STOP_AFTER_FOLD" = 1 ] && ! stage_on B; then
    echo "FATAL: STOP_AFTER_FOLD=1 needs stage B in STAGES (STAGES=$R2S_STAGES has nothing to stop after)"; exit 1
  fi
  if stage_on A; then
    : "${DUMP_GLOB:?DUMP_GLOB=<the pilot dump files, <dir>/dump_*.jsonl> (stage A)}"
    [ -z "${STATES_IN:-}" ] || { echo "FATAL: STATES_IN replaces stage A: STAGES without A, or DUMP_GLOB alone"; exit 1; }
    [ -n "$rest_gpus" ] || { echo "FATAL: TEACHER_DP=$TEACHER_DP takes all $ngpu GPUs: none left for the student server (stage A)"; exit 1; }
  elif stage_on B || stage_on D; then
    [ -f "$states" ] || { echo "FATAL: no states file $states: STATES_IN, or stage A in STAGES"; exit 1; }
  fi
  if stage_on A || stage_on D; then
    [ -f "$POLICY_HF/config.json" ] || { echo "FATAL: no student HF dir $POLICY_HF (POLICY_MODEL / POLICY_HF)"; exit 1; }
  fi
  if stage_on B; then
    [ -z "${FOLD_IN:-}" ] || { echo "FATAL: FOLD_IN replaces stage B: STAGES without B"; exit 1; }
  elif stage_on J || stage_on C || stage_on D; then
    [ -f "$fold" ] || { echo "FATAL: no fold file $fold: FOLD_IN, or stage B in STAGES"; exit 1; }
  fi
  if stage_on J; then
    if [ "$judge_model" = judge ]; then
      [ -n "$rest_gpus" ] || { echo "FATAL: TEACHER_DP=$TEACHER_DP takes all $ngpu GPUs: none left for the judge server (stage J, JUDGE_MODEL is not the teacher)"; exit 1; }
      [ -f "$judge_hf/config.json" ] || { echo "FATAL: no judge HF dir $judge_hf (JUDGE_MODEL / JUDGE_HF)"; exit 1; }
    fi
  elif stage_on C && [ -n "$judge_file" ]; then
    [ -f "$judge" ] || { echo "FATAL: no judge file $judge: stage J in STAGES (check judge)"; exit 1; }
  fi
  if stage_on D && ! stage_on C; then
    [ -f "$tau" ] || { echo "FATAL: no selection $tau: stage C in STAGES"; exit 1; }
  fi
  [ "$teacher_on" = 0 ] || [ -f "$TEACHER_HF/config.json" ] \
    || { echo "FATAL: no teacher HF dir $TEACHER_HF (TEACHER_MODEL / TEACHER_HF)"; exit 1; }
  mkdir -p "$out"
  echo "R2S_PIPELINE_START tag=$TAG env=$RECIPE check=$check stages=$R2S_STAGES teacher=$TEACHER_HF student=$POLICY_HF judge=$judge_hf out=$out gpus=$ngpu"

  # ---- servers: the teacher for A, B, J and D; the student for A ----
  if [ "$teacher_on" = 1 ]; then
    template=$out/teacher_template.jinja
    parser=$(serve_template "$TEACHER_HF" "$template")
    reasoning=$(python3 -m miles_agent.launcher reasoning-parser "$TEACHER_HF")
    start_sglang teacher "$TEACHER_HF" "$R2S_TEACHER_PORT" "$teacher_gpus" "$TEACHER_DP" \
      --mem-fraction-static 0.8 --chunked-prefill-size 4096 --context-length "$R2S_CONTEXT_LEN" \
      --chat-template "$template" --tool-call-parser "$parser" --reasoning-parser "$reasoning" \
      --served-model-name teacher --log-level warning
  fi
  if stage_on A; then
    start_sglang student "$POLICY_HF" "$R2S_STUDENT_PORT" "${rest_gpus%%,*}" 1 \
      --mem-fraction-static 0.8 --chunked-prefill-size 4096 --context-length "$R2S_CONTEXT_LEN" \
      --disable-cuda-graph --log-level warning
  fi
  # `|| return 0` = stopped on the sentinel while the server booted (a failure exits inside wait_server)
  if [ "$teacher_on" = 1 ]; then wait_server teacher "$teacher_url/health_generate" || return 0; fi
  if stage_on A; then wait_server student "http://127.0.0.1:$R2S_STUDENT_PORT/health_generate" || return 0; fi

  # ---- the stages; `|| return 0` = stopped on the sentinel (a failure exits inside run_stage) ----
  if stage_on A; then
    run_stage A DIVERGENCE_DONE "$states" miles_agent.r2s.divergence --env "$RECIPE" --dump-glob "$DUMP_GLOB" \
      --student-url "http://127.0.0.1:$R2S_STUDENT_PORT" --teacher-url "$teacher_url" --tokenizer-path "$POLICY_HF" \
      --top-n "$TOP_N" --max-per-goal "$MAX_PER_GOAL" --workers "$CONCURRENCY" --out "$states" || return 0
    stop_sglang student  # not needed past stage A; its GPU is the judge's
    echo "SGLANG_STOP student"
  fi
  if stage_on A || stage_on B; then
    n_states=$(( $(wc -l < "$states") ))
    [ "$n_states" -ge "$MIN_STATES" ] || { echo "FATAL: $n_states states under MIN_STATES=$MIN_STATES"; exit 1; }
    echo "R2S_DIVERGENCE_PASS states=$n_states"
  fi
  if stage_on B; then
    run_stage B TAKEOVER_DONE "$fold" miles_agent.r2s.takeover --env "$RECIPE" --states "$states" \
      --teacher-url "$teacher_url" --teacher-model teacher --samples "$SAMPLES" --temperature "$TAKEOVER_TEMPERATURE" \
      --concurrency "$CONCURRENCY" --min-replay-ok "$MIN_REPLAY_OK" --out "$fold" || return 0
    echo "R2S_TAKEOVER_PASS fold=$fold"
  fi
  if [ "$STOP_AFTER_FOLD" = 1 ]; then echo "R2S_PIPELINE_PASS fold_only"; return 0; fi
  if stage_on J; then
    if [ "$judge_model" = judge ]; then
      template=$out/judge_template.jinja
      parser=$(serve_template "$judge_hf" "$template")
      reasoning=$(python3 -m miles_agent.launcher reasoning-parser "$judge_hf")
      start_sglang judge "$judge_hf" "$R2S_JUDGE_PORT" "$rest_gpus" "$rest_dp" \
        --mem-fraction-static 0.8 --chunked-prefill-size 4096 --context-length "$R2S_CONTEXT_LEN" \
        --chat-template "$template" --tool-call-parser "$parser" --reasoning-parser "$reasoning" \
        --served-model-name judge --log-level warning
      wait_server judge "$judge_url/health_generate" || return 0
    fi
    run_stage J JUDGE_DONE "$judge" miles_agent.r2s.judge --env "$RECIPE" --fold "$fold" \
      --judge-url "$judge_url" --judge-model "$judge_model" --out "$judge" || return 0
    [ "$judge_model" != judge ] || { stop_sglang judge; echo "SGLANG_STOP judge"; }
    echo "R2S_JUDGE_PASS judge=$judge"
  fi
  if stage_on C; then
    run_stage C SELECT_DONE "$tau" miles_agent.r2s.select --env "$RECIPE" --fold "$fold" \
      ${judge_file:+--judge-file "$judge_file"} --out "$tau" || return 0
    echo "R2S_SELECT_PASS tau=$tau"
  fi
  if stage_on D; then
    run_stage D LABEL_DONE "$labels" miles_agent.r2s.label --env "$RECIPE" --states "$states" --fold "$fold" \
      --tau "$tau" --tokenizer-path "$POLICY_HF" --teacher-url "$teacher_url" --k "$K" --out "$labels" || return 0
    echo "R2S_LABEL_PASS labels=$labels"
  fi
  echo "R2S_PIPELINE_PASS"
}
