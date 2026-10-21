#!/bin/bash
# scripts/run_local.sh — the entry of a training run: one machine, or the head and the other nodes of a multi-node
# run, on a plain box, in a container, or under a job wrapper that exports Contract A (docs/cluster.md). The body is
# the generic half of the old job wrapper; the wrapper half (hardware env, a stale-port sweep,
# a site's bridges and secrets) stays with the wrapper, which ends in `exec bash scripts/run_local.sh`.
#
#   scripts/run_local.sh [--dry-run[=driver]] [--recipe R] [--run-tag T] [--topology X] [--mode M] [--num-nodes N]
#                        [--] [driver args...]
#     --recipe R        alfworld | scienceworld | webshop | search_r1 | tmax (env RECIPE); required
#     --run-tag T       the run's name (env RUN_TAG): checkpoints under $MILES_AGENT_RESULTS/T/, its data under
#                       $MILES_AGENT_DATA/runs/T/, the stop sentinel $MILES_AGENT_RESULTS/T/stop; required
#     --topology X      colocated_1n (default) | async_3n | fully_async_3n | async_4n_teacher | async_2n_teacher |
#                       fully_async (tmax: TRAIN_NODES + ROLLOUT_NODES) (env TOPOLOGY); NNODES must match it
#     --mode M          normal (default) | debug_rollout_only | r2s_pipeline (env MODE)
#     --num-nodes N     the node count of the run (env NNODES, default 1); every node runs this script with its own
#                       NODE_RANK (0 = the head), the same HEAD_IP and RUN_TAG, and a SYNC_DIR they all see
#     -- args           appended to EXTRA_ARGS: the recipe driver's own flags (word-split, as EXTRA_ARGS is)
#     --dry-run         print every command the run would start (`DRY_RUN prepare:`, `probe:`, `ray_head:`, `gym:`,
#                       `env:`, `driver:` lines) and exit 0: no process started, no directory or file created
#     --dry-run=driver  bring the services up for real (data prep, the venv probe, the pre-ray services, the ray
#                       head, the pre-gym services, the gym: GYM_UP, the teacher gate), print `DRY_RUN driver:` with
#                       the driver's argv instead of starting it, and exit 0 through the cleanup (gym killed, ray
#                       stopped, $SYNC_DIR/ENDED raised for the workers): no RUN_INNER_PASS
#   Every flag has its environment name (the flag wins); local.env (scripts/lib/common.sh) sits below the environment.
#
# Knobs (all optional; scripts/lib/common.sh and scripts/lib/gym_ladder.sh document the rest)
#   NODE_RANK NNODES HEAD_IP NODE_IP  the node's place in the run (default 0 / 1 / this node / getent, hostname -I)
#   GPUS_PER_NODE                     else the CUDA_VISIBLE_DEVICES count, else `nvidia-smi -L`; 0 is a FATAL (not
#                                     under --dry-run)
#   POLICY_MODEL POLICY_HF            the policy (Qwen3.5-4B under $MILES_AGENT_MODELS)
#   TEACHER_MODEL / TEACHER_HF / TEACHER_URL   the OPD teacher of a *_teacher topology: the LAST node serves it
#                                     (scripts/teacher_node.sh) unless TEACHER_URL names a serving one (one node fewer)
#   NEMO_GYM_URL                      preset = external agent server(s), a comma list; this run starts no gym (tmax: required)
#   RETRIEVER_URL                     search_r1: the external retriever (a job wrapper's bridge sets it)
#   GYM_VENV_ROOT                     prebuilt gym venvs (scripts/build_gym_venvs.sh); unset = `uv run` from 3rdparty/Gym
#   RUN_SCRIPT                        the driver under recipes/<recipe>/ (default run.py; eval_full.py, opd.py)
#   EXTRA_ARGS                        more driver flags; MODE=debug_rollout_only adds --mode debug_rollout_only
#   WANDB_ENABLE                      0 keeps WANDB_API_KEY out of the driver's environment (default 1)
#   SYNC_DIR PREEMPT_SENTINEL         a job wrapper's; defaults under $MILES_AGENT_RESULTS/<tag>/ (common.sh run_dirs)
#   MILES_ROOT                        where `import miles` must resolve from (the pin check)
#   CONTRACT_VERSION                  set by a job wrapper (1); unset = local mode
# Steps on the head (rank 0): the recipe ladder -> the miles pin check -> the fixed-port pre-check (a holder is
# named, never killed) -> data prep -> the venv probe -> pre-ray services -> the external ray head (a running one is
# reused) -> the GPU wait (multi-node) -> pre-gym services -> the gym (GYM_UP) -> [MODE=r2s_pipeline:
# scripts/r2s_pipeline.sh, then RUN_INNER_PASS] -> the teacher gate -> the driver (--num-nodes <ray nodes>
# --num-gpus-per-node <gpus> ... --preempt-sentinel) -> RUN_INNER_PASS, or RUN_INNER_PREEMPT_STOP rc= durable= when
# the sentinel was raised meanwhile (raised before the driver: "PREEMPT before launch", exit 0). The other ranks are
# ray workers (RAY_WORKER_JOINED / RAY_WORKER_DONE, idle until $SYNC_DIR/ENDED) or the teacher node. TERM / USR1
# raise the sentinel instead of killing the run. To stop a run: touch its sentinel (the driver saves first); remove
# it, then rerun the same tag to resume from that save.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"
. "$LIB/ray.sh"

# ---- argv: every flag has its environment name; the flag wins ----
USAGE_LINE="scripts/run_local.sh [--dry-run[=driver]] --recipe <alfworld|scienceworld|webshop|search_r1|tmax> --run-tag <tag> [--topology <t>] [--mode <m>] [--num-nodes <n>] [-- <driver args>]"
need_value() { [ $# -ge 2 ] || fatal "$1 needs a value: $USAGE_LINE"; }
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run|--dry-run=1|--dry-run=argv) DRY_RUN=1 ;;
    --dry-run=driver) DRY_RUN=driver ;;
    --dry-run=*) fatal "unknown dry-run depth '${1#--dry-run=}' (--dry-run | --dry-run=driver)" ;;
    --recipe) need_value "$@"; RECIPE=$2; shift ;;
    --recipe=*) RECIPE=${1#*=} ;;
    --run-tag) need_value "$@"; RUN_TAG=$2; shift ;;
    --run-tag=*) RUN_TAG=${1#*=} ;;
    --topology) need_value "$@"; TOPOLOGY=$2; shift ;;
    --topology=*) TOPOLOGY=${1#*=} ;;
    --mode) need_value "$@"; MODE=$2; shift ;;
    --mode=*) MODE=${1#*=} ;;
    --num-nodes) need_value "$@"; NNODES=$2; shift ;;
    --num-nodes=*) NNODES=${1#*=} ;;
    --) shift; EXTRA_ARGS="${EXTRA_ARGS:-}${EXTRA_ARGS:+ }$*"; break ;;
    *) fatal "unknown argument '$1': $USAGE_LINE" ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN
contract_check
[ -n "${RECIPE:-}" ] || fatal "--recipe (or RECIPE=) is required: $USAGE_LINE"
[ -n "${RUN_TAG:-}" ] || fatal "--run-tag (or RUN_TAG=) is required: $USAGE_LINE"
# the entry's own defaults (a job wrapper exports its values; a plain machine none of them)
TOPOLOGY=${TOPOLOGY:-colocated_1n}
MODE=${MODE:-normal}
EXTRA_ARGS=${EXTRA_ARGS:-}
WANDB_ENABLE=${WANDB_ENABLE:-1}
case "$MODE" in
  normal|debug_rollout_only|r2s_pipeline) ;;
  *) fatal "unknown MODE=$MODE (normal | debug_rollout_only | r2s_pipeline)" ;;
esac
run_dirs "$RUN_TAG"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
case "$NNODES$NODE_RANK" in *[!0-9]*) fatal "NNODES=$NNODES and NODE_RANK=$NODE_RANK must be numbers" ;; esac
if [ -z "${NODE_IP:-}" ]; then
  if ! NODE_IP=$(node_ip 2>/dev/null); then
    is_dry_run || fatal "could not determine node IP (getent hosts, hostname -I and NODE_IP are all empty)"
    NODE_IP=127.0.0.1
    echo "WARN: node IP unknown: 127.0.0.1 stands in under --dry-run"
  fi
fi
HEAD_IP=${HEAD_IP:-$NODE_IP}
GPUS_PER_NODE=$(measure_gpus)
case "$GPUS_PER_NODE" in ''|*[!0-9]*) fatal "GPUS_PER_NODE=$GPUS_PER_NODE is not a number" ;; esac
[ "$GPUS_PER_NODE" -gt 0 ] || is_dry_run || fatal "no GPU: set GPUS_PER_NODE or CUDA_VISIBLE_DEVICES"
# TERM / USR1 raise the sentinel instead of killing the run (the driver finishes the step, saves and stops)
arm_preempt_trap
is_dry_run || mkdir -p "$SYNC_DIR" "$RUNS_DIR"

# ---- the topology's node count (validated here: a job wrapper only hands over NNODES) ----
case "$TOPOLOGY" in
  colocated_1n) NEED_NODES=1 ;;
  async_3n|fully_async_3n) NEED_NODES=3 ;;
  async_4n_teacher) NEED_NODES=4 ;;
  async_2n_teacher) NEED_NODES=2 ;;
  fully_async)
    # tmax: TRAIN_NODES training nodes + ROLLOUT_NODES rollout nodes (by default the rest of the run's nodes)
    TRAIN_NODES=${TRAIN_NODES:-1}
    ROLLOUT_NODES=${ROLLOUT_NODES:-$((NNODES - TRAIN_NODES))}
    export TRAIN_NODES ROLLOUT_NODES
    NEED_NODES=$((TRAIN_NODES + ROLLOUT_NODES)) ;;
  *) fatal "unknown TOPOLOGY=$TOPOLOGY (colocated_1n | async_3n | fully_async_3n | async_4n_teacher | async_2n_teacher | fully_async)" ;;
esac
case "$TOPOLOGY" in
  *_teacher)
    # the LAST node serves the teacher (scripts/teacher_node.sh); TEACHER_URL names an already serving one instead,
    # so no node is set aside and the topology takes one fewer
    [ -n "${TEACHER_MODEL:-}${TEACHER_HF:-}${TEACHER_URL:-}" ] \
      || fatal "$TOPOLOGY needs TEACHER_MODEL=<HF dir name under $MILES_AGENT_MODELS> (or TEACHER_HF=<dir>, or TEACHER_URL=<an already serving teacher>)"
    [ -z "${TEACHER_URL:-}" ] || NEED_NODES=$(( NEED_NODES - 1 )) ;;
esac
[ "$NNODES" -eq "$NEED_NODES" ] || fatal "TOPOLOGY=$TOPOLOGY needs NNODES=$NEED_NODES, got $NNODES"
export RECIPE RUN_TAG TOPOLOGY MODE NNODES NODE_RANK NODE_IP HEAD_IP GPUS_PER_NODE EXTRA_ARGS WANDB_ENABLE
echo "run_local: RECIPE=$RECIPE RUN_TAG=$RUN_TAG TOPOLOGY=$TOPOLOGY MODE=$MODE NNODES=$NNODES NODE_RANK=$NODE_RANK NODE_IP=$NODE_IP HEAD_IP=$HEAD_IP GPUS_PER_NODE=$GPUS_PER_NODE${DRY_RUN:+ DRY_RUN=$DRY_RUN}"

# ---- teacher node (the *_teacher topologies): the LAST node serves the OPD teacher, outside ray ----
# scripts/teacher_node.sh over scripts/lib/sglang.sh, handing the head its URL through $SYNC_DIR; TEACHER_URL names an
# already serving teacher instead, with any topology (no node is set aside then). The two helpers read MY_IP and
# SRC_MOUNT as they stand (a job wrapper exports both; a plain machine gets this node and this checkout).
export SRC_MOUNT=${SRC_MOUNT:-$REPO} MY_IP=${MY_IP:-$NODE_IP}
. "$LIB/sglang.sh"
. "$REPO/scripts/teacher_node.sh"
TEACHER_RANK=-1
case "$TOPOLOGY" in
  async_4n_teacher|async_2n_teacher) [ -n "${TEACHER_URL:-}" ] || TEACHER_RANK=$(( NNODES - 1 )) ;;
esac
# the ray nodes: the teacher node of a *_teacher topology never joins ray; under TEACHER_URL every node is ray's
RAY_NODES=$NNODES
if [ "$TEACHER_RANK" -ge 0 ]; then RAY_NODES=$(( NNODES - 1 )); fi
if is_dry_run && [ "$NODE_RANK" -eq "$TEACHER_RANK" ]; then dry_run_print teacher_node teacher_node_main; exit 0; fi
if [ "$NODE_RANK" -eq "$TEACHER_RANK" ]; then teacher_node_main; fi

# ---- ray worker branch: everything below this block is head-only ----
if [ "$NODE_RANK" -ne 0 ]; then
  if is_dry_run; then dry_run_print ray_worker ray_worker_join "$HEAD_IP" "$NODE_IP" "$GPUS_PER_NODE" "$NODE_RANK"; exit 0; fi
  ray_worker_join "$HEAD_IP" "$NODE_IP" "$GPUS_PER_NODE" "$NODE_RANK"
  exit 0
fi

# ---- head: the recipe ladder (scripts/lib/gym_ladder.sh: the ONE place that knows the per-recipe bits) ----
# The knob was STUDENT_MODEL / STUDENT_HF on the OPD campaign branches: refused, not silently ignored.
[ -z "${STUDENT_MODEL:-}${STUDENT_HF:-}" ] || { echo "FATAL: STUDENT_MODEL / STUDENT_HF are now POLICY_MODEL / POLICY_HF (one knob for every recipe)"; exit 1; }
# The OPD campaign branches' other knobs, each refused with its replacement.
. "$LIB/retired_knobs.sh"
. "$LIB/gym_ladder.sh"
recipe_vars "$RECIPE"

# ---- head: the miles pin (`import miles` must resolve under MILES_ROOT: the checkout this run was built against) ----
PIN_CHECK="import miles; p = list(miles.__path__)[0]; assert p.startswith('$MILES_ROOT'), p; print('MILES_PIN_ACTIVE path=' + p)"
if is_dry_run; then
  dry_run_print pin_check "$PYTHON" -c "$PIN_CHECK"
else
  "$PYTHON" -c "$PIN_CHECK" || fatal "miles is not imported from MILES_ROOT=$MILES_ROOT (another install first on sys.path? set MILES_ROOT or PYTHONPATH)"
fi

# ---- head: the fixed ports (a holder is named, never killed: a stale-process sweep is a job wrapper's business) ----
# the ports of our own services are skipped: a wrapper's retriever bridge, an external gym / engine / teacher
# shellcheck disable=SC2086
is_dry_run || port_precheck $PROJECT_PORTS -- ${RETRIEVER_URL:-} ${NEMO_GYM_URL:-} ${WEBSHOP_ENGINE_BASE_URL:-} ${TEACHER_URL:-}

# ---- head: data prep (idempotent) ----
"$RECIPE_PREPARE_DATA"

# ---- head: the prebuilt gym venvs <-> checkout compatibility gate ----
venv_probe_gate

# ---- head: pre-ray services (search_r1: the retriever must answer before anything starts) ----
"$RECIPE_PRE_RAY"

# ---- head: the external ray cluster (ALL topologies; scripts/lib/ray.sh says why external, and why before gym) ----
if is_dry_run; then
  dry_run_print ray_head "$RAY_BIN" start --head --node-ip-address="$NODE_IP" --port=6379 --num-gpus "$GPUS_PER_NODE" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --min-worker-port=2000 --max-worker-port=2999
elif ray_running; then
  # a head started by hand (scripts/ray_cluster.sh head) or left by an earlier run on this machine: reused
  echo "ray: a cluster already answers on this machine (ray status): reusing it, MASTER_ADDR=$NODE_IP"
  export MASTER_ADDR=$NODE_IP MILES_SCRIPT_EXTERNAL_RAY=1
else
  echo "ray: starting the head on $NODE_IP:6379 with $GPUS_PER_NODE GPUs"
  ray_head "$NODE_IP" "$GPUS_PER_NODE"
fi
if [ "$NNODES" -gt 1 ]; then
  # the workers (and the teacher node) idle until the head drops ENDED
  is_dry_run || add_cleanup 'touch "$SYNC_DIR/ENDED"'
  MILES_EXPECT_GPUS=$(( RAY_NODES * GPUS_PER_NODE ))
  if is_dry_run; then dry_run_print ray_wait_gpus "$MILES_EXPECT_GPUS" "$NODE_IP:6379"
  else ray_wait_gpus "$MILES_EXPECT_GPUS" "$NODE_IP:6379"; fi
fi

# ---- head: pre-gym services (webshop: the engine; search_r1: the data links; scienceworld: the jar + JDK) ----
"$RECIPE_PRE_GYM"

# ---- head: the gym servers ----
if [ -n "${NEMO_GYM_URL:-}" ]; then
  echo "gym: NEMO_GYM_URL=$NEMO_GYM_URL is preset (an external agent server): this run starts no gym"
else
  render_env_yaml
  gym_start_and_wait
fi

# ---- head: the R2S repair pipeline in place of a run (MODE=r2s_pipeline, scripts/r2s_pipeline.sh) ----
# The stages dial the gym just booted and the sglang servers the script starts itself (no driver, no teacher gate:
# the pipeline's teacher is its own server, alive through stage D). A sentinel raised meanwhile ends it between rows
# with exit 0 (RUN_INNER_PREEMPT_STOP); a job wrapper requeues, and the requeued job resumes per stage and row.
if [ "$MODE" = r2s_pipeline ]; then
  case "$(dry_run_depth)" in argv|driver) dry_run_print r2s_pipeline r2s_pipeline_main; exit 0 ;; esac
  . "$REPO/scripts/r2s_pipeline.sh"
  r2s_pipeline_main
  if [ -n "${PREEMPT_SENTINEL:-}" ] && [ -e "$PREEMPT_SENTINEL" ]; then
    echo "RUN_INNER_PREEMPT_STOP rc=0 durable=$(cat "$PREEMPT_SENTINEL.done" 2>/dev/null || echo none)"; exit 0
  fi
  echo "RUN_INNER_PASS"; exit 0
fi

# ---- head: OPD teacher gate (scripts/teacher_node.sh) ----
# The teacher node publishes its URL once /health_generate answers (TEACHER_URL: an already serving one); a dead
# or mismatched teacher would otherwise only show up as every episode ABORTED at scoring time. The gate hands the
# recipe --teacher-url (TEACHER_URL_ARG, empty without a teacher).
TEACHER_URL_ARG=""
if is_dry_run && { [ "$TEACHER_RANK" -ge 0 ] || [ -n "${TEACHER_URL:-}" ]; }; then
  dry_run_print teacher_gate teacher_gate
  TEACHER_URL_ARG=${TEACHER_URL:+--teacher-url $TEACHER_URL}
elif [ "$TEACHER_RANK" -ge 0 ] || [ -n "${TEACHER_URL:-}" ]; then teacher_gate; fi

# ---- head: launch ----
# a key inherited from the shell is not used when wandb is off (a job wrapper reads its key under WANDB_ENABLE too)
[ "${WANDB_ENABLE:-1}" = 1 ] || unset WANDB_API_KEY
# the session ports right before the driver: a holder is reported with the evidence (miles dies on its own check)
is_dry_run || port_warn_check 7100-7131
MODE_ARGS=""
if [ "$MODE" = "debug_rollout_only" ]; then MODE_ARGS="--mode debug_rollout_only"; fi
PREEMPT_ARGS=""
if [ -n "${PREEMPT_SENTINEL:-}" ]; then
  PREEMPT_ARGS="--preempt-sentinel $PREEMPT_SENTINEL"
  if [ -e "$PREEMPT_SENTINEL" ]; then echo "PREEMPT before launch: not starting the driver"; exit 0; fi
fi
cd "$REPO/recipes/$RECIPE" || fatal "no recipe dir $REPO/recipes/$RECIPE"
# RUN_SCRIPT=eval_full.py swaps in the eval-only launcher (same flag surface); search_r1 has one.
[ -f "${RUN_SCRIPT:-run.py}" ] || { echo "FATAL: recipes/$RECIPE has no ${RUN_SCRIPT:-run.py}"; exit 1; }
case "$(dry_run_depth)" in
  argv|driver)
    # the same words as the launch below (the tests compare the two): --num-nodes counts the ray nodes, the teacher
    # node of a *_teacher topology excluded; $MODE_ARGS $DRIVER_DATA_ARGS $RECIPE_RUN_ARGS $PREEMPT_ARGS
    # $TEACHER_URL_ARG $EXTRA_ARGS are word-split on purpose
    dry_run_print driver "$PYTHON" "${RUN_SCRIPT:-run.py}" --topology "$TOPOLOGY" $MODE_ARGS \
      --num-nodes "$RAY_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
      --base-dir "$TORCH_DIST_DIR" --model-name "$POLICY_MODEL" --hf-checkpoint "$POLICY_HF" \
      --ref-load "$TORCH_DIST_DIR/${POLICY_MODEL}_torch_dist" --save-dir "$MILES_AGENT_RESULTS/$RUN_TAG/" \
      $DRIVER_DATA_ARGS $RECIPE_RUN_ARGS $PREEMPT_ARGS $TEACHER_URL_ARG ${EXTRA_ARGS:-}
    exit 0 ;;
esac
# In the background: bash runs a trap only once its foreground child exits, and
# preempt_term has to run while the driver does.
"$PYTHON" "${RUN_SCRIPT:-run.py}" --topology "$TOPOLOGY" $MODE_ARGS \
  --num-nodes "$RAY_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --base-dir "$TORCH_DIST_DIR" \
  --model-name "$POLICY_MODEL" \
  --hf-checkpoint "$POLICY_HF" \
  --ref-load "$TORCH_DIST_DIR/${POLICY_MODEL}_torch_dist" \
  --save-dir "$MILES_AGENT_RESULTS/$RUN_TAG/" \
  $DRIVER_DATA_ARGS $RECIPE_RUN_ARGS \
  $PREEMPT_ARGS \
  $TEACHER_URL_ARG \
  ${EXTRA_ARGS:-} &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null; do sleep 5 & wait $! || true; done
RUN_RC=0
wait "$RUN_PID" || RUN_RC=$?
if [ -n "${PREEMPT_SENTINEL:-}" ] && [ -e "$PREEMPT_SENTINEL" ]; then
  echo "RUN_INNER_PREEMPT_STOP rc=$RUN_RC durable=$(cat "$PREEMPT_SENTINEL.done" 2>/dev/null || echo none)"
  exit "$RUN_RC"
fi
[ "$RUN_RC" -eq 0 ] || { echo "FATAL: the driver ${RUN_SCRIPT:-run.py} exited with rc=$RUN_RC"; exit "$RUN_RC"; }
echo "RUN_INNER_PASS"
