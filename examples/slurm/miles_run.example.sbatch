#!/bin/bash
# examples/slurm/miles_run.example.sbatch — the smallest Slurm wrapper of scripts/run_local.sh (docs/cluster.md): the
# four portable headers, Contract A exported, one task per node that execs the entry, the USR1 relay of Contract C
# and the two job markers of Contract D. A site's account / partition / QOS / gres / log lines stay comments
# (`##SBATCH`): pass them as sbatch flags, which beat headers. Submit from the checkout:
#
#   sbatch --export=ALL,RUN_TAG=alf1,RECIPE=alfworld [-A <account> -p <partition> --gres=gpu:4] \
#       examples/slurm/miles_run.example.sbatch
# a wrapped --export= list continues at column 0: bash removes the newline, so keep the list on one line
#
# Not here, on purpose (site policy, added by a site's own wrapper around the same entry): requeue after a
# preemption, an idle watchdog, a stale-file sweep, secrets, container flags (one commented srun line shows the
# container form). Resume after JOB_STOPPED_TO_RESUME: submit again with the same RUN_TAG.
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J miles_run
#SBATCH --signal=B:USR1@600
##SBATCH -A <account>
##SBATCH -p <partition>
##SBATCH --qos=<qos>
##SBATCH --gres=gpu:<gpus per node>
##SBATCH -o <log dir>/run_%j.log
set -euo pipefail

# ---- the run: RUN_TAG RECIPE TOPOLOGY from the environment (sbatch --export) ----
[ -n "${RUN_TAG:-}" ] || { echo "FATAL: RUN_TAG is unset (sbatch --export=ALL,RUN_TAG=<name>,RECIPE=<recipe>)"; exit 1; }
[ -n "${RECIPE:-}" ] || { echo "FATAL: RECIPE is unset (alfworld | scienceworld | webshop | search_r1)"; exit 1; }
TOPOLOGY=${TOPOLOGY:-colocated_1n}
export RUN_TAG RECIPE TOPOLOGY

# ---- the checkout: Slurm copies this file to its spool, so the submit dir, not this file's dir, is the repo ----
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "$REPO/scripts/run_local.sh" ] || { echo "FATAL: no $REPO/scripts/run_local.sh (submit from the checkout, or export REPO=<checkout>)"; exit 1; }
export REPO

# ---- Contract A (docs/cluster.md, section A; tests/contract_vars.txt) ----
export CONTRACT_VERSION=1
export JOB_ID=$SLURM_JOB_ID JOB_RESTART=${SLURM_RESTART_COUNT:-0} JOB_NAME=$SLURM_JOB_NAME
export NNODES=$SLURM_JOB_NUM_NODES
# `|| true`: under errexit + pipefail a failing scontrol / getent inside the $( ) would end the job before the FATAL line
HEAD_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1 || true)
HEAD_IP=$(getent hosts "$HEAD_HOST" | awk '{print $1; exit}' || true)
[ -n "$HEAD_IP" ] || { echo "FATAL: could not resolve the head node '$HEAD_HOST' (scontrol show hostnames, getent hosts)"; exit 1; }
export HEAD_IP
# NODE_RANK and NODE_IP are per task (below); GPUS_PER_NODE only when the scheduler says: unset, the entry measures
[ -z "${SLURM_GPUS_ON_NODE:-}" ] || export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
PROJECT_DIR=${PROJECT_DIR:-$HOME/miles-agent-runs}
export PROJECT_DIR
export MODELS_DIR=$PROJECT_DIR/models DATA_DIR=$PROJECT_DIR/data RESULTS_DIR=$PROJECT_DIR/results RUNS_DIR=$PROJECT_DIR/runs
export TOOLS_DIR=$PROJECT_DIR/tools IMAGES_DIR=$PROJECT_DIR/images STAGE_TARS_DIR=$PROJECT_DIR/stage_tars
export SYNC_DIR=$PROJECT_DIR/sync/${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}    # per job and restart: never a stale sentinel
export PREEMPT_SENTINEL=$SYNC_DIR/preempt
mkdir -p "$SYNC_DIR" "$MODELS_DIR" "$DATA_DIR" "$RESULTS_DIR" "$RUNS_DIR"
echo "CONTRACT_A version=1 job=$JOB_ID restart=$JOB_RESTART nodes=$NNODES head=$HEAD_IP project_dir=$PROJECT_DIR"

# ---- Contract C: USR1 (a preemption, or 600 s before the time limit) raises the sentinel; the driver finishes its
# step, saves, writes <sentinel>.done, and the entry ends with RUN_INNER_PREEMPT_STOP ----
trap 'echo "PREEMPT_SIGNAL job=$SLURM_JOB_ID"; touch "$PREEMPT_SENTINEL"' USR1

# ---- the step: one task per node; rank 0 starts the services and the driver, the others join the compute cluster ----
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c 'export NODE_RANK=$SLURM_PROCID; exec bash "$REPO/scripts/run_local.sh"' &
# the same step inside the miles image under a container plugin (the mounts the entry expects; docs/containers.md):
#   srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
#     --container-image=<the image 3rdparty/miles pins, e.g. radixark/miles:v0.1.2> \
#     --container-mounts="$REPO:$REPO,$REPO/3rdparty/miles:/root/miles,$PROJECT_DIR:$PROJECT_DIR" \
#     bash -c 'export NODE_RANK=$SLURM_PROCID; exec bash "$REPO/scripts/run_local.sh"' &
STEP_PID=$!
# `wait` returns early when the trap runs: wait again until the step is really gone
set +e
while :; do
  wait "$STEP_PID"
  STEP_RC=$?
  kill -0 "$STEP_PID" 2>/dev/null || break
done
set -e

# ---- Contract D: the job markers (printed only here, never by an entry) ----
if [ -e "$PREEMPT_SENTINEL" ]; then
  echo "JOB_STOPPED_TO_RESUME_miles_run_$RUN_TAG durable=$(cat "$PREEMPT_SENTINEL.done" 2>/dev/null || echo none)"
  exit 75
fi
[ "$STEP_RC" -eq 0 ] || { echo "FATAL: the main step failed (rc=$STEP_RC)"; exit "$STEP_RC"; }
echo "JOB_PASS_miles_run_$RUN_TAG"
