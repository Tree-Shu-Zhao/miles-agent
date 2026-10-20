#!/bin/bash
# R2S stage E on one GPU machine: full-parameter FT of the base teacher on stage D's labels, torchrun over the
# machine's GPUs (miles_agent.r2s.teacher_ft: FSDP, the campaign's optimizer and schedule, the held-out CE gate),
# promoted to its final name only when the gate passed. No scheduler call: a scheduler wrapper (docs/cluster.md)
# may run it inside the miles container on a cluster node, a plain machine runs it directly.
#   TAG         the output name under $MILES_AGENT_MODELS, by convention r2s-<env>-<teacher>-teacher-for-<student>
#               (e.g. r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b, the student arm's TEACHER_MODEL;
#               launcher.py MODEL_FAMILIES resolves its family by the longest key in the name)
#   LABELS      stage D's labels.jsonl (LABEL_DONE; objective "suffix": teacher_ft refuses any other label file)
#   BASE_MODEL  the base teacher under the same models dir (default Qwen3.5-9B)
#   LR EPOCHS GRAD_ACCUM  the campaign's FT knobs (2e-6, 1, 8); FT_EXTRA more teacher_ft flags, verbatim
#               (--skip-gate exports an unimproved teacher: the report then says passed false)
#   SRC_MOUNT   the repo root (default: this checkout); LOCAL (else the wrapper's NODE_LOCAL) node-local scratch for the JIT
#               caches (/tmp)
# Trains into <models>/<TAG>.tmp and moves it to <models>/<TAG> when torchrun exits 0 (R2S_FT_PASS; under
# --skip-gate the report says passed false and the marker R2S_FT_PASS gate=skipped; r2s_ft_report.json beside the
# weights, printed). A gate refusal (FT_FAIL: the worker exits 2, torchrun itself exits 1 whatever the worker's
# code) leaves <TAG>.tmp holding only the report and exits 2: nothing is promoted. An existing <TAG> is refused; an
# earlier <TAG>.tmp is set aside as <TAG>.failed-<job>, or <TAG>.failed-<job>.<n> with the first free n when that
# name is taken (the same job id again). Nothing is deleted here.
set -euxo pipefail
DRY_RUN=${DRY_RUN:-}
case "${1:-}" in --dry-run|--dry-run=1|--dry-run=argv) DRY_RUN=1 ;; "") ;; *) echo "FATAL: unknown argument $1 (the knobs are environment variables; the one flag is --dry-run)"; exit 1 ;; esac
: "${TAG:?TAG=<output name under the models dir: r2s-<env>-<teacher>-teacher-for-<student>>}"
: "${LABELS:?LABELS=<stage D labels.jsonl>}"
BASE_MODEL=${BASE_MODEL:-Qwen3.5-9B}
SRC=${SRC_MOUNT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -f "$SRC/scripts/lib/common.sh" ] || { echo "FATAL: $SRC is not the miles-agent repo (SRC_MOUNT)"; exit 1; }
set +x; . "$SRC/scripts/lib/common.sh"; set -x   # REPO, local.env, the roots (MILES_AGENT_MODELS <- MODELS_DIR <- $REPO/models), JOB_ID, dry_run_print
MODELS=$MILES_AGENT_MODELS
JOB=$JOB_ID
case "$TAG" in *[!A-Za-z0-9._-]*|.*|"") echo "FATAL: TAG=$TAG: letters, digits, . _ - only"; exit 1 ;; esac
BASE_HF=$MODELS/$BASE_MODEL
OUT_DIR=$MODELS/$TAG
REPORT=r2s_ft_report.json
# `|| true`: grep -c exits 1 on a zero count (its "0" is still printed)
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
if is_dry_run; then
  dry_run_print torchrun torchrun --standalone --nproc-per-node "$NGPU" -m miles_agent.r2s.teacher_ft \
    --data "$LABELS" --model-path "$BASE_HF" --out-dir "$OUT_DIR.tmp" \
    --lr "${LR:-2e-6}" --epochs "${EPOCHS:-1}" --grad-accum "${GRAD_ACCUM:-8}" ${FT_EXTRA:-}
  exit 0
fi
[ -f "$LABELS" ] || { echo "FATAL: no labels file $LABELS"; exit 1; }
[ -f "$BASE_HF/config.json" ] || { echo "FATAL: no HF model dir $BASE_HF (BASE_MODEL under $MODELS)"; exit 1; }
[ -f "$SRC/miles_agent/r2s/teacher_ft.py" ] || { echo "FATAL: $SRC is not the miles-agent repo (SRC_MOUNT)"; exit 1; }
[ ! -e "$OUT_DIR" ] || { echo "FATAL: $OUT_DIR exists; pick a new TAG"; exit 1; }
# an earlier run's .tmp (an FT_FAIL keeps its report there): set aside under the first free name, never deleted
# here (a .failed-<job> already there, the same job id again, would take the .tmp as a subdir)
if [ -e "$OUT_DIR.tmp" ]; then
  aside=$OUT_DIR.failed-$JOB n=0
  while [ -e "$aside" ]; do n=$(( n + 1 )); aside=$OUT_DIR.failed-$JOB.$n; done
  mv "$OUT_DIR.tmp" "$aside"
fi
[ "$NGPU" -ge 1 ] || { echo "FATAL: nvidia-smi lists no GPU on $(hostname)"; exit 1; }
SCRATCH=${LOCAL:-${NODE_LOCAL:-/tmp}}
export TMPDIR=/tmp                                   # /dev/shm may be mounted noexec
export TRITON_CACHE_DIR=$SCRATCH/jit/triton TORCHINDUCTOR_CACHE_DIR=$SCRATCH/jit/inductor   # JIT caches on local disk, never on a network filesystem
export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0        # NVLS needs cumem; keep both off
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SRC${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
echo "R2S_FT_START tag=$TAG labels=$LABELS base=$BASE_HF out=$OUT_DIR ngpu=$NGPU"
rc=0
torchrun --standalone --nproc-per-node "$NGPU" -m miles_agent.r2s.teacher_ft \
  --data "$LABELS" --model-path "$BASE_HF" --out-dir "$OUT_DIR.tmp" \
  --lr "${LR:-2e-6}" --epochs "${EPOCHS:-1}" --grad-accum "${GRAD_ACCUM:-8}" ${FT_EXTRA:-} || rc=$?
if [ "$rc" -ne 0 ]; then
  # the verdict is the report teacher_ft wrote before exiting, not torchrun's code
  if [ -f "$OUT_DIR.tmp/$REPORT" ] && grep -q '"passed": false' "$OUT_DIR.tmp/$REPORT"; then
    echo "R2S_FT_FAIL tag=$TAG: the gate refused the teacher (FT_FAIL above);" \
         "$OUT_DIR.tmp keeps $REPORT, nothing is promoted"
    exit 2
  fi
  echo "FATAL: torchrun exited $rc before the gate verdict; $OUT_DIR.tmp is left as it is"
  exit "$rc"
fi
mv "$OUT_DIR.tmp" "$OUT_DIR"
cat "$OUT_DIR/$REPORT"
if grep -q '"passed": false' "$OUT_DIR/$REPORT"; then
  echo "R2S_FT_PASS gate=skipped out=$OUT_DIR"  # --skip-gate (FT_EXTRA): an unimproved teacher, exported on purpose
else
  echo "R2S_FT_PASS out=$OUT_DIR"
fi
