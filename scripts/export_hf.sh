#!/bin/bash
# scripts/export_hf.sh — a saved torch_dist checkpoint of a run as an HF model dir (tools/convert_ckpt_to_hf.py): the
# trained tensors in the origin's dtypes, the origin's own for what the trained model does not hold (a vision tower,
# an MTP layer), the origin's config and tokenizer, checked against the origin (names, shapes, dtypes, how far each
# tensor moved). CPU only; runs where miles imports (the image, or a machine with MILES_ROOT's dependencies).
#
#   scripts/export_hf.sh [--dry-run] <run tag | results dir> <iter | latest> <out dir>
#                        [--origin-hf-dir D] [--model-args A] [--model-name N]
#     <run tag | results dir>   $MILES_AGENT_RESULTS/<tag>, or a results dir (anything with a / in it)
#     <iter | latest>           the save id (iter_<7 digits> under it), or the one latest_checkpointed_iteration.txt names
#     <out dir>                 the HF dir written; <out dir>/.complete marks a finished export (already there: nothing
#                               is redone, the script says so and exits 0)
#   scripts/export_hf.sh [--dry-run]          the hook form, no positionals: ITER_DIR and OUT_DIR from the environment,
#                                             HF_INPUT (set: an HF export the run wrote itself — miles --save-hf, a
#                                             complete one with its .complete marker — is finished instead of
#                                             converting ITER_DIR) and HF_ORIGIN (set: the origin) — the environment a
#                                             job wrapper exports (docs/cluster.md, PROJECT_CONVERT_CMD)
#   Gates (both forms): <iter dir>/.metadata exists (a torch_dist save) and the run's latest_checkpointed_iteration.txt
#   names this save id or a later one — the tracker is Megatron's completion marker, an iter_ dir appears before its
#   save is complete (the partial-checkpoint rule); the positional form needs the tracker, the hook form only when the
#   run dir has one. The origin needs a config.json.
#   Knobs: HF_MODEL_ARGS (miles' scripts/models/<name>.py, default qwen3.5-4B; --model-args wins), HF_MODEL_NAME (the
#   converter's model family, default qwen3_5; --model-name wins), the origin default
#   ${POLICY_HF:-$MILES_AGENT_MODELS/${POLICY_MODEL:-Qwen3.5-4B}} (--origin-hf-dir / HF_ORIGIN win), MILES_ROOT (the
#   miles checkout: its converter tool, its model scripts, the converter's PYTHONPATH), PYTHON, local.env.
#   --dry-run prints `DRY_RUN convert: <argv>` and exits 0 (a missing input is a WARN there). Ends with
#   EXPORT_HF_OK out=<dir> iter=<n|hf> (plus already=1 when the .complete marker was there before).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/export_hf.sh [--dry-run] [<run tag | results dir> <iter | latest> <out dir>] [--origin-hf-dir D] [--model-args A] [--model-name N]"
need_value() { [ $# -ge 2 ] || fatal "$1 needs a value: $USAGE_LINE"; }
POS1="" POS2="" POS3="" NPOS=0
ORIGIN_FLAG="" MODEL_ARGS_FLAG="" MODEL_NAME_FLAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --origin-hf-dir) need_value "$@"; ORIGIN_FLAG=$2; shift ;;
    --origin-hf-dir=*) ORIGIN_FLAG=${1#*=} ;;
    --model-args) need_value "$@"; MODEL_ARGS_FLAG=$2; shift ;;
    --model-args=*) MODEL_ARGS_FLAG=${1#*=} ;;
    --model-name) need_value "$@"; MODEL_NAME_FLAG=$2; shift ;;
    --model-name=*) MODEL_NAME_FLAG=${1#*=} ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    *) NPOS=$((NPOS + 1))
       case "$NPOS" in 1) POS1=$1 ;; 2) POS2=$1 ;; 3) POS3=$1 ;; *) fatal "too many arguments at '$1': $USAGE_LINE" ;; esac ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

# a gate under --dry-run is a WARN (the argv is still printed); otherwise a FATAL
gate() { if is_dry_run; then echo "WARN: $*"; else fatal "$*"; fi; }
tracker_value() {  # <tracker file>: prints its content; returns 1 when that is no save id (Megatron's `release`) — the
  # caller gates (a FATAL inside the $( ) that captures the value would be swallowed with the output)
  local v
  v=$(tr -d '[:space:]' < "$1")
  printf '%s\n' "$v"
  case "$v" in ''|*[!0-9]*) return 1 ;; esac
}
tracker_gate() {  # <tracker file> <its content>: the gate for a tracker that names no save id
  gate "$1 names no save id ('$2'; Megatron's 'release' marks a converted base)"
}
iter_label() {  # <iter dir>: prints N of its iter_<N> base name; returns 1 for a non-iter dir or a malformed suffix
  local base suffix
  base=$(basename "$1")
  case "$base" in iter_*) suffix=${base#iter_} ;; *) return 1 ;; esac
  case "$suffix" in ''|*[!0-9]*) return 1 ;; esac
  printf '%s\n' "$((10#$suffix))"
}
# <iter dir> <tracker required 1|0>: a torch_dist save whose run tracker has reached it (the partial-checkpoint rule)
check_iter_dir() {
  local dir=$1 need=$2 n tracker saved
  n=$(iter_label "$dir") || { gate "$dir is not an iter_<N> dir"; return 0; }
  [ -d "$dir" ] || { gate "no checkpoint dir $dir"; return 0; }
  [ -e "$dir/.metadata" ] || gate "$dir has no .metadata: not a torch_dist save, or one still being written"
  tracker=$(dirname "$dir")/latest_checkpointed_iteration.txt
  if [ -f "$tracker" ]; then
    if saved=$(tracker_value "$tracker"); then
      [ "$saved" -ge "$n" ] \
        || gate "$tracker names save $saved, before iter $n: that save is not complete yet (the partial-checkpoint rule)"
    else
      tracker_gate "$tracker" "$saved"
    fi
  elif [ "$need" = 1 ]; then
    gate "no $tracker: the run has no completed save"
  fi
}

# ---- the two forms ----
HF_INPUT=${HF_INPUT:-}
if [ "$NPOS" -eq 0 ]; then
  # the hook form: the wrapper resolved the iter dir and the output; HF_INPUT / HF_ORIGIN as it exported them
  [ -n "${ITER_DIR:-}" ] && [ -n "${OUT_DIR:-}" ] \
    || fatal "ITER_DIR and OUT_DIR must both be set in the environment (the hook form), or give <run tag | results dir> <iter | latest> <out dir>: $USAGE_LINE"
  ITER_DIR=${ITER_DIR%/}
  ORIGIN=${ORIGIN_FLAG:-${HF_ORIGIN:-}}
  if [ -n "$HF_INPUT" ]; then
    LABEL=hf
  else
    # the gate first: a non-iter dir or a malformed suffix is its FATAL (a WARN under --dry-run, where the label
    # falls back to the dir's name so that the argv still prints)
    check_iter_dir "$ITER_DIR" 0
    LABEL=$(iter_label "$ITER_DIR") || LABEL=$(basename "$ITER_DIR")
  fi
elif [ "$NPOS" -eq 3 ]; then
  case "$POS1" in */*) RUN_DIR=${POS1%/} ;; *) RUN_DIR=$MILES_AGENT_RESULTS/$POS1 ;; esac
  TRACKER=$RUN_DIR/latest_checkpointed_iteration.txt
  if [ "$POS2" = latest ]; then
    if [ ! -f "$TRACKER" ]; then
      gate "no $TRACKER: the run has no completed save (or $RUN_DIR is not a results dir)"; LABEL=latest
    elif LABEL=$(tracker_value "$TRACKER"); then :
    else tracker_gate "$TRACKER" "$LABEL"; LABEL=latest; fi
  else
    case "$POS2" in ''|*[!0-9]*) fatal "<iter> must be a save id or 'latest', got '$POS2'" ;; esac
    LABEL=$((10#$POS2))
  fi
  case "$LABEL" in latest) ITER_DIR="$RUN_DIR/iter_<latest>" ;; *) ITER_DIR=$RUN_DIR/$(printf 'iter_%07d' "$LABEL") ;; esac
  OUT_DIR=$POS3
  HF_INPUT=""
  ORIGIN=$ORIGIN_FLAG
  [ "$LABEL" = latest ] || check_iter_dir "$ITER_DIR" 1
else
  fatal "three positionals (<run tag | results dir> <iter | latest> <out dir>) or none, got $NPOS: $USAGE_LINE"
fi
OUT_DIR=${OUT_DIR%/}

# ---- the converter's argv (today's convert job's, the in-image literals replaced by MILES_ROOT and the roots) ----
CONVERTER=$REPO/tools/convert_ckpt_to_hf.py
ORIGIN=${ORIGIN:-${POLICY_HF:-$MILES_AGENT_MODELS/${POLICY_MODEL:-Qwen3.5-4B}}}
MODEL_ARGS=${MODEL_ARGS_FLAG:-${HF_MODEL_ARGS:-qwen3.5-4B}}
MODEL_NAME=${MODEL_NAME_FLAG:-${HF_MODEL_NAME:-qwen3_5}}
if [ -n "$HF_INPUT" ]; then
  HF_INPUT=${HF_INPUT%/}
  [ -e "$HF_INPUT/.complete" ] || gate "HF_INPUT=$HF_INPUT has no .complete marker: not a finished HF export of the run"
  INPUT=(--input-hf-dir "$HF_INPUT")
else
  MODEL_ARGS_SCRIPT=$MILES_ROOT/scripts/models/$MODEL_ARGS.py
  [ -f "$MODEL_ARGS_SCRIPT" ] || gate "no model-args script $MODEL_ARGS_SCRIPT (HF_MODEL_ARGS / --model-args names scripts/models/<name>.py of miles)"
  INPUT=(--input-dir "$ITER_DIR" --model-args-script "$MODEL_ARGS_SCRIPT" --model-name "$MODEL_NAME")
fi
[ -f "$ORIGIN/config.json" ] || gate "no origin HF checkpoint at $ORIGIN (config.json missing): --origin-hf-dir / HF_ORIGIN / POLICY_HF"
[ -f "$MILES_ROOT/tools/convert_torch_dist_to_hf.py" ] || gate "MILES_ROOT=$MILES_ROOT has no tools/convert_torch_dist_to_hf.py (the miles checkout the converter builds on)"
ARGV=(env CUDA_VISIBLE_DEVICES= "PYTHONPATH=$MILES_ROOT" "$PYTHON" "$CONVERTER" --miles-root "$MILES_ROOT" "${INPUT[@]}"
      --output-dir "$OUT_DIR" --origin-hf-dir "$ORIGIN" --fill-from-origin --origin-dtypes --check-origin)
if is_dry_run; then
  dry_run_print convert "${ARGV[@]}"
  exit 0
fi
if [ -f "$OUT_DIR/.complete" ]; then
  echo "HF_EXPORT_READY $OUT_DIR (already)"
  echo "EXPORT_HF_OK out=$OUT_DIR iter=$LABEL already=1"
  exit 0
fi
[ ! -e "$OUT_DIR" ] || echo "WARN: $OUT_DIR exists without its .complete marker (a failed or running export): its files are rewritten"
mkdir -p "$OUT_DIR"
echo "export_hf: $ITER_DIR${HF_INPUT:+ (finishing $HF_INPUT)} -> $OUT_DIR (origin $ORIGIN)"
rc=0
"${ARGV[@]}" || rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: the converter exited with rc=$rc (no .complete written to $OUT_DIR)"; exit "$rc"; }
[ -e "$OUT_DIR/.complete" ] || touch "$OUT_DIR/.complete"
echo "HF_EXPORT_READY $OUT_DIR"
echo "EXPORT_HF_OK out=$OUT_DIR iter=$LABEL"
