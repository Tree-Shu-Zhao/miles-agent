#!/bin/bash
# scripts/eval_ckpt.sh — one eval-only pass over a saved checkpoint of a run, or over the policy's HF weights, under
# the recipe's own evaluation protocol (miles_agent/eval_ckpt.py through recipes/<recipe>/eval_ckpt.py): a run of
# scripts/run_local.sh with RUN_SCRIPT=eval_ckpt.py on one node (TOPOLOGY=colocated_1n, the only topology that has
# train.py's eval-only branch). The decode gets a results dir of its own under $MILES_AGENT_RESULTS, a sibling of the
# arm's, which is where eval_ckpt.py looks the arm up (<results>/<arm tag>/iter_<N>, pinned through a load view).
#
#   scripts/eval_ckpt.sh [--dry-run] <recipe> --eval-load-tag <arm tag> [--eval-load-iter N]
#                        [--recipe-script <file>] [--seed S] [--run-tag T] [-- <driver args>]
#   scripts/eval_ckpt.sh [--dry-run] <recipe> --eval-base [--recipe-script <file>] [--seed S] [--run-tag T] [-- <driver args>]
#     <recipe>                 alfworld | scienceworld | webshop (the recipes with an eval_ckpt.py)
#     --eval-load-tag <tag>    the arm whose checkpoint is decoded ($MILES_AGENT_RESULTS/<tag>/)
#     --eval-load-iter N       the arm's save id (default: the one its latest_checkpointed_iteration.txt names now;
#                              save ids are 0-based: a 150-step arm ends at iter_0000149)
#     --eval-base              the HF weights of the policy (POLICY_MODEL / POLICY_HF) instead, rollout-only
#     --recipe-script <file>   the recipe script whose flag surface applies: run.py (default) or opd.py (the OPD arms)
#     --seed S                 the decode's seed (the driver's --seed)
#     --run-tag T              this decode's own run tag (default eval_<arm tag | base>, plus _s<seed> with --seed)
#     -- <driver args>         more driver flags, e.g. the --num-rollout an arm was trained with (the save-id bound)
#     --dry-run                the entry's argv dry run: `DRY_RUN driver:` with the driver argv, nothing started
#   Exactly one of --eval-load-tag / --eval-base. Every other knob is scripts/run_local.sh's (POLICY_MODEL, POLICY_HF,
#   GYM_VENV_ROOT, NEMO_GYM_URL, the roots, local.env); the pass ends with RUN_INNER_PASS, printed by run_local.sh.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/eval_ckpt.sh [--dry-run] <recipe> (--eval-load-tag <tag> [--eval-load-iter N] | --eval-base) [--recipe-script <file>] [--seed S] [--run-tag T] [-- <driver args>]"
need_value() { [ $# -ge 2 ] || fatal "$1 needs a value: $USAGE_LINE"; }
recipe=""
LOAD_TAG=""
LOAD_ITER=""
BASE=0
RECIPE_SCRIPT=""
SEED=""
TAG=""
MORE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --eval-load-tag) need_value "$@"; LOAD_TAG=$2; shift ;;
    --eval-load-tag=*) LOAD_TAG=${1#*=} ;;
    --eval-load-iter) need_value "$@"; LOAD_ITER=$2; shift ;;
    --eval-load-iter=*) LOAD_ITER=${1#*=} ;;
    --eval-base) BASE=1 ;;
    --recipe-script) need_value "$@"; RECIPE_SCRIPT=$2; shift ;;
    --recipe-script=*) RECIPE_SCRIPT=${1#*=} ;;
    --seed) need_value "$@"; SEED=$2; shift ;;
    --seed=*) SEED=${1#*=} ;;
    --run-tag) need_value "$@"; TAG=$2; shift ;;
    --run-tag=*) TAG=${1#*=} ;;
    --) shift; MORE=$*; break ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    *) [ -z "$recipe" ] || fatal "one recipe at most, got '$recipe' and '$1': $USAGE_LINE"
       recipe=$1 ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN
[ -n "$recipe" ] || fatal "<recipe> is required: $USAGE_LINE"
[ -f "$REPO/recipes/$recipe/eval_ckpt.py" ] || fatal "recipes/$recipe has no eval_ckpt.py (alfworld | scienceworld | webshop have one)"
if [ -n "$LOAD_TAG" ] && [ "$BASE" = 1 ]; then
  fatal "--eval-load-tag decodes a saved checkpoint, --eval-base the HF weights: give one of them, not both"
fi
if [ -z "$LOAD_TAG" ] && [ "$BASE" = 0 ]; then
  fatal "one of --eval-load-tag <arm tag> / --eval-base is required: $USAGE_LINE"
fi
[ -z "$LOAD_ITER" ] || [ "$BASE" = 0 ] || fatal "--eval-load-iter goes with --eval-load-tag, not with --eval-base"
case "$LOAD_ITER" in ''|*[!0-9]*) [ -z "$LOAD_ITER" ] || fatal "--eval-load-iter must be a save id, got '$LOAD_ITER'" ;; esac
case "$SEED" in ''|*[!0-9]*) [ -z "$SEED" ] || fatal "--seed must be a number, got '$SEED'" ;; esac

# ---- the decode's own tag and the eval flags, in front of the caller's EXTRA_ARGS ----
if [ -n "$TAG" ]; then RUN_TAG=$TAG
else RUN_TAG=eval_${LOAD_TAG:-base}${SEED:+_s$SEED}; fi
if [ "$BASE" = 1 ]; then EVAL_ARGS="--eval-base"
else EVAL_ARGS="--eval-load-tag $LOAD_TAG${LOAD_ITER:+ --eval-load-iter $LOAD_ITER}"; fi
EVAL_ARGS="$EVAL_ARGS${RECIPE_SCRIPT:+ --recipe-script $RECIPE_SCRIPT}${SEED:+ --seed $SEED}${MORE:+ $MORE}${EXTRA_ARGS:+ $EXTRA_ARGS}"
echo "eval_ckpt: RECIPE=$recipe RUN_TAG=$RUN_TAG RUN_SCRIPT=eval_ckpt.py TOPOLOGY=colocated_1n EXTRA_ARGS=[$EVAL_ARGS]"
DRY_FLAG=""
case "${DRY_RUN:-}" in 1) DRY_FLAG=--dry-run ;; driver) DRY_FLAG=--dry-run=driver ;; esac
export RECIPE=$recipe RUN_TAG RUN_SCRIPT=eval_ckpt.py TOPOLOGY=colocated_1n MODE=normal EXTRA_ARGS=$EVAL_ARGS
# shellcheck disable=SC2086
exec bash "$HERE/run_local.sh" $DRY_FLAG --recipe "$recipe" --run-tag "$RUN_TAG" --topology colocated_1n --mode normal
