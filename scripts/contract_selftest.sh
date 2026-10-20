#!/bin/bash
# scripts/contract_selftest.sh — the no-cluster self-test of the wrapper <-> entry contract (docs/cluster.md): what
# a job wrapper's maintainer runs against a checkout of this repo before pinning its SHA, and what anyone runs after
# scripts/setup.sh — no GPU, no scheduler, no model, nothing started. Under a fake Contract A (`. tests/fake_env.sh`)
# or in a bare shell it
#   1. parses every public script: `bash -n` over scripts/**/*.sh and the example wrappers under examples/slurm/;
#   2. lists every Contract A name of tests/contract_vars.txt (`CONTRACT_VAR name=<n> state=set|unset`);
#   3. runs the entries' contract_check (CONTRACT_VERSION unset = local mode; a major other than 1 is a FATAL);
#   4. runs `scripts/run_local.sh --dry-run --recipe ${RECIPE:-alfworld} --run-tag selftest` with
#      GPUS_PER_NODE=${GPUS_PER_NODE:-1} and, when POLICY_HF is unset, a stub policy dir (an empty config.json at
#      ${TMPDIR:-/tmp}/miles-agent-selftest/policy: ONE dir, created once and overwritten by every run, never removed)
#      so that the dry run does not WARN about the policy, and checks that its `DRY_RUN driver:` line carries
#      `--preempt-sentinel <PREEMPT_SENTINEL>` (the wrapper's, else $MILES_AGENT_RESULTS/selftest/stop) and
#      `--save-dir <MILES_AGENT_RESULTS>/selftest/` (= the wrapper's RESULTS_DIR when it exports one);
#   5. checks that the example wrappers carry none of --account --partition --qos --container-image outside a comment
#      (a site's lines stay comments);
#   and ends with `CONTRACT_SELFTEST_OK vars=<set>/<total>`.
#
#   scripts/contract_selftest.sh [--dry-run]
#     --dry-run (also DRY_RUN=1): the listing of step 2 (reads only), then `DRY_RUN bash_n:`, `DRY_RUN run_local:` and
#     `DRY_RUN grep_example:` with the argv of steps 1, 4 and 5; exit 0 before any side effect, without the marker.
#   Every failure prints `FATAL: <text>` first. Knobs: RECIPE (alfworld), GPUS_PER_NODE (1), POLICY_HF (else the
#   stub), the roots and tool overrides of scripts/lib/common.sh, local.env.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) fatal "unknown argument '$1': scripts/contract_selftest.sh [--dry-run]" ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

VARS_FILE=$REPO/tests/contract_vars.txt
EXAMPLES_DIR=$REPO/examples/slurm
RUN_LOCAL=$REPO/scripts/run_local.sh
[ -f "$VARS_FILE" ] || fatal "no $VARS_FILE (the Contract A names, one per line)"
[ -f "$RUN_LOCAL" ] || fatal "no $RUN_LOCAL"

# ---- 1. every public script parses ----
FILES=()
while IFS= read -r f; do FILES+=("$f"); done < <(find "$REPO/scripts" -type f -name '*.sh' | sort)
for f in "$EXAMPLES_DIR"/*; do
  [ -f "$f" ] || continue
  FILES+=("$f")
done
[ ${#FILES[@]} -gt 0 ] || fatal "no shell file under $REPO/scripts"
if is_dry_run; then
  dry_run_print bash_n bash -n "${FILES[@]}"
else
  for f in "${FILES[@]}"; do
    bash -n "$f" || fatal "bash -n failed for $f"
  done
  echo "selftest: ${#FILES[@]} shell files parse (scripts/**/*.sh and examples/slurm/*)"
fi

# ---- 2. the Contract A names: set or unset in this shell ----
TOTAL=0
SET=0
while IFS= read -r name || [ -n "$name" ]; do
  name=${name%%[[:space:]]*}
  case "$name" in ''|'#'*) continue ;; esac
  TOTAL=$((TOTAL + 1))
  if [ -n "${!name+x}" ]; then
    SET=$((SET + 1))
    echo "CONTRACT_VAR name=$name state=set"
  else
    echo "CONTRACT_VAR name=$name state=unset"
  fi
done < "$VARS_FILE"
echo "selftest: $SET/$TOTAL Contract A variables set"

# ---- 3. the contract version (scripts/lib/common.sh: unset = local mode; a major other than 1 is a FATAL) ----
contract_check
echo "selftest: CONTRACT_VERSION=${CONTRACT_VERSION:-<unset: local mode>}"

# ---- 4. the entry's argv-only dry run ----
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
RECIPE=${RECIPE:-alfworld}
EXPECT_SENTINEL=${PREEMPT_SENTINEL:-$MILES_AGENT_RESULTS/selftest/stop}
EXPECT_SAVE=$MILES_AGENT_RESULTS/selftest/
if is_dry_run; then
  dry_run_print run_local bash "$RUN_LOCAL" --dry-run --recipe "$RECIPE" --run-tag selftest
else
  if [ -z "${POLICY_HF:-}" ]; then
    # the ladder wants the policy's config.json; a stub under $TMPDIR keeps the dry run free of WARNs: ONE dir per
    # TMPDIR, overwritten each run (the scripts remove nothing, so a per-run name would pile up)
    STUB=${TMPDIR:-/tmp}/miles-agent-selftest/policy
    mkdir -p "$STUB"
    printf '{}\n' > "$STUB/config.json"
    export POLICY_HF=$STUB
    echo "selftest: POLICY_HF unset: stub model dir $STUB (an empty config.json)"
  fi
  RC=0
  OUT=$(bash "$RUN_LOCAL" --dry-run --recipe "$RECIPE" --run-tag selftest 2>&1) || RC=$?
  printf '%s\n' "$OUT"
  [ "$RC" -eq 0 ] || fatal "scripts/run_local.sh --dry-run --recipe $RECIPE --run-tag selftest failed (rc=$RC)"
  DRIVER=$(printf '%s\n' "$OUT" | grep '^DRY_RUN driver:' || true)
  [ -n "$DRIVER" ] || fatal "scripts/run_local.sh --dry-run printed no 'DRY_RUN driver:' line"
  # the entry quotes its argv with %q: compare the same spelling (a trailing blank closes the last word)
  Q_SENTINEL=$(printf '%q' "$EXPECT_SENTINEL")
  Q_SAVE=$(printf '%q' "$EXPECT_SAVE")
  case "$DRIVER " in
    *"--preempt-sentinel $Q_SENTINEL "*) ;;
    *) fatal "the driver line lacks --preempt-sentinel $Q_SENTINEL: $DRIVER" ;;
  esac
  case "$DRIVER " in
    *"--save-dir $Q_SAVE "*) ;;
    *) fatal "the driver line lacks --save-dir $Q_SAVE: $DRIVER" ;;
  esac
  echo "selftest: the driver line carries --preempt-sentinel $EXPECT_SENTINEL and --save-dir $EXPECT_SAVE"
fi

# ---- 5. the example wrappers: a site's scheduler flags only as comments ----
ACTIVE_RE='--account|--partition|--qos|--container-image'
for f in "$EXAMPLES_DIR"/*; do
  [ -f "$f" ] || continue
  if is_dry_run; then
    dry_run_print grep_example grep -nE -- "$ACTIVE_RE" "$f"
    continue
  fi
  hits=$(grep -nE -- "$ACTIVE_RE" "$f" | grep -vE '^[0-9]+:[[:space:]]*#' || true)
  if [ -n "$hits" ]; then
    printf '%s\n' "$hits"
    fatal "$f carries a site-specific scheduler flag outside a comment (--account --partition --qos --container-image stay comments)"
  fi
  echo "selftest: $f carries no site-specific scheduler flag outside a comment"
done

if is_dry_run; then
  echo "selftest: dry run - nothing checked for real"
  exit 0
fi
echo "CONTRACT_SELFTEST_OK vars=$SET/$TOTAL"
