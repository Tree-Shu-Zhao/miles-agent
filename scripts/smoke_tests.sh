#!/bin/bash
# scripts/smoke_tests.sh — the contract smoke of an environment that has miles importable (the image, or a machine
# with miles installed): this repo's test suite, then the miles fast tests that cover the patches of patches/miles/
# (session text parts, env/reward metrics, the fp32 LM head, the Mamba seq_idx, the preempt save in both drivers,
# the locks it takes, its flag), and `CONTRACT_TESTS_OK` last. It is what a job wrapper runs as its environment smoke
# (docs/cluster.md, PROJECT_SMOKE_CMD; the old in-repo wrapper's environment-smoke job ran these lines inline), and
# what anyone runs after a rebuild of the image or a bump of the pins.
#
#   scripts/smoke_tests.sh [--dry-run]
#     1. (cd $REPO && $PYTHON -m pytest -q -p no:cacheprovider -rs tests)
#     2. when $MILES_ROOT/tests/fast exists: (cd $MILES_ROOT && env -u CUDA_VISIBLE_DEVICES $PYTHON -m pytest -q
#        -p no:cacheprovider <the fast tests below> --deselect <the one that fails on the image's sglang-router>);
#        the session tests resolve the Qwen/Qwen3-0.6B tokenizer from HF_HOME (the environment's: a wrapper exports
#        the cache it pre-fetched the tokenizer files into; a plain machine downloads them once) and the rollout
#        fixtures declare 8 GPUs, which ray refuses only when CUDA_VISIBLE_DEVICES lists fewer — hence the env -u;
#        no tests/fast under MILES_ROOT: `WARN: no miles fast tests under <root>` and step 2 is skipped
#     3. CONTRACT_TESTS_OK
#     --dry-run (also DRY_RUN=1): `DRY_RUN cwd:` + `DRY_RUN pytest_repo:`, then `DRY_RUN cwd:` + `DRY_RUN pytest_miles:`
#     (or the WARN) with the argv; exit 0, nothing run, no marker
#   A failing suite ends in `FATAL: <which> failed (rc=<n>)` with its exit code. Knobs: MILES_ROOT (scripts/lib/common.sh:
#   /root/miles when that dir exists, else the submodule), PYTHON, HF_HOME, local.env.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) fatal "unknown argument '$1': scripts/smoke_tests.sh [--dry-run]" ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

# the miles fast tests that exercise the patches (the old environment-smoke job's list, verbatim)
MILES_FAST_TESTS=(
  tests/fast/router/test_sessions.py tests/fast/router/test_session_request_args.py
  tests/fast/router/test_session_v1_v2_parity.py
  tests/fast/ray/rollout/test_metrics.py tests/fast/ray/rollout/test_train_data_conversion.py
  tests/fast/test_train.py tests/fast/test_train_async.py tests/fast/test_train_multi_policy.py
  tests/fast/utils/test_preempt.py tests/fast/utils/test_arguments.py
  tests/fast/backends/training_utils/loss/test_fp32_lm_head.py
  tests/fast/backends/megatron_utils/test_packed_seq_idx.py
  tests/fast/utils/test_context_lock.py tests/fast/utils/test_context_lock_failure_patterns.py
  tests/fast/ray/rollout/test_rollout_server_locking.py
  tests/fast/ray/test_train_actor.py tests/fast/ray/test_train_actor_rpc_surface.py
)
# fails on the image's sglang-router, whose --router-allow-requests-without-routing-key the v0.1.2 list does not name
MILES_FAST_DESELECT=tests/fast/utils/test_arguments.py::TestSecretArgumentsAreClassified::test_every_secret_looking_miles_flag_is_either_redacted_or_declared_harmless

# ---- 1. this repo's tests ----
if is_dry_run; then
  dry_run_print cwd "$REPO"
  dry_run_print pytest_repo "$PYTHON" -m pytest -q -p no:cacheprovider -rs tests
else
  echo "smoke_tests: the repo's tests in $REPO"
  RC=0
  (cd "$REPO" && "$PYTHON" -m pytest -q -p no:cacheprovider -rs tests) || RC=$?
  [ "$RC" -eq 0 ] || fatal "the repo's tests failed (rc=$RC)"
fi

# ---- 2. the miles fast tests ----
if [ -d "$MILES_ROOT/tests/fast" ]; then
  if is_dry_run; then
    dry_run_print cwd "$MILES_ROOT"
    dry_run_print pytest_miles env -u CUDA_VISIBLE_DEVICES "$PYTHON" -m pytest -q -p no:cacheprovider \
      "${MILES_FAST_TESTS[@]}" --deselect "$MILES_FAST_DESELECT"
  else
    echo "smoke_tests: the miles fast tests in $MILES_ROOT (HF_HOME=${HF_HOME:-<unset>})"
    RC=0
    (cd "$MILES_ROOT" && env -u CUDA_VISIBLE_DEVICES "$PYTHON" -m pytest -q -p no:cacheprovider \
      "${MILES_FAST_TESTS[@]}" --deselect "$MILES_FAST_DESELECT") || RC=$?
    [ "$RC" -eq 0 ] || fatal "the miles fast tests failed (rc=$RC)"
  fi
else
  echo "WARN: no miles fast tests under $MILES_ROOT (tests/fast missing): step 2 skipped"
fi

if is_dry_run; then
  echo "smoke_tests: dry run - nothing run"
  exit 0
fi
echo "CONTRACT_TESTS_OK"
