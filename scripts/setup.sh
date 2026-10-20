#!/bin/bash
# scripts/setup.sh — a fresh clone made ready for the tests, the data prep and the dry runs: uv (the one tool it needs
# on PATH), the two submodules, the miles patches, the test venv, local.env. Idempotent: a second run touches only
# what is still missing. Training itself runs in the miles image (docs/containers.md) or on a machine with miles
# installed (docs/local_setup.md); this script installs neither.
#
#   scripts/setup.sh [--dry-run]
#     1. uv must be on PATH (UV_BIN, default uv) — else FATAL with the install hint
#     2. git -C <repo> submodule update --init --recursive <the ones of 3rdparty/miles 3rdparty/Gym not yet
#        initialised, or whose HEAD is neither the gitlink nor a descendant of it (the pin moved under the checkout)>
#        (an initialised submodule at the gitlink or ahead of it — 3rdparty/miles with its patch commits — is never
#        touched: a bare `git submodule update` would silently drop those commits; the next step re-applies whatever
#        patch is missing; the three reads behind the decision — ls-files -s, rev-parse, merge-base — run under
#        --dry-run too, and a submodule whose gitlink or HEAD cannot be read is left as it is)
#     3. bash <repo>/scripts/check_patches.sh --apply   (the patches of patches/miles/series onto the miles pin, in
#        order; an applied one is skipped; patches/miles/README.md)
#     4. uv venv --allow-existing --python 3.13.14 <repo>/.venv, then
#        uv pip install -p <repo>/.venv --group <repo>/pyproject.toml:test   (the test venv: `.venv/bin/python -m
#        pytest -q tests`; the repo itself is not a package — pyproject.toml says package = false — the tests and the
#        data prep import it from the checkout, so nothing is installed editable)
#     5. local.env from local.env.example when LOCAL_ENV (default <repo>/local.env) does not exist yet (/dev/null = none)
#     then the next steps, and `SETUP_OK venv=<dir> local_env=<file>` last.
#     --dry-run (also DRY_RUN=1): `DRY_RUN submodules:`, `DRY_RUN check_patches:`, `DRY_RUN uv_venv:`, `DRY_RUN
#     uv_pip:`, `DRY_RUN local_env:` with the argv of each step that would run; exit 0 before any side effect (a
#     missing uv is a WARN there), without the marker.
#   Every failure prints `FATAL: <text>` first. Knobs: UV_BIN GIT (default uv, git), LOCAL_ENV, and
#   scripts/check_patches.sh's MILES_SRC / MILES_PATCHES_DIR; local.env is read when it exists (scripts/lib/common.sh).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) fatal "unknown argument '$1': scripts/setup.sh [--dry-run]" ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

GIT=${GIT:-git}
VENV_DIR=$REPO/.venv
PYTHON_PIN=3.13.14     # pyproject.toml requires-python; the same CPython scripts/build_gym_venvs.sh installs
UV_INSTALL_HINT="install uv: https://docs.astral.sh/uv/getting-started/installation/ (curl -LsSf https://astral.sh/uv/install.sh | sh), then rerun"

# the argv, or its DRY_RUN line under --dry-run
run_or_dry() {  # <label> <argv...>
  local label=$1
  shift
  if is_dry_run; then dry_run_print "$label" "$@"; return 0; fi
  echo "setup: $*"
  "$@"
}

# ---- 1. uv ----
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if is_dry_run; then echo "WARN: uv not found (UV_BIN=$UV_BIN): $UV_INSTALL_HINT"
  else fatal "uv not found (UV_BIN=$UV_BIN): $UV_INSTALL_HINT"; fi
fi

# ---- 2. the submodules: not yet initialised, or initialised at a commit that is neither the gitlink nor ahead of it ----
# An initialised submodule whose HEAD is the gitlink or a descendant of it (3rdparty/miles after step 3's patches) is
# left as it is: a bare `git submodule update` would silently drop the patch commits. One whose HEAD is neither (the
# pin moved under the checkout: a pull of this repo) is updated to the gitlink, recursively, and step 3 re-applies
# the patches. The reads run under --dry-run too (they change nothing); when one of them answers nothing the
# submodule is left as it is.
MISSING=()
for sub in 3rdparty/miles 3rdparty/Gym; do
  if [ -e "$REPO/$sub/.git" ]; then
    gitlink=$("$GIT" -C "$REPO" ls-files -s -- "$sub" 2>/dev/null | awk '{print $2; exit}' || true)
    head=$("$GIT" -C "$REPO/$sub" rev-parse HEAD 2>/dev/null || true)
    if [ -z "$gitlink" ] || [ -z "$head" ]; then
      echo "setup: submodule $sub already initialised: left as it is (gitlink or HEAD unknown)"
    elif [ "$head" = "$gitlink" ] || "$GIT" -C "$REPO/$sub" merge-base --is-ancestor "$gitlink" "$head" 2>/dev/null; then
      echo "setup: submodule $sub already initialised: left as it is"
    else
      echo "setup: submodule $sub is at ${head:0:12}, neither the gitlink ${gitlink:0:12} nor a descendant of it: updated to the gitlink"
      MISSING+=("$sub")
    fi
  else
    MISSING+=("$sub")
  fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
  run_or_dry submodules "$GIT" -C "$REPO" submodule update --init --recursive "${MISSING[@]}" \
    || fatal "git submodule update --init --recursive ${MISSING[*]} failed"
fi

# ---- 3. the miles patches (scripts/check_patches.sh: idempotent, in series order) ----
run_or_dry check_patches bash "$REPO/scripts/check_patches.sh" --apply || fatal "scripts/check_patches.sh --apply failed"

# ---- 4. the test venv ----
run_or_dry uv_venv "$UV_BIN" venv --allow-existing --python "$PYTHON_PIN" "$VENV_DIR" || fatal "uv venv $VENV_DIR failed"
run_or_dry uv_pip "$UV_BIN" pip install -p "$VENV_DIR" --group "$REPO/pyproject.toml:test" \
  || fatal "uv pip install --group test into $VENV_DIR failed"

# ---- 5. local.env ----
if [ -e "$LOCAL_ENV" ]; then
  echo "setup: $LOCAL_ENV already there: kept"
else
  run_or_dry local_env cp "$REPO/local.env.example" "$LOCAL_ENV" || fatal "could not write $LOCAL_ENV"
fi

if is_dry_run; then
  echo "setup: dry run - nothing changed"
  exit 0
fi
cat <<NEXT
setup: next steps
  edit $LOCAL_ENV                                                           # the roots, the ports, the knobs (docs/local_setup.md)
  bash scripts/fetch_data.sh <alfworld|scienceworld|webshop|search_r1>      # an environment's data (docs/data.md)
  bash scripts/build_gym_venvs.sh [<root>]                                  # optional: prebuilt gym venvs (GYM_VENV_ROOT)
  bash scripts/run_local.sh --dry-run --recipe alfworld --run-tag t1        # every command of a run, nothing started
  bash scripts/contract_selftest.sh                                         # the no-cluster self-test
  $VENV_DIR/bin/python -m pytest -q tests                                   # this repo's tests
NEXT
echo "SETUP_OK venv=$VENV_DIR local_env=$LOCAL_ENV"
