#!/bin/bash
# Start nemo-gym servers in the foreground, for local dev and for the env host of a split run.
#
#   scripts/start_gym.sh [--dry-run] <recipe> [+override ...]
#       recipe = alfworld | scienceworld | webshop | search_r1: scripts/lib/gym_ladder.sh picks the config, the
#       agent port and the pre-gym services (webshop's engine, scienceworld's jar + JDK, search_r1's data links),
#       renders gym_root/env.yaml, boots the servers, prints `GYM_UP agent=http://127.0.0.1:$NEMO_GYM_PORT` and
#       waits (TERM / INT stop them). Knobs: RUN_TAG (default gym; the log is $RUNS_DIR/<RUN_TAG>_gym_<JOB_ID>.log),
#       GYM_VENV_ROOT (prebuilt venvs; unset = `uv run` from 3rdparty/Gym), NEMO_GYM_PORT (8200), POLICY_MODEL /
#       POLICY_HF, MILES_AGENT_{MODELS,DATA,RESULTS} and local.env (scripts/lib/common.sh); search_r1 needs
#       RETRIEVER_URL; every other knob is documented in gym_ladder.sh. Further +key=value words are extra gym
#       overrides (word-split: no spaces in a value). tmax starts no gym (NEMO_GYM_URL names its agent server).
#   scripts/start_gym.sh [<gym args> ...]
#       The legacy form (no recipe): wiki_search + tool_call_agent by default; GYM_CONFIG (absolute path — gym runs
#       from 3rdparty/Gym) swaps the server set and every argument goes to `gym env start`, e.g.
#         GYM_CONFIG=$PWD/gym_root/resources_servers/alfworld_env/configs/alfworld.yaml ALFWORLD_DATA=~/alfworld_data scripts/start_gym.sh
#       Requires gym_root/env.yaml (copy env.yaml.example; policy_base_url there is a dummy — every /run carries
#       its own per-episode policy_base_url override).
#   --dry-run anywhere (or DRY_RUN=1): print `DRY_RUN gym: <argv>`, `DRY_RUN env: KEY=VALUE` and `DRY_RUN cwd:
#       <dir>`, start nothing, exit 0 (both forms). -h / --help prints this text.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

args=()
for a in "$@"; do
  if [ "$a" = --dry-run ]; then export DRY_RUN=1; else args+=("$a"); fi
done
set -- ${args[@]+"${args[@]}"}
recipe=""
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  ''|-*|+*|*=*) ;;  # the legacy form: nothing here is a recipe, every word goes to gym
  *) recipe=$1; shift ;;
esac

. "$HERE/lib/common.sh"

if [ -z "$recipe" ]; then
  # ---- the legacy form: `uv run gym env start` from 3rdparty/Gym with GYM_CONFIG and the caller's words ----
  cfg=${GYM_CONFIG:-$REPO/gym_root/resources_servers/wiki_search/configs/wiki_search.yaml}
  gym_argv=("$UV_BIN" run gym env start --config "$cfg" --model-type openai_model "$@")
  if is_dry_run; then
    dry_run_print gym "${gym_argv[@]}"
    printf 'DRY_RUN env: %s\n' "NEMO_GYM_EXTRA_ROOTS=$REPO/gym_root" "RAY_ENABLE_UV_RUN_RUNTIME_ENV=0"
    printf 'DRY_RUN cwd: %s\n' "$REPO/3rdparty/Gym"
    exit 0
  fi
  export NEMO_GYM_EXTRA_ROOTS="$REPO/gym_root"
  # Ray >=2.x replicates the `uv run` env to workers and trips over gym's
  # per-server working dirs; gym only uses Ray as a local task pool.
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
  cd "$REPO/3rdparty/Gym"
  exec "${gym_argv[@]}"
fi

# ---- the recipe form ----
. "$LIB/gym_ladder.sh"
RUN_TAG=${RUN_TAG:-gym}
run_dirs "$RUN_TAG"
recipe_vars "$recipe"
[ -n "$GYMCFG" ] || fatal "RECIPE=$recipe starts no gym here: NEMO_GYM_URL=${NEMO_GYM_URL:-} names its external agent server"
for a in "$@"; do append_override "$a"; done
render_env_yaml
"$RECIPE_PRE_GYM"
gym_start_and_wait
if is_dry_run; then exit 0; fi
# a foreground service: TERM / INT end the wait, the exit runs the cleanup trap (common.sh), which kills the gym
trap 'exit 143' TERM
trap 'exit 130' INT
echo "start_gym: RECIPE=$recipe serving (gym pid $GYM_PID, log $GYM_LOG); TERM or INT stops it"
rc=0
wait "$GYM_PID" || rc=$?
[ "$rc" -eq 0 ] || fatal "gym servers exited with rc=$rc (log: $GYM_LOG)"
