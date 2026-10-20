#!/bin/bash
# scripts/build_gym_venvs.sh — the prebuilt NeMo Gym venv stack a run starts its servers from when GYM_VENV_ROOT is
# set (scripts/start_gym.sh, scripts/lib/gym_ladder.sh: `$GYM_VENV_ROOT/gym_main_venv/bin/gym` plus the flags
# +skip_venv_if_present=true +uv_venv_dir=$GYM_VENV_ROOT/gym_venvs +uv_cache_dir=$GYM_VENV_ROOT/uv_cache). The flat
# layout under ONE root:
#   <root>/gym_main_venv                          the parent venv (uv sync --locked in 3rdparty/Gym)
#   <root>/gym_venvs/<type>/<name>/.venv          one venv per server, the uv_venv_dir layout gym expects
#   <root>/uv_python                              CPython 3.13.14 (uv-managed)
#   <root>/uvbin                                  uv / uvx (written only when GYM_TAR packages the root)
#   $BUILD_DIR/uv_cache, $BUILD_DIR/uvboot        the wheel cache and the bootstrap uv (BUILD_DIR default = <root>;
#                                                 the cache rides in the tar and is mounted as <root>/uv_cache)
# THE ROOT MUST EQUAL THE PATH THE VENVS WILL RUN FROM: uv venvs are not relocatable (shebangs, activate, pyvenv.cfg
# and the editable .pth files bake absolute paths), so build at the GYM_VENV_ROOT the runs will use. Needs network (uv
# downloads the interpreter and the wheels). Gym's own per-server installer is bypassed: every venv carries nemo-gym
# editable from the pinned 3rdparty/Gym checkout next to this repo's servers, and under +skip_venv_if_present=true a
# server whose venv is missing tries to install itself at boot (offline, that fails).
#
#   scripts/build_gym_venvs.sh [--dry-run] [--verify] [<root>] [<server>...]
#     <root>        an absolute path; default GYM_VENV_ROOT, else $REPO/.gym_venvs (git-ignored); a job wrapper passes
#                   its mount root as the argument
#     <server>...   a subset of: wiki_search tool_call_agent alfworld_env webshop_env scienceworld_env env_loop_agent
#                   openai_model legacy_agent (default: all eight, in that order)
#     --verify      after the build: every venv resolves bin/python, has bin/activate and imports nemo_gym from the
#                   checkout; the per-server import probes; the ScienceWorld real-JVM episode through the server
#                   (SCIW_REAL_JVM_OK; JDK = SCIENCEWORLD_JAVA_HOME / SCIW_JAVA_HOME / JAVA_HOME / $TOOLS_DIR/jdk-17 in
#                   the ladder's order (scripts/lib/common.sh scienceworld_java_home), else the java on PATH; jar =
#                   SCIENCEWORLD_JAR / SCIW_JAR, else the wheel's); the offline boot probe of the wiki_search config with the prebuilt
#                   venvs, which must run no installer (GYM_BOOT_OK; placeholder train/validation rows are copied from
#                   example.jsonl when absent, scripts/fetch_data.sh search_r1 moves them aside)
#     GYM_TAR       set: the root is packaged as $GYM_TAR.partial (gym_main_venv gym_venvs uv_python uvbin, plus
#                   $BUILD_DIR/uv_cache); the caller renames it into place
#     BUILD_DIR     scratch for uv_cache and the bootstrap uv (default <root>); a wrapper passes its node-local dir
#     UV_BIN        a specific uv to build with (default: a current uv is bootstrapped into $BUILD_DIR/uvboot — the
#                   box's or the image's uv may predate the 3.13.14 build in uv's interpreter catalog — and reused
#                   when it is already there)
#   --dry-run (also DRY_RUN=1) prints every step as `DRY_RUN <label>: <argv>` (a `DRY_RUN cwd: <dir>` line precedes a
#   step that runs inside a project dir, `DRY_RUN env: KEY=VALUE` lists the uv environment) and exits 0 before any
#   side effect; every failure prints `FATAL: <text>` first; the build ends with GYM_VENVS_OK root=<root> servers=<n>.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/build_gym_venvs.sh [--dry-run] [--verify] [<root>] [<server>...]"
PY_VERSION=3.13.14
ALL_SERVERS="wiki_search tool_call_agent alfworld_env webshop_env scienceworld_env env_loop_agent openai_model legacy_agent"
GYM=$REPO/3rdparty/Gym

server_type() {  # <server> -> its gym server type (the <type> of <root>/gym_venvs/<type>/<name>/.venv)
  case "$1" in
    wiki_search|alfworld_env|webshop_env|scienceworld_env) echo resources_servers ;;
    tool_call_agent|env_loop_agent) echo responses_api_agents ;;
    openai_model) echo responses_api_models ;;
    legacy_agent) echo environment_servers ;;
    *) return 1 ;;
  esac
}
server_dir() {  # <server> -> the project dir its venv is built from (this repo's servers, or the Gym checkout's)
  case "$1" in
    openai_model|legacy_agent) echo "$GYM/$(server_type "$1")/$1" ;;
    *) echo "$REPO/gym_root/$(server_type "$1")/$1" ;;
  esac
}
venv_of() { echo "$ROOT/gym_venvs/$(server_type "$1")/$1/.venv"; }

VERIFY=0 ROOT="" SERVERS=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --verify) VERIFY=1 ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    /*) [ -z "$ROOT" ] || fatal "two roots given ('$ROOT' and '$1'): $USAGE_LINE"
        ROOT=$1 ;;
    *) server_type "$1" >/dev/null || fatal "unknown server '$1' (one of: $ALL_SERVERS; a root is an absolute path): $USAGE_LINE"
       SERVERS="$SERVERS $1" ;;
  esac
  shift
done
[ -z "${DRY_RUN:-}" ] || export DRY_RUN
# the positional root wins, then GYM_VENV_ROOT, then the git-ignored student root
ROOT=${ROOT:-${GYM_VENV_ROOT:-$REPO/.gym_venvs}}
case "$ROOT" in /*) ;; *) fatal "the venv root must be an absolute path, got '$ROOT' (GYM_VENV_ROOT)" ;; esac
ROOT=${ROOT%/}
SERVERS=${SERVERS:-$ALL_SERVERS}
SERVERS=${SERVERS# }
BUILD_DIR=${BUILD_DIR:-$ROOT}
BUILD_DIR=${BUILD_DIR%/}
ARCH=$(uname -m)
# shellcheck disable=SC2086
set -- $SERVERS
NSERVERS=$#

# a step: its DRY_RUN line under --dry-run, else the command; run_in names the dir it runs from (a subshell cd)
run_step() {  # <label> <argv...>
  local label=$1
  shift
  if is_dry_run; then dry_run_print "$label" "$@"; return 0; fi
  "$@"
}
run_in() {  # <dir> <label> <argv...>
  local dir=$1 label=$2
  shift 2
  if is_dry_run; then printf 'DRY_RUN cwd: %s\n' "$dir"; dry_run_print "$label" "$@"; return 0; fi
  ( cd "$dir" && "$@" )
}
# a check under --dry-run is a WARN (the steps are still printed); otherwise a FATAL
gate() { if is_dry_run; then echo "WARN: $*"; else fatal "$*"; fi; }

echo "build_gym_venvs: root=$ROOT build_dir=$BUILD_DIR arch=$ARCH servers=[$SERVERS] verify=$VERIFY${GYM_TAR:+ tar=$GYM_TAR}${DRY_RUN:+ DRY_RUN=$DRY_RUN}"
[ -f "$GYM/pyproject.toml" ] || gate "no Gym checkout at $GYM (git submodule update --init 3rdparty/Gym)"

# the uv environment: the wheel cache and the bootstrap venv under BUILD_DIR, the interpreters under the root
export UV_CACHE_DIR=$BUILD_DIR/uv_cache
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=$ROOT/uv_python
export UV_PYTHON_DOWNLOADS=automatic
if is_dry_run; then
  for kv in "UV_CACHE_DIR=$UV_CACHE_DIR" "UV_LINK_MODE=$UV_LINK_MODE" "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR" \
            "UV_PYTHON_DOWNLOADS=$UV_PYTHON_DOWNLOADS"; do
    printf 'DRY_RUN env: %s\n' "$kv"
  done
fi

# the uv that builds: a bootstrapped one already under BUILD_DIR, an explicit UV_BIN, else a current uv bootstrapped
# now (the box's or the image's uv may predate the cpython-3.13.14 build in the python-build-standalone catalog)
UVBOOT=$BUILD_DIR/uvboot
if [ -x "$UVBOOT/bin/uv" ]; then
  UV=$UVBOOT/bin/uv
  echo "build_gym_venvs: reusing the bootstrapped uv at $UV"
elif [ "$UV_BIN" != uv ]; then
  UV=$UV_BIN
  echo "build_gym_venvs: building with UV_BIN=$UV (no bootstrap)"
else
  UV=$UVBOOT/bin/uv
  py_path=$(command -v "$PYTHON" 2>/dev/null || true)
  [ -n "$py_path" ] || py_path=$PYTHON
  if ! command -v uv >/dev/null 2>&1; then run_step uv_bootstrap_pip "$PYTHON" -m pip install uv; fi
  run_step uv_bootstrap_venv uv venv --python "$py_path" "$UVBOOT"
  run_step uv_bootstrap_upgrade uv pip install --python "$UVBOOT/bin/python3" -U uv
fi
case "$UV" in */*) export PATH=$(dirname "$UV"):$PATH ;; esac
run_step uv_version "$UV" --version
run_step uv_python "$UV" python install "$PY_VERSION"

# 1) the parent Gym venv from the pinned lock, at its fixed runtime path. Gym's setup.cfg points egg_base at cache/ —
# a source tar may strip that dir (it doubles as a uv-cache dump), so restore it before any editable build of nemo-gym.
is_dry_run || mkdir -p "$GYM/cache"
run_in "$GYM" uv_sync env "UV_PROJECT_ENVIRONMENT=$ROOT/gym_main_venv" "$UV" sync --locked
MAIN_PY=$ROOT/gym_main_venv/bin/python
if [ -x "$MAIN_PY" ]; then
  RAY_V=$("$MAIN_PY" -c "import ray; print(ray.__version__)") || fatal "$MAIN_PY cannot import ray"
  OPENAI_V=$("$MAIN_PY" -c "import openai; print(openai.version.VERSION)") || fatal "$MAIN_PY cannot import openai"
else
  is_dry_run || fatal "no $MAIN_PY after uv sync"
  RAY_V='<RAY_V>'
  OPENAI_V='<OPENAI_V>'
fi

# 2) the per-server venvs at the uv_venv_dir layout <type>/<name>/.venv; ray / openai pinned to the parent's
# versions exactly as gym's head_server_deps does
textworld_stub() {  # <venv>: textworld 1.7.0 pre-installed with its setup.sh stubbed out (aarch64 only)
  # textworld has NO linux-aarch64 wheel, and its sdist setup.sh hard-fails on aarch64: it unpacks Inform7 6M62
  # compiler tarballs that only exist for x86. Inform7/glulx are only used to COMPILE .z8/.ulx games; alfworld
  # tw-pddl games never touch them (PDDL planning comes from the separate fast_downward_textworld sdist, which builds
  # from source). With the requirement already satisfied, alfworld_env's `-e .` resolution skips the sdist build.
  local V=$1 TWP=$BUILD_DIR/twpatch
  run_step twp_mkdir mkdir -p "$TWP"
  run_in "$TWP" twp_fetch "$CURL" -fsSL https://files.pythonhosted.org/packages/source/t/textworld/textworld-1.7.0.tar.gz -o textworld-1.7.0.tar.gz
  run_in "$TWP" twp_unpack tar xzf textworld-1.7.0.tar.gz
  if is_dry_run; then
    echo "DRY_RUN twp_stub: $TWP/textworld-1.7.0/setup.sh <- an echo (the Inform7 binaries are x86-only and unused by tw-pddl games)"
  else
    printf '#!/usr/bin/env bash\necho setup.sh skipped: Inform7 binaries are x86-only and unused by tw-pddl games\n' > "$TWP/textworld-1.7.0/setup.sh"
  fi
  run_in "$TWP" uv_pip "$UV" pip install -p "$V" ./textworld-1.7.0
}
build_server() {  # <server>
  local s=$1 dir V
  dir=$(server_dir "$s")
  V=$(venv_of "$s")
  run_in "$dir" uv_venv "$UV" venv --seed --python "$PY_VERSION" "$V"
  if [ "$s" = alfworld_env ] && [ "$ARCH" = aarch64 ]; then textworld_stub "$V"; fi
  case "$s" in
    wiki_search)
      run_in "$dir" uv_pip "$UV" pip install -p "$V" -e "$GYM[dev]" -e . "ray[default]==$RAY_V" "openai==$OPENAI_V" ;;
    alfworld_env|webshop_env|scienceworld_env)
      # plus the shared overlay package (agent_utils.session_gym / procslots) the three servers import
      run_in "$dir" uv_pip "$UV" pip install -p "$V" -e "$GYM[dev]" -e . -e "$REPO/gym_root/agent_utils" "ray[default]==$RAY_V" "openai==$OPENAI_V" ;;
    *)
      # requirements.txt carries the editable nemo-gym[dev] line itself (its relative path resolves against the
      # requirements file's dir -> the pinned checkout), mirroring gym's editable-install branch
      run_in "$dir" uv_pip "$UV" pip install -p "$V" -r requirements.txt "ray[default]==$RAY_V" "openai==$OPENAI_V" ;;
  esac
}
for s in $SERVERS; do build_server "$s"; done

# 3) --verify: the skip_venv_if_present contract (bin/python resolvable, bin/activate present, nemo_gym imported from
# the checkout), the per-server import probes, the ScienceWorld real-JVM episode, the offline boot probe
has_server() { case " $SERVERS " in *" $1 "*) return 0 ;; esac; return 1; }
verify_venvs() {
  local s V
  for s in main $SERVERS; do
    if [ "$s" = main ]; then V=$ROOT/gym_main_venv; else V=$(venv_of "$s"); fi
    if ! is_dry_run; then
      { readlink -e "$V/bin/python" >/dev/null && test -f "$V/bin/activate"; } \
        || fatal "$V is not a complete venv (bin/python unresolvable or bin/activate missing: the skip_venv_if_present contract)"
    fi
    run_step import_check "$V/bin/python" -c "import nemo_gym; assert nemo_gym.__file__.startswith(\"$REPO/\"), nemo_gym.__file__; print(\"OK\", \"$V\", nemo_gym.__file__)"
  done
  ! has_server wiki_search || run_step import_probe "$(venv_of wiki_search)/bin/python" -c "import wiki_search"
  # webshop_env must import with ONLY nemo-gym + httpx (no engine deps present)
  ! has_server webshop_env || run_step import_probe "$(venv_of webshop_env)/bin/python" -c "import webshop_env.app, agent_utils.session_gym, agent_utils.procslots; print('webshop_env OK')"
  # scienceworld_env: server package + simulator wheel + py4j import; the bundled jar must be readable
  ! has_server scienceworld_env || run_step import_probe "$(venv_of scienceworld_env)/bin/python" -c "
import os, scienceworld_env.app, scienceworld_env._worker, scienceworld, py4j
from scienceworld.constants import JAR_PATH
assert os.path.isfile(JAR_PATH), JAR_PATH
print('scienceworld_env OK', scienceworld.__version__ if hasattr(scienceworld, '__version__') else '1.2.3', JAR_PATH)
"
  # tw-pddl runtime path: PddlEnv + the compiled fast-downward planner library
  ! has_server alfworld_env || run_step import_probe "$(venv_of alfworld_env)/bin/python" -c "
import alfworld_env, alfworld, textworld, fast_downward, glob, os
import alfworld_env._worker  # the process-backend module
import agent_utils.session_gym, agent_utils.procslots  # the shared overlay package, installed editable above
assert textworld.__version__ == '1.7.0', textworld.__version__
from textworld.envs.pddl import PddlEnv
libs = glob.glob(os.path.join(os.path.dirname(fast_downward.__file__), '*downward*'))
assert libs, 'libdownward missing'
print('alfworld OK', libs[0])
"
}
verify_real_jvm() {
  # a real-JVM episode through the server (worker process + py4j gateway + the jar): the JDK must run here, or the
  # venv set is useless for RECIPE=scienceworld
  local jdk jar=${SCIENCEWORLD_JAR:-${SCIW_JAR:-}} V
  jdk=$(scienceworld_java_home)  # the ladder's order (scripts/lib/common.sh); "" = the java on PATH
  V=$(venv_of scienceworld_env)
  if [ -n "$jdk" ]; then
    [ -x "$jdk/bin/java" ] || gate "no JDK at $jdk (SCIENCEWORLD_JAVA_HOME / SCIW_JAVA_HOME / JAVA_HOME / TOOLS_DIR): the ScienceWorld real-JVM check needs one"
  else
    command -v java >/dev/null 2>&1 || gate "no java on PATH and no JAVA_HOME (nor SCIENCEWORLD_JAVA_HOME / SCIW_JAVA_HOME): the ScienceWorld real-JVM check needs a JDK >= 11"
  fi
  [ -z "$jar" ] || [ -f "$jar" ] || gate "no ScienceWorld jar at $jar (SCIENCEWORLD_JAR / SCIW_JAR; empty = the wheel's bundled jar)"
  run_in "$REPO/gym_root/resources_servers/scienceworld_env" real_jvm \
    env ${jdk:+"JAVA_HOME=$jdk" "PATH=$jdk/bin:$PATH"} SCIENCEWORLD_REAL_TESTS=1 "SCIENCEWORLD_JAR=$jar" \
    "$V/bin/python" -m pytest -q -p no:cacheprovider tests/test_worker_pool.py -k RealJVM
  is_dry_run || echo "SCIW_REAL_JVM_OK"
}
verify_boot() {
  # start the full server set of the wiki_search config exactly as a run will — prebuilt venvs, zero installs,
  # offline. Placeholder train/validation rows (copies of example.jsonl, moved aside by scripts/fetch_data.sh
  # search_r1) and a dummy env.yaml only stand in for the probe.
  local data=$REPO/gym_root/resources_servers/wiki_search/data log=$BUILD_DIR/gym_boot.log ok i
  local gym=$ROOT/gym_main_venv/bin/gym
  [ -f "$data/example.jsonl" ] || gate "$data/example.jsonl missing (it is a committed fixture)"
  if ! is_dry_run; then
    [ -f "$data/train.jsonl" ] || cp "$data/example.jsonl" "$data/train.jsonl"
    [ -f "$data/validation.jsonl" ] || cp "$data/example.jsonl" "$data/validation.jsonl"
    sed -e "s|http://CHANGE_ME:30000/v1|http://127.0.0.1:9/v1|" -e "s|CHANGE_ME|dummy|g" \
        "$REPO/gym_root/env.yaml.example" > "$REPO/gym_root/env.yaml"
    cp "$REPO/gym_root/env.yaml" "$GYM/env.yaml"
  fi
  export NEMO_GYM_EXTRA_ROOTS=$REPO/gym_root
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
  export RAY_TMPDIR=/tmp
  export TMPDIR=/tmp
  # UV_OFFLINE as at runtime: a server without a prebuilt venv must fail here, not install itself online into the
  # root and ride along into the tar
  if is_dry_run; then
    printf 'DRY_RUN cwd: %s\n' "$GYM"
    dry_run_print gym_boot env UV_OFFLINE=1 "$gym" env start \
      --config "$REPO/gym_root/resources_servers/wiki_search/configs/wiki_search.yaml" --model-type openai_model \
      "+skip_venv_if_present=true" "+uv_venv_dir=$ROOT/gym_venvs" "+uv_cache_dir=$ROOT/uv_cache" \
      "+model_endpoint_readiness_timeout_seconds=0" "+head_server.host=127.0.0.1" "+head_server.port=8100"
    return 0
  fi
  ( cd "$GYM" && exec env UV_OFFLINE=1 "$gym" env start \
      --config "$REPO/gym_root/resources_servers/wiki_search/configs/wiki_search.yaml" --model-type openai_model \
      "+skip_venv_if_present=true" "+uv_venv_dir=$ROOT/gym_venvs" "+uv_cache_dir=$ROOT/uv_cache" \
      "+model_endpoint_readiness_timeout_seconds=0" "+head_server.host=127.0.0.1" "+head_server.port=8100" ) \
    > "$log" 2>&1 &
  GYM_PID=$!
  # the EXIT trap kills the probe's gym too: a failure (or a TERM) between here and the kill below leaves no gym behind
  add_cleanup 'kill $GYM_PID 2>/dev/null'
  # head server /readyz: 200 once every server it spawned is up (gym >= 0.7.0)
  ok=0
  for i in $(seq 84); do
    "$CURL" -sf -o /dev/null http://127.0.0.1:8100/readyz && { ok=1; break; }
    kill -0 "$GYM_PID" 2>/dev/null || break
    sleep 5
  done
  tail -60 "$log"
  kill "$GYM_PID" 2>/dev/null || true
  sleep 3
  [ "$ok" -eq 1 ] || { echo "FATAL: gym boot probe failed"; cat "$log"; exit 1; }
  # the skip path must not have run any installer
  if grep -n "uv pip install\|Creating virtual environment" "$log"; then
    echo "FATAL: a gym server ran its venv installer at boot (missing prebuilt venv)"; exit 1
  fi
  echo "GYM_BOOT_OK"
}
if [ "$VERIFY" = 1 ]; then
  verify_venvs
  ! has_server scienceworld_env || verify_real_jvm
  ! has_server wiki_search || verify_boot
fi

# 4) package (the caller publishes atomically: it renames $GYM_TAR.partial into place)
if [ -n "${GYM_TAR:-}" ]; then
  uv_real=$(command -v "$UV" 2>/dev/null || printf '%s\n' "$UV")
  run_step uvbin_mkdir mkdir -p "$ROOT/uvbin"
  if [ -e "$(dirname "$uv_real")/uvx" ]; then
    run_step uvbin_copy cp "$uv_real" "$(dirname "$uv_real")/uvx" "$ROOT/uvbin/"
  else
    run_step uvbin_copy cp "$uv_real" "$ROOT/uvbin/"
  fi
  run_step tar tar cf "$GYM_TAR.partial" -C "$ROOT" gym_main_venv gym_venvs uv_python uvbin -C "$BUILD_DIR" uv_cache
fi
if is_dry_run; then exit 0; fi
echo "GYM_VENVS_OK root=$ROOT servers=$NSERVERS"
