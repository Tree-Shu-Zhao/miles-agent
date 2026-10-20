#!/usr/bin/env bash
# Launch the WebShop engine server (py3.10, engine-worker pool). Runs OUTSIDE
# the miles/nemo-gym py3.13 world, in the self-contained runtime that setup.sh
# builds under $WEBSHOP_HOME (py3.10 env with the JDK bundled). The py3.13
# webshop_env adapter proxies to it at $WEBSHOP_SERVER_HOST:$WEBSHOP_SERVER_PORT.
#
#   WEBSHOP_HOME=<runtime root> ./launch_engine.sh        # after setup.sh
#
# Every knob is an environment variable defaulted in defaults.sh (the single
# source of truth): WEBSHOP_SERVER_HOST / WEBSHOP_SERVER_PORT,
# WEBSHOP_ENGINE_WORKERS, WEBSHOP_ENGINE_CALL_TIMEOUT, WEBSHOP_ENGINE_SESSION_TTL,
# WEBSHOP_FULL, the protocol (WEBSHOP_GOAL_SEED, WEBSHOP_OBS_MODE,
# WEBSHOP_HUMAN_GOALS) and the layout (WEBSHOP_PYTHON, WEBSHOP_JAVA_HOME,
# WEBSHOP_VENDOR, WEBSHOP_DATA_DIR). All of them reach server.py as explicit
# flags, so its argparse defaults never decide anything here. Extra args pass
# through to server.py after them (last flag wins), e.g.
#   ./launch_engine.sh --max-concurrency 512 --sweep-interval 30
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=defaults.sh
. "${HERE}/defaults.sh"

# The launcher redirects stdout/stderr to a log the inner script tails when the
# health gate fails; block-buffered stdout would hide exactly the boot error
# that matters.
export PYTHONUNBUFFERED=1
# JDK for pyserini's JVM: the conda prefix env bundles it at $WEBSHOP_HOME/env.
export JAVA_HOME="${WEBSHOP_JAVA_HOME}"
export PATH="${JAVA_HOME}/bin:${PATH}"

ARGS=(
    --host "${WEBSHOP_SERVER_HOST}"
    --port "${WEBSHOP_SERVER_PORT}"
    --engine-workers "${WEBSHOP_ENGINE_WORKERS}"
    --engine-call-timeout "${WEBSHOP_ENGINE_CALL_TIMEOUT}"
    --session-ttl "${WEBSHOP_ENGINE_SESSION_TTL}"
    --vendored-path "${WEBSHOP_VENDOR}"
    --data-dir "${WEBSHOP_DATA_DIR}"
    --goal-seed "${WEBSHOP_GOAL_SEED}"
    --observation-mode "${WEBSHOP_OBS_MODE}"
)
if [ "${WEBSHOP_FULL}" = "1" ]; then ARGS+=(--full); else ARGS+=(--small); fi
if [ "${WEBSHOP_HUMAN_GOALS}" = "1" ]; then ARGS+=(--human-goals); fi

exec "${WEBSHOP_PYTHON}" "${HERE}/server.py" "${ARGS[@]}" "$@"
