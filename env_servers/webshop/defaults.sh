# shellcheck shell=bash
# WebShop engine tier: the ONE place shell-side defaults live. Sourced by
# env_servers/webshop/launch_engine.sh, env_servers/webshop/setup.sh and
# the training entry (scripts/run_local.sh). It only assigns variables (no commands, no
# exports, no side effects), so sourcing it anywhere, any number of times, is
# safe; every knob keeps a value already present in the environment
# (VAR=${VAR:-default}). Consumers export what their children need.
#
# The Python-side defaults (argparse in env_servers/webshop/server.py; the
# adapter config in gym_root/resources_servers/webshop_env/app.py and
# configs/webshop.yaml) exist only for manual runs and MUST equal the values
# below; the launcher passes every knob as an explicit flag.
#
# Coupling rules (change one side and you must revisit the other):
#   * adapter engine_timeout_s (240, webshop.yaml) > WEBSHOP_ENGINE_CALL_TIMEOUT
#     (120) + the worker respawn margin (kill/join <= 5 s + replacement-engine
#     boot, run inline on the slot's executor): a wedged engine op then reaches
#     the adapter as the engine's definitive 410/500 rather than a client-side
#     ReadTimeout. The boot is bounded only by SPAWN_READY_TIMEOUT_S (300,
#     _engine_worker.py), so this holds while a replacement boots within
#     ~115 s: true for --small, not guaranteed for --full (raise
#     engine_timeout_s or lower this). Either way the episode is dropped, never
#     trained (ReadTimeout -> EngineError(status=None) -> 502 ->
#     run_transport_failed); only the diagnostics differ.
#   * WEBSHOP_ENGINE_SESSION_TTL (2400) > adapter session_ttl_s (1800,
#     webshop.yaml) > NEMO_GYM_RUN_TIMEOUT (1200, recipes/webshop/run.py): the
#     engine never evicts a session the adapter still considers live, and the
#     adapter never evicts one a running /run may still step.
#   * WEBSHOP_ENGINE_BASE_URL is what the adapter dials; it must reach
#     WEBSHOP_SERVER_HOST:WEBSHOP_SERVER_PORT (set it explicitly if the host is
#     ever not loopback).

# --- engine HTTP endpoint -----------------------------------------------------
# Loopback only: the py3.13 adapter is the sole client and shares the node.
WEBSHOP_SERVER_HOST=${WEBSHOP_SERVER_HOST:-127.0.0.1}
# Below 9000: on a node whose ip_local_port_range starts at 9000, anything
# above can be squatted by an ephemeral socket between our check and bind.
WEBSHOP_SERVER_PORT=${WEBSHOP_SERVER_PORT:-6200}
# The adapter's engine_base_url; the training entry passes it as the gym CLI
# override +webshop_env.resources_servers.webshop_env.engine_base_url=...
WEBSHOP_ENGINE_BASE_URL=${WEBSHOP_ENGINE_BASE_URL:-http://127.0.0.1:${WEBSHOP_SERVER_PORT}}

# --- engine pool --------------------------------------------------------------
# Engine-hosting worker processes, each its own SimServer + Lucene/JVM + spaCy
# (memory = K x one engine). 0 = single shared engine behind a lock (test /
# fallback mode).
WEBSHOP_ENGINE_WORKERS=${WEBSHOP_ENGINE_WORKERS:-8}
# Per-IPC-op budget (seconds); a worker exceeding it is SIGKILLed and respawned,
# its resident episodes 410 on their next op.
WEBSHOP_ENGINE_CALL_TIMEOUT=${WEBSHOP_ENGINE_CALL_TIMEOUT:-120}
# Engine-side idle eviction (seconds) for sessions abandoned by aborted episodes.
WEBSHOP_ENGINE_SESSION_TTL=${WEBSHOP_ENGINE_SESSION_TTL:-2400}
# 1 = full catalog (items_shuffle.json; the index setup.sh built must match),
# 0 = the 1000-item small catalog (items_*_1000.json).
WEBSHOP_FULL=${WEBSHOP_FULL:-0}

# --- protocol (docs/webshop_integration.md "TCOD protocol"): the split is the
# mainstream WebShop held-out layout; rendering / feedback / prompt / eval
# sampling below are TCOD-aligned. ---
# Which split layout recipes/webshop/prepare_data.py emits and recipes/webshop/
# run.py evaluates: canonical (WebShop paper: test = idx 0..499, fixed and
# never trained, plus a 256-goal pool sample; train = the rest) or tcod (TCOD's
# layout: train = goal idx 0..4095, test = 4096..4195 + a 400-goal held-out
# companion; its train range contains the canonical test set).
WEBSHOP_PROTOCOL=${WEBSHOP_PROTOCOL:-canonical}
# Goal-shuffle seed defining goal index -> goal (engine --goal-seed). 233 is
# the ORIGINAL WebShop constant TCOD inherits; 0 was the pre-TCOD value of
# this repo (wstrain1/wstrain2).
WEBSHOP_GOAL_SEED=${WEBSHOP_GOAL_SEED:-233}
# 1 = human-written instructions (items_human_ins.json; WebShop paper goal set,
# needs WEBSHOP_FULL=1: the 1000-item catalog carries 13 of them), 0 = synthetic
# goals (6910 on the small catalog). TCOD ran the small catalog; its code says
# human_goals=True, which cannot serve task ids up to 4195 there, so the goal
# source it actually used is unknown (docs/webshop_integration.md "TCOD protocol").
WEBSHOP_HUMAN_GOALS=${WEBSHOP_HUMAN_GOALS:-0}
# Page rendering of the vendored env, passed to BOTH tiers (engine
# --observation-mode, adapter observation_mode): text_rich (TCOD / original
# baselines) or text (verl-agent [SEP] form).
WEBSHOP_OBS_MODE=${WEBSHOP_OBS_MODE:-text_rich}
# Adapter reply to a non-admissible command: error (TCOD: the turn is spent,
# the env is NOT stepped, the model reads the error text) or noop (forward it;
# WebShop no-ops it and the unchanged page is the feedback — wstrain1/2).
WEBSHOP_INVALID_FEEDBACK=${WEBSHOP_INVALID_FEEDBACK:-error}
# The system prompt is a job-level knob: PROMPT=<name> picks prompts/webshop/<name>.py
# (think = TCOD, reason in <think></think> before the tool call; none = tool call only).

# --- runtime layout (built once by setup.sh) ----------------------------------
# Root of the self-contained py3.10 runtime: under the data root when one is
# set (scripts/fetch_data.sh stages the catalog there), else /data/webshop.
WEBSHOP_HOME=${WEBSHOP_HOME:-${MILES_AGENT_DATA:+${MILES_AGENT_DATA}/webshop}}
WEBSHOP_HOME=${WEBSHOP_HOME:-/data/webshop}
# conda PREFIX env (py3.10 + JDK 11 + the pins): its shebangs bake this path,
# which resolves unchanged inside the container when the data root is mounted at the same path.
WEBSHOP_PYTHON=${WEBSHOP_PYTHON:-${WEBSHOP_HOME}/env/bin/python}
# JDK for pyserini's JVM. conda-forge openjdk puts the JDK under
# <prefix>/lib/jvm (bin/java is symlinked into <prefix>/bin, libjvm.so lives at
# lib/jvm/lib/server/libjvm.so); pyjnius resolves libjvm.so from JAVA_HOME, so
# JAVA_HOME must be the JDK dir itself, not the conda prefix.
WEBSHOP_JAVA_HOME=${WEBSHOP_JAVA_HOME:-${WEBSHOP_HOME}/env/lib/jvm}
# Writable copy of the committed fork code (env_servers/webshop/webshop), with
# the catalog and the built Lucene index inside it.
WEBSHOP_VENDOR=${WEBSHOP_VENDOR:-${WEBSHOP_HOME}/webshop}
# Directory holding items_*.json (server.py --data-dir).
WEBSHOP_DATA_DIR=${WEBSHOP_DATA_DIR:-${WEBSHOP_VENDOR}/data}
