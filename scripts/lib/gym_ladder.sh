#!/bin/bash
# scripts/lib/gym_ladder.sh — the recipe ladder: the ONE place that knows the per-recipe bits of a run (sourced after
# scripts/lib/common.sh; functions only until recipe_vars runs; no scheduler call, no site path). Everything
# recipe-specific is a variable or a function chosen by recipe_vars; the generic steps of an entry (data prep, the venv
# probe, the pre-ray / pre-gym services, the gym start, the driver line) only consume them. Adding a recipe = one arm.
#
# recipe_vars <recipe>        alfworld | scienceworld | webshop | search_r1 | tmax (anything else is a FATAL); needs
#                             run_dirs <tag> first (RUN_DATA_DIR). Sets
#   POLICY_MODEL POLICY_HF TORCH_DIST_DIR   Qwen3.5-4B under $MILES_AGENT_MODELS (POLICY_HF overrides the dir; its
#                             config.json must exist: a FATAL, a WARN under --dry-run); $MILES_TORCH_DIST_DIR
#   PROMPT_DATA / EVAL_DATA   the miles rows (built + validated by RECIPE_PREPARE_DATA) under $RUN_DATA_DIR
#                             (search_r1: under $MILES_AGENT_DATA; tmax: PROMPT_DATA must be preset, no EVAL_DATA)
#   GYMCFG                    the nemo-gym config under gym_root/resources_servers/<env>/configs/ (tmax: none)
#   AGENT_PORT_OVERRIDE       pins the agent server to :$NEMO_GYM_PORT (8200)
#   GYM_EXTRA_OVERRIDES       further +key=value gym CLI overrides (a string, word-split on purpose; values carry no
#                             spaces): scienceworld's jar / JDK / workers, webshop's engine URL / rendering / feedback,
#                             search_r1's retriever_base_url=$RETRIEVER_URL (the config file is never edited)
#   DRIVER_DATA_ARGS          `--prompt-data P --eval-data E` (tmax: `--prompt-data P`)
#   RECIPE_RUN_ARGS           tmax only: `--run-tag --eval-prompt-data --nemo-gym-url --session-server-external-host`
#   RECIPE_PREPARE_DATA       function: build + validate the data files
#   RECIPE_PROBE              function: import-probe the prebuilt gym venvs it will start (`:` without GYM_VENV_ROOT)
#   RECIPE_PRE_RAY            function: services needed before ray comes up (search_r1: the retriever probe)
#   RECIPE_PRE_GYM            function: services the gym servers dial (webshop: the engine; search_r1: the data
#                             links; scienceworld: the jar + JDK the worker JVMs run on)
#   and the environment knobs, new name first, then the old alias, then the default:
#   ALFWORLD_DATA             $MILES_AGENT_DATA/alfworld (exported: the alfworld_env server resolves game files under it)
#   SCIENCEWORLD_JAR          SCIW_JAR, then $MILES_AGENT_DATA/scienceworld/scienceworld.jar when that file exists,
#                             else "" = the scienceworld wheel's bundled jar
#   SCIENCEWORLD_JAVA_HOME    SCIW_JAVA_HOME, JAVA_HOME, then $TOOLS_DIR/jdk-17 when TOOLS_DIR is set and that dir
#                             exists, else "" = the java on PATH (scripts/lib/common.sh scienceworld_java_home: the
#                             one order, shared with scripts/fetch_data.sh and scripts/build_gym_venvs.sh)
#   SCIENCEWORLD_ENV_WORKERS  SCIW_ENV_WORKERS, then 128 (worker JVMs = concurrent episodes)
#   WEBSHOP_HOME              $MILES_AGENT_DATA/webshop, then env_servers/webshop/defaults.sh derives the rest
#                             (WEBSHOP_SERVER_PORT 6200 and every other WEBSHOP_* knob stay overridable there)
#   WIKI_SEARCH_DATA          $MILES_AGENT_DATA/wiki_search (scripts/fetch_data.sh search_r1 writes it)
#   RETRIEVER_URL             search_r1: required, the external retriever (http://<host>:<port>)
#   NEMO_GYM_URL              tmax: required, the external agent server (the run starts no gym)
# prepare_data_<env>          the data builders (DRY_RUN prints `DRY_RUN prepare: <argv>`)
# venv_probe_gate             GYM_VENV_PROBE_OK after RECIPE_PROBE + the relay probe under $GYM_VENV_ROOT/gym_venvs/;
#                             `GYM_VENV_PROBE_OK mode=uv_run` without a root (nothing to probe: `uv run` resolves),
#                             `GYM_VENV_PROBE_OK mode=no_gym` for a recipe that starts no gym (tmax)
# gym_argv / gym_env / gym_cwd
#                             the gym start, one argv element / KEY=VALUE per line, from $REPO/3rdparty/Gym: with
#                             GYM_VENV_ROOT `$GYM_VENV_ROOT/gym_main_venv/bin/gym` + the three prebuilt-venv flags +
#                             UV_OFFLINE=1, else `$UV_BIN run gym`; both with head 127.0.0.1:8100, the agent on
#                             :$NEMO_GYM_PORT, siblings 8300-8999
# render_env_yaml             gym_root/env.yaml and 3rdparty/Gym/env.yaml from env.yaml.example with a dummy
#                             endpoint (every /run carries its own policy_base_url); skipped under --dry-run
# gym_start_and_wait          the background start (log $RUNS_DIR/<RUN_TAG>_gym_<JOB_ID>.log, GYM_PID = the gym
#                             process itself, killed by the cleanup trap), the /readyz + /openapi.json gate (84 x 5 s),
#                             `GYM_UP agent=http://127.0.0.1:$NEMO_GYM_PORT`, export NEMO_GYM_URL; under --dry-run
#                             `DRY_RUN gym: <argv>`, `DRY_RUN env: KEY=VALUE` lines and `DRY_RUN cwd: <dir>` only
# pre_ray_search_r1           the retriever probe against $RETRIEVER_URL/retrieve, RETRIEVER_BRIDGE_OK
# pre_gym_webshop             the engine server, its boot gate, the protocol and goal-set gates, WEBSHOP_ENGINE_UP
# pre_gym_scienceworld        the jar / JDK checks, JAVA_HOME + PATH for the worker JVMs, SCIW_RUNTIME
# pre_gym_search_r1           links $WIKI_SEARCH_DATA/{train,validation}.jsonl into the wiki_search data dir
# Under the argv-only dry run (DRY_RUN=1) every function prints its argv and starts nothing; existence checks are WARNs.

# the argv, or its DRY_RUN line under the argv-only depth
run_or_dry() {  # <label> <argv...>
  local label=$1
  shift
  if is_dry_run; then dry_run_print "$label" "$@"; return 0; fi
  "$@"
}

sha256_16() {  # <file>: the first 16 hex digits of its sha256
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-16
}

default_scienceworld_jar() {
  local jar=$MILES_AGENT_DATA/scienceworld/scienceworld.jar
  [ -f "$jar" ] && printf '%s\n' "$jar"
  return 0
}

count_eval_rows() {  # sets EV = the row count of $EVAL_DATA; a FATAL when the converter wrote no such file (a bare
  # `wc -l <` on a missing file ends the caller with a redirection error under set -e, without a FATAL line)
  [ -f "$EVAL_DATA" ] || fatal "prepare wrote no eval file $EVAL_DATA"
  EV=$(wc -l < "$EVAL_DATA" | tr -d " ")
}

append_override() {  # <+key=value>
  GYM_EXTRA_OVERRIDES="${GYM_EXTRA_OVERRIDES:+$GYM_EXTRA_OVERRIDES }$1"
}

recipe_vars() {  # <recipe>
  RECIPE=${1:-${RECIPE:-}}
  [ -n "$RECIPE" ] || fatal "recipe_vars: a recipe is required (alfworld | scienceworld | webshop | search_r1 | tmax)"
  [ -n "${RUN_DATA_DIR:-}" ] || fatal "recipe_vars: run_dirs <tag> must run first (RUN_DATA_DIR is unset)"
  # ---- policy model ----
  # POLICY_MODEL (default Qwen3.5-4B) is a full HF dir under $MILES_AGENT_MODELS (POLICY_HF overrides the path);
  # run.py's prepare converts its torch_dist twin under $TORCH_DIST_DIR when missing. MILES_TORCH_DIST_DIR defaults
  # to a dir of its own: miles >= 0.1.2 redoes an older conversion IN PLACE (stamp miles_conversion_version.txt,
  # layout 5), and its Qwen3.5 GDN layout is unreadable by the older pins whose campaigns load the plain twins.
  POLICY_MODEL=${POLICY_MODEL:-Qwen3.5-4B}
  POLICY_HF=${POLICY_HF:-$MILES_AGENT_MODELS/$POLICY_MODEL}
  if [ ! -f "$POLICY_HF/config.json" ]; then
    if is_dry_run; then echo "WARN: no policy model HF dir $POLICY_HF (config.json missing)"
    else echo "FATAL: no policy model HF dir $POLICY_HF"; exit 1; fi
  fi
  TORCH_DIST_DIR=$MILES_TORCH_DIST_DIR
  # ---- the ladder ----
  GYM_EXTRA_OVERRIDES=""
  RECIPE_RUN_ARGS=""
  case "$RECIPE" in
    alfworld)
      # exported: the alfworld_env server resolves metadata.game_file under it
      export ALFWORLD_DATA=${ALFWORLD_DATA:-$MILES_AGENT_DATA/alfworld}
      PROMPT_DATA=$RUN_DATA_DIR/alfworld_train.jsonl
      EVAL_DATA=$RUN_DATA_DIR/alfworld_eval274.jsonl
      GYMCFG=$REPO/gym_root/resources_servers/alfworld_env/configs/alfworld.yaml
      AGENT_PORT_OVERRIDE="+alfworld_agent.responses_api_agents.env_loop_agent.port=$NEMO_GYM_PORT"
      RECIPE_PREPARE_DATA=prepare_data_alfworld
      RECIPE_PROBE=probe_gym_venvs_alfworld
      RECIPE_PRE_RAY=:
      RECIPE_PRE_GYM=:
      ;;
    scienceworld)
      # ScienceWorld env stack (gym_root/resources_servers/scienceworld_env); rows from
      # recipes/scienceworld/prepare_data.py (prompts/scienceworld/<PROMPT>.py). The simulator jar and the JDK the
      # worker JVMs run on: the new names first, then the SCIW_* aliases, then the data root / TOOLS_DIR, else the
      # wheel's bundled jar and the java on PATH (the yaml's own "" defaults; docs/scienceworld_integration.md).
      SCIENCEWORLD_JAR=${SCIENCEWORLD_JAR:-${SCIW_JAR:-$(default_scienceworld_jar)}}
      SCIENCEWORLD_JAVA_HOME=$(scienceworld_java_home)
      SCIENCEWORLD_ENV_WORKERS=${SCIENCEWORLD_ENV_WORKERS:-${SCIW_ENV_WORKERS:-128}}
      PROMPT_DATA=$RUN_DATA_DIR/scienceworld_train.jsonl
      EVAL_DATA=$RUN_DATA_DIR/scienceworld_eval274.jsonl
      GYMCFG=$REPO/gym_root/resources_servers/scienceworld_env/configs/scienceworld.yaml
      AGENT_PORT_OVERRIDE="+scienceworld_agent.responses_api_agents.env_loop_agent.port=$NEMO_GYM_PORT"
      # Pinned jar + JDK for the worker JVMs; an empty value keeps the yaml default (the wheel jar, the PATH java),
      # so its override is left out rather than passed empty.
      [ -z "$SCIENCEWORLD_JAR" ] || append_override "+scienceworld_env.resources_servers.scienceworld_env.jar_path=$SCIENCEWORLD_JAR"
      [ -z "$SCIENCEWORLD_JAVA_HOME" ] || append_override "+scienceworld_env.resources_servers.scienceworld_env.java_home=$SCIENCEWORLD_JAVA_HOME"
      append_override "+scienceworld_env.resources_servers.scienceworld_env.env_workers=$SCIENCEWORLD_ENV_WORKERS"
      RECIPE_PREPARE_DATA=prepare_data_scienceworld
      RECIPE_PROBE=probe_gym_venvs_scienceworld
      RECIPE_PRE_RAY=:
      RECIPE_PRE_GYM=pre_gym_scienceworld
      ;;
    webshop)
      # WebShop's shell defaults (host/port/workers/timeouts/layout) live ONLY in env_servers/webshop/defaults.sh,
      # which keeps a value already present: the runtime root is set first so that it hangs under the data root.
      WEBSHOP_HOME=${WEBSHOP_HOME:-$MILES_AGENT_DATA/webshop}
      . "$REPO/env_servers/webshop/defaults.sh"
      # Per run and protocol: a tcod run never picks up canonical rows.
      PROMPT_DATA=$RUN_DATA_DIR/webshop_$WEBSHOP_PROTOCOL/webshop_train.jsonl
      EVAL_DATA=$RUN_DATA_DIR/webshop_$WEBSHOP_PROTOCOL/webshop_eval.jsonl
      GYMCFG=$REPO/gym_root/resources_servers/webshop_env/configs/webshop.yaml
      AGENT_PORT_OVERRIDE="+webshop_agent.responses_api_agents.env_loop_agent.port=$NEMO_GYM_PORT"
      # The adapter dials the engine at the URL defaults.sh derives from WEBSHOP_SERVER_PORT, and must render
      # pages / answer invalid commands the way the engine and the protocol expect; override the yaml so one knob
      # moves both tiers.
      append_override "+webshop_env.resources_servers.webshop_env.engine_base_url=$WEBSHOP_ENGINE_BASE_URL"
      append_override "+webshop_env.resources_servers.webshop_env.observation_mode=$WEBSHOP_OBS_MODE"
      append_override "+webshop_env.resources_servers.webshop_env.invalid_action_feedback=$WEBSHOP_INVALID_FEEDBACK"
      RECIPE_PREPARE_DATA=prepare_data_webshop
      RECIPE_PROBE=probe_gym_venvs_webshop
      RECIPE_PRE_RAY=:
      RECIPE_PRE_GYM=pre_gym_webshop
      ;;
    search_r1)
      WIKI_SEARCH_DATA=${WIKI_SEARCH_DATA:-$MILES_AGENT_DATA/wiki_search}
      [ -n "${RETRIEVER_URL:-}" ] || fatal "RECIPE=search_r1 needs RETRIEVER_URL: the retriever is external: RETRIEVER_URL=http://<host>:<port> (a FlashRAG-style /retrieve endpoint)"
      PROMPT_DATA=$MILES_AGENT_DATA/search_r1_train.jsonl
      EVAL_DATA=$MILES_AGENT_DATA/search_r1_eval893.jsonl
      GYMCFG=$REPO/gym_root/resources_servers/wiki_search/configs/wiki_search.yaml
      AGENT_PORT_OVERRIDE="+wiki_search_agent.responses_api_agents.tool_call_agent.port=$NEMO_GYM_PORT"
      # the wiki_search server dials the retriever named here (the config's 127.0.0.1:8000 is a local default); a
      # trailing slash is dropped: the server appends its own paths to the base
      append_override "+wiki_search.resources_servers.wiki_search.retriever_base_url=${RETRIEVER_URL%/}"
      RECIPE_PREPARE_DATA=prepare_data_search_r1
      RECIPE_PROBE=probe_gym_venvs_search_r1
      RECIPE_PRE_RAY=pre_ray_search_r1
      RECIPE_PRE_GYM=pre_gym_search_r1
      ;;
    tmax)
      # No gym ladder: the agent server is external (NEMO_GYM_URL), the rows come from recipes/tmax/prepare.py, and
      # recipes/tmax/run.py takes the run tag and the URLs itself (it is not on BaseScriptArgs yet).
      [ -n "${NEMO_GYM_URL:-}" ] || fatal "RECIPE=tmax needs NEMO_GYM_URL=http://<env host>:8200: the agent server is external (this run starts no gym)"
      [ -n "${PROMPT_DATA:-}" ] || fatal "RECIPE=tmax needs PROMPT_DATA=<rows.jsonl> (recipes/tmax/prepare.py writes them)"
      [ -n "${RUN_TAG:-}" ] || fatal "RECIPE=tmax needs RUN_TAG"
      EVAL_DATA=""
      GYMCFG=""
      AGENT_PORT_OVERRIDE=""
      RECIPE_PREPARE_DATA=:
      RECIPE_PROBE=:
      RECIPE_PRE_RAY=:
      RECIPE_PRE_GYM=:
      # EVAL_PROMPT_DATA is a PATH: run.py renders `--eval-prompt-data tmax <path>` and adds the `tmax` name itself
      RECIPE_RUN_ARGS="--run-tag $RUN_TAG ${EVAL_PROMPT_DATA:+--eval-prompt-data $EVAL_PROMPT_DATA} --nemo-gym-url $NEMO_GYM_URL ${SESSION_SERVER_EXTERNAL_HOST:+--session-server-external-host $SESSION_SERVER_EXTERNAL_HOST}"
      ;;
    *) echo "FATAL: unknown RECIPE=$RECIPE (alfworld | scienceworld | webshop | search_r1 | tmax)"; exit 1 ;;
  esac
  if [ "$RECIPE" = tmax ]; then
    DRIVER_DATA_ARGS="--prompt-data $PROMPT_DATA"
  else
    DRIVER_DATA_ARGS="--prompt-data $PROMPT_DATA --eval-data $EVAL_DATA"
  fi
  # Without a prebuilt venv root `uv run` resolves every server's environment itself: nothing to probe.
  [ -n "${GYM_VENV_ROOT:-}" ] || RECIPE_PROBE=:
}

# ---- data prep (idempotent) ----
prepare_data_alfworld() {
  # exported: the alfworld_env server resolves metadata.game_file under it
  export ALFWORLD_DATA=${ALFWORLD_DATA:-$MILES_AGENT_DATA/alfworld}
  if [ ! -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    if is_dry_run; then echo "WARN: no $ALFWORLD_DATA/json_2.1.1 (scripts/fetch_data.sh alfworld stages the ALFWorld game files)"
    else echo "FATAL: no $ALFWORLD_DATA/json_2.1.1 — stage the ALFWorld game files first (scripts/fetch_data.sh alfworld)"; exit 1; fi
  fi
  # Rebuilt every launch (a walk over ~3.8k game files): the rows carry the
  # policy model's own tool-call markup. PROMPT=<name> picks prompts/alfworld/<name>.py.
  run_or_dry prepare "$PYTHON" "$REPO/recipes/alfworld/prepare_data.py" \
    --data-root "$ALFWORLD_DATA" --output-dir "$(dirname "$PROMPT_DATA")" --model "$POLICY_HF" \
    ${PROMPT:+--prompt "$PROMPT"}
  if is_dry_run; then return 0; fi
  count_eval_rows
  [ "$EV" -eq 274 ] || { echo "FATAL: eval set has $EV rows, expected 274 (seen-then-unseen order!)"; exit 1; }
}
prepare_data_scienceworld() {
  # Rebuilt every launch (stdlib + the env server's task_splits.py; the frozen
  # TCOD test taskset is the list scripts/fetch_data.sh scienceworld writes,
  # SCIENCEWORLD_TEST_LIST else the data root): the rows carry the policy
  # model's own tool-call markup. PROMPT=<name> picks prompts/scienceworld/<name>.py.
  run_or_dry prepare "$PYTHON" "$REPO/recipes/scienceworld/prepare_data.py" \
    --output-dir "$(dirname "$PROMPT_DATA")" --model "$POLICY_HF" ${PROMPT:+--prompt "$PROMPT"}
  if is_dry_run; then return 0; fi
  count_eval_rows
  [ "$EV" -eq 274 ] || { echo "FATAL: eval set has $EV rows, expected 274 (140 valid_seen then 134 valid_unseen!)"; exit 1; }
}
prepare_data_webshop() {
  # Rows address goals by CONTENT: metadata.goal_id = sha1 of the goal's scoring
  # fields (gym_root/agent_utils/webshop_goals.py), taken IN FILE ORDER from the
  # goals file scripts/fetch_data.sh webshop --regenerate-goals writes (seed-233 engine
  # order at export time). Positions in that FILE define the splits, so the
  # dataset depends on no runtime knob; the engine resolves goal_id -> goal on
  # /reset and refuses ids it does not know.
  # Layout per WEBSHOP_PROTOCOL (defaults.sh; recipes/webshop/run.py reads the
  # same variable for its eval slices):
  #   tcod      train = pos 0..4095; eval = 4096..4195 (TCOD test, 100 rows)
  #             + 4196..4595 (held-out companion, 400 rows)      -> 500 rows
  #   canonical eval = pos 0..499 + 256 pool-sampled goals; train = 500..
  #                                                          -> 756 rows
  # ALWAYS regenerated (a few seconds: the policy model's tokenizer renders the example
  # call): a file left by another protocol / prompt
  # variant / policy model must never be reused. prepare_data.py verifies every stored
  # goal_id against its content and writes protocol.json.goals_digest, which
  # pre_gym_webshop gates against the engine's /health.goals_digest.
  local want
  case "$WEBSHOP_PROTOCOL" in
    tcod) want=500 ;;
    canonical) want=756 ;;
    *) echo "FATAL: WEBSHOP_PROTOCOL=$WEBSHOP_PROTOCOL is not tcod|canonical"; exit 1 ;;
  esac
  # PROMPT=<name> picks prompts/webshop/<name>.py (default think = TCOD).
  run_or_dry prepare "$PYTHON" "$REPO/recipes/webshop/prepare_data.py" --output-dir "$(dirname "$PROMPT_DATA")" \
    --protocol "$WEBSHOP_PROTOCOL" --model "$POLICY_HF" ${PROMPT:+--prompt "$PROMPT"}
  if is_dry_run; then return 0; fi
  count_eval_rows
  [ "$EV" -eq "$want" ] || { echo "FATAL: eval set has $EV rows, expected $want for protocol $WEBSHOP_PROTOCOL"; exit 1; }
  cat "$(dirname "$PROMPT_DATA")/protocol.json"
}
prepare_data_search_r1() {
  # miles rows from the gym rows scripts/fetch_data.sh search_r1 wrote under $WIKI_SEARCH_DATA
  local split
  if is_dry_run; then
    dry_run_print prepare "$PYTHON" "$REPO/recipes/search_r1/prepare_data.py" \
      --input "$WIKI_SEARCH_DATA/train.jsonl" --output "$PROMPT_DATA"
    dry_run_print prepare "$PYTHON" "$REPO/recipes/search_r1/prepare_data.py" \
      --input "$WIKI_SEARCH_DATA/validation.jsonl" --output "$EVAL_DATA"
    return 0
  fi
  if [ ! -f "$PROMPT_DATA" ] || [ ! -f "$EVAL_DATA" ]; then
    for split in train validation; do
      [ -f "$WIKI_SEARCH_DATA/$split.jsonl" ] \
        || fatal "no $WIKI_SEARCH_DATA/$split.jsonl: run scripts/fetch_data.sh search_r1 (WIKI_SEARCH_DATA=$WIKI_SEARCH_DATA)"
    done
    "$PYTHON" "$REPO/recipes/search_r1/prepare_data.py" \
      --input "$WIKI_SEARCH_DATA/train.jsonl" --output "$PROMPT_DATA"
    "$PYTHON" "$REPO/recipes/search_r1/prepare_data.py" \
      --input "$WIKI_SEARCH_DATA/validation.jsonl" --output "$EVAL_DATA"
  fi
  count_eval_rows
  [ "$EV" -eq 893 ] || { echo "FATAL: eval set has $EV rows, expected 893 (source-block order!)"; exit 1; }
}

# ---- the prebuilt gym venvs <-> checkout compatibility gate ----
# The gym venvs are built once (scripts/build_gym_venvs.sh <root>); a set built from an older checkout (e.g. before
# agent_utils existed) makes gym die at boot with a bare ImportError long after ray came up. Probe the venvs this
# recipe will actually start (both probes run so the log shows every stale venv).
probe_venv() {  # <type> <name> <imports>
  local py=$GYM_VENV_ROOT/gym_venvs/$1/$2/.venv/bin/python
  if is_dry_run; then dry_run_print probe "$py" -c "import $3"; return 0; fi
  "$py" -c "import $3" 2>/dev/null
}
probe_gym_venvs_alfworld() {
  local ok=1
  probe_venv responses_api_agents env_loop_agent "agent_utils.chat_dialect, nemo_gym" || ok=0
  probe_venv resources_servers alfworld_env "alfworld_env.app, alfworld, textworld" || ok=0
  [ "$ok" -eq 1 ]
}
probe_gym_venvs_scienceworld() {
  local ok=1
  probe_venv responses_api_agents env_loop_agent "agent_utils.chat_dialect, nemo_gym" || ok=0
  # The simulator wheel (scienceworld) and the py4j gateway it drives live in
  # this venv; the jar and the JDK come from SCIENCEWORLD_JAR / SCIENCEWORLD_JAVA_HOME (pre_gym_scienceworld).
  probe_venv resources_servers scienceworld_env "scienceworld_env.app, scienceworld, py4j" || ok=0
  [ "$ok" -eq 1 ]
}
probe_gym_venvs_webshop() {
  local ok=1
  probe_venv responses_api_agents env_loop_agent "agent_utils.chat_dialect, nemo_gym" || ok=0
  # The webshop_env adapter is a THIN py3.13 proxy — it must import with only
  # nemo-gym + httpx (no pyserini/JVM); those live in the py3.10 engine env.
  probe_venv resources_servers webshop_env "webshop_env.app" || ok=0
  [ "$ok" -eq 1 ]
}
probe_gym_venvs_search_r1() {
  local ok=1
  probe_venv responses_api_agents tool_call_agent "agent_utils.chat_dialect, nemo_gym" || ok=0
  probe_venv resources_servers wiki_search "wiki_search.app" || ok=0
  [ "$ok" -eq 1 ]
}
# Every recipe: gym >= 0.7.0 fronts each agent with a generated legacy_agent
# relay environment server, which boots from its own prebuilt venv.
probe_gym_relay_venv() {
  probe_venv environment_servers legacy_agent "nemo_gym.base_environment_server, httptools"
}
venv_probe_gate() {
  if [ -z "${GYMCFG:-}" ]; then  # tmax: this run starts no gym, so there is no venv to probe
    if is_dry_run; then dry_run_print probe; return 0; fi
    echo "GYM_VENV_PROBE_OK mode=no_gym"
    return 0
  fi
  if [ -z "${GYM_VENV_ROOT:-}" ]; then
    if is_dry_run; then dry_run_print probe; return 0; fi
    echo "GYM_VENV_PROBE_OK mode=uv_run"
    return 0
  fi
  if is_dry_run; then "$RECIPE_PROBE"; probe_gym_relay_venv; return 0; fi
  # Fail fast with the fix spelled out (see the probe functions above).
  if ! "$RECIPE_PROBE" || ! probe_gym_relay_venv; then
    echo "FATAL: gym venv probe failed for RECIPE=$RECIPE — the prebuilt gym venvs under $GYM_VENV_ROOT are stale for this checkout."
    echo "Rebuild them with scripts/build_gym_venvs.sh $GYM_VENV_ROOT, then rerun."
    exit 1
  fi
  echo "GYM_VENV_PROBE_OK"
}

# ---- pre-ray services (search_r1: the retriever must answer before anything starts) ----
# Function body kept unindented: the probe heredoc terminator must stay at column 0.
pre_ray_search_r1() {
[ -n "${RETRIEVER_URL:-}" ] || fatal "RECIPE=search_r1 needs RETRIEVER_URL=http://<host>:<port> (the external retriever)"
# HARD GATE: a real query must return document contents before anything
# starts — a dead retriever otherwise degrades into a silent no-retrieval run
export MILES_PROBE_URL="${RETRIEVER_URL%/}/retrieve"
if is_dry_run; then dry_run_print retriever_probe "$PYTHON" -; return 0; fi
"$PYTHON" - <<'PYEOF' || { echo "FATAL: retriever probe failed: no document contents from $MILES_PROBE_URL (RETRIEVER_URL=$RETRIEVER_URL)"; exit 1; }
import json, os, sys, time, urllib.request
url = os.environ["MILES_PROBE_URL"]
payload = json.dumps({"queries": ["capital of France"], "topk": 1, "return_scores": True}).encode()
deadline = time.time() + 120
while time.time() < deadline:
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        body = urllib.request.urlopen(req, timeout=10).read().decode()
        if "contents" in body:
            print("probe ok:", body[:120]); sys.exit(0)
        print("probe: unexpected body:", body[:200])
    except Exception as e:
        print("probe retry:", e)
    time.sleep(5)
sys.exit(1)
PYEOF
echo "RETRIEVER_BRIDGE_OK"
}

# ---- pre-gym services ----
pre_gym_search_r1() {
  # link the corpus rows into the gym's data dir (the config file is not edited: the retriever URL travels as the
  # +override recipe_vars put into GYM_EXTRA_OVERRIDES)
  local data=$REPO/gym_root/resources_servers/wiki_search/data split
  for split in train validation; do
    [ -e "$data/$split.jsonl" ] && continue
    if [ ! -f "$WIKI_SEARCH_DATA/$split.jsonl" ]; then
      if is_dry_run; then echo "WARN: no $WIKI_SEARCH_DATA/$split.jsonl (scripts/fetch_data.sh search_r1 writes it)"; continue; fi
      fatal "no $WIKI_SEARCH_DATA/$split.jsonl: run scripts/fetch_data.sh search_r1 (WIKI_SEARCH_DATA=$WIKI_SEARCH_DATA)"
    fi
    if is_dry_run; then dry_run_print link ln -s "$WIKI_SEARCH_DATA/$split.jsonl" "$data/$split.jsonl"; continue; fi
    ln -s "$WIKI_SEARCH_DATA/$split.jsonl" "$data/$split.jsonl" \
      || fatal "cannot link $WIKI_SEARCH_DATA/$split.jsonl into $data (a stale link there? scripts/fetch_data.sh search_r1 replaces it)"
  done
}

# WebShop py3.10 engine server. The heavy engine (catalog + Lucene/JVM search +
# spaCy reward) cannot share the py3.13 gym venv (its pyserini/spaCy/gym pins
# have no py3.13 wheels), so it runs as a separate loopback HTTP server that the
# py3.13 webshop_env adapter proxies to (the wiki_search-over-retriever
# pattern). It hosts K engine workers, each its own SimServer + JVM, hosting
# many lightweight per-episode envs: K-way parallel search under the rollout
# wave (env_servers/webshop). The whole py3.10 runtime is self-contained under
# $WEBSHOP_HOME, built once by env_servers/webshop/setup.sh — no dependency on
# any other repo. Must be up and healthy BEFORE the adapter boots.
pre_gym_webshop() {
  # defaults.sh (sourced by recipe_vars) resolved every knob; export them so
  # launch_engine.sh (which re-sources defaults.sh) and run.py (WEBSHOP_PROTOCOL,
  # WEBSHOP_* reward knobs) see the same values.
  export WEBSHOP_HOME WEBSHOP_PYTHON WEBSHOP_JAVA_HOME WEBSHOP_VENDOR WEBSHOP_DATA_DIR WEBSHOP_FULL \
         WEBSHOP_SERVER_HOST WEBSHOP_SERVER_PORT WEBSHOP_ENGINE_WORKERS \
         WEBSHOP_ENGINE_CALL_TIMEOUT WEBSHOP_ENGINE_SESSION_TTL \
         WEBSHOP_PROTOCOL WEBSHOP_GOAL_SEED WEBSHOP_HUMAN_GOALS WEBSHOP_OBS_MODE \
         WEBSHOP_INVALID_FEEDBACK
  local catalog=items_shuffle_1000.json
  [ "$WEBSHOP_FULL" = "1" ] && catalog=items_shuffle.json
  # Human goals exist for 13 of the 1000 small-catalog products: the WebShop
  # paper goal set needs the full catalog (items_shuffle.json + items_ins_v2.json).
  if [ "$WEBSHOP_HUMAN_GOALS" = "1" ] && [ "$WEBSHOP_FULL" != "1" ]; then
    echo "FATAL: WEBSHOP_HUMAN_GOALS=1 needs WEBSHOP_FULL=1 (the 1000-item catalog carries 13 human goals)"; exit 1
  fi
  if is_dry_run; then
    [ -x "$WEBSHOP_PYTHON" ] || echo "WARN: no webshop py3.10 interpreter at $WEBSHOP_PYTHON (env_servers/webshop/setup.sh, WEBSHOP_HOME=$WEBSHOP_HOME)"
    [ -d "$WEBSHOP_VENDOR/web_agent_site" ] || echo "WARN: no vendored WebShop tree at $WEBSHOP_VENDOR"
    [ -f "$WEBSHOP_DATA_DIR/$catalog" ] || echo "WARN: no catalog at $WEBSHOP_DATA_DIR ($catalog, WEBSHOP_FULL=$WEBSHOP_FULL)"
    dry_run_print webshop_engine bash "$REPO/env_servers/webshop/launch_engine.sh"
    return 0
  fi
  [ -x "$WEBSHOP_PYTHON" ] || { echo "FATAL: no webshop py3.10 interpreter at $WEBSHOP_PYTHON — build it once with env_servers/webshop/setup.sh (WEBSHOP_HOME=$WEBSHOP_HOME) or set WEBSHOP_PYTHON"; exit 1; }
  [ -d "$WEBSHOP_VENDOR/web_agent_site" ] || { echo "FATAL: no vendored WebShop tree at $WEBSHOP_VENDOR — run env_servers/webshop/setup.sh (WEBSHOP_HOME=$WEBSHOP_HOME) or set WEBSHOP_VENDOR"; exit 1; }
  [ -f "$WEBSHOP_DATA_DIR/$catalog" ] || { echo "FATAL: no catalog at $WEBSHOP_DATA_DIR ($catalog, WEBSHOP_FULL=$WEBSHOP_FULL) — run env_servers/webshop/setup.sh or set WEBSHOP_DATA_DIR"; exit 1; }
  mkdir -p "$RUNS_DIR"
  ENGINE_LOG=$RUNS_DIR/${RUN_TAG}_webshop_engine_${JOB_ID}.log
  ( exec bash "$REPO/env_servers/webshop/launch_engine.sh" > "$ENGINE_LOG" 2>&1 ) &
  ENGINE_PID=$!
  add_cleanup 'kill $ENGINE_PID 2>/dev/null'
  # Boot budget covers K engines building in parallel (each: catalog load +
  # Lucene open + spaCy + JVM): 10 min for the small catalog, 60 min for the
  # full one (a 5 GB JSON per worker). If the pool constructor raises, the
  # process exits and the pid check ends the wait early with the real cause
  # in the log.
  local health="http://127.0.0.1:$WEBSHOP_SERVER_PORT/health" boot_tries=120
  [ "$WEBSHOP_FULL" = "1" ] && boot_tries=720
  wait_http_ready "$health" "$ENGINE_PID" "$ENGINE_LOG" "$boot_tries" 5 \
    || { echo "FATAL: webshop engine server failed to boot on :$WEBSHOP_SERVER_PORT"; exit 1; }
  # PROTOCOL GATE: the engine must have been started with the seed / page
  # rendering / goal source this job's data and adapter assume (a stale engine
  # or a hand-edited launch would otherwise silently change goal identity).
  local want_catalog=small proto
  [ "$WEBSHOP_FULL" = "1" ] && want_catalog=full
  proto=$("$CURL" -sf "$health" | "$PYTHON" -c '
import json, sys
p = json.load(sys.stdin)["protocol"]
print(p["seed"], p["observation_mode"], int(bool(p["human_goals"])), p["catalog"])') \
    || { echo "FATAL: $health has no protocol block"; "$CURL" -s "$health"; exit 1; }
  [ "$proto" = "$WEBSHOP_GOAL_SEED $WEBSHOP_OBS_MODE $WEBSHOP_HUMAN_GOALS $want_catalog" ] \
    || { echo "FATAL: engine protocol ($proto) != job knobs (seed=$WEBSHOP_GOAL_SEED obs=$WEBSHOP_OBS_MODE human_goals=$WEBSHOP_HUMAN_GOALS catalog=$want_catalog)"; exit 1; }
  # GOAL-SET GATE: the data addresses goals by content id, so the engine must
  # have built exactly the goal set the frozen export was taken from. Any drift
  # (catalog file, vendored goal code, goal seed — the seed also draws the
  # price caps, so it changes goal CONTENT, not just order) would otherwise
  # surface as 400s on /reset for part of the rows, hours in. The order-free
  # SET digest is the gate (repeated content at several list positions is one
  # question on both sides); the counts are printed for the log only.
  local goalset engine_digest engine_goals engine_unique data_digest data_goals data_unique
  goalset=$("$CURL" -sf "$health" | "$PYTHON" -c 'import json,sys; h=json.load(sys.stdin); print(h["goals_digest"], h["goals"], h.get("unique_goals", h["goals"]))') \
    || { echo "FATAL: $health has no goals_digest / goals field (engine predates content-addressed goals?)"; "$CURL" -s "$health"; exit 1; }
  read -r engine_digest engine_goals engine_unique <<< "$goalset"
  goalset=$("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["goals_digest"], p["goals"], p.get("unique_goals", p["goals"]))' \
    "$(dirname "$PROMPT_DATA")/protocol.json") \
    || { echo "FATAL: $(dirname "$PROMPT_DATA")/protocol.json has no goals_digest / goals field"; exit 1; }
  read -r data_digest data_goals data_unique <<< "$goalset"
  [ "$engine_digest" = "$data_digest" ] \
    || { echo "FATAL: engine goal set (digest $engine_digest, $engine_goals rows / $engine_unique unique) != dataset (digest $data_digest, $data_goals rows / $data_unique unique): catalog / goal code / seed changed since the goals file was exported (scripts/fetch_data.sh webshop --regenerate-goals)"; exit 1; }
  echo "WEBSHOP_ENGINE_UP port=$WEBSHOP_SERVER_PORT workers=$WEBSHOP_ENGINE_WORKERS goals=$engine_goals unique_goals=$engine_unique goals_digest=$engine_digest protocol=$WEBSHOP_PROTOCOL seed=$WEBSHOP_GOAL_SEED obs=$WEBSHOP_OBS_MODE human_goals=$WEBSHOP_HUMAN_GOALS catalog=$want_catalog invalid_feedback=$WEBSHOP_INVALID_FEEDBACK prompt=${PROMPT:-think}"
}

# ScienceWorld runtime: the scienceworld + py4j wheels live in the scienceworld_env gym venv; the simulator jar and
# the JDK are SCIENCEWORLD_JAR / SCIENCEWORLD_JAVA_HOME (recipe_vars; empty = the wheel's bundled jar / the java on
# PATH). SCIENCEWORLD_ENV_WORKERS = worker JVMs = concurrent episodes; a 512-rollout wave queues on /reset's slot
# wait (docs/scienceworld_integration.md).
pre_gym_scienceworld() {
  local jar_desc sha java_desc
  if [ -n "$SCIENCEWORLD_JAR" ] && [ ! -f "$SCIENCEWORLD_JAR" ]; then
    if is_dry_run; then echo "WARN: missing ScienceWorld jar $SCIENCEWORLD_JAR (SCIENCEWORLD_JAR; empty = the wheel's bundled jar)"
    else echo "FATAL: missing ScienceWorld jar $SCIENCEWORLD_JAR — copy the pinned jar there (docs/scienceworld_integration.md) or set SCIENCEWORLD_JAR (empty = the wheel's bundled jar)"; exit 1; fi
  fi
  if [ -n "$SCIENCEWORLD_JAVA_HOME" ] && [ ! -x "$SCIENCEWORLD_JAVA_HOME/bin/java" ]; then
    if is_dry_run; then echo "WARN: missing JDK $SCIENCEWORLD_JAVA_HOME/bin/java (JAVA_HOME; empty = the java on PATH)"
    else echo "FATAL: missing JDK $SCIENCEWORLD_JAVA_HOME/bin/java — stage a JDK 17 there or set JAVA_HOME (empty = the java on PATH)"; exit 1; fi
  fi
  if is_dry_run; then return 0; fi
  if [ -n "$SCIENCEWORLD_JAVA_HOME" ]; then
    # The env server's worker processes put $JAVA_HOME/bin on their own PATH
    # (py4j execs `java`); exporting here too makes the gym log's `java -version`
    # line and any manual probe agree with what the workers run.
    export JAVA_HOME=$SCIENCEWORLD_JAVA_HOME
    export PATH=$SCIENCEWORLD_JAVA_HOME/bin:$PATH
    java_desc=$("$SCIENCEWORLD_JAVA_HOME/bin/java" -version 2>&1 | head -1 || true)
  else
    command -v java >/dev/null 2>&1 || { echo "FATAL: no java on PATH for the ScienceWorld worker JVMs — set JAVA_HOME to a JDK 17 or put java on PATH"; exit 1; }
    java_desc=$(java -version 2>&1 | head -1 || true)
  fi
  if [ -n "$SCIENCEWORLD_JAR" ]; then
    jar_desc=$SCIENCEWORLD_JAR
    sha=$(sha256_16 "$SCIENCEWORLD_JAR")
  else
    jar_desc=bundled
    sha=-
  fi
  echo "SCIW_RUNTIME jar=$jar_desc sha256=$sha java=$java_desc workers=$SCIENCEWORLD_ENV_WORKERS prompt=${PROMPT:-default}"
}

# ---- the gym servers ----
gym_cwd() { printf '%s\n' "$REPO/3rdparty/Gym"; }

gym_env() {  # KEY=VALUE per line
  printf '%s\n' "NEMO_GYM_EXTRA_ROOTS=$REPO/gym_root"
  # Ray >= 2.x replicates the `uv run` env to workers and trips over gym's
  # per-server working dirs; gym only uses Ray as a local task pool.
  printf '%s\n' "RAY_ENABLE_UV_RUN_RUNTIME_ENV=0"
  # PRIVATE ray temp root: keeps gym's local ray cluster invisible to miles'
  # ray auto-discovery AND (with MILES_SCRIPT_EXTERNAL_RAY=1) safe from
  # miles' `ray stop --force` preamble — otherwise gym's raylet gets
  # SIGKILLed mid-run and every /run errors out (reward=None).
  printf '%s\n' "RAY_TMPDIR=/tmp/gymray"
  # the prebuilt venvs are complete: uv must never reach for the network
  [ -z "${GYM_VENV_ROOT:-}" ] || printf '%s\n' "UV_OFFLINE=1"
}

gym_build_argv() {  # -> GYM_ARGV
  # ALL gym ports live below 9000: an ephemeral port range starting at 9000 can
  # squat anything above it — pinned or "picked free" — between check and bind
  # (EADDRINUSE at boot). head=8100, agent=$NEMO_GYM_PORT (8200), auto-assigned
  # siblings 8300-8999; all clear of ray workers (2000-2999), the retriever
  # (6100), the webshop engine (6200), GCS (6379), sessions (7100-7131), the
  # router (7200) and the ray dashboard (8265).
  # readiness_timeout=0 skips gym probing the static openai_base_url — a
  # dummy here, the agent dials per-request policy_base_url instead — which
  # otherwise tears the whole gym stack down ~600s into the run.
  # $GYM_EXTRA_OVERRIDES is intentionally unquoted (word-split; empty for
  # recipes without extra overrides).
  if [ -n "${GYM_VENV_ROOT:-}" ]; then
    GYM_ARGV=("$GYM_VENV_ROOT/gym_main_venv/bin/gym" env start --config "$GYMCFG" --model-type openai_model
              "+skip_venv_if_present=true" "+uv_venv_dir=$GYM_VENV_ROOT/gym_venvs" "+uv_cache_dir=$GYM_VENV_ROOT/uv_cache")
  else
    GYM_ARGV=("$UV_BIN" run gym env start --config "$GYMCFG" --model-type openai_model)
  fi
  # shellcheck disable=SC2206
  GYM_ARGV=("${GYM_ARGV[@]}"
            "+model_endpoint_readiness_timeout_seconds=0"
            "+head_server.host=127.0.0.1" "+head_server.port=8100"
            "$AGENT_PORT_OVERRIDE"
            "+port_range_low=8300" "+port_range_high=8999"
            $GYM_EXTRA_OVERRIDES)
}

gym_argv() {  # one argv element per line (callers and --dry-run share it)
  gym_build_argv
  printf '%s\n' "${GYM_ARGV[@]}"
}

render_env_yaml() {
  if is_dry_run; then return 0; fi
  # the same gate gym_start_and_wait has, first: without the checkout the cp below would fail without a FATAL line
  [ -d "$(gym_cwd)" ] || fatal "no gym checkout at $(gym_cwd): git submodule update --init 3rdparty/Gym"
  sed -e "s|http://CHANGE_ME:30000/v1|http://127.0.0.1:9/v1|" -e "s|CHANGE_ME|dummy|g" \
      "$REPO/gym_root/env.yaml.example" > "$REPO/gym_root/env.yaml"
  cp "$REPO/gym_root/env.yaml" "$(gym_cwd)/env.yaml"
}

# Gate on the head's /readyz: 200 only once every server it spawned (agent,
# resources server, policy_model and, from gym 0.7.0, the legacy_agent relay)
# answers — a log-line count passes before the relay is up, and a relay that
# dies later takes the whole stack down. Then the agent must answer on :$NEMO_GYM_PORT.
gym_all_bound() { "$CURL" -sf -o /dev/null "http://127.0.0.1:8100/readyz"; }

gym_start_and_wait() {
  local kv
  [ -n "${RUN_TAG:-}" ] || fatal "gym_start_and_wait: RUN_TAG is required (the gym log is \$RUNS_DIR/<RUN_TAG>_gym_<JOB_ID>.log)"
  [ -n "${GYMCFG:-}" ] || fatal "gym_start_and_wait: RECIPE=${RECIPE:-} has no gym config (recipe_vars first; tmax starts no gym)"
  gym_build_argv
  if is_dry_run; then
    [ -d "$(gym_cwd)" ] || echo "WARN: no gym checkout at $(gym_cwd) (git submodule update --init 3rdparty/Gym)"
    dry_run_print gym "${GYM_ARGV[@]}"
    gym_env | while IFS= read -r kv; do printf 'DRY_RUN env: %s\n' "$kv"; done
    printf 'DRY_RUN cwd: %s\n' "$(gym_cwd)"
    return 0
  fi
  [ -d "$(gym_cwd)" ] || fatal "no gym checkout at $(gym_cwd): git submodule update --init 3rdparty/Gym"
  GYM_LOG=$RUNS_DIR/${RUN_TAG}_gym_${JOB_ID}.log
  mkdir -p /tmp/gymray "$RUNS_DIR"
  # The exec makes $! the gym process itself, not a wrapper subshell: the cleanup trap's kill reaches it.
  (
    cd "$(gym_cwd)"
    while IFS= read -r kv; do export "$kv"; done <<EOF
$(gym_env)
EOF
    exec "${GYM_ARGV[@]}" > "$GYM_LOG" 2>&1
  ) &
  GYM_PID=$!
  add_cleanup 'kill $GYM_PID 2>/dev/null'
  wait_http_ready "http://127.0.0.1:$NEMO_GYM_PORT/openapi.json" "$GYM_PID" "$GYM_LOG" 84 5 gym_all_bound \
    || { echo "FATAL: gym servers failed to boot"; exit 1; }
  echo "GYM_UP agent=http://127.0.0.1:$NEMO_GYM_PORT"
  export NEMO_GYM_URL=http://127.0.0.1:$NEMO_GYM_PORT
}
