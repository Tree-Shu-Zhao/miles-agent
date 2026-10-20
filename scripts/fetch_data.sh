#!/bin/bash
# scripts/fetch_data.sh — an environment's data: download / verify the UPSTREAM files (every dataset belongs to its
# upstream project; this repo ships processing code and digests only) and run this repo's converters, so that every
# derived file is rebuilt from the official source and checked against its pinned digest.
#
#   scripts/fetch_data.sh [--dry-run] alfworld
#       alfworld-download --data-dir $ALFWORLD_DATA (default $MILES_AGENT_DATA/alfworld): the venv binary
#       $GYM_VENV_ROOT/gym_venvs/resources_servers/alfworld_env/.venv/bin/alfworld-download when GYM_VENV_ROOT is
#       set, else the alfworld-download on PATH (none: a FATAL printing the command to run inside the alfworld_env
#       venv); then $ALFWORLD_DATA/json_2.1.1 must exist.                           FETCH_DATA_OK env=alfworld dir=
#   scripts/fetch_data.sh [--dry-run] scienceworld [--from <TCOD campaign test.jsonl> | --synthesize]
#       the simulator jar ships inside the scienceworld wheel (nothing to download; SCIENCEWORLD_JAR, alias SCIW_JAR,
#       pins another one); the JDK is checked (java >= 11 under SCIENCEWORLD_JAVA_HOME, alias SCIW_JAVA_HOME, else
#       JAVA_HOME, else $TOOLS_DIR/jdk-17 when that dir exists, else the java on PATH: the ladder's order, shared
#       through scripts/lib/common.sh scienceworld_java_home). The eval list recipes/scienceworld/prepare_data.py
#       reads is DERIVED data and is not in the repository: TCOD's eval taskset, 1308 tasks
#       (recipes/scienceworld/data/README.md). --from re-serialises the campaign's test.jsonl (task_splits.load_test
#       checks it) to ${SCIENCEWORLD_TEST_LIST:-$MILES_AGENT_DATA/scienceworld/scienceworld_test1308.jsonl} and
#       FATALs unless its sha256 is the pinned one (${SCIENCEWORLD_TEST_SHA256:-the paper's list}); --synthesize
#       writes the same tasks in sorted (task_name, var_num) order there (its own pinned sha256; the valid_unseen
#       slice then differs from the paper's); neither flag: an existing list is classified by digest
#       (order=frozen|synthesized, else a WARN) and the upstream pointer printed (DATA_UPSTREAM); none present: the
#       manual step (DATA_MANUAL_STEP) and exit 2.        FETCH_DATA_OK env=scienceworld test_list= sha256= [order=]
#       (order=synthesized after --synthesize; order=frozen|synthesized|unknown without a flag; none after --from)
#   scripts/fetch_data.sh [--dry-run] webshop --from <dir> [--full]
#       copies the WebShop authors' catalog files (items_shuffle_1000.json items_ins_v2_1000.json
#       items_human_ins.json; --full adds items_shuffle.json items_ins_v2.json) from <dir> into the engine's data dir
#       — $WEBSHOP_DATA_DIR of env_servers/webshop/defaults.sh, i.e. $WEBSHOP_HOME/webshop/data, where
#       env_servers/webshop/setup.sh looks before it tries a download; WEBSHOP_HOME default $MILES_AGENT_DATA/webshop —
#       after checking each md5 against env_servers/webshop/data_md5.txt (`<name> <md5>` lines; an entry may be a
#       PREFIX of the md5 — the repo only has 8-hex prefixes — and a full 32-hex entry must match whole;
#       WEBSHOP_MD5_TABLE overrides the file; a name without an entry is copied with a WARN). Without --from: the
#       manual step is printed (DATA_MANUAL_STEP) and the exit code is 2. The Lucene index build stays
#       env_servers/webshop/setup.sh's.                                                FETCH_DATA_OK env=webshop dir=
#   scripts/fetch_data.sh [--dry-run] webshop --regenerate-goals [--engine-url U]
#       rebuilds the goals export webshop_goals.jsonl (WEBSHOP_GOALS_FILE; default the recipe's data dir, where it is
#       gitignored: docs/data.md) from a RUNNING engine (U, else ${WEBSHOP_ENGINE_BASE_URL:-http://127.0.0.1:
#       $WEBSHOP_SERVER_PORT}; start it with env_servers/webshop/launch_engine.sh) through
#       env_servers/webshop/export_goals.py and FATALs unless the exporter's goals_digest equals
#       ${WEBSHOP_GOALS_DIGEST:-455c9eaa96191605} (the exported file stays in place for inspection); on success the
#       exporter's sidecar is copied to the name webshop_goals.meta.json beside it.
#                                                                   FETCH_DATA_OK env=webshop goals= goals_digest=
#   scripts/fetch_data.sh [--dry-run] search_r1 [--splits <split>...]
#       gym_root/resources_servers/wiki_search/data/prepare.py --output-dir $WIKI_SEARCH_DATA (default
#       $MILES_AGENT_DATA/wiki_search; network + the huggingface_hub package: the FlashRAG files come from the Hub;
#       --splits takes split names only, train validation test, and the next flag or environment name ends the list),
#       then train.jsonl and validation.jsonl are linked into gym_root/resources_servers/wiki_search/data/ (the gym
#       config's paths; a stale link is replaced, a placeholder copy of example.jsonl — the venv build's boot probe
#       leaves one — is moved aside as placeholder_<split>.jsonl, any other regular file is a FATAL).
#                                                                             FETCH_DATA_OK env=search_r1 dir=
#   --dry-run (also DRY_RUN=1) prints every command as `DRY_RUN <label>: <argv>` and exits 0 before any side effect
#   (a missing file or tool is a WARN there; a manual step too); every failure prints `FATAL: <text>` first.
#   Knobs come from the environment, then local.env (scripts/lib/common.sh): MILES_AGENT_DATA ALFWORLD_DATA
#   GYM_VENV_ROOT SCIENCEWORLD_JAVA_HOME JAVA_HOME TOOLS_DIR SCIENCEWORLD_JAR SCIENCEWORLD_TEST_LIST
#   SCIENCEWORLD_TEST_SHA256 WEBSHOP_HOME WEBSHOP_SERVER_PORT WEBSHOP_ENGINE_BASE_URL WEBSHOP_MD5_TABLE
#   WEBSHOP_GOALS_FILE WEBSHOP_GOALS_DIGEST WIKI_SEARCH_DATA PYTHON CURL.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

. "$HERE/lib/common.sh"

USAGE_LINE="scripts/fetch_data.sh [--dry-run] <alfworld | scienceworld [--from F | --synthesize] | webshop (--from DIR [--full] | --regenerate-goals [--engine-url U]) | search_r1 [--splits S...]>"
need_value() { [ $# -ge 2 ] || fatal "$1 needs a value: $USAGE_LINE"; }
ENV_NAME="" FROM="" FULL=0 REGENERATE=0 SYNTHESIZE=0 ENGINE_URL_FLAG="" SPLITS=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --from) need_value "$@"; FROM=$2; shift ;;
    --from=*) FROM=${1#*=} ;;
    --full) FULL=1 ;;
    --regenerate-goals) REGENERATE=1 ;;
    --synthesize) SYNTHESIZE=1 ;;
    --engine-url) need_value "$@"; ENGINE_URL_FLAG=$2; shift ;;
    --engine-url=*) ENGINE_URL_FLAG=${1#*=} ;;
    --splits)
      need_value "$@"
      shift
      # split names only (prepare.py's train validation test): the next flag or environment name ends the list
      n=0
      while [ $# -gt 0 ]; do
        case "$1" in train|validation|test) SPLITS="$SPLITS $1"; n=$((n + 1)); shift ;; *) break ;; esac
      done
      [ "$n" -gt 0 ] || fatal "--splits takes split names (train validation test), not '$1': $USAGE_LINE"
      continue ;;
    -*) fatal "unknown argument '$1': $USAGE_LINE" ;;
    *) [ -z "$ENV_NAME" ] || fatal "one environment at a time ('$ENV_NAME' and '$1'): $USAGE_LINE"
       ENV_NAME=$1 ;;
  esac
  shift
done
[ -n "$ENV_NAME" ] || fatal "an environment is required: $USAGE_LINE"
[ -z "${DRY_RUN:-}" ] || export DRY_RUN
SPLITS=${SPLITS# }

# a check under --dry-run is a WARN (the argv is still printed); otherwise a FATAL
gate() { if is_dry_run; then echo "WARN: $*"; else fatal "$*"; fi; }
sha256_file() {  # <file>: its sha256, 64 hex (need_sha256_tool first: a FATAL inside $(...) would be swallowed)
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'
}
need_sha256_tool() {
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
    || fatal "no sha256 tool (sha256sum or shasum) on PATH"
}
md5_file() {  # <file>: its md5, 32 hex (need_md5_tool first: a FATAL inside $(...) would be swallowed)
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then md5 -q "$1"
  else openssl dgst -md5 "$1" | awk '{print $NF}'; fi
}
need_md5_tool() {
  command -v md5sum >/dev/null 2>&1 || command -v md5 >/dev/null 2>&1 || command -v openssl >/dev/null 2>&1 \
    || fatal "no md5 tool (md5sum, md5 or openssl) on PATH"
}
same_entry() {  # <a> <b>: both exist and are the same file system entry (their dirs resolved physically)
  local da db
  [ -e "$1" ] && [ -e "$2" ] || return 1
  da=$(cd "$(dirname "$1")" && pwd -P) || return 1
  db=$(cd "$(dirname "$2")" && pwd -P) || return 1
  [ "$da/$(basename "$1")" = "$db/$(basename "$2")" ]
}
only_for() {  # <env> <flag> <value>: a flag of another environment is a usage error
  [ -z "$3" ] || [ "$3" = 0 ] || fatal "$2 is not an option of $1: $USAGE_LINE"
}

# ---- alfworld: the game files through alfworld's own downloader ----
fetch_alfworld() {
  only_for alfworld --from "$FROM"; only_for alfworld --regenerate-goals "$REGENERATE"; only_for alfworld --splits "$SPLITS"
  only_for alfworld --synthesize "$SYNTHESIZE"
  export ALFWORLD_DATA=${ALFWORLD_DATA:-$MILES_AGENT_DATA/alfworld}
  local bin
  if [ -n "${GYM_VENV_ROOT:-}" ]; then
    bin=$GYM_VENV_ROOT/gym_venvs/resources_servers/alfworld_env/.venv/bin/alfworld-download
    [ -x "$bin" ] || gate "no $bin: build the alfworld_env venv first (scripts/build_gym_venvs.sh $GYM_VENV_ROOT alfworld_env), or unset GYM_VENV_ROOT to use an alfworld-download on PATH"
  elif command -v alfworld-download >/dev/null 2>&1; then
    bin=alfworld-download
  else
    bin=alfworld-download
    gate "no alfworld-download on PATH (and no GYM_VENV_ROOT): run it inside the alfworld_env venv — cd gym_root/resources_servers/alfworld_env && uv run alfworld-download --data-dir $ALFWORLD_DATA — or build the venvs (scripts/build_gym_venvs.sh) and set GYM_VENV_ROOT"
  fi
  if is_dry_run; then
    dry_run_print alfworld_download "$bin" --data-dir "$ALFWORLD_DATA"
    return 0
  fi
  mkdir -p "$ALFWORLD_DATA"
  "$bin" --data-dir "$ALFWORLD_DATA" || fatal "alfworld-download exited with rc=$? (ALFWORLD_DATA=$ALFWORLD_DATA)"
  [ -d "$ALFWORLD_DATA/json_2.1.1" ] || fatal "no $ALFWORLD_DATA/json_2.1.1 after the download: alfworld-download unpacks the game files under --data-dir; check its output"
  echo "FETCH_DATA_OK env=alfworld dir=$ALFWORLD_DATA"
}

# ---- scienceworld: the JDK check, the eval list (TCOD's taskset: re-serialised from the campaign file, or synthesized) ----
# the paper's list (--from the campaign file; SCIENCEWORLD_TEST_SHA256 pins another campaign's) and the sorted
# stand-in (--synthesize): recipes/scienceworld/data/README.md
SCIENCEWORLD_TEST_SHA256=${SCIENCEWORLD_TEST_SHA256:-b2e1024f82a09e63011f37e4f423036c94b175b49b79c8d131eba7994cba2739}
SCIENCEWORLD_SYNTHESIZED_SHA256=f98541246a3145ce0ae2f97831a5f7dc7cd19e083a7b9e782f52dda4a1a8c11c
SCIENCEWORLD_UPSTREAM="DATA_UPSTREAM scienceworld: TCOD_examples/scienceworld/get_sciworld_data.py (arXiv 2604.24005), campaign test.jsonl sha256 9632fe4f...ebf8 — obtain that file from the TCOD code and rebuild the list with: scripts/fetch_data.sh scienceworld --from <test.jsonl>"

java_major() {  # <java binary>: the major version of `java -version` (17 for "17.0.12", 8 for "1.8.0_292"), or nothing
  local desc major
  desc=$("$1" -version 2>&1 | head -1 || true)
  major=$(printf '%s\n' "$desc" | sed -n 's/^[^"]*"\([0-9][0-9]*\)[^"]*".*/\1/p')
  [ "$major" != 1 ] || major=$(printf '%s\n' "$desc" | sed -n 's/^[^"]*"1\.\([0-9][0-9]*\)[^"]*".*/\1/p')
  printf '%s\n' "$major"
  [ -z "$desc" ] || echo "scienceworld: $desc" >&2
}
check_scienceworld_digest() {  # <file> <what>: its sha256 must be the pinned one; the digest lands in SCIW_DIGEST
  # (not a $(...) helper: a FATAL inside a command substitution would be swallowed with the output; a check, so a
  # WARN under --dry-run)
  need_sha256_tool
  SCIW_DIGEST=$(sha256_file "$1")
  [ "$SCIW_DIGEST" = "$SCIENCEWORLD_TEST_SHA256" ] \
    || gate "scienceworld $2 digest $SCIW_DIGEST != pinned $SCIENCEWORLD_TEST_SHA256 (SCIENCEWORLD_TEST_SHA256): not the TCOD campaign's taskset (recipes/scienceworld/data/README.md)"
}
fetch_scienceworld() {
  only_for scienceworld --full "$FULL"; only_for scienceworld --regenerate-goals "$REGENERATE"; only_for scienceworld --splits "$SPLITS"
  [ -z "$FROM" ] || [ "$SYNTHESIZE" = 0 ] || fatal "--from and --synthesize are exclusive: $USAGE_LINE"
  # 1) the JVM the worker processes run on (scienceworld 1.2.3 needs >= 11; the runs used 17), resolved in the
  # ladder's order (scripts/lib/common.sh scienceworld_java_home): SCIENCEWORLD_JAVA_HOME, SCIW_JAVA_HOME, JAVA_HOME,
  # $TOOLS_DIR/jdk-17, else the java on PATH
  local java_home java="" major
  java_home=$(scienceworld_java_home)
  if [ -n "$java_home" ]; then
    java=$java_home/bin/java
    [ -x "$java" ] || { gate "no $java (the JDK resolves to $java_home: SCIENCEWORLD_JAVA_HOME, alias SCIW_JAVA_HOME, else JAVA_HOME, else TOOLS_DIR/jdk-17): point it at a JDK 17"; java=""; }
  elif command -v java >/dev/null 2>&1; then
    java=java
  else
    gate "no java on PATH and no JAVA_HOME: install a JDK 17 or set JAVA_HOME (SCIENCEWORLD_JAVA_HOME, alias SCIW_JAVA_HOME, wins over it)"
  fi
  if [ -n "$java" ]; then
    major=$(java_major "$java")
    if [ -z "$major" ]; then echo "WARN: could not parse the java version of $java"
    elif [ "$major" -lt 11 ]; then gate "java $major at $java is too old: ScienceWorld needs a JDK >= 11 (17 is what the runs used)"
    else echo "scienceworld: java $major ok"; fi
  fi
  echo "scienceworld: the simulator jar ships inside the scienceworld wheel (nothing to download); SCIENCEWORLD_JAR (alias SCIW_JAR) pins another jar, SCIENCEWORLD_JAVA_HOME (alias SCIW_JAVA_HOME, else JAVA_HOME) the JDK the worker JVMs run on"
  # 2) the eval list (not in the repository): the SETS are rebuilt in-repo (task_splits.build_*, synthesized_test), the
  # ORDER of the paper's list is one shuffle of the original campaign and comes only from that campaign's file, so --from
  # re-serialises that file and pins the result by its digest; --synthesize writes the same tasks in sorted order
  local out=${SCIENCEWORLD_TEST_LIST:-$MILES_AGENT_DATA/scienceworld/scienceworld_test1308.jsonl}
  local list d order
  if [ -n "$FROM" ]; then
    [ -f "$FROM" ] || gate "no --from file $FROM"
    if is_dry_run; then
      dry_run_print reserialise "$PYTHON" - "$REPO" "$FROM" "$out"
      return 0
    fi
    mkdir -p "$(dirname "$out")"
    # the compact rows ({"task_name", "var_num"}, json.dumps default separators, one per line: what
    # task_splits.write_test_list writes) reproduce the paper's list byte for byte; load_test accepts the campaign's raw rows too
    "$PYTHON" - "$REPO" "$FROM" "$out.tmp" <<'PY' || fatal "re-serialising $FROM failed (task_splits.load_test refused it, see above): nothing written to $out"
import json
import sys
from pathlib import Path

repo, src, dst = sys.argv[1:4]
sys.path.insert(0, str(Path(repo) / "gym_root" / "resources_servers"))
from scienceworld_env.task_splits import load_test  # noqa: E402

tasks = load_test(src)
with open(dst, "w", encoding="utf-8") as fh:
    for task_name, var_num in tasks:
        fh.write(json.dumps({"task_name": task_name, "var_num": var_num}) + "\n")
print(f"reserialised {len(tasks)} rows -> {dst}")
PY
    mv "$out.tmp" "$out"
    check_scienceworld_digest "$out" "test list (re-serialised from $FROM)"
    d=$SCIW_DIGEST
    list=$out
  elif [ "$SYNTHESIZE" = 1 ]; then
    # the same 1308 tasks in sorted (task_name, var_num) order (task_splits.synthesized_test + write_test_list), pinned
    # by its own digest; the paper's list (the frozen pin) is never overwritten by it
    if [ -f "$out" ]; then
      need_sha256_tool
      d=$(sha256_file "$out")
      [ "$d" != "$SCIENCEWORLD_TEST_SHA256" ] \
        || gate "$out is the frozen campaign list; not overwriting it with the synthesized order (sha256 $d = SCIENCEWORLD_TEST_SHA256)"
    fi
    if is_dry_run; then
      dry_run_print synthesize "$PYTHON" - "$REPO" "$out"
      return 0
    fi
    mkdir -p "$(dirname "$out")"
    "$PYTHON" - "$REPO" "$out.tmp" <<'PY' || fatal "synthesizing the eval list failed (see above): nothing written to $out"
import sys
from pathlib import Path

repo, dst = sys.argv[1:3]
sys.path.insert(0, str(Path(repo) / "gym_root" / "resources_servers"))
from scienceworld_env.task_splits import load_test, synthesized_test, write_test_list  # noqa: E402

tasks = synthesized_test()
write_test_list(tasks, dst)
assert load_test(dst) == tasks, "the synthesized list does not read back"
print(f"synthesized {len(tasks)} rows (sorted task_name, var_num) -> {dst}")
PY
    mv "$out.tmp" "$out"
    need_sha256_tool
    d=$(sha256_file "$out")
    [ "$d" = "$SCIENCEWORLD_SYNTHESIZED_SHA256" ] \
      || fatal "synthesized list digest $d != $SCIENCEWORLD_SYNTHESIZED_SHA256: task_splits.py changed (recipes/scienceworld/data/README.md)"
    echo "WARN: synthesized order (same task set as the paper's; valid_unseen differs): the first 134 rows of the list are the valid_unseen slice, so the scienceworld_unseen numbers are another subset than the paper's (recipes/scienceworld/data/README.md)"
    echo "FETCH_DATA_OK env=scienceworld test_list=$out sha256=$d order=synthesized"
    return 0
  else
    if [ ! -f "$out" ]; then
      is_dry_run && echo "WARN: no ScienceWorld eval list at $out" || echo "FATAL: no ScienceWorld eval list at $out"
      echo "$SCIENCEWORLD_UPSTREAM"
      echo "DATA_MANUAL_STEP scienceworld: obtain the TCOD campaign test.jsonl (TCOD_examples/scienceworld/get_sciworld_data.py of github.com/kokolerk/TCOD; the paper's file has sha256 9632fe4f...ebf8) and rerun with --from <file>, or rerun with --synthesize for the same task set in sorted order"
      is_dry_run && return 0
      exit 2
    fi
    need_sha256_tool
    d=$(sha256_file "$out")
    if [ "$d" = "$SCIENCEWORLD_TEST_SHA256" ]; then order=frozen
    elif [ "$d" = "$SCIENCEWORLD_SYNTHESIZED_SHA256" ]; then order=synthesized
    else
      order=unknown
      echo "WARN: unknown list digest $d at $out: neither the frozen pin $SCIENCEWORLD_TEST_SHA256 (SCIENCEWORLD_TEST_SHA256) nor the synthesized list $SCIENCEWORLD_SYNTHESIZED_SHA256"
    fi
    echo "scienceworld: eval list $out sha256 $d order=$order"
    echo "$SCIENCEWORLD_UPSTREAM"
    is_dry_run && return 0
    echo "FETCH_DATA_OK env=scienceworld test_list=$out sha256=$d order=$order"
    return 0
  fi
  echo "FETCH_DATA_OK env=scienceworld test_list=$list sha256=$d"
}

# ---- webshop: the authors' catalog (copied + md5-checked), or the goals export against a running engine ----
WEBSHOP_SMALL_FILES="items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json"
WEBSHOP_FULL_FILES="items_shuffle.json items_ins_v2.json"

md5_entry() {  # <table> <name>: the md5 (or prefix) the table gives for the name, or nothing
  awk -v n="$2" '$1 == n { print $2; exit }' "$1"
}
fetch_webshop() {
  only_for webshop --splits "$SPLITS"; only_for webshop --synthesize "$SYNTHESIZE"
  # WEBSHOP_HOME first, then defaults.sh derives the port, the engine URL and the layout (as the recipe ladder does)
  WEBSHOP_HOME=${WEBSHOP_HOME:-$MILES_AGENT_DATA/webshop}
  . "$REPO/env_servers/webshop/defaults.sh"
  if [ "$REGENERATE" = 1 ]; then
    only_for "webshop --regenerate-goals" --from "$FROM"; only_for "webshop --regenerate-goals" --full "$FULL"
    regenerate_webshop_goals
    return 0
  fi
  only_for "webshop --from" --engine-url "$ENGINE_URL_FLAG"
  local table=${WEBSHOP_MD5_TABLE:-$REPO/env_servers/webshop/data_md5.txt}
  local dest=$WEBSHOP_DATA_DIR names=$WEBSHOP_SMALL_FILES n src expected actual
  [ "$FULL" != 1 ] || names="$names $WEBSHOP_FULL_FILES"
  if [ -z "$FROM" ]; then
    is_dry_run && echo "WARN: webshop: no --from <dir>" \
      || echo "FATAL: webshop: the catalog is the WebShop authors' data and is not downloaded here (no --from <dir>)"
    echo "DATA_MANUAL_STEP webshop: obtain items_shuffle_1000.json, items_ins_v2_1000.json, items_human_ins.json (the WebShop authors' small catalog) and rerun with --from <dir>; md5 prefixes: $(awk 'NF >= 2 { printf "%s %s; ", $1, $2 }' "$table" 2>/dev/null)(env_servers/webshop/data_md5.txt)"
    is_dry_run && return 0
    exit 2
  fi
  [ -f "$table" ] || fatal "no md5 table $table (WEBSHOP_MD5_TABLE)"
  need_md5_tool
  [ -d "$FROM" ] || gate "no --from dir $FROM"
  for n in $names; do
    src=$FROM/$n
    if [ ! -s "$src" ]; then gate "no $src"; continue; fi
    expected=$(md5_entry "$table" "$n")
    if [ -z "$expected" ]; then
      echo "WARN: no md5 entry for $n in $table: copied unverified"
    else
      actual=$(md5_file "$src")
      case "$actual" in
        "$expected"*) echo "webshop: $n md5 $actual matches the ${#expected}-hex entry $expected" ;;
        *) gate "webshop: $n md5 $actual does not match $expected ($table): not the WebShop authors' file" ;;
      esac
    fi
    if is_dry_run; then dry_run_print copy cp "$src" "$dest/$n"; continue; fi
    mkdir -p "$dest"
    if same_entry "$src" "$dest/$n"; then
      echo "webshop: $n is already in place at $dest"
    else
      cp "$src" "$dest/$n.tmp"
      mv "$dest/$n.tmp" "$dest/$n"
      echo "webshop: $n -> $dest"
    fi
  done
  # the hint names --full only when this run copied the big catalog (FULL is 0 or 1, never empty)
  local full_flag=""
  [ "$FULL" != 1 ] || full_flag=" --full"
  echo "webshop: catalog dir $dest; the Lucene index build is env_servers/webshop/setup.sh's: WEBSHOP_HOME=$WEBSHOP_HOME bash env_servers/webshop/setup.sh$full_flag"
  is_dry_run && return 0
  echo "FETCH_DATA_OK env=webshop dir=$dest"
}
regenerate_webshop_goals() {
  local url=${ENGINE_URL_FLAG:-${WEBSHOP_ENGINE_BASE_URL:-http://127.0.0.1:$WEBSHOP_SERVER_PORT}}
  local goals=${WEBSHOP_GOALS_FILE:-$REPO/recipes/webshop/data/webshop_goals.jsonl}
  local pinned=${WEBSHOP_GOALS_DIGEST:-455c9eaa96191605} meta d n
  url=${url%/}
  meta=$goals.meta.json
  if is_dry_run; then
    echo "WARN: the engine health check ($url/health) is skipped under --dry-run"
    dry_run_print export_goals "$PYTHON" "$REPO/env_servers/webshop/export_goals.py" --engine-url "$url" --output "$goals"
    return 0
  fi
  "$CURL" -sf -o /dev/null "$url/health" \
    || fatal "no WebShop engine at $url: start it with env_servers/webshop/launch_engine.sh (WEBSHOP_HOME=$WEBSHOP_HOME) and rerun"
  mkdir -p "$(dirname "$goals")"
  "$PYTHON" "$REPO/env_servers/webshop/export_goals.py" --engine-url "$url" --output "$goals" \
    || fatal "export_goals.py exited with rc=$? (it writes nothing unless its own checks pass)"
  [ -f "$meta" ] || fatal "export_goals.py wrote no sidecar $meta"
  d=$(sed -n 's/.*"goals_digest": *"\([0-9a-fA-F]*\)".*/\1/p' "$meta" | head -1)
  n=$(sed -n 's/.*"goals": *\([0-9][0-9]*\).*/\1/p' "$meta" | head -1)
  [ -n "$d" ] || fatal "no goals_digest in $meta"
  [ "$d" = "$pinned" ] \
    || fatal "regenerated goals digest $d != pinned $pinned (WEBSHOP_GOALS_DIGEST): the catalog, goal code or seed differs from the frozen export, see recipes/webshop/data/README.md"
  # the provenance sidecar name beside the goals file (gitignored like it; nothing reads it programmatically:
  # prepare_data.py recomputes
  # the digest from the rows)
  cp "$meta" "$(dirname "$goals")/webshop_goals.meta.json"
  echo "FETCH_DATA_OK env=webshop goals=$n goals_digest=$d"
}

# ---- search_r1: the FlashRAG rows through prepare.py, linked where the gym config reads them ----
fetch_search_r1() {
  only_for search_r1 --from "$FROM"; only_for search_r1 --full "$FULL"; only_for search_r1 --regenerate-goals "$REGENERATE"
  only_for search_r1 --synthesize "$SYNTHESIZE"
  WIKI_SEARCH_DATA=${WIKI_SEARCH_DATA:-$MILES_AGENT_DATA/wiki_search}
  local prep=$REPO/gym_root/resources_servers/wiki_search/data/prepare.py
  local data=$REPO/gym_root/resources_servers/wiki_search/data split src dst
  if is_dry_run; then
    # shellcheck disable=SC2086
    dry_run_print prepare "$PYTHON" "$prep" --output-dir "$WIKI_SEARCH_DATA" ${SPLITS:+--splits $SPLITS}
    for split in train validation; do
      dry_run_print link ln -sfn "$WIKI_SEARCH_DATA/$split.jsonl" "$data/$split.jsonl"
    done
    return 0
  fi
  "$PYTHON" -c 'import huggingface_hub' 2>/dev/null \
    || fatal "the python package huggingface_hub is missing for $PYTHON: prepare.py downloads the FlashRAG files with hf_hub_download (pip install huggingface_hub)"
  mkdir -p "$WIKI_SEARCH_DATA"
  # shellcheck disable=SC2086
  "$PYTHON" "$prep" --output-dir "$WIKI_SEARCH_DATA" ${SPLITS:+--splits $SPLITS} || fatal "prepare.py exited with rc=$?"
  for split in train validation; do
    src=$WIKI_SEARCH_DATA/$split.jsonl
    dst=$data/$split.jsonl
    [ -f "$src" ] || { echo "WARN: no $src (not among --splits?): $dst not linked"; continue; }
    if same_entry "$src" "$dst"; then echo "search_r1: $dst is the output itself"; continue; fi
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
      if cmp -s "$dst" "$data/example.jsonl"; then
        mv "$dst" "$data/placeholder_$split.jsonl"
        echo "search_r1: the placeholder copy of example.jsonl at $dst moved to $data/placeholder_$split.jsonl"
      else
        fatal "$dst is a regular file, not a link: move it away (the gym config reads this path; a link to $src belongs there)"
      fi
    fi
    ln -sfn "$src" "$dst"
    echo "search_r1: $dst -> $src"
  done
  echo "FETCH_DATA_OK env=search_r1 dir=$WIKI_SEARCH_DATA"
}

case "$ENV_NAME" in
  alfworld) fetch_alfworld ;;
  scienceworld) fetch_scienceworld ;;
  webshop) fetch_webshop ;;
  search_r1) fetch_search_r1 ;;
  *) fatal "unknown environment '$ENV_NAME' (alfworld | scienceworld | webshop | search_r1): $USAGE_LINE" ;;
esac
