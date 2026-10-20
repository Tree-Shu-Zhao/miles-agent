#!/usr/bin/env bash
# Builds one harness pack of miles_harbor.harnesses (gym_root/miles_harbor/miles_harbor/harnesses.py)
# at $HARBOR_PACK_ROOT/<name> (default $HOME/.cache/miles-harness): the tree for glibc images, then VERSION,
# written last (a pack without it is no pack). Runs as uid 0 in a glibc build container with that dir
# mounted read-write, at the path every sandbox mounts the packs at read-only, so the absolute paths
# baked in (shebangs, pyvenv.cfg, the python links) are the sandboxes'. HARBOR_PACK_ROOT is the one
# contract on both sides: the runtime (harnesses.py) reads the packs from the same variable, and a
# wrapper job that mounts the pack's dir elsewhere than the default puts HARBOR_PACK_ROOT=<its mount
# point> into this script's environment (a wrapper mounting at the default exports nothing).
# This script lives in gym_root/miles_harbor/packs/ with the files it installs (miles_mini_ext.py) and
# reads them from its own directory, wherever the two are copied to together.
#   harbor_packs.sh <name> <version>
# mini-swe-agent: what Harbor's install() puts in a trial's image (uv's CPython 3.12, `uv tool
# install mini-swe-agent==<version> --with litellm --with orjson --with fastapi`), resolved once
# for every image instead of in each: wheels only, for manylinux_2_28, so the tree runs on
# glibc >= 2.28 (its GLIBC_MIN, which install() checks; at manylinux_2_17 the resolution falls
# back to older litellm, numpy, tiktoken, whose newest wheels are 2_28 only). FREEZE.txt records
# what was resolved. No musl tree yet: no image we run is musl (install() fails on one).
# A local version, <version>+miles<N>, is the harness's <version> and miles' own extensions to it, taken from this
# script's directory (N counts their revisions): for mini-swe-agent, miles_mini_ext.py (InterpreterLocalEnvironment,
# which configs/mini_swebench.yaml names) as a top-level module of the tool's python, its sha256 in EXTENSIONS.txt.
set -euo pipefail
NAME=${1:?harbor_packs.sh <name> <version>}
VERSION=${2:?harbor_packs.sh <name> <version>}
PACK=${HARBOR_PACK_ROOT:-$HOME/.cache/miles-harness}/$NAME
UPSTREAM=${VERSION%%+*}
LABEL=${VERSION#"$UPSTREAM"}
case "$LABEL" in ""|+miles[1-9]*) ;; *) echo "FATAL: version $VERSION: the only local label is +miles<N>"; exit 1 ;; esac
UV_VERSION=0.13.1
GLIBC_MIN=2.28
[ ! -e "$PACK/VERSION" ] || { echo "FATAL: $PACK has a VERSION already"; exit 1; }
# uv only builds the pack: it stays in the container, as do its caches
export UV_INSTALL_DIR=/root/uv UV_NO_MODIFY_PATH=1 UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
UV=/root/uv/uv
case "$NAME" in
  mini-swe-agent)
    T=$PACK/glibc
    export UV_PYTHON_INSTALL_DIR=$T/python UV_TOOL_DIR=$T/tools UV_TOOL_BIN_DIR=$T/bin
    "$UV" python install --no-bin --compile-bytecode 3.12
    "$UV" tool install --python 3.12 --python-preference only-managed \
      --python-platform "$(uname -m)-manylinux_${GLIBC_MIN/./_}" --no-build --compile-bytecode \
      "mini-swe-agent==$UPSTREAM" --with litellm --with orjson --with fastapi
    PY=$T/tools/mini-swe-agent/bin/python
    ext=
    if [ -n "$LABEL" ]; then
      site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
      cp "$(dirname "$0")/miles_mini_ext.py" "$site/"
      "$PY" -m compileall -q "$site/miles_mini_ext.py"
      "$PY" -c "import miles_mini_ext"
      echo "miles_mini_ext.py sha256=$(sha256sum "$(dirname "$0")/miles_mini_ext.py" | cut -d' ' -f1)" > "$T/EXTENSIONS.txt"
      ext=" $(cat "$T/EXTENSIONS.txt")"
    fi
    # Harbor's install() ends with this check
    "$T/bin/mini-swe-agent" --help > /dev/null
    "$PY" -c "import litellm, minisweagent"
    "$UV" pip freeze --python "$PY" > "$T/FREEZE.txt"
    echo "$GLIBC_MIN" > "$T/GLIBC_MIN"
    summary="python=$("$PY" -c 'import platform; print(platform.python_version())') $(grep -E '^litellm==' "$T/FREEZE.txt")$ext"
    ;;
  *) echo "FATAL: no pack recipe for '$NAME' (mini-swe-agent)"; exit 1 ;;
esac
echo "$VERSION" > "$PACK/VERSION"
echo "PACK_OK $NAME $VERSION glibc>=$GLIBC_MIN $summary"
