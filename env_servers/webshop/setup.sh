#!/usr/bin/env bash
# Build the SELF-CONTAINED py3.10 WebShop engine runtime under $WEBSHOP_HOME.
# miles-agent depends on no other repo: this materializes everything from the
# committed vendored fork (env_servers/webshop/webshop) plus gdown downloads.
#
# Layout it builds (WEBSHOP_HOME defaults to $MILES_AGENT_DATA/webshop when that
# root is set, else /data/webshop; see defaults.sh):
#   $WEBSHOP_HOME/env/                             conda/mamba env (py3.10 + JDK 11 + deps)
#   $WEBSHOP_HOME/webshop/                         writable copy of the committed fork code
#   $WEBSHOP_HOME/webshop/data/                    catalog (items_*.json, downloaded)
#   $WEBSHOP_HOME/webshop/search_engine/indexes/   built Lucene index (symlink -> indexes_<mode>)
# Nothing is written back into the committed tree; the copy under $WEBSHOP_HOME
# is what launch_engine.sh serves (WEBSHOP_VENDOR / WEBSHOP_DATA_DIR).
#
# The env is a PREFIX env (conda -p) so its shebangs bake $WEBSHOP_HOME/env and
# resolve unchanged when the data root is mounted into the container at the same path.
#
# Usage: [WEBSHOP_HOME=...] ./setup.sh [--full]
#   default   1000-item catalog (items_*_1000.json) — fast to download and index.
#   --full    full catalog (~6 GB downloads + long index build). The server's
#             --small/--full flag (WEBSHOP_FULL) must match the mode last built here.
# Idempotent: an existing env / downloads / index are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_VENDOR="${HERE}/webshop"                 # committed fork code (source of truth)
# shellcheck source=defaults.sh
. "${HERE}/defaults.sh"                      # WEBSHOP_HOME / WEBSHOP_VENDOR / WEBSHOP_DATA_DIR
VENDOR="${WEBSHOP_VENDOR}"                   # writable runtime copy (default $WEBSHOP_HOME/webshop)
ENV_PREFIX="${WEBSHOP_HOME}/env"
DATA="${WEBSHOP_DATA_DIR}"                   # default $VENDOR/data

SMALL=1
for arg in "$@"; do
    case "${arg}" in
        --small) SMALL=1 ;;
        --full) SMALL=0 ;;
        *) echo "unknown argument: ${arg} (usage: $0 [--full])" >&2; exit 1 ;;
    esac
done

[ -d "${SRC_VENDOR}/web_agent_site" ] || {
    echo "ERROR: committed vendored code missing at ${SRC_VENDOR} — run ${HERE}/vendor.sh" >&2; exit 1; }

# 0) writable copy of the committed fork code (data/index rebuilt below, so skip).
# Skipped when WEBSHOP_VENDOR points at the committed tree itself (dev shortcut:
# the catalog + index then land inside the checkout; .gitignore keeps them out).
mkdir -p "${WEBSHOP_HOME}" "${VENDOR}"
if [ "$(cd "${SRC_VENDOR}" && pwd -P)" = "$(cd "${VENDOR}" && pwd -P)" ]; then
    echo "WEBSHOP_VENDOR is the committed tree itself — skipping the copy"
else
    rsync -a \
        --exclude 'data/' \
        --exclude 'search_engine/resources*/' \
        --exclude 'search_engine/indexes*' \
        --exclude '__pycache__/' \
        "${SRC_VENDOR}/" "${VENDOR}/"
fi

# 1) python env at an explicit prefix (relocatable-mounted; JDK bundled)
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    CONDA_BIN="${WEBSHOP_CONDA_BIN:-}"
    if [ -z "${CONDA_BIN}" ]; then
        for c in micromamba mamba conda; do command -v "$c" >/dev/null 2>&1 && { CONDA_BIN="$c"; break; }; done
    fi
    if [ -n "${CONDA_BIN}" ]; then
        # -c conda-forge on create too: micromamba ships with NO default channels
        # (conda/mamba would fall back to defaults; conda-forge is what we want anyway).
        "${CONDA_BIN}" create -p "${ENV_PREFIX}" -y -c conda-forge python=3.10
        "${CONDA_BIN}" install -p "${ENV_PREFIX}" -y -c conda-forge openjdk=11 faiss-cpu
    else
        echo "no conda/mamba/micromamba found — falling back to venv (needs python3.10 + JDK 11+ on PATH)"
        command -v python3.10 >/dev/null 2>&1 || { echo "ERROR: python3.10 not found" >&2; exit 1; }
        java -version >/dev/null 2>&1 || { echo "ERROR: java (JVM 11+) not found; pyserini needs it" >&2; exit 1; }
        python3.10 -m venv "${ENV_PREFIX}"
    fi
fi
PYBIN="${ENV_PREFIX}/bin/python"
# conda-forge openjdk installs the JDK under <prefix>/lib/jvm (pyjnius needs
# JAVA_HOME to be that dir to find lib/server/libjvm.so); for the venv fallback
# rely on PATH.
if [ -d "${ENV_PREFIX}/lib/jvm" ]; then
    export JAVA_HOME="${ENV_PREFIX}/lib/jvm"
elif [ -x "${ENV_PREFIX}/bin/java" ]; then
    export JAVA_HOME="${ENV_PREFIX}"
fi
[ -n "${JAVA_HOME:-}" ] && export PATH="${JAVA_HOME}/bin:${PATH}"

"${PYBIN}" -m pip install -r "${HERE}/requirements.txt"
# Pinned model wheel instead of `spacy download`: the downloader resolves the
# model version from GitHub's compatibility table and, when that lookup comes
# back empty, builds a bogus URL (.../download/-en_core_web_sm/...) and 404s
# (seen 2026-09). 3.7.1 is the en_core_web_sm release for
# spaCy 3.7.x (requirements.txt pins spacy==3.7.2).
SPACY_MODEL_WHL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
"${PYBIN}" -c 'import spacy; spacy.load("en_core_web_sm")' 2>/dev/null \
    || "${PYBIN}" -m pip install "${SPACY_MODEL_WHL}"

# 2) catalog downloads (gdown ids from the fork's setup.sh). items_human_ins.json
# is required even for synthetic goals (engine.py loads it unconditionally).
mkdir -p "${DATA}"
gdl() {  # gdl <gdrive-id> <filename>
    if [ -s "${DATA}/$2" ]; then
        echo "data/$2 already present — skipping"
    else
        # 2026-09: Google Drive answers these public ids with a sign-in page
        # (gdown: "Cannot retrieve the public link"), from servers and
        # laptops alike. Pre-stage the files into ${DATA} instead (any copy of
        # the WebShop data works: the 1k catalog is ~10 MB; md5 of the
        # canonical files: items_shuffle_1000 df2b2610..., items_ins_v2_1000
        # a3346e8b..., items_human_ins 738c99a2...) and re-run; the download is
        # only a convenience path.
        (cd "${DATA}" && "${ENV_PREFIX}/bin/gdown" "https://drive.google.com/uc?id=$1" -O "$2") || {
            rm -f "${DATA}/$2"
            echo "ERROR: could not download $2 from Google Drive; pre-stage it into ${DATA} and re-run" >&2
            exit 1
        }
    fi
}
gdl 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib items_shuffle_1000.json   # product info (1k)
gdl 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu items_ins_v2_1000.json    # product attributes (1k)
if [ "${SMALL}" -eq 0 ]; then
    gdl 1A2whVgOO0euk5O13n2iYDM0bQRkkRduB items_shuffle.json     # full product info (~5.1 GB)
    gdl 1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi items_ins_v2.json      # full attributes (~587 MB)
fi
gdl 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O items_human_ins.json       # always required

# 3) Lucene index build directly into the writable vendored copy
if [ "${SMALL}" -eq 1 ]; then
    MODE=small
    FILE_PATH="${DATA}/items_shuffle_1000.json"
    ATTR_PATH="${DATA}/items_ins_v2_1000.json"
else
    MODE=full
    FILE_PATH="${DATA}/items_shuffle.json"
    ATTR_PATH="${DATA}/items_ins_v2.json"
fi
RESOURCES="${VENDOR}/search_engine/resources_${MODE}"
INDEXES="${VENDOR}/search_engine/indexes_${MODE}"

if [ -d "${INDEXES}" ] && [ -n "$(ls -A "${INDEXES}" 2>/dev/null)" ]; then
    echo "index ${INDEXES} already built — skipping"
else
    mkdir -p "${RESOURCES}" "${INDEXES}"
    VENDOR="${VENDOR}" FILE_PATH="${FILE_PATH}" ATTR_PATH="${ATTR_PATH}" RESOURCES="${RESOURCES}" \
    "${PYBIN}" - <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["VENDOR"])
from web_agent_site.engine.engine import load_products

all_products, *_ = load_products(
    filepath=os.environ["FILE_PATH"], attrpath=os.environ["ATTR_PATH"])
with open(os.path.join(os.environ["RESOURCES"], "documents.jsonl"), "w") as f:
    for p in all_products:
        option_texts = []
        for option_name, option_contents in p.get("options", {}).items():
            option_texts.append(f'{option_name}: {", ".join(option_contents)}')
        doc = {
            "id": p["asin"],
            "contents": " ".join([
                p["Title"], p["Description"], p["BulletPoints"][0],
                ", and ".join(option_texts),
            ]).lower(),
            "product": p,
        }
        f.write(json.dumps(doc) + "\n")
print(f"wrote {len(all_products)} documents")
PYEOF
    "${PYBIN}" -m pyserini.index.lucene \
        --collection JsonCollection \
        --input "${RESOURCES}" \
        --index "${INDEXES}" \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1 \
        --storePositions --storeDocvectors --storeRaw
fi

# the vendored engine loads its index from the default relative 'indexes' dir
ln -sfn "${INDEXES}" "${VENDOR}/search_engine/indexes"

echo "setup complete (catalog=${MODE}, WEBSHOP_HOME=${WEBSHOP_HOME})"
echo "launch: WEBSHOP_HOME=${WEBSHOP_HOME} bash ${HERE}/launch_engine.sh$([ "${SMALL}" -eq 1 ] || echo ' (set WEBSHOP_FULL=1)')"
