#!/usr/bin/env bash
# (Re)populate the vendored WebShop fork (princeton-nlp/WebShop, text-env
# subset) in env_servers/webshop/webshop/ from an upstream source. The fork's
# CODE is committed to miles-agent so a normal checkout needs nothing — this is
# only for MAINTAINERS refreshing it against a newer upstream. miles-agent is
# self-contained: it carries no dependency on any other repo.
#
# Usage: ./vendor.sh <source>
#   <source> is EITHER a verl-agent-style checkout (…/env_package/webshop/webshop
#   is used) OR an already-vendored webshop tree (has web_agent_site/). Defaults
#   to $WEBSHOP_VENDOR_SRC.
#
# Excluded (kept OUT of git): downloaded data + built Lucene indexes (owned by
# setup.sh), caches, RL baseline models, and the 16 MB Selenium chromedriver —
# the text env (SimServer / WebAgentTextEnv) uses none of them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ARG="${1:-${WEBSHOP_VENDOR_SRC:-}}"
DEST="${HERE}/webshop"

[ -n "${SRC_ARG}" ] || { echo "ERROR: pass a source (verl-agent checkout or a vendored webshop tree)" >&2; exit 1; }

# Resolve: a verl-agent checkout nests the tree one level down.
if [ -d "${SRC_ARG}/web_agent_site" ]; then
    SRC="${SRC_ARG}"
elif [ -d "${SRC_ARG}/agent_system/environments/env_package/webshop/webshop/web_agent_site" ]; then
    SRC="${SRC_ARG}/agent_system/environments/env_package/webshop/webshop"
else
    echo "ERROR: no web_agent_site/ under ${SRC_ARG}" >&2; exit 1
fi

rsync -a --delete \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude 'data/' \
    --exclude 'search_engine/resources*/' \
    --exclude 'search_engine/indexes*/' \
    --exclude 'baseline_models/' \
    --exclude 'user_session_logs/' \
    --exclude 'web_agent_site/envs/chromedriver' \
    "${SRC}/" "${DEST}/"

echo "vendored $(find "${DEST}" -type f | wc -l | tr -d ' ') files into ${DEST}"
echo "next: ./setup.sh [--full] to install deps, download data, and build the index"
