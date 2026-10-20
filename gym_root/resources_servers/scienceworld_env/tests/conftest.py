import sys
from pathlib import Path

# Make `scienceworld_env`, `agent_utils` (shared session base) and
# `resources_servers.gymnasium` importable when running pytest straight from
# the checkout (on cluster the packages are installed editable into the server
# venv, so these inserts are harmless no-ops there).
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))  # gym_root -> agent_utils
sys.path.insert(0, str(_HERE.parents[2]))  # resources_servers -> import scienceworld_env
