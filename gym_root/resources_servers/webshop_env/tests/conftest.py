import sys
from pathlib import Path

# Make `webshop_env` and `resources_servers.gymnasium` importable when running
# pytest straight from the checkout (on cluster the package is installed
# editable into the server venv, so these inserts are harmless no-ops there).
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))  # gym_root  -> resources_servers.gymnasium, nemo_gym overlay
sys.path.insert(0, str(_HERE.parents[2]))  # resources_servers -> import webshop_env
