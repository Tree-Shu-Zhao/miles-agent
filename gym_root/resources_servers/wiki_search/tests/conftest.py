import sys
from pathlib import Path

# Make `wiki_search` importable when running pytest straight from the checkout
# (on cluster the package is installed editable into the server venv, so this
# insert is a harmless no-op there).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
