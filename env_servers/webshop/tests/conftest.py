import sys
from pathlib import Path

# The engine tier is a set of top-level modules (session_server, _webshop_engine,
# _engine_worker, server) rather than a package, so put its dir on sys.path when
# running pytest from the checkout. tests/ itself must also be importable as a
# top-level location so `fake_engine_factory` resolves BOTH here and inside the
# spawn'd worker processes (spawn children inherit the parent's sys.path).
# gym_root is added too so tests import the shared, stdlib-only
# agent_utils.webshop_goals directly (the engine modules reach it by the same
# path fallback; see _webshop_engine.py).
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "gym_root"))  # agent_utils.webshop_goals
sys.path.insert(0, str(_HERE.parents[1]))  # env_servers/webshop
sys.path.insert(0, str(_HERE.parent))  # env_servers/webshop/tests -> fake_engine_factory
