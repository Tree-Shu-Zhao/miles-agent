"""recipes/webshop/reward.py: training reward = SCALE * won - penalties.

Loaded from its file with a controlled environment (the coefficients are read
at import), so the defaults and the env overrides are both pinned here.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REWARD_PY = Path(__file__).resolve().parent.parent / "recipes" / "webshop" / "reward.py"


def _load(monkeypatch, **env):
    for key in (
        "WEBSHOP_REWARD_SCALE",
        "WEBSHOP_DENSE_REWARD",
        "WEBSHOP_INVALID_PENALTY",
        "WEBSHOP_FORMAT_PENALTY",
        "WEBSHOP_STEP_PENALTY",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("_webshop_reward", REWARD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample(**env_metrics):
    return SimpleNamespace(metadata={"env_metrics": env_metrics})


def test_defaults_are_ten_iff_won_minus_format_penalty(monkeypatch):
    r = _load(monkeypatch)
    assert (r._SCALE, r._DENSE, r._INVALID_PENALTY, r._FORMAT_PENALTY, r._STEP_PENALTY) == (10.0, False, 0.0, 0.1, 0.0)
    assert r.shape(1.0, _sample(won=1, task_score=1.0, action_format_errors=0, invalid_actions=3)) == 10.0
    assert r.shape(1.0, _sample(won=1, task_score=1.0, action_format_errors=2)) == 10.0 - 0.2
    # a partial-score purchase is not a win: base 0, only the penalty remains
    assert r.shape(0.75, _sample(won=0, task_score=0.75, action_format_errors=1)) == -0.1


def test_env_overrides(monkeypatch):
    r = _load(monkeypatch, WEBSHOP_FORMAT_PENALTY="0", WEBSHOP_INVALID_PENALTY="0.5", WEBSHOP_DENSE_REWARD="1")
    assert r.shape(0.75, _sample(won=0, task_score=0.75, action_format_errors=4, invalid_actions=2)) == 7.5 - 1.0
