"""Fake engine factory resolved INSIDE spawn'd worker processes (tests only).

The worker pool resolves build_engine/make_env/step_env/drop_engine_state via
``EngineConfig.factory_module`` (importlib) precisely so tests can substitute a
fake stack that carries no pyserini/JVM/spaCy — monkeypatching cannot cross the
spawn boundary, but a module name can. Importable top-level because pytest puts
tests/ on sys.path and spawn children inherit it (see conftest.py).

The engine's ``goals`` is a LIST of deterministic goal dicts (``fake_goal``)
with every ``agent_utils.webshop_goals.GOAL_FIELDS`` key populated, so the host
computes real content ids over it and tests can address goals by goal_id
exactly as the frozen dataset does. ``make_env`` still takes the list index
(the host resolves ids before calling it).

Failure modes are scripted by SENTINEL goal indices (above the real small
catalog's 6910 so a legitimate goal can never trip them, but inside the fake
engine's goal count so the parent-side range check lets them through):
  GOAL_HANG blocks past any test timeout, GOAL_DIE kills the worker outright,
  GOAL_BOOM raises inside make_env (worker survives). A build failure is keyed
  off a ``BUILD-BOOM`` substring in the vendored path (permanent) or a ``BOOM``
  marker FILE inside it (toggle it to script "respawn fails, later succeeds");
  a ``SKEW`` marker file makes every process build a DIFFERENT goal list (the
  pool must refuse to boot on disagreeing goal-set digests).
Per-step: action ``"explode"`` raises inside step_env (session survives).
Terminal: ``"click[buy now]"`` ends the episode with task_score 1.0 / won.
"""

import os
import time

GOAL_HANG = 9_999
GOAL_DIE = 9_998
GOAL_BOOM = 9_997
# > every sentinel and > the real small catalog's 6910, yet small enough that
# each spawned worker hashes its whole goal list in well under a second.
N_GOALS = 10_000

_SEARCH_PAGE = {"has_search_bar": True, "clickables": []}
_ITEM_PAGE = {"has_search_bar": False, "clickables": ["buy now"]}


class FakeEngineEnv:
    """One per-episode env; mirrors the real env's ``session_prefix`` attribute
    (drop_engine_state keys off it) and holds an independent step counter so
    same-worker sessions are provably isolated."""

    def __init__(self, goal_idx: int, prefix: str, observation_mode: str = "text_rich"):
        self.goal_idx = goal_idx
        self.session_prefix = prefix
        self.observation_mode = observation_mode
        self.steps = 0


def fake_goal(i: int) -> dict:
    """Deterministic engine-shaped goal dict for list position ``i``. Every
    GOAL_FIELDS key is set (so goal_id is a real content hash); ``weight`` is
    present to show the identity ignores it, as it does for the real engine."""
    return {
        "asin": f"B{i:05d}",
        "name": f"Fake Product {i}",
        "category": f"cat{i % 5}",
        "query": f"fake query {i % 7}",
        "product_category": f"Fake › Category {i % 5}",
        "instruction_text": f"fake goal {i}",
        "attributes": [f"attr{i % 3}"],
        "goal_options": {"size": str(i % 2)},
        "price_upper": float(10 + i),
        "weight": 1,
    }


class FakeEngine:
    """Stand-in SimServer: a prefix-keyed ``user_sessions`` dict (so cleanup is
    observable) and a ``goals`` list of engine-shaped goal dicts."""

    def __init__(self, skew: str | None = None):
        self.user_sessions: dict[str, dict] = {}
        self.goals = [fake_goal(i) for i in range(N_GOALS)]
        if skew is not None:
            # One extra goal whose content differs per process -> per-worker digests differ.
            self.goals.append({**fake_goal(N_GOALS), "instruction_text": f"skewed goal {skew}"})


def build_engine(cfg):
    vendored = str(cfg.vendored_path)
    if "BUILD-BOOM" in vendored or os.path.exists(os.path.join(vendored, "BOOM")):
        raise RuntimeError("scripted build failure")
    skew = str(os.getpid()) if os.path.exists(os.path.join(vendored, "SKEW")) else None
    return FakeEngine(skew), FakeEngineEnv, ("items.json", "attrs.json")


def make_env(engine, env_cls, file_paths, goal_idx: int, prefix: str, observation_mode: str = "text_rich"):
    if goal_idx == GOAL_HANG:
        time.sleep(60)
    if goal_idx == GOAL_DIE:
        os._exit(1)
    if goal_idx == GOAL_BOOM:
        raise RuntimeError("scripted load failure")
    env = env_cls(goal_idx, prefix, observation_mode)
    engine.user_sessions[f"{prefix}{goal_idx}"] = {"goal": goal_idx}
    obs = f"fake obs for goal {goal_idx}"
    # observation_mode rides in info so pool tests can prove the cfg reached the child.
    info = {
        "available_actions": dict(_SEARCH_PAGE),
        "task_score": 0.0,
        "won": False,
        "goal_idx": goal_idx,
        "observation_mode": observation_mode,
    }
    return env, obs, info


def step_env(engine, env, action: str):
    if action == "explode":
        raise RuntimeError("scripted step failure")
    env.steps += 1
    if action == "click[buy now]":
        drop_engine_state(engine, env.session_prefix)
        return (
            "Thank you for shopping with us!",
            1.0,
            True,
            {"available_actions": dict(_SEARCH_PAGE), "task_score": 1.0, "won": True},
        )
    return (
        f"echo:{action}:{env.steps}",
        0.0,
        False,
        {"available_actions": dict(_ITEM_PAGE), "task_score": 0.0, "won": False},
    )


def drop_engine_state(engine, prefix):
    if not prefix:
        return
    for k in [k for k in engine.user_sessions if k.startswith(prefix)]:
        del engine.user_sessions[k]
