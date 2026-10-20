"""gym_root/agent_utils/webshop_goals.py: content identity of WebShop goals.

Stdlib module loaded from its file (the root .venv has no nemo_gym); the same
code runs in the py3.10 engine and in recipes/webshop/prepare_data.py.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wg():
    path = REPO / "gym_root" / "agent_utils" / "webshop_goals.py"
    spec = importlib.util.spec_from_file_location("_webshop_goals", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOAL = {
    "asin": "B0001",
    "category": "beauty",
    "query": "hair oil",
    "name": "Argan Hair Oil 100ml",
    "product_category": "Beauty & Personal Care > Hair Care",
    "instruction_text": "i need argan hair oil, size: 100ml, and price lower than 30.00 dollars",
    "attributes": ["argan", "sulfate free"],
    "price_upper": 30.0,
    "goal_options": {"size": "100ml"},
    "weight": 1,
}


def test_goal_id_is_content_only_and_stable(wg):
    gid = wg.goal_id(GOAL)
    assert len(gid) == 16 and int(gid, 16) >= 0
    # position / weight / extra bookkeeping never enter the identity
    assert wg.goal_id({**GOAL, "weight": 7, "idx": 3}) == gid
    # the record round-trips through JSON and back to the same id
    rec = wg.goal_record(GOAL)
    assert set(rec) == set(wg.GOAL_FIELDS) and wg.goal_id(rec) == gid
    assert wg.goal_id(wg.goal_record(rec)) == gid


def test_goal_id_changes_with_any_scoring_field(wg):
    base = wg.goal_id(GOAL)
    for key, other in [
        ("asin", "B0002"),
        ("instruction_text", "something else"),
        ("attributes", ["argan"]),
        ("goal_options", {"size": "50ml"}),
        ("price_upper", 40.0),
        ("name", "Other Oil"),
        ("query", "oil"),
        ("product_category", "Other"),
    ]:
        assert wg.goal_id({**GOAL, key: other}) != base, key


def test_numeric_normalization(wg):
    # engine goals carry price_upper as int (1000000 = no cap) or float
    assert wg.goal_id({**GOAL, "price_upper": 30}) == wg.goal_id({**GOAL, "price_upper": 30.0})
    rec = wg.goal_record({**GOAL, "price_upper": 1000000})
    assert rec["price_upper"] == 1000000.0


def test_set_digest_is_order_free_and_ignores_repeats(wg):
    ids = [wg.goal_id({**GOAL, "asin": f"B{i:04d}"}) for i in range(5)]
    assert wg.goals_set_digest(ids) == wg.goals_set_digest(reversed(ids))
    assert wg.goals_set_digest(ids) != wg.goals_set_digest(ids[1:])
    # a goal repeated at another list position is the same question (WebShop's
    # synthetic generator does this when a product lists an option value twice)
    assert wg.goals_set_digest(ids + [ids[2]]) == wg.goals_set_digest(ids)
    assert len(wg.goals_set_digest(ids)) == 16
