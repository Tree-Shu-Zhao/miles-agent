"""Content identity for WebShop goals (stdlib only).

A WebShop "question" is a goal dict the engine builds at boot from the product
catalog (instruction text = the question; target asin, required attributes,
required options and price cap = the rubric ``get_reward`` scores against). The
engine keeps those goals in a seed-shuffled list, so a list position is NOT a
stable identity: change the goal seed, the catalog file or the goal code and
position 37 is another question. The frozen datasets under
``recipes/webshop/data`` therefore address goals by CONTENT:

  goal_id            = sha1(canonical JSON of the scoring-relevant fields)[:16]
  goals_set_digest   = sha1("\\n".join(sorted(goal_ids)))[:16]  (order-free)

Shared by the py3.10 engine (``env_servers/webshop``, which resolves goal_id ->
list index and serves ``GET /goals``), the py3.13 adapter tests, and
``recipes/webshop/prepare_data.py`` (which runs under the bare container python3
and imports this module by path). Keep it stdlib-only and import-side-effect free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

# Every field ``web_agent_site.engine.goal.get_reward`` reads (attributes,
# goal_options, name, price_upper, product_category, query) plus the question
# text and the target asin. ``weight`` (sampling weight) is deliberately not
# part of the identity.
GOAL_FIELDS: tuple[str, ...] = (
    "asin",
    "name",
    "category",
    "query",
    "product_category",
    "instruction_text",
    "attributes",
    "goal_options",
    "price_upper",
)


def goal_record(goal: Mapping[str, Any]) -> dict[str, Any]:
    """The normalized, JSON-serializable content record of a goal.

    Accepts either an engine goal dict (``SimServer.goals[i]``) or a record
    previously produced by this function (round-trips unchanged).
    """
    rec: dict[str, Any] = {}
    for key in GOAL_FIELDS:
        value = goal.get(key)
        if key == "price_upper":
            rec[key] = None if value is None else float(value)
        elif key == "attributes":
            rec[key] = [str(a) for a in (value or [])]
        elif key == "goal_options":
            if isinstance(value, Mapping):
                rec[key] = {str(k): str(v) for k, v in value.items()}
            else:
                rec[key] = [str(v) for v in (value or [])]
        else:
            rec[key] = None if value is None else str(value)
    return rec


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def goal_id(goal: Mapping[str, Any]) -> str:
    """16-hex content id of a goal (same input content -> same id, everywhere)."""
    return hashlib.sha1(canonical_json(goal_record(goal)).encode("utf-8")).hexdigest()[:16]


def goals_set_digest(goal_ids: Iterable[str]) -> str:
    """Order-free digest of a goal SET; equal digests <=> the same set of goal_id.

    Set, not multiset: WebShop's synthetic generator emits identical goals at
    several list positions when a product lists the same option value twice
    (9 such pairs on the 1000-product catalog). They are one question, so
    repeats never change the digest, and consumers collapse them the same way
    (the engine serves an id from its first position; prepare_data keeps the
    first row).
    """
    return hashlib.sha1("\n".join(sorted(set(goal_ids))).encode("utf-8")).hexdigest()[:16]
