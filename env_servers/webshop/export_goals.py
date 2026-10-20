#!/usr/bin/env python3
"""Freeze a running WebShop engine's goal list into the recipe's goals file.

    python3 export_goals.py --engine-url http://127.0.0.1:6200 \\
        --output recipes/webshop/data/webshop_goals.jsonl

Why a frozen file: the engine builds its goals at boot from the catalog under a
shuffle seed, so a list position is only a stable question while catalog, goal
code and seed all stay fixed. recipes/webshop/prepare_data.py therefore reads
this export (one JSON row per goal, in the engine's list order, keyed by the
content id ``goal_id``) and the job refuses to start unless the live engine's
/health ``goals_digest`` equals the digest of the ids in this file.

What is written (only after every check passes; nothing is written otherwise):
  <output>            JSONL, one goal per line in goal_idx order, keys ROW_FIELDS
  <output>.meta.json  {"goals", "goals_digest", "protocol", "exported_at", "engine_url"}

Checks: /health carries ``protocol`` + ``goals_digest`` (i.e. this IS a WebShop
engine server of the current protocol), /goals rows have contiguous goal_idx,
each row's stored goal_id equals the id recomputed from its content, and the
set digest of the rows equals /health's (the engine did not change under us).

stdlib only and py3.10-compatible: it runs under the engine's micromamba python
(env_servers/webshop/launch_engine.sh's runtime) as well as the
repo's root venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from agent_utils.webshop_goals import goal_id, goals_set_digest
except ImportError:
    # Same path fallback as _webshop_engine.py: gym_root/ is a sibling of
    # env_servers/ under the checkout root and cannot be pip-installed here.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gym_root"))
    from agent_utils.webshop_goals import goal_id, goals_set_digest

# Row key order: identity first, then the question text, then the rubric
# fields, so the file diffs and greps naturally.
ROW_FIELDS: tuple[str, ...] = (
    "goal_id",
    "goal_idx",
    "instruction_text",
    "asin",
    "name",
    "category",
    "query",
    "product_category",
    "attributes",
    "goal_options",
    "price_upper",
)


def fetch_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback engine URL
        return json.loads(resp.read().decode("utf-8"))


def export(engine_url: str, output: Path, timeout: float = 600.0) -> dict[str, Any]:
    """Fetch, verify and write; returns the sidecar metadata. Raises SystemExit
    with a one-line reason on any check failure (nothing written)."""
    base = engine_url.rstrip("/")
    health = fetch_json(f"{base}/health", timeout)
    for key in ("protocol", "goals_digest"):
        if not isinstance(health, dict) or key not in health:
            raise SystemExit(f"{base}/health carries no {key!r}: not a WebShop engine server of the current protocol")

    rows = fetch_json(f"{base}/goals", timeout)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{base}/goals returned no goal rows")
    rows = sorted(rows, key=lambda r: r["goal_idx"])
    if [r["goal_idx"] for r in rows] != list(range(len(rows))):
        raise SystemExit("refusing to write: /goals goal_idx values are not the contiguous range 0..n-1")
    for row in rows:
        if goal_id(row) != row["goal_id"]:
            raise SystemExit(
                f"refusing to write: goal_idx {row['goal_idx']} carries goal_id {row['goal_id']!r} but its content "
                f"hashes to {goal_id(row)!r} (engine and export disagree on the identity fields)"
            )
    digest = goals_set_digest(r["goal_id"] for r in rows)
    if digest != health["goals_digest"]:
        raise SystemExit(
            f"refusing to write: digest of the {len(rows)} exported rows ({digest}) != /health goals_digest "
            f"({health['goals_digest']}); the engine's goal set changed between the two requests"
        )
    if health.get("goals") not in (None, len(rows)):
        raise SystemExit(f"refusing to write: /goals has {len(rows)} rows but /health reports goals={health['goals']}")

    meta = {
        "goals": len(rows),
        "goals_digest": digest,
        "protocol": health["protocol"],
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 - py3.10 runtime
        "engine_url": engine_url,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in ROW_FIELDS}, ensure_ascii=False) + "\n")
    Path(f"{output}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine-url", default="http://127.0.0.1:6200", help="running WebShop engine server")
    parser.add_argument("--output", type=Path, required=True, help="JSONL path to write (plus <output>.meta.json)")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    args = parser.parse_args(argv)
    meta = export(args.engine_url, args.output, args.timeout)
    print(
        f"exported {meta['goals']} goals (goals_digest {meta['goals_digest']}, protocol "
        f"{json.dumps(meta['protocol'], sort_keys=True)}) -> {args.output} (+ .meta.json)"
    )


if __name__ == "__main__":
    main()
