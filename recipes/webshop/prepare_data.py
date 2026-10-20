"""Build Miles prompt data for the webshop_env environment.

Rows address goals by CONTENT, not by position. Each row carries
``metadata.goal_id`` = sha1 of the goal's scoring-relevant fields
(``gym_root/agent_utils/webshop_goals.py``: asin, name, category, query,
product_category, instruction_text, attributes, goal_options, price_upper). The
engine (env_servers/webshop) resolves goal_id -> its own goal at /reset and
refuses ids it does not know, and the training entry refuses to start when
the engine's goal SET digest differs from the dataset's. A list position was
never a stable identity: change the goal seed, the catalog file or the vendored
goal code and position 37 is another question, silently.

The goals come from the frozen export ``webshop_goals.jsonl`` (written from the
real engine by ``env_servers/webshop/export_goals.py``, rebuilt bit-for-bit by
``scripts/fetch_data.sh webshop --regenerate-goals`` and named by ``WEBSHOP_GOALS_FILE``;
not committed, provenance + digest in ``recipes/webshop/data/README.md``) and are consumed IN FILE
ORDER: positions in that FILE define the splits below, after collapsing rows
that repeat an earlier row's content (same question; 9 such rows on the small
catalog, the first at raw position 1367). The file order is the engine's
seed-233 list order (the original WebShop / TCOD constant), so up to the first
twin a position is TCOD's task_id; past it our positions sit up to 9 places
before TCOD's raw ids. The dataset itself no longer depends on any runtime
knob: the engine resolves goal_id -> goal, and the job refuses to start when
the engine's goal SET digest differs from the file's.

Two split layouts (--protocol; default canonical), over distinct-question positions:

  tcod       TCOD_examples/webshop/get_webshop_data.py: train = pos 0..4095
             (4096 goals), test = pos 4096..4195 (100 goals). webshop_eval.jsonl
             = that test set (rows 0..99, split "tcod_test") followed by a
             400-goal held-out companion, pos 4196..4595 (rows 100..499, split
             "heldout"; never trained, only there to read the curve with less
             noise than 100 goals give). NOTE: TCOD's train range contains the
             WebShop paper's test set (pos < 500), so a tcod-trained model is
             not comparable to leaderboard numbers.
  canonical  (default) WebShop paper (baseline_models/env.py): test = pos
             0..499, fixed and never trained; train = the rest.
             webshop_eval.jsonl = the 500 held-out goals (split "eval") plus
             POOL_GOAL_COUNT goals sampled from the WHOLE pool with the constant
             POOL_SAMPLE_SEED (split "eval_pool", the distribution
             verl-agent/GiGPO validate on).

Stdlib plus two repo modules imported by path (no package install): the shared
goal identity and miles_agent/tool_call_markup.py, so it runs under the bare
container python3 (the engine deps live only in the py3.10 engine env). Actions
travel as calls to the `act` tool declared per row (the engine's tool-call
parser turns them into structured tool_calls; the webshop_env adapter reads
arguments["command"], no text parsing). The prompt is prompts/webshop/<--prompt>.py
(prompts/README.md): "think" (default; TCOD protocol: reason inside <think></think>
first, then the tool call) or "none" (tool call only; wstrain1/wstrain2), rendered
for --model (the HF checkpoint dir) by miles_agent/prompts.py, so its example call
is the policy model's own markup, the rows belong to that model family and the job
builds them per run. Miles row (consumed with --prompt-data ... --input-key prompt
--metadata-key metadata):

    {"prompt": <the rendered prompt's messages>,
     "metadata": {"tools": [ACT_TOOL], "goal_id": <16 hex>,
                  "instruction_text": <the question, for humans reading logs>,
                  "split": ..., "max_episode_steps": ..., "data_source": "webshop"}}

A protocol.json sidecar records every parameter (the model, the prompt's name
and sha256 included) plus the goal-set digest (``goals_digest`` =
order-free digest of ALL ids in the goals file; the job compares it with the
engine's /health.goals_digest) next to the two files.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

# Import the shared modules by path: this script runs as
# `python3 recipes/webshop/prepare_data.py` from the repo root (in the container too)
# and from the repo root locally, never from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gym_root"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_utils.webshop_goals import goal_id, goals_set_digest  # noqa: E402

from miles_agent import prompts  # noqa: E402
from miles_agent.tool_call_markup import hf_render_chat  # noqa: E402

# Where scripts/fetch_data.sh webshop --regenerate-goals writes the export (gitignored: data/README.md);
# WEBSHOP_GOALS_FILE moves both the exporter and this reader.
DEFAULT_GOALS_PATH = Path(__file__).resolve().parent / "data" / "webshop_goals.jsonl"


def default_goals_path() -> Path:
    return Path(os.environ.get("WEBSHOP_GOALS_FILE") or DEFAULT_GOALS_PATH)


# Constant seed of the canonical protocol's pool sample over FILE positions. A
# constant (not a CLI knob) because the dataset must be a pure function of the
# goals file: 233 for continuity with the export order's engine seed, nothing
# else depends on it.
POOL_SAMPLE_SEED = 233
PROTOCOLS = ("tcod", "canonical")
DEFAULT_PROMPT = "think"  # prompts/webshop/think.py (TCOD protocol)

# --- tcod layout (TCOD_examples/webshop/get_webshop_data.py) ---
TCOD_TRAIN_GOALS = 4096  # distinct-question pos 0..4095
TCOD_TEST_GOALS = 100  # distinct-question pos 4096..4195
HELDOUT_GOALS = 400  # our companion slice: pos 4196..4595, never trained
TCOD_EVAL_ROWS = TCOD_TEST_GOALS + HELDOUT_GOALS  # 500

# --- canonical layout (WebShop paper) ---
EVAL_GOAL_COUNT = 500  # test = file pos < 500
# Second validation view: goals sampled uniformly from the WHOLE pool (trained
# goals included). verl-agent/GiGPO validate on exactly this distribution: each
# of their env workers shuffles the goal list with its own seed, so "test =
# idx < 500" there is a uniform draw from all goals, re-drawn every validation
# (val_data_size 128 x 2). A fixed seeded sample keeps our curve comparable
# to the paper's Score/Success while staying reproducible across runs.
POOL_GOAL_COUNT = 256
CANONICAL_EVAL_ROWS = EVAL_GOAL_COUNT + POOL_GOAL_COUNT  # 756

ACT_TOOL = {
    "type": "function",
    "name": "act",
    "description": "Execute exactly one admissible WebShop command and observe the resulting page.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "One command copied verbatim from the admissible actions list, "
                "e.g. search[red ceramic mug] or click[b09xyz] or click[buy now].",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}


def prompt_messages(name: str, render_chat) -> list[dict]:
    """prompts/webshop/<name>.py rendered for the model: the rows' "prompt" value."""
    return prompts.render(prompts.load("webshop", name), render_chat, [ACT_TOOL])


def load_goals(path: Path) -> tuple[list[dict], int]:
    """The frozen goals export, in file order, content-verified.

    Every row must carry a ``goal_id`` equal to the id recomputed from its own
    fields (a hand edit or a changed identity function would otherwise ship
    ids the engine cannot resolve). Rows repeating an earlier row's content
    (WebShop's synthetic generator emits identical goals when a product lists
    an option value twice; 9 pairs on the 1000-product catalog) are one
    question: only the FIRST occurrence is kept, so a held-out goal can never
    be trained through a twin and the layouts below run over distinct
    questions. Returns (unique goals in file order, rows read).
    """
    if not path.is_file():
        raise ValueError(
            f"goals file {path} not found — rebuild it with scripts/fetch_data.sh webshop --regenerate-goals "
            "(env_servers/webshop/export_goals.py against a running engine; WEBSHOP_GOALS_FILE names another "
            "path; recipes/webshop/data/README.md)"
        )
    goals: list[dict] = []
    seen: set[str] = set()
    rows_read = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows_read += 1
            stored, actual = row.get("goal_id"), goal_id(row)
            if stored != actual:
                raise ValueError(
                    f"{path}:{lineno}: stored goal_id {stored!r} != content id {actual!r} — the row was edited "
                    "after export or the identity fields changed; re-export the goals file"
                )
            if stored in seen:
                continue  # same content as an earlier row: same question, first position serves it
            seen.add(stored)
            goals.append(row)
    if not goals:
        raise ValueError(f"goals file {path} is empty")
    return goals, rows_read


def write_jsonl(
    path: Path,
    rows: Iterable[tuple[int, str]],
    goals: list[Mapping],
    max_episode_steps: int,
    prompt: list[dict],
) -> int:
    """rows = (file position, split label) pairs, written in order; each
    position is resolved to the goal at that position of the goals file and the
    row carries its content id. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for pos, split in rows:
            n += 1
            goal = goals[pos]
            row = {
                "prompt": [dict(m) for m in prompt],
                "metadata": {
                    "tools": [ACT_TOOL],
                    "goal_id": goal["goal_id"],
                    "instruction_text": goal.get("instruction_text"),
                    "max_episode_steps": max_episode_steps,
                    "data_source": "webshop",
                    "split": split,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {n} rows -> {path}")
    return n


def split_rows(protocol: str, num_goals: int) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(eval rows, train rows) as (file position, split) pairs for a protocol;
    eval row ORDER is contractual (recipes/webshop/run.py reads the file as
    index slices). num_goals = rows in the goals file."""
    if protocol == "tcod":
        need = TCOD_TRAIN_GOALS + TCOD_EVAL_ROWS
        if num_goals < need:
            raise ValueError(
                f"tcod protocol needs at least {need} goals (train 4096 + eval {TCOD_EVAL_ROWS}), got {num_goals}"
            )
        test_end = TCOD_TRAIN_GOALS + TCOD_TEST_GOALS
        eval_rows = [(i, "tcod_test") for i in range(TCOD_TRAIN_GOALS, test_end)]
        eval_rows += [(i, "heldout") for i in range(test_end, test_end + HELDOUT_GOALS)]
        train_rows = [(i, "train") for i in range(TCOD_TRAIN_GOALS)]
        return eval_rows, train_rows
    if protocol == "canonical":
        if num_goals <= EVAL_GOAL_COUNT:
            raise ValueError(f"canonical protocol needs more than {EVAL_GOAL_COUNT} goals, got {num_goals}")
        pool = random.Random(POOL_SAMPLE_SEED).sample(range(num_goals), POOL_GOAL_COUNT)
        eval_rows = [(i, "eval") for i in range(EVAL_GOAL_COUNT)] + [(i, "eval_pool") for i in pool]
        train_rows = [(i, "train") for i in range(EVAL_GOAL_COUNT, num_goals)]
        return eval_rows, train_rows
    raise ValueError(f"unknown protocol {protocol!r} (choose from {PROTOCOLS})")


def eval_slices(protocol: str) -> dict[str, tuple[int, int]]:
    """Row ranges of webshop_eval.jsonl per eval slice name (mirrored by
    recipes/webshop/run.py EVAL_SLICES)."""
    if protocol == "tcod":
        return {"webshop": (0, TCOD_TEST_GOALS), "webshop_heldout": (TCOD_TEST_GOALS, TCOD_EVAL_ROWS)}
    if protocol == "canonical":
        return {"webshop": (0, EVAL_GOAL_COUNT), "webshop_pool": (EVAL_GOAL_COUNT, CANONICAL_EVAL_ROWS)}
    raise ValueError(f"unknown protocol {protocol!r} (choose from {PROTOCOLS})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--goals",
        type=Path,
        default=None,
        help="frozen goals export (JSONL with goal_id per row, consumed in file order; default WEBSHOP_GOALS_FILE, "
        f"else {DEFAULT_GOALS_PATH})",
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-episode-steps", type=int, default=15)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="canonical", help="split layout (default canonical)")
    ap.add_argument("--model", required=True, help="policy model HF checkpoint dir; its chat template renders the example call")
    ap.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help=f"prompts/webshop/<name>.py (default {DEFAULT_PROMPT} = TCOD)"
    )
    args = ap.parse_args(argv)
    goals_path = args.goals or default_goals_path()
    try:
        goals, rows_read = load_goals(goals_path)
        eval_rows, train_rows = split_rows(args.protocol, len(goals))
    except ValueError as e:
        ap.error(str(e))

    out = args.output_dir
    prompt = prompt_messages(args.prompt, hf_render_chat(args.model))
    n_eval = write_jsonl(out / "webshop_eval.jsonl", eval_rows, goals, args.max_episode_steps, prompt)
    n_train = write_jsonl(out / "webshop_train.jsonl", train_rows, goals, args.max_episode_steps, prompt)
    digest = goals_set_digest(g["goal_id"] for g in goals)
    sidecar = {
        "protocol": args.protocol,
        "prompt": args.prompt,
        "model": str(args.model),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt, ensure_ascii=False).encode()).hexdigest(),
        "goals": rows_read,
        "unique_goals": len(goals),
        "duplicate_rows": rows_read - len(goals),
        "goals_digest": digest,
        "goals_file": str(goals_path),
        "goals_file_sha256": hashlib.sha256(goals_path.read_bytes()).hexdigest(),
        "max_episode_steps": args.max_episode_steps,
        "eval_rows": n_eval,
        "train_rows": n_train,
        "eval_slices": {k: list(v) for k, v in eval_slices(args.protocol).items()},
        "train_file_positions": [train_rows[0][0], train_rows[-1][0]],
    }
    (out / "protocol.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"protocol={args.protocol} prompt={args.prompt} goals={rows_read} unique={len(goals)} "
        f"goals_digest={digest} "
        f"-> {out / 'protocol.json'}"
    )


if __name__ == "__main__":
    main()
