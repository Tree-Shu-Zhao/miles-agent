"""Build Miles prompt data for the alfworld_env environment from $ALFWORLD_DATA.

Pure filesystem walk over the ALFWorld game files — no alfworld/textworld
imports, so it runs under the bare container python3 (those packages live
only in the resources-server venv). The env server re-resolves
metadata.game_file under its own $ALFWORLD_DATA at /reset time, so the path
is stored RELATIVE to --data-root: prep host and serving host need not share
a layout.

The prompt is prompts/alfworld/<--prompt>.py (default "default"; prompts/README.md),
rendered for --model (the HF checkpoint dir) by miles_agent/prompts.py: its example
act call is the policy model's own tool-call markup, so the rows belong to that
model family and the job builds them per run.

Miles row (consumed with --prompt-data ... --input-key prompt --metadata-key
metadata; actions travel as calls to the `act` tool declared per row — the
engine's tool-call parser turns them into structured tool_calls, and the
alfworld_env server reads arguments["command"], no text parsing anywhere):
    {"prompt": <the rendered prompt's messages>,
     "metadata": {"tools": [ACT_TOOL], "game_file": ...,
                  "max_episode_steps": ...,
                  "data_source": "alfworld/<task_type>", "split": ...}}

Layout: json_2.1.1/<split>/<task_type>-<params...>/<trial_id>/game.tw-pddl.
Games whose game.tw-pddl carries "solvable": false are skipped (the oracle
planner failed on them; they cannot yield reward 1.0).

Outputs:
    alfworld_train.jsonl    train split
    alfworld_eval274.jsonl  valid_seen rows FIRST, then valid_unseen — the
                            eval slice offsets in run.py index into this order
"""

import argparse
import json
import os
import sys
from pathlib import Path

# The repo root (miles_agent): this runs as a script from
# the training entry's checkout and from the repo root locally, never as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miles_agent import prompts  # noqa: E402
from miles_agent.tool_call_markup import hf_render_chat  # noqa: E402

DEFAULT_PROMPT = "default"  # prompts/alfworld/default.py

# Responses-form flat tool schema (same shape as wiki_search rows); gym_bridge
# moves metadata["tools"] into responses_create_params and the agents'
# to_chat_tools() nests it for chat completions. The name must match the env
# server's action_tool_name config (default "act").
ACT_TOOL = {
    "type": "function",
    "name": "act",
    "description": "Execute exactly one admissible command in the household environment and observe the result.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "One command copied verbatim from the admissible actions list.",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    # Responses-form FunctionToolParam requires strict; wiki_search rows set it too.
    "strict": True,
}


def prompt_messages(name: str, render_chat) -> list[dict]:
    """prompts/alfworld/<name>.py rendered for the model: the rows' "prompt" value."""
    return prompts.render(prompts.load("alfworld", name), render_chat, [ACT_TOOL])


TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

SPLITS = ("train", "valid_seen", "valid_unseen")


def is_solvable(game_file: Path) -> bool:
    """A game is skipped only on an explicit "solvable": false; unreadable or
    malformed JSON keeps the game (the env server is the authority on it)."""
    try:
        with open(game_file) as f:
            return json.load(f).get("solvable", True) is not False
    except (OSError, ValueError):
        return True


def collect_split(data_root: Path, split: str, max_episode_steps: int, prompt: list[dict]):
    rows = []
    unsolvable = 0
    for game_file in sorted(data_root.glob(f"json_2.1.1/{split}/**/game.tw-pddl")):
        if not is_solvable(game_file):
            unsolvable += 1
            continue
        task_dir = game_file.parent.parent.name
        task_type = task_dir.split("-")[0]
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown task type {task_type!r} in {game_file}")
        rows.append(
            {
                "prompt": [dict(m) for m in prompt],
                "metadata": {
                    "tools": [ACT_TOOL],
                    "game_file": str(game_file.relative_to(data_root)),
                    "max_episode_steps": max_episode_steps,
                    "data_source": f"alfworld/{task_type}",
                    "split": split,
                },
            }
        )
    return rows, unsolvable


def write_jsonl(rows, out_path: Path) -> None:
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(name: str, rows) -> None:
    by_type = {}
    for row in rows:
        by_type[row["metadata"]["data_source"]] = by_type.get(row["metadata"]["data_source"], 0) + 1
    print(f"{name}: {len(rows)} rows")
    for source, n in sorted(by_type.items()):
        print(f"  {source}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("ALFWORLD_DATA"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-episode-steps", type=int, default=30)
    ap.add_argument("--model", required=True, help="policy model HF checkpoint dir; its chat template renders the example call")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"prompts/alfworld/<name>.py (default {DEFAULT_PROMPT})")
    args = ap.parse_args()
    if not args.data_root:
        ap.error("--data-root is required (or set ALFWORLD_DATA)")

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_messages(args.prompt, hf_render_chat(args.model))
    print(f"prompt {args.prompt} for {args.model} ({len(prompt)} messages):\n{prompt[0]['content']}\n")

    splits = {}
    for split in SPLITS:
        rows, unsolvable = collect_split(data_root, split, args.max_episode_steps, prompt)
        splits[split] = rows
        report(split, rows)
        if unsolvable:
            print(f"  skipped {unsolvable} unsolvable games")

    write_jsonl(splits["train"], out_dir / "alfworld_train.jsonl")
    # Slice order is load-bearing: run.py evaluates alfworld_seen on
    # [0:len(valid_seen)] and alfworld_unseen on the remainder.
    write_jsonl(splits["valid_seen"] + splits["valid_unseen"], out_dir / "alfworld_eval274.jsonl")
    print(f"wrote {len(splits['train'])} rows -> {out_dir / 'alfworld_train.jsonl'}")
    print(
        f"wrote {len(splits['valid_seen']) + len(splits['valid_unseen'])} rows "
        f"({len(splits['valid_seen'])} valid_seen + {len(splits['valid_unseen'])} valid_unseen) "
        f"-> {out_dir / 'alfworld_eval274.jsonl'}"
    )


if __name__ == "__main__":
    main()
