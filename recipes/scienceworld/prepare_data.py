"""Build Miles prompt data for the scienceworld_env environment (TCOD-aligned task splits).

Stdlib plus the env server's own task enumeration (gym_root/resources_servers/
scienceworld_env/task_splits.py, the module the server validates rows against; no
scienceworld / py4j / java here): a row names its task by task_name + var_num and the
server loads it into a worker JVM at /reset time. The frozen TCOD eval taskset
scienceworld_test1308.jsonl (1308 rows of {"task_name", "var_num"}) is not shipped:
scripts/fetch_data.sh scienceworld writes it and SCIENCEWORLD_TEST_LIST names it
(data/README.md).

The prompt is prompts/scienceworld/<--prompt>.py (default "default"; prompts/README.md),
rendered for --model (the HF checkpoint dir) by miles_agent/prompts.py: a prompt that
demonstrates act calls (r2s_oneshot) carries the policy model's own tool-call markup, so
the rows belong to that model family and the job builds them per run.

Miles row (consumed with --prompt-data ... --input-key prompt --metadata-key
metadata; actions travel as calls to the `act` tool declared per row — the
engine's tool-call parser turns them into structured tool_calls, and the
scienceworld_env server reads arguments["command"], no text parsing anywhere):
    {"prompt": <the rendered prompt's messages>,
     "metadata": {"tools": [ACT_TOOL], "task_name": ..., "var_num": ..., "split": ...,
                  "max_episode_steps": ..., "data_source": "scienceworld/<task_name>"}}

Splits (TCOD-aligned, arXiv 2604.24005; see task_splits.py):
    train        2294 rows - 17 train task types x first 50% variations
    valid_seen    140 rows - seed-42 sample of train_heldout (train task types, variations
                             DISJOINT from train; our addition, TCOD has no seen-task eval)
    valid_unseen  134 rows - the FIRST 134 rows of the frozen TCOD test taskset (file order;
                             the file is already seed-42 shuffled by the repro campaign)
    test_full    1308 rows - the full frozen TCOD eval set

Outputs:
    scienceworld_train.jsonl     train split
    scienceworld_eval274.jsonl   valid_seen rows FIRST, then valid_unseen — the
                                 eval slice offsets in run.py index into this order
    scienceworld_test1308.jsonl  the full frozen TCOD eval set (a final evaluation)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The repo root (miles_agent) and the env servers (scienceworld_env.task_splits): this
# runs as a script from the training entry's checkout and from the repo root locally,
# never as an installed package.
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "gym_root" / "resources_servers"))
from scienceworld_env.task_splits import build_splits  # noqa: E402

from miles_agent import prompts  # noqa: E402
from miles_agent.tool_call_markup import hf_render_chat  # noqa: E402

DEFAULT_PROMPT = "default"  # prompts/scienceworld/default.py
VALID_SEEN_N = 140
VALID_UNSEEN_N = 134
SEED = 42


def default_test_jsonl() -> Path:
    """SCIENCEWORLD_TEST_LIST, else <MILES_AGENT_DATA or <repo>/data>/scienceworld/scienceworld_test1308.jsonl:
    where scripts/fetch_data.sh scienceworld writes the frozen TCOD eval taskset (data/README.md). Resolved at
    call time (the tests set the environment per case); launcher.data_path is not imported here because
    miles_agent.launcher imports miles at module level and this script runs under the bare container python."""
    named = os.environ.get("SCIENCEWORLD_TEST_LIST")
    if named:
        return Path(named)
    data_root = Path(os.environ.get("MILES_AGENT_DATA") or _HERE.parents[1] / "data")
    return data_root / "scienceworld" / "scienceworld_test1308.jsonl"


# Responses-form flat tool schema (same shape as the alfworld rows); gym_bridge
# moves metadata["tools"] into responses_create_params and the agents'
# to_chat_tools() nests it for chat completions. The name must match the env
# server's action_tool_name config (default "act").
ACT_TOOL = {
    "type": "function",
    "name": "act",
    "description": "Execute exactly one action in the ScienceWorld environment and observe the result.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "One action formed from an available action command and object(s), "
                    "e.g. 'look around', 'open door to kitchen' or 'focus on thermometer'."
                ),
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}


def prompt_messages(name: str, render_chat) -> list[dict]:
    """prompts/scienceworld/<name>.py rendered for the model: the rows' "prompt" value."""
    return prompts.render(prompts.load("scienceworld", name), render_chat, [ACT_TOOL])


def make_row(task_name: str, var_num: int, split: str, max_episode_steps: int, prompt: list[dict]) -> dict:
    return {
        "prompt": [dict(m) for m in prompt],
        "metadata": {
            "tools": [ACT_TOOL],
            "task_name": task_name,
            "var_num": int(var_num),
            "split": split,
            "max_episode_steps": max_episode_steps,
            "data_source": f"scienceworld/{task_name}",
        },
    }


def build_rows(test_jsonl: Path, max_episode_steps: int, prompt: list[dict]) -> dict[str, list[dict]]:
    splits = build_splits(test_jsonl)
    train, heldout, test = splits["train"], splits["train_heldout"], splits["test"]
    seen_idx = sorted(random.Random(SEED).sample(range(len(heldout)), VALID_SEEN_N))
    return {
        "train": [make_row(t, v, "train", max_episode_steps, prompt) for t, v in train],
        "valid_seen": [make_row(*heldout[i], "train_heldout", max_episode_steps, prompt) for i in seen_idx],
        "valid_unseen": [make_row(t, v, "test", max_episode_steps, prompt) for t, v in test[:VALID_UNSEEN_N]],
        "test_full": [make_row(t, v, "test", max_episode_steps, prompt) for t, v in test],
    }


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
        help="frozen TCOD eval taskset (default: SCIENCEWORLD_TEST_LIST, else "
        "<MILES_AGENT_DATA>/scienceworld/scienceworld_test1308.jsonl; data/README.md)",
    )
    ap.add_argument("--max-episode-steps", type=int, default=30)
    ap.add_argument(
        "--model", required=True, help="policy model HF checkpoint dir; its chat template renders the prompt's calls"
    )
    ap.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help=f"prompts/scienceworld/<name>.py (default {DEFAULT_PROMPT})"
    )
    args = ap.parse_args(argv)

    test_jsonl = args.test_jsonl or default_test_jsonl()
    if not test_jsonl.is_file():
        sys.exit(
            f"FATAL: ScienceWorld eval taskset {test_jsonl} not found: run scripts/fetch_data.sh scienceworld "
            "(--from <TCOD campaign test.jsonl> for the paper's order, --synthesize for the same task set in sorted "
            "order) or set SCIENCEWORLD_TEST_LIST (recipes/scienceworld/data/README.md)"
        )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_messages(args.prompt, hf_render_chat(args.model))
    print(f"prompt {args.prompt} for {args.model} ({len(prompt)} messages):\n{prompt[0]['content']}\n")

    rows = build_rows(test_jsonl, args.max_episode_steps, prompt)
    for split in ("train", "valid_seen", "valid_unseen", "test_full"):
        report(split, rows[split])

    write_jsonl(rows["train"], out_dir / "scienceworld_train.jsonl")
    # Slice order is load-bearing: run.py evaluates scienceworld_seen on
    # [0:len(valid_seen)] and scienceworld_unseen on the remainder.
    write_jsonl(rows["valid_seen"] + rows["valid_unseen"], out_dir / "scienceworld_eval274.jsonl")
    write_jsonl(rows["test_full"], out_dir / "scienceworld_test1308.jsonl")
    n_eval = len(rows["valid_seen"]) + len(rows["valid_unseen"])
    print(f"wrote {len(rows['train'])} rows -> {out_dir / 'scienceworld_train.jsonl'}")
    print(
        f"wrote {n_eval} rows ({len(rows['valid_seen'])} valid_seen + {len(rows['valid_unseen'])} valid_unseen) "
        f"-> {out_dir / 'scienceworld_eval274.jsonl'}"
    )
    print(f"wrote {len(rows['test_full'])} rows -> {out_dir / 'scienceworld_test1308.jsonl'}")


if __name__ == "__main__":
    main()
