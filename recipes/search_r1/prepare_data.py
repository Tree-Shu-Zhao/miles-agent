"""Convert nemo-gym wiki_search JSONL rows into Miles prompt data.

Gym row:
    {"responses_create_params": {"input": [messages], "tools": [SEARCH_TOOL]},
     "expected_answer": [...], "question": ..., "data_source": ...}

Miles row (consumed with --prompt-data ... --input-key prompt --metadata-key metadata):
    {"prompt": [messages],
     "metadata": {"tools": [...], "expected_answer": ..., "data_source": ...}}

NO template rendering here — that was the slime Mode A design. In the Miles
black-box flow the session server owns tokenization (fixed per-family TITO
template), so data prep is a pure field shuffle: the agent function rebuilds
the /run body from prompt + metadata.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input) as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            params = row["responses_create_params"]
            fout.write(
                json.dumps(
                    {
                        "prompt": params["input"],
                        "metadata": {
                            "tools": params.get("tools") or [],
                            "expected_answer": row["expected_answer"],
                            "data_source": row.get("data_source"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
