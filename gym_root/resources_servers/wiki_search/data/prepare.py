"""FlashRAG -> wiki_search gym rows: train.jsonl, validation.jsonl and test.jsonl of the search_r1 recipe.
Caller: scripts/fetch_data.sh search_r1 (runs it with --output-dir $WIKI_SEARCH_DATA, then links train/validation into this dir).
Sources: RUC-NLPIR/FlashRAG_datasets on the Hugging Face Hub (NQ, HotpotQA, TriviaQA, PopQA, 2WikiMultihopQA, MuSiQue, Bamboogle; licenses: docs/data.md).

Convert FlashRAG splits into wiki_search gym JSONL (Search-R1 recipe).

train = nq.train + hotpotqa.train, merged and shuffled (nq_hotpotqa_train).
validation = the paper's 7 eval benchmarks: 128 seeded-random rows from each
of nq/triviaqa/popqa/hotpotqa/2wikimultihopqa/musique (test split where one
exists with public answers, dev otherwise) plus all 125 of bamboogle.test -
893 rows total. Row format matches data/example.jsonl exactly;
expected_answer keeps FlashRAG golden_answers as-is (aliases included) for
the binary-EM verifier.

Run anywhere with network access to huggingface.co:
    python prepare.py [--output-dir DIR] [--limit N] [--seed 42]
Requires: huggingface_hub (datasets library not needed - FlashRAG stores
plain JSONL files, downloaded directly via hf_hub_download).
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"

# (data_source, flashrag file, sample_n) per output split; sample_n = None takes
# every row, otherwise a seeded random sample of that many.
SPLITS = {
    "train": [
        ("nq", "nq/train.jsonl", None),
        ("hotpotqa", "hotpotqa/train.jsonl", None),
    ],
    "validation": [
        ("nq", "nq/test.jsonl", 128),
        ("triviaqa", "triviaqa/test.jsonl", 128),
        ("popqa", "popqa/test.jsonl", 128),
        ("hotpotqa", "hotpotqa/dev.jsonl", 128),
        ("2wikimultihopqa", "2wikimultihopqa/dev.jsonl", 128),
        ("musique", "musique/dev.jsonl", 128),
        ("bamboogle", "bamboogle/test.jsonl", None),  # only 125 rows - use all
    ],
    # Full Search-R1 eval protocol - the same 7 benchmarks, every row.
    "test": [
        ("nq", "nq/test.jsonl", None),
        ("triviaqa", "triviaqa/test.jsonl", None),
        ("popqa", "popqa/test.jsonl", None),
        ("hotpotqa", "hotpotqa/dev.jsonl", None),
        ("2wikimultihopqa", "2wikimultihopqa/dev.jsonl", None),
        ("musique", "musique/dev.jsonl", None),
        ("bamboogle", "bamboogle/test.jsonl", None),
    ],
}

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

USER_PROMPT_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. If you find you lack "
    "some knowledge, you can call the search tool to look up external "
    "information. You can search as many times as you want. When you are "
    "confident, provide your final answer inside <answer> and </answer> without "
    "detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {question}"
)

SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": "Searches the web for relevant information based on the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "A list of fully-formed semantic queries. The tool will return search results for each query."
                ),
            }
        },
        "required": ["query_list"],
        "additionalProperties": False,
    },
    "strict": True,
}


def make_row(question: str, golden_answers: list[str], data_source: str) -> dict:
    question = question.strip()
    # Search-R1 normalization: FlashRAG nq questions have no trailing '?'.
    if not question.endswith("?"):
        question += "?"
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
            ],
            "tools": [SEARCH_TOOL],
        },
        "expected_answer": golden_answers,
        "question": question,
        "data_source": data_source,
    }


def load_split(
    data_source: str,
    fpath_in_repo: str,
    limit: int | None,
    sample_n: int | None,
    rng: random.Random,
) -> list[dict]:
    local = hf_hub_download(FLASHRAG_REPO, fpath_in_repo, repo_type="dataset")
    rows, skipped = [], 0
    with open(local) as f:
        for line in f:
            raw = json.loads(line)
            question = (raw.get("question") or "").strip()
            answers = [a for a in raw.get("golden_answers") or [] if a and a.strip()]
            if not question or not answers:
                skipped += 1
                continue
            rows.append(make_row(question, answers, data_source))
            if limit is not None and len(rows) >= limit:
                break
    sampled = ""
    if sample_n is not None and len(rows) > sample_n:
        rows = rng.sample(rows, sample_n)
        sampled = f", sampled {sample_n}"
    print(f"  {fpath_in_repo}: {len(rows)} rows ({skipped} skipped empty{sampled})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--limit", type=int, default=None, help="Max rows per source split (smoke runs).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the merged train split.")
    parser.add_argument(
        "--splits", nargs="+", choices=sorted(SPLITS), default=sorted(SPLITS), help="Which splits to (re)generate."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split, sources in SPLITS.items():
        if split not in args.splits:
            continue
        print(f"[{split}]")
        rows = [
            row for source, fpath, sample_n in sources for row in load_split(source, fpath, args.limit, sample_n, rng)
        ]
        if split == "train":
            random.Random(args.seed).shuffle(rows)
        out = args.output_dir / f"{split}.jsonl"
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  -> {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
