"""gym_root/resources_servers/wiki_search/data/prepare.py (the FlashRAG -> gym row builder that scripts/fetch_data.sh
search_r1 runs) against local fixtures: a fake `huggingface_hub` module is put into sys.modules BEFORE the module is
loaded from its file, so its `hf_hub_download` hands back fixture files instead of reaching the Hub (the real package
is installed in the test venv, so a missing-module assumption would silently exercise it). The paper's validation
composition — 128 seeded rows from each of six benchmarks plus all 125 of bamboogle = 893 — is what the recipe ladder
gates on (EV == 893), so it is pinned here together with the row schema of data/example.jsonl.
"""

import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREPARE = REPO / "gym_root" / "resources_servers" / "wiki_search" / "data" / "prepare.py"
EXAMPLE = PREPARE.parent / "example.jsonl"
# the FlashRAG files prepare.py asks for, with a fixture row count each: 200 everywhere (above the 128 sample size)
# except bamboogle, whose real file has 125 rows and is taken whole
SOURCES = {
    "nq/train.jsonl": 200,
    "hotpotqa/train.jsonl": 200,
    "nq/test.jsonl": 200,
    "triviaqa/test.jsonl": 200,
    "popqa/test.jsonl": 200,
    "hotpotqa/dev.jsonl": 200,
    "2wikimultihopqa/dev.jsonl": 200,
    "musique/dev.jsonl": 200,
    "bamboogle/test.jsonl": 125,
}


@pytest.fixture
def fixtures(tmp_path):
    """The FlashRAG layout under tmp: `{question, golden_answers}` rows, plus two rows per file that prepare.py must
    skip (an empty question, no answers)."""
    root = tmp_path / "flashrag"
    for rel, n in SOURCES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i in range(n):
                f.write(json.dumps({"id": f"{rel}-{i}", "question": f"question {i} of {rel}",
                                    "golden_answers": [f"answer {i}", f"alias {i}"]}) + "\n")
            f.write(json.dumps({"question": "   ", "golden_answers": ["x"]}) + "\n")
            f.write(json.dumps({"question": "no answers", "golden_answers": ["", " "]}) + "\n")
    return root


def _load_prepare(monkeypatch, fixtures):
    """prepare.py loaded by path with a fake huggingface_hub whose hf_hub_download returns the fixture file; returns
    (module, the recorded download calls)."""
    calls = []
    fake = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename, repo_type=None, **kwargs):
        calls.append((repo_id, filename, repo_type))
        return str(fixtures / filename)

    fake.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    spec = importlib.util.spec_from_file_location("_wiki_search_prepare", PREPARE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.hf_hub_download is hf_hub_download
    return mod, calls


def _rows(out, split):
    return [json.loads(line) for line in (out / f"{split}.jsonl").read_text().splitlines()]


def _run(mod, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prepare.py", *args])
    mod.main()


def test_prepare_builds_the_893_row_validation_split_from_local_fixtures(tmp_path, monkeypatch, fixtures):
    mod, calls = _load_prepare(monkeypatch, fixtures)
    assert mod.FLASHRAG_REPO == "RUC-NLPIR/FlashRAG_datasets"
    out = tmp_path / "out"
    _run(mod, monkeypatch, "--output-dir", str(out))
    assert sorted(p.name for p in out.iterdir()) == ["test.jsonl", "train.jsonl", "validation.jsonl"]
    assert all(c == (mod.FLASHRAG_REPO, c[1], "dataset") for c in calls) and len(calls) == 16
    assert [c[1] for c in calls][:2] == ["nq/train.jsonl", "hotpotqa/train.jsonl"]
    # validation: 128 seeded rows from each of the six sampled benchmarks + all 125 bamboogle rows = 893
    validation = _rows(out, "validation")
    assert len(validation) == 893
    assert Counter(r["data_source"] for r in validation) == {
        "nq": 128, "triviaqa": 128, "popqa": 128, "hotpotqa": 128, "2wikimultihopqa": 128, "musique": 128,
        "bamboogle": 125,
    }
    # train = the union of nq.train and hotpotqa.train (every row, the skipped two excepted), shuffled
    train = _rows(out, "train")
    assert len(train) == 400 and Counter(r["data_source"] for r in train) == {"nq": 200, "hotpotqa": 200}
    assert {r["question"] for r in train} == {f"question {i} of {rel}?" for rel in ("nq/train.jsonl", "hotpotqa/train.jsonl")
                                             for i in range(200)}
    assert [r["question"] for r in train] != sorted(r["question"] for r in train)
    # test = every row of the seven benchmarks
    test = _rows(out, "test")
    assert len(test) == 6 * 200 + 125
    # the row schema is data/example.jsonl's: the same keys, the same system message and search tool, the question
    # inside the user prompt with its trailing "?", FlashRAG's golden_answers kept whole as expected_answer
    example = json.loads(EXAMPLE.read_text().splitlines()[0])
    for row in (validation[0], train[0], test[-1]):
        assert set(row) == set(example)
        assert set(row["responses_create_params"]) == set(example["responses_create_params"])
        assert row["responses_create_params"]["tools"] == example["responses_create_params"]["tools"]
        assert row["responses_create_params"]["input"][0] == example["responses_create_params"]["input"][0]
        user = row["responses_create_params"]["input"][1]
        assert user["role"] == "user" and user["content"].endswith("Question: " + row["question"])
        assert row["question"].endswith("?") and len(row["expected_answer"]) == 2
    assert test[-1]["data_source"] == "bamboogle"
    # deterministic: a second run reproduces every file byte for byte (--seed 42)
    again = tmp_path / "again"
    _run(mod, monkeypatch, "--output-dir", str(again))
    for split in ("train", "validation", "test"):
        assert (again / f"{split}.jsonl").read_bytes() == (out / f"{split}.jsonl").read_bytes(), split
    # --limit N caps every source at N rows (below the sample size nothing is sampled); --splits selects the files
    limited = tmp_path / "limited"
    _run(mod, monkeypatch, "--output-dir", str(limited), "--limit", "10", "--splits", "validation", "train")
    assert sorted(p.name for p in limited.iterdir()) == ["train.jsonl", "validation.jsonl"]
    assert len(_rows(limited, "validation")) == 70 and len(_rows(limited, "train")) == 20
    assert all(int(r["question"].split()[1]) < 10 for r in _rows(limited, "train"))


def test_prepare_defaults_to_its_own_dir_and_only_imports_the_hub_downloader():
    """The output dir defaults to the module's own data dir (whose .gitignore covers every file it writes) and the
    imports are the stdlib plus huggingface_hub (no datasets library, no credentials)."""
    text = PREPARE.read_text()
    assert 'parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)' in text
    ignored = (PREPARE.parent / ".gitignore").read_text().split()
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert any(Path(name).match(pattern) for pattern in ignored), name
    imports = [line for line in text.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["import argparse", "import json", "import random", "from pathlib import Path",
                       "from huggingface_hub import hf_hub_download"]
    for word in ("token", "HF_TOKEN", "password"):
        assert word not in text
    head = text.split('"""')[1].splitlines()
    assert head[1].startswith("Caller: scripts/fetch_data.sh search_r1")
    assert "RUC-NLPIR/FlashRAG_datasets" in head[2] and "docs/data.md" in head[2]
