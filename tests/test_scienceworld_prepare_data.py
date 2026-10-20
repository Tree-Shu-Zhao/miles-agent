"""recipes/scienceworld/prepare_data.py: TCOD-aligned splits, the eval order contract, row shape,
prompt selection, output files.

Bare-python3 module (it runs under the container python3), loaded from its file so the test
needs no package install. The policy model's chat template comes from miles' fixed Qwen
templates instead of a checkpoint (tests/conftest.py template_render_chat; --model is then
only recorded). The paper's own eval274 rows (the campaign branch, Qwen3 and Qwen3.5
families) were compared row by row with this script's --prompt r2s_oneshot output rendered
through the HF templates on 2026-10-17: 274/274 byte-identical prompts and metadata.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
MODEL = ["--model", "/models/Qwen3.5-4B"]
XML_OPEN_DOOR = (
    "<tool_call>\n<function=act>\n<parameter=command>\nopen door to kitchen\n</parameter>\n</function>\n</tool_call>"
)
HERMES_OPEN_DOOR = '<tool_call>\n{"name": "act", "arguments": {"command": "open door to kitchen"}}\n</tool_call>'
# recipes/scienceworld/data/README.md: the paper's list (--from the campaign file) and the sorted stand-in (--synthesize)
FROZEN_TEST_SHA256 = "b2e1024f82a09e63011f37e4f423036c94b175b49b79c8d131eba7994cba2739"
SYNTHESIZED_TEST_SHA256 = "f98541246a3145ce0ae2f97831a5f7dc7cd19e083a7b9e782f52dda4a1a8c11c"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pd(template_render_chat):
    mod = _load(REPO / "recipes" / "scienceworld" / "prepare_data.py", "_scienceworld_prepare_data")
    qwen35 = template_render_chat("qwen3.5_fixed.jinja")
    mod.hf_render_chat = lambda model_dir: qwen35  # this module object only
    return mod


@pytest.fixture(scope="module")
def prompt(pd, template_render_chat):
    return pd.prompt_messages(pd.DEFAULT_PROMPT, template_render_chat("qwen3.5_fixed.jinja"))


@pytest.fixture(scope="module")
def test_list(pd, tmp_path_factory) -> Path:
    """The frozen TCOD list when SCIENCEWORLD_TEST_LIST names one (never committed: data/README.md),
    else the set-identical sorted stand-in task_splits.synthesized_test() written to tmp."""
    named = os.environ.get("SCIENCEWORLD_TEST_LIST")
    if named and Path(named).is_file():
        return Path(named)
    ts = pd.build_splits.__globals__  # the task_splits module's namespace (already used by test_splits_disjoint_...)
    path = tmp_path_factory.mktemp("sciworld") / "scienceworld_test1308.jsonl"
    ts["write_test_list"](ts["synthesized_test"](), path)
    return path


def _is_frozen(path: Path) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_TEST_SHA256


@pytest.fixture(scope="module")
def rows(pd, prompt, test_list):
    return pd.build_rows(test_list, 30, prompt)


def _key(row: dict) -> tuple[str, int]:
    return row["metadata"]["task_name"], row["metadata"]["var_num"]


def test_split_sizes_and_eval_order(rows):
    assert {k: len(v) for k, v in rows.items()} == {
        "train": 2294,
        "valid_seen": 140,
        "valid_unseen": 134,
        "test_full": 1308,
    }
    # run.py slices: scienceworld_seen @[0:140] then scienceworld_unseen @[140:274]
    assert {r["metadata"]["split"] for r in rows["train"]} == {"train"}
    assert {r["metadata"]["split"] for r in rows["valid_seen"]} == {"train_heldout"}
    assert {r["metadata"]["split"] for r in rows["valid_unseen"]} == {"test"}
    assert [r["metadata"] for r in rows["valid_unseen"]] == [r["metadata"] for r in rows["test_full"][:134]]


def test_splits_are_the_papers(pd, rows):
    """The task sets and their order are the paper's (the campaign branch's recipes/scienceworld_opd):
    train in sorted-name order, valid_seen = the seed-42 sample of train_heldout, valid_unseen = the
    list's head (the campaign's order only for the frozen list: the next test). Pinned on the paper
    code's rows so the sample can never move."""
    train = [_key(r) for r in rows["train"]]
    assert train[0] == ("boil", 0) and train[-1] == ("use-thermometer", 269) and train == sorted(train)
    seen = [_key(r) for r in rows["valid_seen"]]
    assert seen[:3] == [("boil", 17), ("boil", 28), ("change-the-state-of-matter-of", 26)]
    assert seen[-1] == ("use-thermometer", 526) and seen == sorted(seen)
    assert (pd.SEED, pd.VALID_SEEN_N, pd.VALID_UNSEEN_N) == (42, 140, 134)


def test_frozen_test_head_is_the_campaigns(pd, rows, test_list):
    """valid_unseen = the frozen list's head in the campaign's order: checkable only against that list
    (SCIENCEWORLD_TEST_LIST; never committed, data/README.md)."""
    if not _is_frozen(test_list):
        pytest.skip("frozen TCOD list not present (set SCIENCEWORLD_TEST_LIST)")
    unseen = [_key(r) for r in rows["valid_unseen"]]
    assert unseen[0] == ("test-conductivity-of-unknown-substances", 282)
    assert unseen[-1] == ("measure-melting-point-unknown-substance", 76)


def test_splits_disjoint_and_test_head_is_the_files(pd, rows, test_list):
    train = {_key(r) for r in rows["train"]}
    assert not train & {_key(r) for r in rows["valid_seen"]}
    assert not train & {_key(r) for r in rows["test_full"]}
    train_types = set(pd.build_splits.__globals__["TRAIN_TASK_NAMES"])
    assert {r["metadata"]["task_name"] for r in rows["valid_seen"]} <= train_types
    assert not {r["metadata"]["task_name"] for r in rows["test_full"]} & train_types
    first = json.loads(test_list.read_text().splitlines()[0])
    assert _key(rows["test_full"][0]) == (first["task_name"], first["var_num"])


def test_default_test_list_follows_the_env_and_the_data_root(pd, monkeypatch):
    """No list in the repository: SCIENCEWORLD_TEST_LIST names it, else <MILES_AGENT_DATA or <repo>/data>/
    scienceworld/scienceworld_test1308.jsonl, where scripts/fetch_data.sh scienceworld writes it (resolved per call)."""
    monkeypatch.setenv("SCIENCEWORLD_TEST_LIST", "/x/l.jsonl")
    monkeypatch.setenv("MILES_AGENT_DATA", "/d")
    assert pd.default_test_jsonl() == Path("/x/l.jsonl")
    monkeypatch.delenv("SCIENCEWORLD_TEST_LIST")
    assert pd.default_test_jsonl() == Path("/d/scienceworld/scienceworld_test1308.jsonl")
    monkeypatch.delenv("MILES_AGENT_DATA")
    assert pd.default_test_jsonl() == REPO / "data" / "scienceworld" / "scienceworld_test1308.jsonl"
    assert not hasattr(pd, "DEFAULT_TEST_JSONL")


def test_frozen_test_list_digest_is_pinned(test_list):
    """What --from must produce (the pin stays here as documentation; the check runs when the frozen list is present)."""
    if not _is_frozen(test_list):
        pytest.skip("frozen TCOD list not present (set SCIENCEWORLD_TEST_LIST)")
    assert hashlib.sha256(test_list.read_bytes()).hexdigest() == FROZEN_TEST_SHA256
    first = json.loads(test_list.read_text().splitlines()[0])
    assert (first["task_name"], first["var_num"]) == ("test-conductivity-of-unknown-substances", 282)


def test_synthesized_test_list_is_set_identical_sorted_and_pinned(pd, test_list, tmp_path):
    ts = pd.build_splits.__globals__
    synth = ts["synthesized_test"]()
    assert len(synth) == 1308 and len(set(synth)) == 1308 and synth == sorted(synth)
    assert synth[0] == ("chemistry-mix-paint-tertiary-color", 0)
    assert synth[-1] == ("test-conductivity-of-unknown-substances", 299)
    assert set(synth) == set(ts["load_test"](test_list))
    out = tmp_path / "synth.jsonl"
    ts["write_test_list"](synth, out)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SYNTHESIZED_TEST_SHA256
    assert ts["load_test"](out) == synth


def test_row_shape(pd, rows, prompt):
    row = rows["train"][0]
    assert row["prompt"] == prompt and [m["role"] for m in row["prompt"]] == ["system"]
    md = row["metadata"]
    assert list(md) == ["tools", "task_name", "var_num", "split", "max_episode_steps", "data_source"]
    assert md["tools"] == [pd.ACT_TOOL] and md["tools"][0]["name"] == "act" and md["tools"][0]["strict"] is True
    assert md["max_episode_steps"] == 30 and md["data_source"] == f"scienceworld/{md['task_name']}"
    assert isinstance(md["var_num"], int)
    # every row carries its own copy of the messages
    assert row["prompt"] is not rows["train"][1]["prompt"]


def test_prompt_selection(pd, template_render_chat):
    hermes, xml = template_render_chat("qwen3_fixed.jinja"), template_render_chat("qwen3.5_fixed.jinja")
    # default: the paper's plain system message, no example call
    default = pd.prompt_messages("default", hermes)
    assert [m["role"] for m in default] == ["system"] and "<tool_call>" not in default[0]["content"]
    assert default[0]["content"].startswith("You are an expert agent operating in the ScienceWorld text environment")
    assert pd.DEFAULT_PROMPT == "default"
    # r2s_oneshot: the same system message, then the worked example in the model's markup
    for render_chat, call in ((hermes, HERMES_OPEN_DOOR), (xml, XML_OPEN_DOOR)):
        oneshot = pd.prompt_messages("r2s_oneshot", render_chat)
        assert [m["role"] for m in oneshot] == ["system", "user", "assistant", "user", "assistant", "user"]
        assert oneshot[0] == default[0] and oneshot[2]["content"].endswith(call)
        assert oneshot[5]["content"] == "<tool_response>\n[example ended; a new task follows]\n</tool_response>"
    with pytest.raises(FileNotFoundError, match="available: default, r2s_oneshot"):
        pd.prompt_messages("nope", hermes)


def test_main_writes_three_files(pd, tmp_path, test_list):
    pd.main([*MODEL, "--output-dir", str(tmp_path), "--test-jsonl", str(test_list)])
    counts = {p.name: sum(1 for _ in p.open()) for p in tmp_path.glob("*.jsonl")}
    assert counts == {
        "scienceworld_train.jsonl": 2294,
        "scienceworld_eval274.jsonl": 274,
        "scienceworld_test1308.jsonl": 1308,
    }
    ev = [json.loads(line) for line in (tmp_path / "scienceworld_eval274.jsonl").open()]
    assert [r["metadata"]["split"] for r in ev] == ["train_heldout"] * 140 + ["test"] * 134
    assert all([m["role"] for m in r["prompt"]] == ["system"] for r in ev)
    # --prompt picks the file; the rows then carry the model's markup (qwen3.5 here)
    out = tmp_path / "oneshot"
    pd.main([*MODEL, "--output-dir", str(out), "--prompt", "r2s_oneshot", "--max-episode-steps", "20",
             "--test-jsonl", str(test_list)])
    row = json.loads((out / "scienceworld_train.jsonl").open().readline())
    assert XML_OPEN_DOOR in row["prompt"][2]["content"] and row["metadata"]["max_episode_steps"] == 20


def test_main_refuses_a_missing_test_list(pd, tmp_path):
    """A missing list is a FATAL naming the command that rebuilds it (the list is not shipped: data/README.md),
    before anything is written."""
    nope = tmp_path / "nope.jsonl"
    with pytest.raises(SystemExit) as e:
        pd.main([*MODEL, "--output-dir", str(tmp_path / "out"), "--test-jsonl", str(nope)])
    msg = str(e.value)
    assert msg.startswith("FATAL: ScienceWorld eval taskset") and str(nope) in msg
    assert "scripts/fetch_data.sh scienceworld" in msg and "--synthesize" in msg and "SCIENCEWORLD_TEST_LIST" in msg
    assert not (tmp_path / "out").exists()


def test_reward_shaping_is_identity():
    mod = _load(REPO / "recipes" / "scienceworld" / "reward.py", "_scienceworld_reward")
    sample = SimpleNamespace(metadata={"env_metrics": {"won": 0.0, "task_score": 0.35, "step_count": 30.0}})
    assert mod.shape(0.35, sample) == 0.35 and mod.shape(0.0, sample) == 0.0 and mod.shape(1.0, sample) == 1.0
