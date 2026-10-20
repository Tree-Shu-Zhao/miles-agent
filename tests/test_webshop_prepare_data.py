"""recipes/webshop/prepare_data.py: content-addressed rows, split layouts,
prompt variants, sidecar + goals-file integrity.

Bare-python3 module (it runs under the container python3), loaded from its file
so the test needs no package install. The frozen goals export is not in the
repository (data/README.md); in these tests a synthetic 6910-goal file (the small-catalog count) with
ids computed by the shared identity module stands in for it. The policy model's
chat template comes from miles' fixed Qwen3.5 template instead of a checkpoint
(--model is then only recorded).
"""

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
N_GOALS = 6910  # synthetic goals on the 1000-product small catalog


MODEL = ["--model", "/models/Qwen3.5-4B"]
XML_SEARCH = "<tool_call>\n<function=act>\n<parameter=command>\nsearch[red ceramic mug]\n</parameter>\n</function>\n</tool_call>"


@pytest.fixture(scope="module")
def pd(template_render_chat):
    spec = importlib.util.spec_from_file_location(
        "_webshop_prepare_data", REPO / "recipes" / "webshop" / "prepare_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    qwen35 = template_render_chat("qwen3.5_fixed.jinja")
    mod.hf_render_chat = lambda model_dir: qwen35  # this module object only
    return mod


def _fake_goal(i: int) -> dict:
    # Same shape env_servers/webshop/export_goals.py writes: goal_idx + every GOAL_FIELDS key.
    return {
        "goal_idx": i,
        "asin": f"B{i:05d}",
        "name": f"Product {i}",
        "category": ["beauty", "garden", "electronics"][i % 3],
        "query": f"query {i % 97}",
        "product_category": f"Cat > Sub {i % 11}",
        "instruction_text": f"i need fake goal {i}, and price lower than {10 + i}.00 dollars",
        "attributes": [f"attr{i % 3}", f"attr{i % 5}"],
        "goal_options": {"size": str(i % 2)},
        "price_upper": float(10 + i),
    }


def _write_goals(path: Path, goals: list[dict]) -> None:
    path.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in goals), encoding="utf-8")


@pytest.fixture(scope="module")
def goals(pd) -> list[dict]:
    rows = []
    for i in range(N_GOALS):
        g = _fake_goal(i)
        rows.append({"goal_id": pd.goal_id(g), **g})
    return rows


@pytest.fixture(scope="module")
def goals_file(tmp_path_factory, goals) -> Path:
    path = tmp_path_factory.mktemp("goals") / "webshop_goals.jsonl"
    _write_goals(path, goals)
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ids(rows: list[dict]) -> list[str]:
    return [r["metadata"]["goal_id"] for r in rows]


def test_default_goals_path_and_constant_pool_seed(pd, monkeypatch):
    # The dataset is a pure function of the goals file: no CLI knob may move it. The file is not
    # shipped; WEBSHOP_GOALS_FILE names it, else the recipe's data dir (where fetch_data.sh writes it).
    monkeypatch.delenv("WEBSHOP_GOALS_FILE", raising=False)
    assert pd.DEFAULT_GOALS_PATH == REPO / "recipes" / "webshop" / "data" / "webshop_goals.jsonl"
    assert pd.default_goals_path() == pd.DEFAULT_GOALS_PATH
    monkeypatch.setenv("WEBSHOP_GOALS_FILE", "/g/goals.jsonl")
    assert pd.default_goals_path() == Path("/g/goals.jsonl")
    assert pd.POOL_SAMPLE_SEED == 233
    assert pd.goal_id(_fake_goal(0)) == pd.goal_id({**_fake_goal(0), "goal_idx": 99, "goal_id": "x"})


def test_legacy_knobs_are_gone(pd, goals_file, tmp_path):
    for flag in (["--goal-seed", "0"], ["--num-goals", "6910"]):
        with pytest.raises(SystemExit) as e:
            pd.main([*MODEL, "--output-dir", str(tmp_path), "--goals", str(goals_file), *flag])
        assert e.value.code == 2


def test_tcod_layout(pd, goals, goals_file, tmp_path):
    pd.main([*MODEL, "--output-dir", str(tmp_path), "--goals", str(goals_file), "--protocol", "tcod"])  # defaults: think
    ev, tr = _rows(tmp_path / "webshop_eval.jsonl"), _rows(tmp_path / "webshop_train.jsonl")
    gid = [g["goal_id"] for g in goals]
    assert _ids(tr) == gid[:4096]  # file positions 0..4095, in file order
    assert all(r["metadata"]["split"] == "train" for r in tr)
    assert _ids(ev[:100]) == gid[4096:4196]  # TCOD test set
    assert all(r["metadata"]["split"] == "tcod_test" for r in ev[:100])
    assert _ids(ev[100:]) == gid[4196:4596]  # held-out companion
    assert all(r["metadata"]["split"] == "heldout" for r in ev[100:]) and len(ev) == 500
    # no eval goal is ever trained under tcod
    assert not (set(_ids(ev)) & set(_ids(tr)))
    side = json.loads((tmp_path / "protocol.json").read_text())
    assert side["protocol"] == "tcod" and side["prompt"] == "think"
    assert side["eval_slices"] == {"webshop": [0, 100], "webshop_heldout": [100, 500]}
    assert (side["eval_rows"], side["train_rows"], side["train_file_positions"]) == (500, 4096, [0, 4095])
    # the slices the recipe reads must match the sidecar
    assert pd.eval_slices("tcod") == {"webshop": (0, 100), "webshop_heldout": (100, 500)}
    row = ev[0]
    assert row["prompt"][0]["role"] == "system" and "<think> </think>" in row["prompt"][0]["content"]
    assert XML_SEARCH in row["prompt"][0]["content"]  # the policy model's own markup
    assert row["metadata"]["tools"] == [pd.ACT_TOOL] and row["metadata"]["max_episode_steps"] == 15
    assert row["metadata"]["data_source"] == "webshop"


def test_rows_carry_content_ids_not_positions(pd, goals, goals_file, tmp_path):
    pd.main([*MODEL, "--output-dir", str(tmp_path), "--goals", str(goals_file)])
    by_id = {g["goal_id"]: g for g in goals}
    for path in (tmp_path / "webshop_eval.jsonl", tmp_path / "webshop_train.jsonl"):
        for r in _rows(path):
            md = r["metadata"]
            assert "task_id" not in md
            assert isinstance(md["goal_id"], str) and len(md["goal_id"]) == 16
            # the question travels with the id so logs stay readable; it must be that goal's own text
            assert md["instruction_text"] == by_id[md["goal_id"]]["instruction_text"]


def test_canonical_layout_and_none_prompt(pd, goals, goals_file, tmp_path):
    pd.main(
        [*MODEL, "--output-dir", str(tmp_path), "--goals", str(goals_file), "--protocol", "canonical", "--prompt", "none"]
    )
    ev, tr = _rows(tmp_path / "webshop_eval.jsonl"), _rows(tmp_path / "webshop_train.jsonl")
    gid = [g["goal_id"] for g in goals]
    assert len(ev) == 756 and len(tr) == N_GOALS - 500
    assert _ids(ev[:500]) == gid[:500]
    assert all(r["metadata"]["split"] == "eval" for r in ev[:500])
    pool = _ids(ev[500:])
    assert all(r["metadata"]["split"] == "eval_pool" for r in ev[500:]) and len(set(pool)) == 256
    # the pool sample is a fixed draw over FILE positions (POOL_SAMPLE_SEED = 233); pin it
    expected_pos = random.Random(233).sample(range(N_GOALS), 256)
    assert expected_pos[:5] == [5590, 6573, 5147, 1428, 4302]
    assert pool == [gid[p] for p in expected_pos]
    assert _ids(tr) == gid[500:]
    prompt = ev[0]["prompt"][0]["content"]
    assert "<think>" not in prompt and "Respond ONLY with a call to the act tool" in prompt
    side = json.loads((tmp_path / "protocol.json").read_text())
    assert side["eval_slices"] == {"webshop": [0, 500], "webshop_pool": [500, 756]}
    assert side["train_file_positions"] == [500, N_GOALS - 1]


def test_sidecar_carries_goal_set_digest(pd, goals, goals_file, tmp_path):
    pd.main([*MODEL, "--output-dir", str(tmp_path), "--goals", str(goals_file)])
    side = json.loads((tmp_path / "protocol.json").read_text())
    ids = [g["goal_id"] for g in goals]
    assert side["goals"] == N_GOALS
    assert side["goals_digest"] == pd.goals_set_digest(ids) == pd.goals_set_digest(reversed(ids))
    assert len(side["goals_digest"]) == 16
    assert side["goals_file_sha256"] == hashlib.sha256(goals_file.read_bytes()).hexdigest()
    assert side["goals_file"] == str(goals_file)
    for legacy in ("goal_seed", "num_goals", "train_goal_range"):
        assert legacy not in side
    for kept in (
        "protocol",
        "prompt",
        "model",
        "prompt_sha256",
        "eval_slices",
        "max_episode_steps",
        "eval_rows",
        "train_rows",
    ):
        assert kept in side


def test_default_protocol_is_canonical(pd, goals_file, tmp_path):
    # No --protocol: the mainstream held-out layout, think prompt
    # (env_servers/webshop/defaults.sh WEBSHOP_PROTOCOL must agree).
    pd.main([*MODEL, "--output-dir", str(tmp_path), "--goals", str(goals_file)])
    side = json.loads((tmp_path / "protocol.json").read_text())
    assert side["protocol"] == "canonical" and side["prompt"] == "think"
    assert side["eval_slices"] == {"webshop": [0, 500], "webshop_pool": [500, 756]}
    sh = (REPO / "env_servers" / "webshop" / "defaults.sh").read_text()
    assert "WEBSHOP_PROTOCOL=${WEBSHOP_PROTOCOL:-canonical}" in sh


def test_pool_sample_is_deterministic_and_flag_free(pd):
    ev_a, tr_a = pd.split_rows("canonical", N_GOALS)
    ev_b, tr_b = pd.split_rows("canonical", N_GOALS)
    assert ev_a == ev_b and tr_a == tr_b
    assert ev_a[:500] == [(i, "eval") for i in range(500)]
    assert sorted(p for p, _ in ev_a[500:]) != [p for p, _ in ev_a[500:]]  # a sample, not a prefix
    # the tcod layout never consults the sampler
    ev_t, tr_t = pd.split_rows("tcod", N_GOALS)
    assert [p for p, _ in ev_t] == list(range(4096, 4596)) and [p for p, _ in tr_t] == list(range(4096))


def test_missing_goals_file_is_a_usage_error(pd, tmp_path):
    with pytest.raises(SystemExit) as e:
        pd.main([*MODEL, "--output-dir", str(tmp_path), "--goals", str(tmp_path / "nope.jsonl")])
    assert e.value.code == 2
    with pytest.raises(ValueError, match="export_goals.py"):
        pd.load_goals(tmp_path / "nope.jsonl")


def test_main_refuses_a_missing_goals_file(pd, tmp_path, monkeypatch, capsys):
    """No --goals: WEBSHOP_GOALS_FILE names the export; a missing file is a usage error whose text gives the
    command that rebuilds it (the file is not shipped: data/README.md)."""
    nope = tmp_path / "nope.jsonl"
    monkeypatch.setenv("WEBSHOP_GOALS_FILE", str(nope))
    with pytest.raises(SystemExit) as e:
        pd.main([*MODEL, "--output-dir", str(tmp_path)])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert f"goals file {nope} not found" in err and "scripts/fetch_data.sh webshop --regenerate-goals" in err
    with pytest.raises(ValueError, match="scripts/fetch_data.sh webshop --regenerate-goals") as v:
        pd.load_goals(pd.default_goals_path())
    assert "WEBSHOP_GOALS_FILE" in str(v.value) and "export_goals.py" in str(v.value)
    assert not (tmp_path / "webshop_eval.jsonl").exists()


def test_tampered_goal_id_is_a_usage_error(pd, goals, tmp_path):
    # (a) the stored id no longer matches the content
    rows = [dict(g) for g in goals]
    rows[7]["goal_id"] = pd.goal_id(_fake_goal(8))
    bad = tmp_path / "tampered_id.jsonl"
    _write_goals(bad, rows)
    with pytest.raises(SystemExit) as e:
        pd.main([*MODEL, "--output-dir", str(tmp_path / "out"), "--goals", str(bad)])
    assert e.value.code == 2
    with pytest.raises(ValueError, match="stored goal_id"):
        pd.load_goals(bad)
    # (b) a scoring field edited under an unchanged id
    rows = [dict(g) for g in goals]
    rows[3]["price_upper"] = 999.0
    bad2 = tmp_path / "tampered_field.jsonl"
    _write_goals(bad2, rows)
    with pytest.raises(ValueError, match="stored goal_id"):
        pd.load_goals(bad2)


def test_repeated_content_collapses_to_the_first_row(pd, goals, tmp_path):
    # WebShop's synthetic generator emits identical goals at several list
    # positions (9 pairs on the small catalog). Same content = same question:
    # the first row serves it, the twin is neither a second held-out goal nor a
    # back door into training, and the set digest is unchanged.
    rows = [dict(g) for g in goals]
    twin = dict(rows[3], goal_idx=len(rows))  # exact copy of row 3 appended at the end
    dup_file = tmp_path / "dup_goals.jsonl"
    _write_goals(dup_file, rows + [twin])
    unique, rows_read = pd.load_goals(dup_file)
    assert rows_read == len(rows) + 1 and len(unique) == len(rows)
    assert [g["goal_id"] for g in unique] == [r["goal_id"] for r in rows]
    out = tmp_path / "dup_out"
    pd.main([*MODEL, "--goals", str(dup_file), "--output-dir", str(out)])
    side = json.loads((out / "protocol.json").read_text())
    assert (side["goals"], side["unique_goals"], side["duplicate_rows"]) == (len(rows) + 1, len(rows), 1)
    assert side["goals_digest"] == pd.goals_set_digest(r["goal_id"] for r in rows)
    emitted = [
        json.loads(line)["metadata"]["goal_id"]
        for name in ("webshop_eval.jsonl", "webshop_train.jsonl")
        for line in (out / name).read_text().splitlines()
    ]
    assert set(emitted) == {r["goal_id"] for r in rows}  # every question reachable, no twin


def test_too_few_goals_is_a_usage_error(pd, goals, tmp_path):
    few = tmp_path / "few.jsonl"
    _write_goals(few, goals[:4500])  # tcod needs 4596
    with pytest.raises(SystemExit) as e:
        pd.main([*MODEL, "--output-dir", str(tmp_path / "out"), "--goals", str(few), "--protocol", "tcod"])
    assert e.value.code == 2
    tiny = tmp_path / "tiny.jsonl"
    _write_goals(tiny, goals[:500])  # canonical needs > 500
    with pytest.raises(SystemExit) as e:
        pd.main([*MODEL, "--output-dir", str(tmp_path / "out"), "--goals", str(tiny), "--protocol", "canonical"])
    assert e.value.code == 2
    with pytest.raises(ValueError):
        pd.split_rows("bogus", N_GOALS)
    with pytest.raises(ValueError, match="empty"):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n")
        pd.load_goals(empty)


def test_prompt_variants_share_the_task_text(pd, template_render_chat):
    hermes = template_render_chat("qwen3_fixed.jinja")
    think, none = (pd.prompt_messages(name, hermes)[0]["content"] for name in ("think", "none"))
    head = "You are an expert autonomous shopping agent"
    assert think.startswith(head) and none.startswith(head)
    for p in (think, none):
        assert '<tool_call>\n{"name": "act", "arguments": {"command": "search[red ceramic mug]"}}\n</tool_call>' in p
        assert "click[buy now]" in p
    assert "MUST be enclosed within <think> </think> tags" in think and "<think>" not in none


def test_prompt_markup_is_the_models(pd, template_render_chat):
    """The prompt files name no markup; the example call is rendered by the model's template."""
    from miles_agent import prompts

    assert "think" in prompts.available("webshop") and pd.DEFAULT_PROMPT == "think"
    for name in ("think", "none"):
        system = prompts.load("webshop", name).system
        assert "<tool_call>" not in system and system.count("{example_call}") == 1
    xml = pd.prompt_messages("none", template_render_chat("qwen3.5_fixed.jinja"))[0]["content"]
    assert XML_SEARCH in xml and '"arguments"' not in xml
