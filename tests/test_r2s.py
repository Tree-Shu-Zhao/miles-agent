"""miles_agent/r2s: the per-env selection table, the one selection rule, the transcript split and the io
every stage shares (design.md sections 4.1, 4.2; C9), stage A's divergence mining (section 4.3, F8; C10),
stage B's proxy-free takeover (section 4.4, D5, D6, D16, F5; C11), stage J's WebShop judge (section 4.5,
D6, D15; C12), stage D's token-spliced privileged labels (section 4.6, D9, F4, F8; C13) and stage E's teacher FT
around the GPU (section 4.7, D18; C14).

The selection tests build stage B rows by hand in REF's schema (state_id, sample, replay_ok, won, done,
takeover_steps, last_observation, terminal_key); the one-shot prompt comes from prompts/alfworld/r2s_oneshot.py
as a dump row carries it (role/content messages, the demonstration turns flattened to text). Stage A's CLI runs
over a dump in the fake tokenizer's vocabulary with a fake prefill (the student-teacher gap is tok / 100 at
every scored position) behind a stand-in transformers; its F8 default reads the real MODEL_FAMILIES through
test_launcher's stubs (launcher imports miles at module level). Stage B's CLI runs over a fake POST <gym>/run
that answers what env_loop_agent returns (app.py:279-296) from a script, and once over the real agent with
test_env_loop_agent's fakes, so the body the stage builds and the reply it reads are the agent's own. Stage J's
CLI runs against a chat-completions endpoint stood in by httpx.MockTransport over text_rich pages as webshop_env
renders them. Stage D's splice is checked against REF's literal surgery, re-implemented here as the oracle (the
only place the ChatML literals appear), on the real one-shot prompt rendered through the fixed templates; its CLI
runs over states whose prompt ids are such renders, with a fake prefill whose top-k names the scored position.
Stage E's split, label checks, CLI defaults, eval schedule, export sidecars and report run without torch; the
loss runs on CPU tensors where torch is installed and skips elsewhere.
"""

import argparse
import asyncio
import collections
import dataclasses
import functools
import hashlib
import json
import math
import os
import re
import sys
import types

import httpx
import pytest
from conftest import MILES_TEMPLATES
from test_launcher import stubs  # noqa: F401  (the fixture joins this module)
from test_tool_call_markup import ACT, ACT_FLAT, FakeTokenizer, tito_stub  # noqa: F401

from miles_agent import prompts
from miles_agent.opd import teacher_scoring
from miles_agent.r2s import common, divergence, judge, label, select, takeover, teacher_ft
from miles_agent.tool_call_markup import generation_prompt_ids, tito_render_chat, user_message_ids

ALF, SW, WS = (select.ENV_DEFAULTS[e] for e in ("alfworld", "scienceworld", "webshop"))
OBS0 = "obs0: You are in the middle of a room. Your task is to: put a clean mug on the desk."
OBS1 = "obs1: On the desk 1, you see a mug 2."
HINT = "Nothing happens. Reply with one act call."
OBS3 = "obs3: You open the drawer 1."


def _take(
    sample,
    terminal_key,
    n_steps,
    *,
    replay_ok=True,
    done=True,
    won=False,
    format_ok=True,
    last_observation="You put the mug 1 in/on the desk 1.",
    state_id="s0",
):
    """One stage B row (REF takeover.py:299-317 keys); format_ok=False faults the first step."""
    steps = [
        {
            "turn": i,
            "delib": "d",
            "command": "go to desk 1",
            "call": {"name": "act", "arguments": {"command": "go to desk 1"}},
            "format_ok": True,
            "obs_before": "o",
        }
        for i in range(n_steps)
    ]
    if steps and not format_ok:
        steps[0]["format_ok"] = False
    return {
        "state_id": state_id,
        "sample": sample,
        "replay_ok": replay_ok,
        "won": won,
        "done": done,
        "takeover_steps": steps,
        "last_observation": last_observation,
        "terminal_key": terminal_key,
    }


def _act(call_id, command, content=""):
    """An assistant message as the engine parser returns it (env_loop_agent appends it verbatim)."""
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "act", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": content, "tool_calls": [call]}


def _oneshot_prompt() -> list[dict]:
    """prompts/alfworld/r2s_oneshot.py as a dump row carries it: role/content messages, the demonstration's
    assistant and tool turns flattened to text as prompts.render does (what the text says is the template's
    business and plays no part here)."""
    p = prompts.load("alfworld", "r2s_oneshot")
    flat = [{"role": "system", "content": p.system}]
    for t in p.turns:
        flat.append({"role": "user" if t["role"] == "tool" else t["role"], "content": t.get("content") or "(call)"})
    return flat


def test_env_defaults_are_the_appendix_values():
    """tab:main's selection (x_appendix_analysis_method.tex:62) and the campaign knobs, baked in: no hand-filled
    knob for a paper cell."""
    assert set(select.ENV_DEFAULTS) == {"alfworld", "scienceworld", "webshop"}
    fields = ("goal_keys", "vote_key", "check", "m", "done_regex", "label_frac_gate", "max_steps")
    row = lambda cfg: tuple(getattr(cfg, f) for f in fields)  # noqa: E731
    assert row(ALF) == (("game_file",), "terminal", "vote", 2, "", 0.25, 30)
    assert row(SW) == (("task_name", "var_num"), "terminal", "vote", 1, "Task completed.", 0.25, 30)
    assert row(WS) == (("goal_id",), "purchased_asin", "judge", 1, "", 0.0, 15)
    for cfg in (ALF, SW, WS):
        assert (cfg.allow_invalid, cfg.require_action, cfg.action_tool) == (True, True, "act")
    # the campaign's exact export, dot unescaped: the simulator's success line matches, its failure line does not
    assert re.search(SW.done_regex, "You see a lit stove.\nTask completed.") and not re.search(
        SW.done_regex, "Task finished."
    )
    assert (select.TOP_N, select.MAX_PER_GOAL, select.MIN_DELIB_TOKENS, select.MIN_DIV_TURN) == (2400, 3, 3, 0)
    assert (select.K, select.TAKEOVER_T, select.TOP_P, select.MAX_TEACHER_TOKENS) == (8, 1.0, 1.0, 512)
    assert (select.LABEL_K, select.MAX_CTX, select.MIN_STATES) == (16, 32768, 100)
    assert (select.JUDGE_THINKING, select.JUDGE_MAX_TOKENS, select.JUDGE_CONCURRENCY, select.JUDGE_T) == (
        True,
        2048,
        64,
        0.0,
    )
    assert (select.FT_EPOCHS, select.FT_LR, select.FT_GRAD_ACCUM, select.FT_WARMUP_STEPS) == (1, 2e-6, 8, 10)
    assert (select.FT_MIN_LR_RATIO, select.FT_CLIP, select.FT_VAL_PCT, select.SEED) == (0.1, 1.0, 10, 42)
    with pytest.raises(ValueError, match="check in"):
        select.EnvConfig(("g",), check="gt")  # the ground-truth selector lives on the branch, not in the table


def test_env_config_file_and_flag_overrides(tmp_path):
    """ENV_DEFAULTS, then the --env-config table's entry, then the flags that were given; unknown fields and
    unknown envs fail loudly; goal_keys come back as a tuple."""
    assert select.load_env_config("alfworld") == ALF
    table = tmp_path / "env.json"
    table.write_text(json.dumps({"alfworld": {"m": 3, "done_regex": "Done"}, "lab": {"goal_keys": ["task"], "m": 1}}))
    cfg = select.load_env_config("alfworld", str(table))
    assert (cfg.m, cfg.done_regex, cfg.goal_keys, cfg.label_frac_gate) == (3, "Done", ("game_file",), 0.25)
    cfg = select.load_env_config("alfworld", str(table), m=1, done_regex=None, check="judge")
    assert (cfg.m, cfg.done_regex, cfg.check) == (1, "Done", "judge")  # a flag given wins; None is "not given"
    lab = select.load_env_config("lab", str(table))
    assert (lab.goal_keys, lab.m, lab.check, lab.max_steps) == (("task",), 1, "vote", 30)
    with pytest.raises(ValueError, match="unknown env 'lab'"):
        select.load_env_config("lab")
    with pytest.raises(ValueError, match=r"unknown EnvConfig field\(s\) \['vote_min'\]"):
        select.load_env_config("alfworld", vote_min=2)
    table.write_text(json.dumps({"alfworld": {"vote_min": 2}}))
    with pytest.raises(ValueError, match="vote_min"):
        select.load_env_config("alfworld", str(table))
    with pytest.raises(ValueError, match="check in"):
        select.load_env_config("alfworld", check="gt")


def test_load_env_config_needs_goal_keys(tmp_path):
    """An env the --env-config table alone defines must name its goal_keys (stage A's per-goal cap and stage E's
    split have no default key): an entry without them, or with [], is a ValueError naming the fix, not a KeyError;
    a listed env's goal_keys can be replaced but not emptied; a lone key may be a string."""
    table = tmp_path / "env.json"
    table.write_text(json.dumps({"lab": {"m": 1}, "bare": {"goal_keys": [], "m": 1}, "alfworld": {"goal_keys": []}}))
    for env in ("lab", "bare", "alfworld"):
        with pytest.raises(ValueError, match=rf"env '{env}' needs goal_keys \(--env-config\)"):
            select.load_env_config(env, str(table))
    with pytest.raises(ValueError, match="needs goal_keys"):
        select.load_env_config("alfworld", goal_keys=())
    table.write_text(json.dumps({"lab": {"goal_keys": "task", "m": 1}}))
    assert select.load_env_config("lab", str(table)).goal_keys == ("task",)


def test_goal_of_per_env():
    """The per-goal key (stage A's cap, stage E's split) is the env's goal_keys joined with "/"; ScienceWorld
    rows carry task_name and var_num, no task_id (D17)."""
    game = "json_2.1.1/train/pick_and_place_simple-Mug-None-Desk-308/trial_T20190909_023119_283521/game.tw-pddl"
    assert select.goal_of({"game_file": game, "split": "train", "data_source": "alfworld"}, ALF) == game
    assert select.goal_of({"goal_id": 1234, "split": "train"}, WS) == "1234"
    assert select.goal_of({"task_name": "boil", "var_num": 7, "data_source": "scienceworld"}, SW) == "boil/7"
    with pytest.raises(KeyError, match="var_num"):
        select.goal_of({"task_name": "boil", "data_source": "scienceworld"}, SW)


def test_policy_turns_split_prompt_with_demo_turns():
    """The split is the prompt's length: the one-shot prompt's own demonstration assistant turns are not
    policy turns, the reset observation is the first episode message, and the observations a replay must
    reproduce are read from the dump transcript and from a takeover's /run reply alike."""
    prompt = _oneshot_prompt()
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user", "assistant", "user"]
    t0 = _act("c1", "go to desk 1", "The desk is where the mug should end up; I check it first.")
    t1 = {"role": "assistant", "content": "I open the drawer."}  # a turn without a call: the env's hint rides as user
    t2 = _act("c2", "open drawer 1")
    transcript = [
        *prompt,
        {"role": "user", "content": OBS0},
        t0,
        {"role": "tool", "tool_call_id": "c1", "content": OBS1},
        t1,
        {"role": "user", "content": HINT},
        t2,
        {"role": "tool", "tool_call_id": "c2", "content": OBS3},
    ]
    row = {"prompt": prompt, "transcript": transcript, "tokens": [1, 2], "response_length": 1}
    assert common.policy_turns(row) == [t0, t1, t2]
    assert common.episode_messages(row)[0] == {"role": "user", "content": OBS0}
    assert common.observations_after_prefix(row, 0) == [OBS0]
    assert common.observations_after_prefix(row, 1) == [OBS0, OBS1]
    assert common.observations_after_prefix(row, 2) == [OBS0, OBS1, HINT]
    assert common.observations_after_prefix(row, 3) == [OBS0, OBS1, HINT, OBS3]
    # a takeover's /run reply: the same prompt, the replayed prefix, then the teacher's turns
    reply = [
        *prompt,
        {"role": "user", "content": OBS0},
        t0,
        {"role": "tool", "tool_call_id": "c1", "content": "obs1:  On the desk 1,\nyou see a mug 2."},
        _act("x", "take mug 2 from desk 1"),
    ]
    got = common.observations_after_prefix(row, 1, reply)
    assert got != [OBS0, OBS1] and [common.norm_text(o) for o in got] == [common.norm_text(o) for o in (OBS0, OBS1)]
    assert common.fingerprint(got[1]) == common.fingerprint(OBS1) and len(common.fingerprint(OBS1)) == 12
    # a campaign v1 row has no transcript (nothing to replay); a bare-string prompt is one user message
    assert common.policy_turns({"prompt": prompt, "transcript": None}) == [] and common.policy_turns({}) == []
    bare = {"prompt": "go", "transcript": [{"role": "user", "content": "go"}, {"role": "user", "content": OBS0}, t1]}
    assert common.policy_turns(bare) == [t1] and common.observations_after_prefix(bare, 0) == [OBS0]


def test_select_tau_vote_shortest_in_modal_cluster():
    """REF label.py:66-109 case table: eligibility (replay_ok, done, a terminal key, a step), the modal key with
    the (count, key) tie-break, the shortest takeover of the modal cluster then the lowest sample, and
    allow_invalid admitting format-faulted takeovers (the ALFWorld protocol)."""
    group = [
        _take(0, "A", 4),
        _take(1, "A", 2),
        _take(2, "B", 3),
        _take(3, "B", 1, format_ok=False),
        _take(4, "A", 1, replay_ok=False),  # never replayed
        _take(5, "A", 1, done=False),  # ran out of turns
        _take(6, "A", 0),  # no teacher step
        _take(7, None, 1),  # no terminal key
    ]
    tau, stats = select.select_tau(group, ALF)
    assert tau is group[3]  # A and B tie at 2 votes, the greater key wins, the invalid 1-step takeover is admitted
    assert stats == {"n_group": 7, "p_won": 0.0, "vote_n_eligible": 4, "vote_modal_count": 2}
    tau, stats = select.select_tau(group, dataclasses.replace(ALF, allow_invalid=False))
    assert tau is group[1] and stats["vote_n_eligible"] == 3 and stats["vote_modal_count"] == 2
    tau, _ = select.select_tau([_take(5, "A", 2, won=True), _take(2, "A", 2), _take(9, "A", 3)], ALF)
    assert tau["sample"] == 2  # equal length: the lowest sample
    assert select.select_tau([_take(0, "A", 2), _take(1, "B", 2)], ALF) == (
        None,
        {"n_group": 2, "p_won": 0.0, "vote_n_eligible": 2},
    )  # two clusters of one: no cluster reaches m=2
    assert select.select_tau([], ALF) == (None, {"n_group": 0, "p_won": 0.0, "vote_n_eligible": 0})
    _, stats = select.select_tau(
        [_take(0, "A", 1, won=True), _take(1, "A", 1), _take(2, "A", 1, replay_ok=False, won=True)], ALF
    )
    assert stats["p_won"] == 0.5  # p_won over the replayed takeovers


def test_select_tau_done_regex_and_m():
    """ALFWorld votes with m=2 over the takeovers that end inside the turn budget; ScienceWorld with m=1 over
    those whose terminal observation carries the simulator's completion message."""
    completed = _take(0, "A", 5, last_observation="The stove is on.\nTask completed.")
    finished = _take(1, "B", 1, last_observation="You dropped it.\nTask finished.")
    assert select.select_tau([completed], ALF) == (None, {"n_group": 1, "p_won": 0.0, "vote_n_eligible": 1})
    tau, stats = select.select_tau([completed], SW)
    assert tau is completed and stats["vote_modal_count"] == 1
    assert select.select_tau([finished], SW) == (None, {"n_group": 1, "p_won": 0.0, "vote_n_eligible": 0})
    tau, stats = select.select_tau([finished, completed], SW)
    assert tau is completed and stats["vote_n_eligible"] == 1  # the shorter failure never enters the vote
    tau, _ = select.select_tau([finished, completed], dataclasses.replace(SW, done_regex=""))
    assert tau is finished  # without the regex both end "done" and the tie goes to the greater key, then shortest
    assert select.select_tau([completed], dataclasses.replace(SW, m=2))[0] is None


def test_select_tau_judge_requires_match_ok():
    """WebShop: only takeovers the judge approved (match_ok) vote, grouped by the item bought; a takeover without
    a judge row is not approved and is counted (F9); the judge rows are required with check judge."""
    group = [
        _take(0, "b000000000", 3, won=True),
        _take(1, "b000000000", 2),
        _take(2, "b000000001", 1),
        _take(3, "b000000000", 1),  # judged by nobody
    ]
    judge = {
        ("s0", 0): {"match_ok": True},
        ("s0", 1): {"match_ok": False, "match_text": "MISMATCH"},
        ("s0", 2): {"match_ok": True},
    }
    tau, stats = select.select_tau(group, WS, judge)
    assert tau is group[2]  # two approved items with one vote each: the greater ASIN wins at m=1
    assert stats == {
        "n_group": 4,
        "p_won": 0.25,
        "vote_n_eligible": 4,
        "judge_missing": 1,
        "judge_n_ok": 2,
        "vote_modal_count": 1,
    }
    assert select.select_tau(group, dataclasses.replace(WS, m=2), judge)[0] is None
    tau, stats = select.select_tau(group, dataclasses.replace(WS, check="vote"))
    assert tau is group[3] and "judge_n_ok" not in stats  # the reward-free vote alone: three buys of the first item
    with pytest.raises(ValueError, match="judge rows"):
        select.select_tau(group, WS)
    assert select.select_tau(group, WS, {})[0] is None and select.select_tau(group, WS, {})[1]["judge_missing"] == 4


def test_select_cli_writes_tau_and_coverage(tmp_path, capsys):
    """python -m miles_agent.r2s.select: tau rows {state_id, sample, group_size, n_eligible}, the summary with
    coverage, the resolved config and SELECT_DONE as key and trailing line; the DONE skip, --force, the
    per-field flags, --limit, the judge path and the stop file."""
    fold = tmp_path / "fold.jsonl"
    common.write_jsonl(
        str(fold),
        [
            _take(0, "A", 4, state_id="s0"),
            _take(1, "A", 2, state_id="s0"),
            _take(2, "B", 1, state_id="s0"),
            _take(0, "C", 2, state_id="s1"),  # one eligible takeover: not enough for ALFWorld's m=2
            _take(1, "C", 2, state_id="s1", replay_ok=False),
            _take(0, None, 0, state_id="s2", replay_ok=False, done=False),
        ],
    )
    out = tmp_path / "tau.jsonl"
    assert select.main(["--env", "alfworld", "--fold", str(fold), "--out", str(out)]) == 0
    assert common.read_jsonl(str(out)) == [{"state_id": "s0", "sample": 1, "group_size": 3, "n_eligible": 3}]
    summary_path = f"{out}.summary.json"
    summary = common.read_summary(summary_path)
    assert (summary["n_states"], summary["n_selected"], summary["coverage"]) == (3, 1, pytest.approx(1 / 3))
    assert (summary["n_takeovers"], summary["eligible_total"], summary["judge_missing"]) == (6, 4, 0)
    assert summary["config"] == {**dataclasses.asdict(ALF), "goal_keys": ["game_file"]}
    assert summary["SELECT_DONE"] is True and open(summary_path).read().endswith("}\nSELECT_DONE\n")
    assert capsys.readouterr().out.rstrip().endswith("SELECT_DONE")
    # done: a rerun leaves the files alone unless --force; --m 1 then selects s1 too
    with open(out, "a") as f:
        f.write("stale\n")
    assert select.main(["--env", "alfworld", "--fold", str(fold), "--out", str(out)]) == 0
    assert "SELECT_DONE already in" in capsys.readouterr().out and open(out).read().endswith("stale\n")
    assert select.main(["--env", "alfworld", "--fold", str(fold), "--out", str(out), "--force", "--m", "1"]) == 0
    assert [r["state_id"] for r in common.read_jsonl(str(out))] == ["s0", "s1"]
    summary = common.read_summary(summary_path)
    assert summary["config"]["m"] == 1 and summary["coverage"] == pytest.approx(2 / 3)
    assert select.main(["--env", "alfworld", "--fold", str(fold), "--out", str(out), "--force", "--limit", "1"]) == 0
    assert common.read_summary(summary_path)["n_states"] == 1
    # the judge path: webshop needs --judge-file, approved buys grouped by ASIN
    with pytest.raises(SystemExit, match="--judge-file is required"):
        select.main(["--env", "webshop", "--fold", str(fold), "--out", str(tmp_path / "ws.jsonl")])
    judge = tmp_path / "judge.jsonl"
    common.write_jsonl(str(judge), [{"state_id": "s0", "sample": 2, "match_ok": True, "match_text": "MATCH"}])
    ws_out = tmp_path / "ws.jsonl"
    assert (
        select.main(["--env", "webshop", "--fold", str(fold), "--judge-file", str(judge), "--out", str(ws_out)]) == 0
    )
    assert common.read_jsonl(str(ws_out)) == [{"state_id": "s0", "sample": 2, "group_size": 3, "n_eligible": 1}]
    assert common.read_summary(f"{ws_out}.summary.json")["judge_missing"] == 3  # s0: 2 unjudged, s1: 1
    # the preempt sentinel: exit 0, no rows, no marker
    stop = tmp_path / "STOP"
    stop.write_text("")
    stopped = tmp_path / "stopped.jsonl"
    argv = ["--env", "alfworld", "--fold", str(fold), "--out", str(stopped), "--stop-file", str(stop)]
    assert select.main(argv) == 0 and not stopped.exists() and common.read_summary(f"{stopped}.summary.json") is None


def test_select_summary_keys_on_empty_fold(tmp_path):
    """An empty fold (every takeover failed to replay, or a smoke over nothing) writes the summary keys a full one
    has: n_takeovers, eligible_total and judge_missing at 0 beside n_states 0, n_selected 0 and coverage None, so a
    reader finds every key on every run."""
    fold = tmp_path / "fold.jsonl"
    fold.write_text("")
    out = tmp_path / "tau.jsonl"
    assert select.main(["--env", "alfworld", "--fold", str(fold), "--out", str(out)]) == 0
    summary = common.read_summary(f"{out}.summary.json")
    keys = ("n_states", "n_selected", "coverage", "n_takeovers", "eligible_total", "judge_missing")
    assert [summary[k] for k in keys] == [0, 0, None, 0, 0, 0]
    assert common.read_jsonl(str(out)) == [] and summary["SELECT_DONE"] is True


def test_io_summary_done_markers_and_force(tmp_path, capsys):
    """write_summary / read_summary / summary_done (marker as key and trailing line), the stage flags with the
    --summary default and the DONE skip under --force, existing_keys over a file with a cut last row,
    append_jsonl, stop_requested and read_dump over v2 and v1 rows."""
    summary = tmp_path / "x.summary.json"
    assert common.read_summary(str(summary)) is None and not common.summary_done(str(summary), "X_DONE")
    common.write_summary(str(summary), {"n": 1, "config": {"m": 2}}, "X_DONE")
    text = summary.read_text()
    assert text.endswith("}\nX_DONE\n") and json.loads(text[: text.rfind("}") + 1]) == {
        "n": 1,
        "config": {"m": 2},
        "X_DONE": True,
    }
    assert common.read_summary(str(summary)) == {"X_DONE": True, "config": {"m": 2}, "n": 1}
    assert common.summary_done(str(summary), "X_DONE") and not common.summary_done(str(summary), "Y_DONE")
    assert capsys.readouterr().out.splitlines()[-1] == "X_DONE"
    summary.write_text('{"n": 1, "X_DO')  # cut while writing
    assert common.read_summary(str(summary)) is None
    # the shared flags
    ap = common.stage_parser("Stage doc.\n\nMore.")
    assert ap.description == "Stage doc."
    out = tmp_path / "x.jsonl"
    args = common.parse_stage_args(ap, ["--env", "alfworld", "--out", str(out)])
    assert (args.summary, args.force, args.limit, args.seed, args.stop_file, args.env_config) == (
        f"{out}.summary.json",
        False,
        None,
        42,
        "",
        None,
    )
    assert not common.skip_if_done(args, "X_DONE")
    common.write_summary(args.summary, {}, "X_DONE")
    assert common.skip_if_done(args, "X_DONE") and "X_DONE already in" in capsys.readouterr().out
    args = common.parse_stage_args(ap, ["--env", "alfworld", "--out", str(out), "--force", "--summary", str(summary)])
    assert not common.skip_if_done(args, "X_DONE") and args.summary == str(summary)
    # resume: the keys of the complete rows; a cut last row is dropped from the file and redone
    assert common.existing_keys(str(out), ("state_id", "sample")) == set()
    common.append_jsonl(str(out), {"state_id": "s0", "sample": 0})
    common.append_jsonl(str(out), {"state_id": "s0", "sample": 1})
    with open(out, "a") as f:
        f.write('{"state_id": "s1", "sam')
    assert common.existing_keys(str(out), ("state_id", "sample")) == {("s0", 0), ("s0", 1)}
    assert common.read_jsonl(str(out)) == [{"state_id": "s0", "sample": 0}, {"state_id": "s0", "sample": 1}]
    with open(out, "w") as f:
        f.write('{"state_id": "s0", "sam\n{"state_id": "s1", "sample": 0}\n')
    with pytest.raises(json.JSONDecodeError):
        common.existing_keys(str(out), ("state_id",))  # a cut row elsewhere is a corrupt file, not a resume
    assert not common.stop_requested("") and not common.stop_requested(str(tmp_path / "STOP"))
    (tmp_path / "STOP").write_text("")
    assert common.stop_requested(str(tmp_path / "STOP"))
    # the dump: v2 and v1 rows, a blank line, a cut row and a tokenless row, keys "<basename>:<line>"
    v2 = {
        "rollout_id": 0,
        "index": 0,
        "status": "completed",
        "reward": 1.0,
        "metadata": {"game_file": "g", "split": "train", "max_episode_steps": 30},
        "env_metrics": {"won": 1.0},
        "tokens": [1, 2, 3, 10, 11, 7, 12, 90, 7, 13],
        "response_length": 7,
        "loss_mask": [1, 1, 1, 1, 0, 1, 1],
        "turns": [[0, 4, 2], [5, 7, 5]],
        "v": 2,
        "prompt": [{"role": "system", "content": "sys"}],
        "transcript": [{"role": "system", "content": "sys"}, {"role": "user", "content": OBS0}],
        "tito_model": "qwen35",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    v1 = {
        "status": "truncated",
        "metadata": {"game_file": "h", "split": "unseen"},
        "tokens": [1, 2, 3],
        "response_length": 2,
        "loss_mask": None,
        "turns": [[0, 2, 2]],
    }
    (tmp_path / "dump_b_1.jsonl").write_text(json.dumps(v2) + "\n\n" + json.dumps(v1)[:-5] + "\n")
    (tmp_path / "dump_a_2.jsonl").write_text(
        json.dumps({"tokens": [], "response_length": 0}) + "\n" + json.dumps(v1) + "\n"
    )
    eps = common.read_dump(str(tmp_path / "dump_*.jsonl"))
    assert [ep.key for ep in eps] == ["dump_a_2.jsonl:1", "dump_b_1.jsonl:0"]
    assert common.read_dump(str(tmp_path / "dump_*.jsonl"), limit=1)[0].key == "dump_a_2.jsonl:1"
    old, new = eps
    assert (old.loss_mask, old.split, old.status, old.reward, old.prompt, old.transcript, old.tito_model) == (
        [1, 1],
        "unseen",
        "truncated",
        None,
        None,
        None,
        None,
    )
    assert (new.prompt_length, new.split, new.tito_model, new.chat_template_kwargs) == (
        3,
        "train",
        "qwen35",
        {"enable_thinking": False},
    )
    assert [(t.start, t.end, t.delib_end) for t in new.turns] == [(0, 4, 2), (5, 7, 5)]
    assert new.abs_span(new.turns[1]) == (8, 8, 10) and new.turns[0].delib_tokens == 2 and new.turns[1].has_action
    assert common.policy_turns(dataclasses.asdict(new)) == []  # the dump's transcript holds the reset observation only
    # a row without turns is read, not skipped: stage A re-segments it with the student tokenizer (F8)
    assert common.episode_from_row({"tokens": [1, 2], "response_length": 1}, "k").turns == []


def test_existing_keys_truncates_an_unterminated_last_line(tmp_path):
    """A last line without its newline was cut by a kill whether or not it parses: a complete row kept would have the
    next append_jsonl glue two rows onto one line (read_jsonl then raises Extra data). existing_keys truncates the
    file at that line's byte offset (os.truncate on the bytes: a multibyte row before it does not shift the cut),
    returns the keys of the rows before it, and the restart redoes its row; a cut line anywhere else still raises
    and leaves the file alone."""
    out = tmp_path / "fold.jsonl"
    head = '{"state_id": "s0", "sample": 0}\n{"state_id": "é:0", "sample": 1}\n'.encode()
    for tail in (b'{"state_id": "s1", "sam', b'{"state_id": "s1", "sample": 0}'):
        out.write_bytes(head + tail)
        assert common.existing_keys(str(out), ("state_id", "sample")) == {("s0", 0), ("é:0", 1)}
        assert out.read_bytes() == head
        common.append_jsonl(str(out), {"state_id": "s1", "sample": 0})
        assert common.read_jsonl(str(out))[-1] == {"state_id": "s1", "sample": 0}
        assert common.existing_keys(str(out), ("state_id", "sample")) == {("s0", 0), ("é:0", 1), ("s1", 0)}
        assert out.read_bytes() == head + b'{"state_id": "s1", "sample": 0}\n'
    # what keeping the complete unterminated row did: the next append glued onto it is one line of two rows
    out.write_bytes(head + b'{"state_id": "s1", "sample": 0}')
    common.append_jsonl(str(out), {"state_id": "s1", "sample": 1})
    with pytest.raises(json.JSONDecodeError, match="Extra data"):
        common.read_jsonl(str(out))
    # a cut line elsewhere is a corrupt file, not a resume: nothing is truncated
    corrupt = b'{"state_id": "s0", "sam\n' + head
    out.write_bytes(corrupt)
    with pytest.raises(json.JSONDecodeError):
        common.existing_keys(str(out), ("state_id",))
    assert out.read_bytes() == corrupt
    assert common.existing_keys(str(tmp_path / "absent.jsonl"), ("state_id",)) == set()


def test_stage_parser_limit_refuses_zero(capsys):
    """--limit is a positive int: REF's "0 = unlimited" would make a stage run over nothing and write its marker
    (judge sliced taus[:0] under a JUDGE_DONE; read_dump(limit=0) keeps one row), so 0 and negatives are refused at
    the parser with "N >= 1"; a non-integer is argparse's own refusal; 1 is the smallest smoke."""
    ap = common.stage_parser("Stage doc.\n\nMore.")
    argv = ["--env", "alfworld", "--out", "x.jsonl"]
    assert common.parse_stage_args(ap, [*argv, "--limit", "1"]).limit == 1
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit) as refused:
            common.parse_stage_args(ap, [*argv, "--limit", bad])
        assert refused.value.code == 2 and f"--limit: N >= 1, got {bad}" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        common.parse_stage_args(ap, [*argv, "--limit", "x"])
    assert "invalid positive_int value: 'x'" in capsys.readouterr().err


def test_generate_url_is_the_base_urls_generate_endpoint():
    """The stages' --student-url / --teacher-url are base URLs, as the launcher's --teacher-url: the prefill
    endpoint is <base>/generate (teacher_scoring.prefill posts to the URL it is given), a trailing slash is not
    doubled, and a URL that already carries a path (the endpoint itself included) is refused at the flag, as
    launcher._validate_opd refuses it."""
    assert common.generate_url("http://127.0.0.1:13142", "--student-url") == "http://127.0.0.1:13142/generate"
    assert common.generate_url("http://h:13141/", "--teacher-url") == "http://h:13141/generate"
    for bad in ("http://h:13141/generate", "http://h:13141/v1"):
        with pytest.raises(SystemExit, match=f"--teacher-url {re.escape(bad)}: pass the server's base URL"):
            common.generate_url(bad, "--teacher-url")


def test_base_url_refuses_path_query_and_fragment():
    """The stages' server flags (--student-url, --teacher-url, --judge-url, --gym-url) are base URLs: a trailing slash
    is dropped; a path (the endpoint itself included), a query or a fragment is refused at the flag with the fix
    named, never appended to (http://h:13141?x=1 became ...?x=1/generate), as launcher._validate_opd refuses
    --teacher-url."""
    assert common.base_url("http://h:8200", "--gym-url", "/run") == "http://h:8200"
    assert common.base_url("http://h:8200/", "--gym-url", "/run") == "http://h:8200"
    for bad in ("http://h:8200/run", "http://h:8200?x=1", "http://h:8200/?x=1", "http://h:8200#f", "http://h:8200/#f"):
        with pytest.raises(SystemExit) as refused:
            common.base_url(bad, "--gym-url", "/run")
        assert str(refused.value) == (
            f"--gym-url {bad}: pass the server's base URL (http://host:13141, no path, query or fragment); "
            "the stage appends /run"
        )
    with pytest.raises(SystemExit, match=r"--teacher-url .*no path, query or fragment.*appends /generate"):
        common.generate_url("http://h:13141?x=1", "--teacher-url")


# ---------------------------------------------------------------- stage A: divergence.py (C10)

MARK = 7  # a call-opening token of the unit tests' vocabulary (a dump row stores the spans, not the marker)
STUDENT, TEACHER = "http://127.0.0.1:13142", "http://127.0.0.1:13141"


def _episode(n_turns=3, prompt=4, delib=3, act=2, obs=2, key="d:0"):
    """A dump row in token space (REF tests/test_r2s.py:30-48): the prompt, then per turn ``delib`` tokens, the
    marker and act - 1 argument tokens, with ``obs`` observation tokens (loss_mask 0) between turns."""
    tokens = list(range(1000, 1000 + prompt))
    mask, triples, pos = [], [], 0
    for t in range(n_turns):
        tokens += [1] * delib + [MARK] + [2] * (act - 1)
        mask += [1] * (delib + act)
        triples.append([pos, pos + delib + act, pos + delib])
        pos += delib + act
        if t < n_turns - 1:
            tokens += [3] * obs
            mask += [0] * obs
            pos += obs
    row = {
        "tokens": tokens,
        "response_length": pos,
        "loss_mask": mask,
        "turns": triples,
        "metadata": {"game_file": "g1", "split": "train"},
        "reward": 0.0,
        "status": "completed",
    }
    return common.episode_from_row(row, key)


def test_turn_divergences_mean_over_delib_span():
    """Per turn, mean(lp_S - lp_T) over the deliberation span [start, delib_end) in absolute positions (the paper's
    divergence score, sampled estimator) and over the whole turn (div_all); a position either server left unscored
    is skipped and not counted; under min_delib_tokens scored positions div_delib is None."""
    ep = _episode()  # prompt 4; turns at response [0, 5), [7, 12), [14, 19), each deliberating for 3 tokens
    n = len(ep.tokens)
    lp_t = [None] + [-1.0] * (n - 1)
    lp_s = [None] + [-1.0] * (n - 1)
    s1, d1, e1 = ep.abs_span(ep.turns[1])
    assert (s1, d1, e1) == (11, 14, 16)
    for p in range(s1, d1):
        lp_s[p] = -0.2  # the student is surer than the teacher of turn 1's deliberation ...
    lp_s[e1 - 1] = -3.0  # ... and less sure of its action: only div_all sees that
    rows = divergence.turn_divergences(ep, lp_s, lp_t, min_delib_tokens=3)
    assert [(r["turn"], r["n_delib"], r["has_action"]) for r in rows] == [(0, 3, True), (1, 3, True), (2, 3, True)]
    assert [r["div_delib"] for r in rows] == [pytest.approx(0.0), pytest.approx(0.8), pytest.approx(0.0)]
    assert rows[1]["div_all"] == pytest.approx((0.8 * 3 + 0.0 - 2.0) / 5)
    assert set(rows[0]) == {"turn", "div_delib", "div_all", "n_delib", "has_action"}
    # an unscored position (None on either side) leaves the mean and the count
    lp_t[s1] = None
    rows = divergence.turn_divergences(ep, lp_s, lp_t, 3)
    assert rows[1]["n_delib"] == 2 and rows[1]["div_delib"] is None
    assert rows[1]["div_all"] == pytest.approx((0.8 * 2 - 2.0) / 4)
    assert divergence.turn_divergences(ep, lp_s, lp_t, 2)[1]["div_delib"] == pytest.approx(0.8)
    # a turn that ended without a call deliberates to its end and has no action; a deliberation shorter than
    # min_delib_tokens is not scored (None, its count kept)
    row = {
        "tokens": [1000, 1001, 1, 1, MARK, 2, 3, 1, 1, 1, 1],
        "response_length": 9,
        "loss_mask": [1, 1, 1, 1, 0, 1, 1, 1, 1],
        "turns": [[0, 4, 2], [5, 9, 9]],
    }
    ep = common.episode_from_row(row, "d:1")
    lp_t = [None] + [-1.0] * 10
    lp_s = [None] + [-0.5] * 10
    rows = divergence.turn_divergences(ep, lp_s, lp_t, 3)
    assert [(r["n_delib"], r["div_delib"], r["div_all"], r["has_action"]) for r in rows] == [
        (2, None, pytest.approx(0.5), True),
        (4, pytest.approx(0.5), pytest.approx(0.5), False),
    ]


def test_pick_div_turn_requires_action_and_min_delib():
    """The argmax of div_delib over the eligible turns: turn >= min_div_turn, a scored deliberation (div_delib None
    = under min_delib_tokens) and, with require_action, a tool call at the end; the first turn wins a tie. A turn
    that ended without a call is a format failure, not a deliberation (REF: 266/266 such states collapsed the
    student to zero tool calls); ENV_DEFAULTS require the action everywhere."""
    rows = [
        {"turn": 0, "div_delib": 0.9, "div_all": 0.5, "n_delib": 5, "has_action": False},
        {"turn": 1, "div_delib": None, "div_all": 0.7, "n_delib": 2, "has_action": True},
        {"turn": 2, "div_delib": 0.5, "div_all": 0.1, "n_delib": 3, "has_action": True},
        {"turn": 3, "div_delib": 0.5, "div_all": 0.9, "n_delib": 4, "has_action": True},
        {"turn": 4, "div_delib": 0.4, "div_all": 0.9, "n_delib": 4},  # a row without the key counts as acting
    ]
    assert divergence.pick_div_turn(rows, 0)["turn"] == 0  # without the rule the no-call turn wins
    assert divergence.pick_div_turn(rows, 0, require_action=True)["turn"] == 2  # strict >: the first of the tie
    assert divergence.pick_div_turn(rows, 3, require_action=True)["turn"] == 3
    assert divergence.pick_div_turn(rows, 4, require_action=True)["turn"] == 4
    assert divergence.pick_div_turn(rows, 5, require_action=True) is None
    assert divergence.pick_div_turn(rows[1:2], 0) is None and divergence.pick_div_turn([], 0) is None
    assert all(cfg.require_action for cfg in (ALF, SW, WS))
    # through turn_divergences: min_delib_tokens 4 leaves the 3-token deliberations unscored
    ep = _episode()
    lps = [None] + [-1.0] * (len(ep.tokens) - 1)
    assert divergence.pick_div_turn(divergence.turn_divergences(ep, lps, lps, 4), 0) is None
    assert divergence.pick_div_turn(divergence.turn_divergences(ep, lps, lps, 3), 0)["turn"] == 0  # all 0.0: first


def test_select_states_cap_per_goal_then_top_n():
    """Highest divergence first within each goal, at most max_per_goal of them, then the global top_n of the pool
    (REF's two cuts in that order: the per-goal cap comes before the budget)."""

    def st(goal, score, key):
        return {"goal": goal, "div_score": score, "episode_key": key}

    states = [st("g1", 3, "c"), st("g2", 2, "d"), st("g1", 5, "a"), st("g3", 1, "e"), st("g1", 4, "b")]
    out = divergence.select_states(states, max_per_goal=2, top_n=3)
    assert [(s["goal"], s["div_score"]) for s in out] == [("g1", 5), ("g1", 4), ("g2", 2)]
    assert [s["episode_key"] for s in divergence.select_states(states, 3, 10)] == ["a", "b", "c", "d", "e"]
    assert [s["episode_key"] for s in divergence.select_states(states, 1, 10)] == ["a", "d", "e"]
    assert divergence.select_states(states, 3, 0) == [] and divergence.select_states([], 3, 5) == []
    # equal scores keep the input order (stable sorts), so a rerun over the same dump cuts the same set
    ties = [st("g1", 1.0, "x"), st("g1", 1.0, "y"), st("g2", 1.0, "z")]
    assert [s["episode_key"] for s in divergence.select_states(ties, 1, 2)] == ["x", "z"]


def _dump_row(d0, d1, marker, *, goal="g1", reward=None, status="completed", **fields):
    """A dump row in the fake tokenizer's vocabulary: prompt [1, 2, 3]; turn 0 = three d0 tokens, the call (the
    marker and two argument tokens), an observation; turn 1 = three d1 tokens and its call. The fake prefill's
    student-teacher gap is tok / 100 per position, so turn t diverges by d_t / 100."""
    response = [d0, d0, d0, marker, 20, 21, 90, 91, 92, d1, d1, d1, marker, 30]
    return {
        "status": status,
        "reward": reward,
        "metadata": {"game_file": goal, "split": "train", "max_episode_steps": 30, "tools": [ACT_FLAT]},
        "tokens": [1, 2, 3, *response],
        "response_length": len(response),
        "loss_mask": [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        **fields,
    }


@pytest.mark.usefixtures("stubs", "tito_stub")  # launcher under test_launcher's miles stand-ins; the TITO registry
def test_divergence_resegments_v1_rows_with_tokenizer_path(template_render, tmp_path, monkeypatch, capsys):
    """python -m miles_agent.r2s.divergence over a dump of v2 rows, campaign v1 rows and rows without usable
    spans: the dump's spans serve a row of the requested family (stored here with a different delib_end to tell
    the paths apart) and a v1 row, which names no family (F8); a row without spans, or with another family's, is
    re-cut at the marker of --tokenizer-path's tokenizer under the dir's MODEL_FAMILIES template, loaded once.
    State rows = the Episode fields + REF's keys in divergence order after the per-goal cap and top-n; the summary
    counts long_ctx, prefill_failed, no_eligible_turn, resegmented and ends in DIVERGENCE_DONE; the prefill goes to
    the flags' /generate (the fake asserts the endpoint); the DONE skip, --force, --limit, the missing tokenizer
    path, an unlisted dir, a URL that carries a path and the stop file."""
    from miles_agent.tool_call_markup import marker_token_id, tito_render_chat

    if not (MILES_TEMPLATES / "qwen3.5_fixed.jinja").is_file():
        pytest.skip("3rdparty/miles templates not checked out")
    tokenizer = FakeTokenizer(template_render)
    marker = marker_token_id(tokenizer, tito_render_chat("qwen35", tokenizer, '{"enable_thinking": false}'), ACT_FLAT)
    loads = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code=False):
            loads.append(path)
            return tokenizer

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    divergence.marker_id.cache_clear()
    calls = []

    async def prefill(url, ids, top_k=0, **kw):
        assert url in (STUDENT + "/generate", TEACHER + "/generate"), url  # the endpoint, never the flag's base URL
        calls.append((url, tuple(ids), kw))
        if ids[0] == 999:
            raise teacher_scoring.TeacherScoringError(f"{url} down")
        gap = 0.0 if url == STUDENT + "/generate" else 1.0
        return {"lps": [None] + [-1.0 - gap * tok / 100 for tok in ids[1:]], "top": [None] * len(ids)}

    monkeypatch.setattr(divergence.teacher_scoring, "prefill", prefill)
    own = [[0, 6, 3], [9, 14, 12]]  # what segment_turns cuts at the marker
    prompt = [{"role": "system", "content": "sys"}]
    v2 = _dump_row(
        12,
        6,
        marker,
        reward=1.0,
        turns=[[0, 6, 4], [9, 14, 12]],
        v=2,
        prompt=prompt,
        transcript=[*prompt, {"role": "user", "content": OBS0}],
        tito_model="qwen35",
        chat_template_kwargs={"enable_thinking": False},
    )
    v1 = _dump_row(40, 80, marker, reward=0.0, turns=own)  # a campaign row: spans, no family
    other = _dump_row(50, 24, marker, goal="g2", tito_model="qwen3", turns=[[0, 14, 14]])  # another family's spans
    bare = _dump_row(10, 32, marker, reward=1.0)  # no spans at all
    long = _dump_row(10, 10, marker, goal="g3", turns=own)
    long["tokens"] = [7] * 10 + long["tokens"]
    silent = {**v1, "tokens": [1, 2, 3, 90, 91], "response_length": 2, "loss_mask": [0, 0], "turns": []}
    aborted = _dump_row(10, 10, marker, status="aborted", turns=own)
    down = _dump_row(10, 10, marker, goal="g3", turns=own)
    down["tokens"][0] = 999
    dump = tmp_path / "dump"
    dump.mkdir()
    common.write_jsonl(str(dump / "dump_a.jsonl"), [v2, v1, other])
    common.write_jsonl(str(dump / "dump_b.jsonl"), [bare, long, silent, aborted, down])
    urls = ["--student-url", STUDENT, "--teacher-url", TEACHER]
    base = ["--env", "alfworld", "--dump-glob", str(dump / "dump_*.jsonl"), *urls, "--max-ctx", "20"]
    tok_path = "/models/Qwen3.5-4B"
    out = tmp_path / "states.jsonl"
    summary_path = f"{out}.summary.json"
    assert divergence.main([*base, "--out", str(out), "--tokenizer-path", tok_path]) == 0
    states = common.read_jsonl(str(out))
    by_divergence = ["dump_a.jsonl:1:1", "dump_a.jsonl:2:0", "dump_b.jsonl:0:1", "dump_a.jsonl:0:0"]
    assert [s["state_id"] for s in states] == by_divergence
    assert [s["div_score"] for s in states] == [pytest.approx(x) for x in (0.80, 0.50, 0.32, 0.1025)]
    assert [s["turns"] for s in states] == [own, own, own, [[0, 6, 4], [9, 14, 12]]]
    assert loads == [tok_path]  # F8: the dir's row is qwen35; one load for both re-cut rows
    first = states[0]
    assert (first["episode_key"], first["goal"], first["split"]) == ("dump_a.jsonl:1", "g1", "train")
    assert first["div_turn"] == 1 and set(first) == {
        *("state_id", "episode_key", "goal", "split", "div_turn", "div_score", "div_rows"),  # REF's state keys
        *("tokens", "response_length", "loss_mask", "turns", "metadata", "reward", "status"),  # REF's row keys
        *("prompt", "transcript", "tito_model", "chat_template_kwargs"),  # the v2 fields
    }
    div_all = [(120 + marker + 20 + 21) / 600, (240 + marker + 30) / 500]  # the call's tokens count in div_all
    assert first["div_rows"] == [
        {"turn": t, "div_delib": pytest.approx(d), "div_all": pytest.approx(a), "n_delib": 3, "has_action": True}
        for t, d, a in ((0, 0.40, div_all[0]), (1, 0.80, div_all[1]))
    ]
    assert {k: first[k] for k in ("tokens", "loss_mask", "metadata", "reward", "status")} == {
        k: v1[k] for k in ("tokens", "loss_mask", "metadata", "reward", "status")
    }
    assert (first["prompt"], first["transcript"], first["tito_model"], first["chat_template_kwargs"]) == (None,) * 4
    v2_state = states[3]
    assert (v2_state["prompt"], v2_state["tito_model"], v2_state["chat_template_kwargs"], v2_state["reward"]) == (
        prompt,
        "qwen35",
        {"enable_thinking": False},
        1.0,
    )
    assert v2_state["transcript"] == v2["transcript"] and states[2]["goal"] == "g1" and states[1]["goal"] == "g2"
    summary = common.read_summary(summary_path)
    counted = ("episodes_read", "scored", "resegmented", "long_ctx", "no_eligible_turn", "prefill_failed")
    assert [summary[k] for k in counted] == [7, 4, 2, 1, 1, 1]  # 8 rows, the aborted one never read
    assert (summary["states_out"], summary["goals_out"], summary["div_turn_hist"]) == (4, 2, {"0": 2, "1": 2})
    assert summary["won_frac_in_states"] == 0.5 and summary["div_score_min"] == pytest.approx(0.1025)
    assert summary["div_score_median"] == pytest.approx(0.41)
    knobs = ("top_n", "max_per_goal", "min_delib_tokens", "min_div_turn", "max_ctx", "tito_model")
    assert [summary[k] for k in knobs] == [2400, 3, 3, 0, 20, "qwen35"]
    assert summary["config"] == {**dataclasses.asdict(ALF), "goal_keys": ["game_file"]}
    assert summary["DIVERGENCE_DONE"] is True and open(summary_path).read().endswith("}\nDIVERGENCE_DONE\n")
    printed = capsys.readouterr().out
    assert printed.rstrip().endswith("DIVERGENCE_DONE") and "prefill failed for dump_b.jsonl:4" in printed
    # one prefill of the whole stream on each server's /generate per episode under --workers; long_ctx and the
    # silent row never reach a server
    assert len(calls) == 10 and {c[0] for c in calls} == {STUDENT + "/generate", TEACHER + "/generate"}
    assert {c[2]["concurrency"] for c in calls} == {32}
    assert {(c[0], c[1]) for c in calls} >= {(u + "/generate", tuple(v1["tokens"])) for u in (STUDENT, TEACHER)}
    assert all(len(c[1]) <= 20 for c in calls) and tuple(silent["tokens"]) not in {c[1] for c in calls}
    # done: a rerun does nothing; --force with the cap and the budget (and a trailing slash on a base URL)
    calls.clear()
    assert divergence.main([*base, "--out", str(out), "--tokenizer-path", tok_path]) == 0
    assert "DIVERGENCE_DONE already in" in capsys.readouterr().out and calls == []
    argv = [*base, "--out", str(out), "--tokenizer-path", tok_path, "--force"]
    assert divergence.main([*argv, "--max-per-goal", "1", "--top-n", "2", "--student-url", STUDENT + "/"]) == 0
    assert {c[0] for c in calls} == {STUDENT + "/generate", TEACHER + "/generate"}  # the slash is not doubled
    assert [s["state_id"] for s in common.read_jsonl(str(out))] == ["dump_a.jsonl:1:1", "dump_a.jsonl:2:0"]
    summary = common.read_summary(summary_path)
    assert [summary[k] for k in ("states_out", "goals_out", "max_per_goal", "top_n")] == [2, 2, 1, 2]
    # an explicit family: the other row's stored span now serves (it ends without a call: eligible under
    # --no-require-action alone) and the v2 row's spans are re-cut at the qwen3 template's marker (the same
    # token, one more tokenizer load)
    assert divergence.main([*argv, "--no-require-action", "--tito-model", "qwen3", "--chat-template-kwargs", ""]) == 0
    states = common.read_jsonl(str(out))
    assert [s["state_id"] for s in states] == by_divergence
    assert [s["turns"] for s in states] == [own, [[0, 14, 14]], own, own]
    assert states[1]["div_score"] == pytest.approx(576 / 1400) and states[3]["div_score"] == pytest.approx(0.12)
    summary = common.read_summary(summary_path)
    assert [summary[k] for k in ("resegmented", "tito_model", "states_out")] == [2, "qwen3", 4]
    assert summary["config"]["require_action"] is False and loads == [tok_path, tok_path]
    # --limit: the first two rows carry usable spans, so no tokenizer is needed; without one the others are refused
    # before any prefill, as is a dir no family lists
    lim = tmp_path / "lim.jsonl"
    assert divergence.main([*base, "--out", str(lim), "--limit", "2"]) == 0
    summary = common.read_summary(f"{lim}.summary.json")
    assert [summary[k] for k in ("episodes_read", "resegmented", "states_out", "tito_model")] == [2, 0, 2, None]
    calls.clear()
    with pytest.raises(SystemExit, match="--tokenizer-path"):
        divergence.main([*base, "--out", str(tmp_path / "none.jsonl")])
    with pytest.raises(SystemExit, match="no MODEL_FAMILIES entry for 'mystery'"):
        divergence.main([*base, "--out", str(tmp_path / "none.jsonl"), "--tokenizer-path", "/models/mystery"])
    # a URL that already carries a path (the endpoint itself included) is refused: the stage appends /generate
    with pytest.raises(SystemExit, match="--teacher-url .*13141/generate: pass the server's base URL"):
        divergence.main([*base, "--out", str(tmp_path / "none.jsonl"), "--teacher-url", TEACHER + "/generate"])
    assert calls == [] and loads == [tok_path, tok_path] and not (tmp_path / "none.jsonl").exists()
    # the preempt sentinel: exit 0, no states, no marker
    stop = tmp_path / "STOP"
    stop.write_text("")
    stopped = tmp_path / "stopped.jsonl"
    argv = [*base, "--out", str(stopped), "--tokenizer-path", tok_path, "--stop-file", str(stop)]
    assert divergence.main(argv) == 0 and not stopped.exists()
    assert common.read_summary(f"{stopped}.summary.json") is None
    divergence.marker_id.cache_clear()


def test_divergence_counts_long_ctx_before_resegmenting(tmp_path, monkeypatch):
    """The --max-ctx cut runs before the turn spans are settled: an over-length campaign v1 row without spans and
    without metadata["tools"] (nothing to derive the marker from) is counted as long_ctx instead of raising inside
    resegment, needs no --tokenizer-path and never reaches a server; the rows under the cap are scored as before,
    and under a cap that admits the same row the stage still asks for the tokenizer before any prefill."""
    calls = []

    async def prefill(url, ids, top_k=0, **kw):
        calls.append(tuple(ids))
        gap = 0.0 if url == STUDENT + "/generate" else 1.0
        return {"lps": [None] + [-1.0 - gap * tok / 100 for tok in ids[1:]], "top": [None] * len(ids)}

    monkeypatch.setattr(divergence.teacher_scoring, "prefill", prefill)
    v1 = _dump_row(40, 80, MARK, turns=[[0, 6, 3], [9, 14, 12]])
    long_bare = _dump_row(10, 10, MARK, goal="g2")  # no spans, no family
    long_bare["tokens"] = [7] * 10 + long_bare["tokens"]
    del long_bare["metadata"]["tools"]
    common.write_jsonl(str(tmp_path / "dump_a.jsonl"), [v1, long_bare])
    out = tmp_path / "states.jsonl"
    argv = ["--env", "alfworld", "--dump-glob", str(tmp_path / "dump_*.jsonl"), "--student-url", STUDENT]
    argv += ["--teacher-url", TEACHER, "--max-ctx", "20", "--out", str(out)]
    assert divergence.main(argv) == 0
    summary = common.read_summary(f"{out}.summary.json")
    counted = ("episodes_read", "long_ctx", "resegmented", "scored", "states_out")
    assert [summary[k] for k in counted] == [2, 1, 0, 1, 1]
    assert [s["state_id"] for s in common.read_jsonl(str(out))] == ["dump_a.jsonl:0:1"]
    assert len(calls) == 2 and all(len(ids) <= 20 for ids in calls)  # the v1 row on each server; the long one never
    with pytest.raises(SystemExit, match="--tokenizer-path"):
        divergence.main([*argv, "--out", str(tmp_path / "x.jsonl"), "--max-ctx", "40"])
    assert len(calls) == 2 and not (tmp_path / "x.jsonl").exists()


# ---------------------------------------------------------------- stage B: takeover.py (C11)

GYM = "http://127.0.0.1:8200"
T0 = _act("c1", "go to desk 1", "The desk is where the mug should end up; I check it first.")
T1 = {"role": "assistant", "content": "I open the drawer."}  # no call: the env's hint comes back as a user message
T2 = _act("c2", "open drawer 1")
EPISODE = [
    {"role": "user", "content": OBS0},
    T0,
    {"role": "tool", "tool_call_id": "c1", "content": OBS1},
    T1,
    {"role": "user", "content": HINT},
    T2,
    {"role": "tool", "tool_call_id": "c2", "content": OBS3},
]
WON_OBS = "You put the mug 2 in/on the desk 1.\nTask completed."
# REF takeover.py:274-317; the port adds the agent's messages
REF_ROW_KEYS = (
    *("state_id", "goal", "split", "div_turn", "sample", "replay_ok", "won", "done", "truncated", "reward"),
    *("turns_total", "invalid_actions", "takeover_steps", "last_observation", "terminal_key"),
)


def _state(div_turn=2, *, state_id="dump_a.jsonl:0:2", goal="g1", prompt=None, episode=EPISODE, n_turns=3, **meta):
    """A stage A row (the dump row's fields + REF's state keys) over the one-shot prompt and a three-turn
    episode; ``meta`` lays over the metadata (max_episode_steps, tools, ...)."""
    prompt = _oneshot_prompt() if prompt is None else prompt
    return {
        "state_id": state_id,
        "episode_key": state_id.rsplit(":", 1)[0],
        "goal": goal,
        "split": "train",
        "div_turn": div_turn,
        "div_score": 0.5,
        "div_rows": [],
        "tokens": list(range(40)),
        "response_length": 30,
        "loss_mask": [1] * 30,
        "turns": [[0, 10, 4], [10, 20, 20], [20, 30, 24]][:n_turns],
        "metadata": {
            "game_file": goal,
            "split": "train",
            "data_source": "alfworld",
            "max_episode_steps": 30,
            "tools": [ACT_FLAT],
            **meta,
        },
        "reward": 0.0,
        "status": "completed",
        "prompt": prompt,
        "transcript": [*prompt, *episode],
        "tito_model": "qwen35",
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _observation(msg, text):
    """The observation as env_loop_agent appends it: a tool reply to the turn's first call, a user message
    after a turn without one (app.py:157-177)."""
    calls = msg.get("tool_calls") or []
    return (
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": text}
        if calls
        else {"role": "user", "content": text}
    )


def _reply(
    state,
    teacher,
    *,
    obs=None,
    scripted_steps=None,
    won=False,
    terminated=None,
    truncated=False,
    policy_400=0,
    reward=None,
    invalid_actions=0.0,
):
    """What env_loop_agent returns for the stage's body (app.py:279-296): the prompt re-emitted, the reset
    observation, the replayed prefix with the dump's observations (``obs`` overrides them by index,
    ``scripted_steps`` cuts the replay short), then ``teacher`` = [(assistant message, observation), ...]."""
    div_turn = state["div_turn"]
    prefix = common.policy_turns(state)[:div_turn]
    want = [(obs or {}).get(i, o) for i, o in enumerate(common.observations_after_prefix(state, div_turn))]
    n = len(prefix) if scripted_steps is None else scripted_steps
    messages = [*state["prompt"], {"role": "user", "content": want[0]}]
    for i in range(n):
        messages += [prefix[i], _observation(prefix[i], want[i + 1])]
    for msg, observation in teacher:
        messages += [msg, _observation(msg, observation)]
    return {
        "reward": float(won) if reward is None else reward,
        "won": float(won),
        "invalid_actions": invalid_actions,
        "scripted_steps": float(n),
        "scripted_terminated": float(n < len(prefix)),
        "turns": float(len(teacher)),
        "env_truncated": float(truncated),
        "terminated": float(won if terminated is None else terminated),
        "policy_400_truncated": policy_400,
        "last_observation": teacher[-1][1] if teacher else want[n],
        "messages": messages,
    }


class FakeGym:
    """POST <gym>/run stood in: the bodies are recorded and the replies come from ``script`` keyed by the
    body's game_file (a list consumed in sample order: a reply, or an exception to raise); ``stop_file`` is
    written as the first call arrives (the preempt sentinel rising mid-stage)."""

    def __init__(self, script, stop_file=None):
        self.script = {k: list(v) for k, v in script.items()}
        self.stop_file = stop_file
        self.bodies = []

    async def post_json(self, url, body):
        assert url == GYM + "/run", url
        self.bodies.append(body)
        reply = self.script[body["game_file"]].pop(0)
        if self.stop_file:
            with open(self.stop_file, "w") as f:
                f.write("")
        await asyncio.sleep(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def served(self):
        return collections.Counter(b["game_file"] for b in self.bodies)


def test_takeover_body_uses_scripted_prefix_and_teacher_endpoint():
    """The /run body is gym_bridge.build_run_body over the row's own prompt, metadata and sampling, the teacher
    server as the per-request policy (policy_base_url <teacher>/v1, its served model name, api key EMPTY) and the
    student's turns before the divergence turn as the scripted prefix (D5); nothing of the row beyond its metadata
    rides along, the one-shot prompt's demonstration turns stay prompt, and --teacher-url is a base URL."""
    state = _state(div_turn=2)
    kwargs = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512}
    body = takeover.run_body(state, TEACHER, "teacher", kwargs)
    assert body["policy_base_url"] == TEACHER + "/v1" and body["policy_api_key"] == "EMPTY"
    assert body["scripted_prefix"] == [T0, T1]
    params = body["responses_create_params"]
    assert params["input"] == state["prompt"] and params["tools"] == [ACT_FLAT] and params["model"] == "teacher"
    assert (params["temperature"], params["top_p"], params["max_output_tokens"]) == (1.0, 1.0, 512)
    assert (body["game_file"], body["max_episode_steps"], body["split"], body["data_source"]) == (
        "g1",
        30,
        "train",
        "alfworld",
    )
    assert not {"tools", "tokens", "transcript", "prompt", "turns", "div_turn", "state_id"} & set(body)
    assert sum(m["role"] == "assistant" for m in params["input"]) == 2  # the demonstration's, never prefix
    json.dumps(body)
    assert takeover.run_body(_state(div_turn=0), TEACHER, "t", kwargs)["scripted_prefix"] == []
    assert takeover.run_body(_state(div_turn=3), TEACHER, "t", kwargs)["scripted_prefix"] == [T0, T1, T2]
    assert takeover.run_body(state, TEACHER, "t", {"temperature": 0.4})["responses_create_params"] == {
        "input": state["prompt"],
        "tools": [ACT_FLAT],
        "temperature": 0.4,
        "model": "t",
    }
    # the flag is the server's base URL (the launcher's --teacher-url rule): a trailing slash is dropped, a path
    # (the endpoint itself included) refused
    assert common.base_url(TEACHER + "/", "--teacher-url", "/v1") == TEACHER
    with pytest.raises(SystemExit, match="--teacher-url .*13141/v1: pass the server's base URL .* appends /v1"):
        common.base_url(TEACHER + "/v1", "--teacher-url", "/v1")


def test_takeover_row_from_messages():
    """REF takeover.py:274-317's row from the agent's /run reply: the teacher's steps read from the structured
    transcript after the replayed prefix (turn = div_turn + i, the first call as emitted, the command under the
    env's rule, format_ok, obs_before), won / truncated / reward / invalid_actions from the numerics, turns_total
    and the done formula over scripted_steps + turns against the row's own max_episode_steps (D16), the
    terminal-observation fingerprint of a finished takeover, and the transcript kept."""
    state = _state(div_turn=2)
    teacher = [
        (_act("t1", "take mug 2 from desk 1", "I take it."), "You pick up the mug 2."),
        ({"role": "assistant", "content": "done?"}, HINT),
        (_act("t2", "put mug 2 in/on desk 1"), WON_OBS),
    ]
    row = takeover.takeover_row(state, 3, _reply(state, teacher, won=True, reward=1.0), ALF)
    assert set(row) == {*REF_ROW_KEYS, "messages"}
    assert [row[k] for k in ("state_id", "goal", "split", "div_turn", "sample")] == [
        state["state_id"],
        "g1",
        "train",
        2,
        3,
    ]
    assert [row[k] for k in ("replay_ok", "won", "done", "truncated", "reward")] == [True, True, True, False, 1.0]
    assert (row["turns_total"], row["invalid_actions"], row["last_observation"]) == (5.0, 0.0, WON_OBS)
    assert row["terminal_key"] == common.fingerprint(WON_OBS) and row["messages"] == _reply(state, teacher)["messages"]
    assert row["takeover_steps"] == [
        {
            "turn": 2,
            "delib": "I take it.",
            "command": "take mug 2 from desk 1",
            "call": {"name": "act", "arguments": json.dumps({"command": "take mug 2 from desk 1"})},
            "format_ok": True,
            "obs_before": HINT,
        },
        {
            "turn": 3,
            "delib": "done?",
            "command": None,
            "call": None,
            "format_ok": False,
            "obs_before": "You pick up the mug 2.",
        },
        {
            "turn": 4,
            "delib": "",
            "command": "put mug 2 in/on desk 1",
            "call": {"name": "act", "arguments": json.dumps({"command": "put mug 2 in/on desk 1"})},
            "format_ok": True,
            "obs_before": HINT,
        },
    ]
    # the done formula (REF :297 with scripted_steps): a truncated takeover is not done and has no key; a reward
    # of 1 is a win; a takeover the env ended neither way counts prefix + teacher turns against the row's budget
    # (the config's max_steps standing in for a row without one)
    lost = [(_act("t1", "look"), "Nothing happens.")] * 3
    row = takeover.takeover_row(state, 0, _reply(state, lost, truncated=True), ALF)
    assert (row["done"], row["terminal_key"], row["won"], row["truncated"]) == (False, None, False, True)
    assert takeover.takeover_row(state, 0, _reply(state, lost[:1], policy_400=1, truncated=True), ALF)["truncated"]
    assert takeover.takeover_row(state, 0, _reply(state, lost, reward=1.0, terminated=False), ALF)["won"] is True
    row = takeover.takeover_row(state, 0, _reply(state, lost, terminated=False), ALF)
    assert (row["done"], row["turns_total"], row["terminal_key"]) == (
        True,
        5.0,
        common.fingerprint("Nothing happens."),
    )
    tight = _state(max_episode_steps=5)
    assert not takeover.takeover_row(tight, 0, _reply(tight, lost, terminated=False), ALF)["done"]
    bare = _state()
    del bare["metadata"]["max_episode_steps"]
    assert takeover.takeover_row(bare, 0, _reply(bare, lost, terminated=False), ALF)["done"]
    assert not takeover.takeover_row(
        bare, 0, _reply(bare, lost, terminated=False), dataclasses.replace(ALF, max_steps=5)
    )["done"]
    # invalid_actions and reward pass through as floats; a key only for a finished takeover with an observation
    row = takeover.takeover_row(state, 0, _reply(state, lost, terminated=True, invalid_actions=2, reward=0.25), ALF)
    assert (row["invalid_actions"], row["reward"], row["won"], row["done"]) == (2.0, 0.25, False, True)
    assert (
        takeover.takeover_row(state, 0, {**_reply(state, lost, terminated=True), "last_observation": ""}, ALF)[
            "terminal_key"
        ]
        is None
    )


def test_replay_ok_requires_full_prefix_and_matching_observations():
    """replay_ok needs the whole prefix stepped, every observation from the reset one through the one after the
    last replayed turn equal to the dump's under norm_text (REF's proxy answered 400 obs_mismatch@t for the same
    inside the episode; a teacher-side observation is the teacher's business) and a teacher turn; a failed replay's
    row is REF's short row plus the transcript; a state whose transcript and spans disagree on the number of
    turns never runs."""
    state = _state(div_turn=2)
    teacher = [(_act("t1", "take mug 2 from desk 1"), WON_OBS)]
    assert takeover.replay_fail(state, _reply(state, teacher, won=True)) is None
    spaced = {0: OBS0.replace(" ", "  "), 1: OBS1 + "\n", 2: " " + HINT}
    assert takeover.replay_fail(state, _reply(state, teacher, obs=spaced, won=True)) is None
    assert takeover.replay_fail(state, _reply(state, teacher, scripted_steps=1)) == "prefix_ended@1"
    assert takeover.replay_fail(state, _reply(state, teacher, scripted_steps=0)) == "prefix_ended@0"
    for i in range(3):
        assert takeover.replay_fail(state, _reply(state, teacher, obs={i: "something else"})) == f"obs_mismatch@{i}"
    assert takeover.replay_fail(state, _reply(state, [])) == "no_teacher_turn"
    assert takeover.replay_fail(state, {**_reply(state, teacher), "messages": []}) == "obs_mismatch@0"
    assert takeover.replay_fail(state, _reply(state, [(_act("t1", "look"), "You pick it up.")])) is None
    # div_turn 0: nothing to replay but the reset observation; div_turn 3: the whole episode
    s0 = _state(div_turn=0)
    assert takeover.replay_fail(s0, _reply(s0, teacher)) is None
    assert takeover.replay_fail(s0, _reply(s0, teacher, obs={0: "x"})) == "obs_mismatch@0"
    s3 = _state(div_turn=3)
    assert takeover.replay_fail(s3, _reply(s3, teacher)) is None
    assert takeover.replay_fail(s3, _reply(s3, teacher, obs={3: "x"})) == "obs_mismatch@3"
    # the failed replay's row, and the row of a takeover that never came back
    row = takeover.takeover_row(state, 4, _reply(state, teacher, obs={1: "x"}), ALF)
    assert set(row) == {"state_id", "goal", "split", "div_turn", "sample", "replay_ok", "fail", "messages"}
    assert (row["replay_ok"], row["fail"], row["sample"], len(row["messages"])) == (False, "obs_mismatch@1", 4, 13)
    assert takeover.fail_row(state, 1, "run_transport:ConnectError") == {
        "state_id": state["state_id"],
        "goal": "g1",
        "split": "train",
        "div_turn": 2,
        "sample": 1,
        "replay_ok": False,
        "fail": "run_transport:ConnectError",
    }
    # the state-level check: three policy turns in the transcript, two spans in the tokens; no policy turn at all
    assert takeover.state_fail(state) is None
    assert takeover.state_fail(_state(n_turns=2)) == "turn_count_mismatch"
    assert takeover.state_fail({**state, "transcript": [*state["prompt"], EPISODE[0]]}) == "turn_count_mismatch"
    # a sample miles cut at the per-turn cap: the spans end at the cut turn while the agent played on, so a longer
    # transcript is no disagreement; a shorter one still is
    assert takeover.state_fail({**_state(n_turns=2), "status": "truncated"}) is None
    assert takeover.state_fail({**state, "status": "truncated", "transcript": [*state["prompt"], EPISODE[0]]}) == "turn_count_mismatch"


def test_terminal_key_modes():
    """terminal: the fingerprint of a finished takeover's terminal observation; purchased_asin: the last
    product-code click over the student's prefix and the teacher's steps (REF takeover.py:87-107; the Search answer
    key is not ported); command_from_call follows the env's rule on every shape (session_gym.extract_command)."""
    cmds = ["search[l'eau d'issey 6.76 fl oz]", "click[B000VOHH8I]", "click[6.76 fl oz (pack of 1)]", "click[buy now]"]
    assert takeover.purchased_asin(cmds) == "b000vohh8i"
    assert (
        takeover.purchased_asin(cmds[:2] + ["click[< prev]", "click[b09xyzabcd]", None, "click[buy now]"])
        == "b09xyzabcd"
    )
    assert takeover.purchased_asin(["search[mug]", "click[next >]", "click[buy now]"]) is None
    assert takeover.terminal_key_of("purchased_asin", True, "obs", cmds) == "b000vohh8i"
    assert takeover.terminal_key_of("purchased_asin", False, "obs", cmds) is None
    assert takeover.terminal_key_of("terminal", True, "Task completed.", cmds) == common.fingerprint("Task completed.")
    assert takeover.terminal_key_of("terminal", True, "", cmds) is None
    assert takeover.terminal_key_of("terminal", False, "x", []) is None
    assert takeover.terminal_key_of("terminal", True, "a  b\n", []) == takeover.terminal_key_of(
        "terminal", True, "a b", []
    )
    ok = json.dumps({"command": "click[buy now]"})
    assert takeover.command_from_call("act", ok, "act") == "click[buy now]"
    assert takeover.command_from_call("act", json.dumps({"command": " click[buy now] "}), "act") == "click[buy now]"
    assert takeover.command_from_call("act", {"command": "look"}, "act") == "look"  # parsed arguments too
    two = json.dumps({"command": "click[buy now]", "reason": "done"})
    assert takeover.command_from_call("act", two, "act") == "click[buy now]"  # the env takes `command`, not the JSON
    # wrong tool name / wrong key / empty command / bad JSON / non-object args / no args / no name: a format error
    bad = (
        ("search", ok),
        ("act", json.dumps({"cmd": "x"})),
        ("act", json.dumps({"command": "  "})),
        ("act", "{not json"),
        ("act", json.dumps(["x"])),
        ("act", None),
        (None, ok),
    )
    assert [takeover.command_from_call(name, arguments, "act") for name, arguments in bad] == [None] * len(bad)
    # through a message: the first call counts (the env steps it), a call-less message has no command
    assert takeover.first_call(T0) == {"name": "act", "arguments": json.dumps({"command": "go to desk 1"})}
    assert takeover.first_call(T1) is None and takeover.message_command(T1, "act") is None
    assert takeover.message_command(T0, "act") == "go to desk 1" and takeover.message_command(T0, "search") is None
    calls = [*_act("a", "click[b000000001]")["tool_calls"], *_act("b", "click[b000000002]")["tool_calls"]]
    assert (
        takeover.message_command({"role": "assistant", "content": "", "tool_calls": calls}, "act")
        == "click[b000000001]"
    )
    # a WebShop takeover's key spans the student's prefix and the teacher's steps; the student's own click decides
    # when the teacher only buys; under the terminal mode the same rows key on the observation
    ws_episode = [
        {"role": "user", "content": "WebShop\nInstruction: a mug\n[button] Search [button_]"},
        _act("c1", "search[mug]"),
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
        _act("c2", "click[b000000001]"),
        {"role": "tool", "tool_call_id": "c2", "content": "item 1"},
    ]
    ws1 = _state(div_turn=1, episode=ws_episode, n_turns=2, goal="goal", max_episode_steps=15)
    teacher = [(_act("t1", "click[B000000002]"), "item 2"), (_act("t2", "click[buy now]"), "Thank you for shopping")]
    row = takeover.takeover_row(ws1, 0, _reply(ws1, teacher, won=True, reward=0.6), WS)
    assert (row["terminal_key"], row["won"], row["reward"], row["turns_total"]) == ("b000000002", True, 0.6, 3.0)
    ws2 = _state(div_turn=2, episode=ws_episode, n_turns=2, goal="goal", max_episode_steps=15)
    buy = [(_act("t1", "click[buy now]"), "Thank you for shopping")]
    assert takeover.takeover_row(ws2, 0, _reply(ws2, buy, terminated=True), WS)["terminal_key"] == "b000000001"
    assert takeover.takeover_row(ws2, 0, _reply(ws2, buy, truncated=True), WS)["terminal_key"] is None
    assert takeover.takeover_row(ws2, 0, _reply(ws2, buy, terminated=True), ALF)["terminal_key"] == common.fingerprint(
        "Thank you for shopping"
    )
    nothing = [(_act("t1", "click[next >]"), "page 2"), (_act("t2", "click[buy now]"), "Thank you")]
    ws0 = _state(div_turn=0, episode=ws_episode, n_turns=2, goal="goal", max_episode_steps=15)
    assert takeover.takeover_row(ws0, 0, _reply(ws0, nothing, terminated=True), WS)["terminal_key"] is None


def test_takeover_resumes_after_partial_fold(tmp_path, monkeypatch, capsys):
    """python -m miles_agent.r2s.takeover: K bodies per state through POST <gym>/run (the fake records them), rows
    appended per state, the stop file honoured between states (the state in flight finishes, no marker, exit 0),
    a restart that redoes only the (state_id, sample) pairs missing (a cut last row included), the summary over the
    whole fold with TAKEOVER_DONE as key and trailing line, the DONE skip, --force and --limit."""
    states = [_state(div_turn=2, state_id=f"d:{i}:2", goal=f"g{i}") for i in range(3)]
    teacher = [(_act("t1", "take mug 2 from desk 1", "Taking it."), WON_OBS)]

    def won(i):
        return _reply(states[i], teacher, won=True)

    states_path = tmp_path / "states.jsonl"
    common.write_jsonl(str(states_path), states)
    out = tmp_path / "fold.jsonl"
    stop = tmp_path / "STOP"
    gym = FakeGym({f"g{i}": [won(i)] * 2 for i in range(3)}, stop_file=str(stop))
    monkeypatch.setattr(takeover, "post_json", gym.post_json)
    base = ["--env", "alfworld", "--states", str(states_path), "--teacher-url", TEACHER, "--out", str(out)]
    base += ["--gym-url", GYM, "--samples", "2", "--concurrency", "2"]
    assert takeover.main([*base, "--stop-file", str(stop)]) == 0
    assert [(r["state_id"], r["sample"]) for r in common.read_jsonl(str(out))] == [("d:0:2", 0), ("d:0:2", 1)]
    assert gym.served() == {"g0": 2} and not os.path.exists(f"{out}.summary.json")
    printed = capsys.readouterr().out
    assert "d:0:2: 2/2 replayed" in printed and "2 states not taken over; exiting without TAKEOVER_DONE" in printed
    for body in gym.bodies:  # the teacher endpoint, the prefix, the served name and the campaign sampling, every time
        assert body["policy_base_url"] == TEACHER + "/v1" and body["scripted_prefix"] == [T0, T1]
        params = body["responses_create_params"]
        assert (params["model"], params["temperature"], params["top_p"], params["max_output_tokens"]) == (
            "teacher",
            1.0,
            1.0,
            512,
        )
    # a row cut by a kill is dropped and redone; the restart posts the missing pairs only
    with open(out, "a") as f:
        f.write('{"state_id": "d:1:2", "sam')
    gym2 = FakeGym({"g1": [won(1)] * 2, "g2": [won(2)] * 2})
    monkeypatch.setattr(takeover, "post_json", gym2.post_json)
    assert takeover.main(base) == 0
    rows = common.read_jsonl(str(out))
    assert sorted((r["state_id"], r["sample"]) for r in rows) == [(f"d:{i}:2", s) for i in range(3) for s in range(2)]
    assert gym2.served() == {"g1": 2, "g2": 2}
    summary_path = f"{out}.summary.json"
    summary = common.read_summary(summary_path)
    assert [summary[k] for k in ("n_takeovers", "replay_ok", "replay_ok_frac", "states_total")] == [6, 6, 1.0, 3]
    frac = ("won_frac", "done_frac", "states_with_vote_ge2", "mean_teacher_turns", "format_ok_frac")
    assert [summary[k] for k in frac] == [1.0, 1.0, 3, 1.0, 1.0]
    knobs = ("samples", "temperature", "top_p", "max_teacher_tokens", "min_replay_ok", "teacher_model")
    assert [summary[k] for k in knobs] == [2, 1.0, 1.0, 512, 0.7, "teacher"]
    assert summary["config"] == {**dataclasses.asdict(ALF), "goal_keys": ["game_file"]}
    assert summary["TAKEOVER_DONE"] is True and open(summary_path).read().endswith("}\nTAKEOVER_DONE\n")
    assert capsys.readouterr().out.rstrip().endswith("TAKEOVER_DONE")
    # done: nothing runs; --force truncates the fold and redoes every pair (a trailing slash on --gym-url dropped);
    # --limit takes the first states
    gym3 = FakeGym({f"g{i}": [won(i)] * 2 for i in range(3)})
    monkeypatch.setattr(takeover, "post_json", gym3.post_json)
    assert takeover.main(base) == 0 and gym3.bodies == []
    assert "TAKEOVER_DONE already in" in capsys.readouterr().out
    assert takeover.main([*base, "--force", "--gym-url", GYM + "/"]) == 0
    assert gym3.served() == {"g0": 2, "g1": 2, "g2": 2} and len(common.read_jsonl(str(out))) == 6
    lim = tmp_path / "lim.jsonl"
    gym4 = FakeGym({"g0": [won(0)] * 2})
    monkeypatch.setattr(takeover, "post_json", gym4.post_json)
    assert takeover.main([*base, "--out", str(lim), "--limit", "1"]) == 0
    assert common.read_summary(f"{lim}.summary.json")["states_total"] == 1 and gym4.served() == {"g0": 2}


def test_takeover_min_replay_ok_gate(tmp_path, monkeypatch, capsys):
    """Below --min-replay-ok (0.7, REF's default, F5) the stage writes TAKEOVER_FAIL as key and trailing line and
    exits 2, the failures counted by kind (obs_mismatch, turn_count_mismatch, run_transport:<error>; the dropped
    state's rows leave the fraction's denominator); at a lower gate the same fold passes without a new takeover;
    states without a transcript (a campaign v1 dump) are refused before any run, as is a --teacher-url carrying a
    path."""
    names = ("good", "bad", "cut", "lost")
    good, bad, cut, lost = (_state(div_turn=2, state_id=f"d:{i}:2", goal=g) for i, g in enumerate(names))
    cut = {**cut, "turns": cut["turns"][:2]}  # three policy turns in the transcript over two spans
    teacher = [(_act("t1", "take mug 2 from desk 1"), WON_OBS)]
    gym = FakeGym(
        {
            "good": [_reply(good, teacher, won=True)] * 2,
            "bad": [_reply(bad, teacher, obs={1: "a different desk"})] * 2,
            "lost": [httpx.ConnectError("refused"), TimeoutError()],
        }
    )
    monkeypatch.setattr(takeover, "post_json", gym.post_json)
    states_path = tmp_path / "states.jsonl"
    common.write_jsonl(str(states_path), [good, bad, cut, lost])
    out = tmp_path / "fold.jsonl"
    base = ["--env", "alfworld", "--states", str(states_path), "--teacher-url", TEACHER, "--out", str(out)]
    base += ["--samples", "2"]
    assert takeover.main(base) == 2
    summary_path = f"{out}.summary.json"
    summary = common.read_summary(summary_path)
    assert [summary[k] for k in ("n_takeovers", "replay_ok", "states_total")] == [8, 2, 4]
    # over the six replayed rows: the dropped state's two never ran
    assert summary["replay_ok_frac"] == pytest.approx(1 / 3)
    assert (summary["replayed"], summary["turn_count_mismatch"]) == (6, 1)
    counted = ("fail:obs_mismatch", "fail:turn_count_mismatch", "fail:run_transport:ConnectError")
    assert [summary[k] for k in counted] == [2, 2, 1] and summary["fail:run_transport:TimeoutError"] == 1
    assert summary["TAKEOVER_FAIL"] is True and "TAKEOVER_DONE" not in summary
    assert open(summary_path).read().endswith("}\nTAKEOVER_FAIL\n")
    assert capsys.readouterr().out.rstrip().endswith("TAKEOVER_FAIL")
    assert gym.served() == {"good": 2, "bad": 2, "lost": 2}  # the mismatched state never runs
    rows = {(r["state_id"], r["sample"]): r for r in common.read_jsonl(str(out))}
    assert len(rows) == 8 and rows[("d:1:2", 0)]["fail"] == "obs_mismatch@1" and rows[("d:1:2", 0)]["messages"]
    assert rows[("d:2:2", 1)] == {
        "state_id": "d:2:2",
        "goal": "cut",
        "split": "train",
        "div_turn": 2,
        "sample": 1,
        "replay_ok": False,
        "fail": "turn_count_mismatch",
    }
    assert {rows[("d:3:2", s)]["fail"] for s in (0, 1)} == {"run_transport:ConnectError", "run_transport:TimeoutError"}
    # the same fold at a lower gate: no new takeover, TAKEOVER_DONE
    assert takeover.main([*base, "--min-replay-ok", "0.25"]) == 0 and gym.served() == {"good": 2, "bad": 2, "lost": 2}
    summary = common.read_summary(summary_path)
    assert summary["TAKEOVER_DONE"] is True and summary["min_replay_ok"] == 0.25 and "TAKEOVER_FAIL" not in summary
    # a v1 dump's states (no transcript) are refused before any run, as is a --teacher-url carrying a path
    common.write_jsonl(str(tmp_path / "v1.jsonl"), [{**good, "prompt": None, "transcript": None}, good])
    with pytest.raises(SystemExit, match="1 of 2 states carry no transcript"):
        takeover.main([*base, "--states", str(tmp_path / "v1.jsonl"), "--out", str(tmp_path / "v1_fold.jsonl")])
    with pytest.raises(SystemExit, match="--teacher-url .*13141/v1: pass the server's base URL"):
        takeover.main([*base, "--teacher-url", TEACHER + "/v1", "--out", str(tmp_path / "x.jsonl")])
    assert gym.served() == {"good": 2, "bad": 2, "lost": 2}
    assert not (tmp_path / "v1_fold.jsonl").exists() and not (tmp_path / "x.jsonl").exists()


def test_takeover_gate_ignores_turn_count_mismatch(tmp_path, monkeypatch, capsys):
    """A turn_count_mismatch state never runs (the design's word is dropped): its K short rows are written, so the
    resume has every (state_id, sample), but they are no replays and leave the F5 gate: replay_ok_frac is over the
    rows that were replayed, and the summary counts the dropped states (turn_count_mismatch) and the replayed rows
    (replayed). A fold of one dropped state and one replayed state passes --min-replay-ok 1.0 with replay_ok_frac
    1.0; the rerun posts nothing; a fold of dropped states alone has no fraction and is refused (TAKEOVER_FAIL)."""
    good, cut = (_state(div_turn=2, state_id=f"d:{i}:2", goal=g) for i, g in enumerate(("good", "cut")))
    cut = {**cut, "turns": cut["turns"][:2]}  # three policy turns in the transcript over two spans
    teacher = [(_act("t1", "take mug 2 from desk 1"), WON_OBS)]
    gym = FakeGym({"good": [_reply(good, teacher, won=True)] * 2})
    monkeypatch.setattr(takeover, "post_json", gym.post_json)
    states_path = tmp_path / "states.jsonl"
    common.write_jsonl(str(states_path), [cut, good])
    out = tmp_path / "fold.jsonl"
    base = ["--env", "alfworld", "--states", str(states_path), "--teacher-url", TEACHER, "--out", str(out)]
    base += ["--samples", "2", "--min-replay-ok", "1.0"]
    assert takeover.main(base) == 0
    summary_path = f"{out}.summary.json"
    summary = common.read_summary(summary_path)
    keys = ("n_takeovers", "replayed", "replay_ok", "replay_ok_frac", "turn_count_mismatch")
    assert [summary[k] for k in keys] == [4, 2, 2, 1.0, 1] and summary["fail:turn_count_mismatch"] == 2
    assert (summary["states_total"], summary["TAKEOVER_DONE"]) == (2, True) and "TAKEOVER_FAIL" not in summary
    assert capsys.readouterr().out.rstrip().endswith("TAKEOVER_DONE")
    rows = {(r["state_id"], r["sample"]): r for r in common.read_jsonl(str(out))}
    assert sorted(rows) == [("d:0:2", 0), ("d:0:2", 1), ("d:1:2", 0), ("d:1:2", 1)] and gym.served() == {"good": 2}
    assert rows[("d:1:2", 0)]["fail"] == "turn_count_mismatch" and rows[("d:0:2", 1)]["replay_ok"] is True
    # the rerun has every pair: nothing posted, the same summary
    os.unlink(summary_path)
    gym2 = FakeGym({})
    monkeypatch.setattr(takeover, "post_json", gym2.post_json)
    assert takeover.main(base) == 0 and gym2.bodies == []
    assert [common.read_summary(summary_path)[k] for k in keys] == [4, 2, 2, 1.0, 1]
    # dropped states alone: nothing was replayed, no fraction, TAKEOVER_FAIL (nothing to select from; exit 2)
    only = tmp_path / "only.jsonl"
    common.write_jsonl(str(only), [cut])
    alone = tmp_path / "alone.jsonl"
    assert takeover.main([*base, "--states", str(only), "--out", str(alone)]) == 2 and gym2.bodies == []
    summary = common.read_summary(f"{alone}.summary.json")
    assert [summary[k] for k in ("n_takeovers", "replayed", "turn_count_mismatch")] == [2, 0, 1]
    assert summary["replay_ok_frac"] is None and summary["TAKEOVER_FAIL"] is True and "TAKEOVER_DONE" not in summary
    capsys.readouterr()


def test_takeover_refuses_gym_url_with_path(tmp_path, monkeypatch):
    """--gym-url is the agent server's base URL (the stage posts to <url>/run): http://h:8200/run would post to
    /run/run and every takeover would end run_transport:HTTPStatusError, so a path, a query or a fragment is refused
    at the flag before any state is read, as --teacher-url is (a trailing slash is dropped:
    test_takeover_resumes_after_partial_fold)."""
    gym = FakeGym({})
    monkeypatch.setattr(takeover, "post_json", gym.post_json)
    states_path = tmp_path / "states.jsonl"
    common.write_jsonl(str(states_path), [_state(div_turn=2)])
    out = tmp_path / "fold.jsonl"
    base = ["--env", "alfworld", "--states", str(states_path), "--teacher-url", TEACHER, "--out", str(out)]
    refusal = "pass the server's base URL .* appends /run"
    for bad in (GYM + "/run", GYM + "?x=1", GYM + "#f"):
        with pytest.raises(SystemExit, match=f"--gym-url {re.escape(bad)}: {refusal}"):
            takeover.main([*base, "--gym-url", bad])
    assert gym.bodies == [] and not out.exists() and not os.path.exists(f"{out}.summary.json")


async def test_takeover_body_drives_env_loop_agent(monkeypatch):
    """The stage's body is what env_loop_agent accepts (EnvLoopAgentRunRequest) and the agent's reply is what
    takeover_row reads: the prefix replayed through /step before the first policy call (the teacher's first call
    sees it with its observations), the teacher's turn and observation after it, replay_ok over the dump's
    observations, an env that answers a replayed turn differently read as obs_mismatch, a prefix the env ends
    early as prefix_ended (test_env_loop_agent's fakes drive the real agent)."""
    pytest.importorskip("nemo_gym", reason="nemo-gym not importable here (uv sync --group gym)")
    from test_env_loop_agent import (
        ACT_TOOL,
        POLICY_URL,
        STEP_FORMAT_ERROR,
        STEP_MID,
        STEP_WON,
        FakeEnvServer,
        _act_msg,
        _fake_policy,
        _load_module,
        _make_agent,
    )

    mod = _load_module()
    prompt = [{"role": "system", "content": "sys prompt"}]
    prefix = [_act_msg("go north", content="North first.", start=1), _act_msg("open box", start=2)]
    episode = [
        {"role": "user", "content": "obs0"},
        prefix[0],
        {"role": "tool", "tool_call_id": "tc1", "content": "obs1"},
        prefix[1],
        {"role": "tool", "tool_call_id": "tc2", "content": "obs1"},
        _act_msg("look", start=3),
        {"role": "tool", "tool_call_id": "tc3", "content": "obs1"},
    ]
    # the recipe rows' tool (strict, as prepare_data declares it) and the fakes' policy URL
    state = _state(div_turn=2, prompt=prompt, episode=episode, max_episode_steps=5, tools=[ACT_TOOL])
    kwargs = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512}
    body = takeover.run_body(state, POLICY_URL.removesuffix("/v1"), "teacher", kwargs)
    request = mod.EnvLoopAgentRunRequest.model_validate(body)
    assert request.scripted_prefix == prefix and request.policy_base_url == POLICY_URL
    assert "scripted_prefix" not in request.model_dump(exclude=mod._AGENT_ONLY_FIELDS)
    env = FakeEnvServer([STEP_MID, STEP_MID, STEP_WON])
    policy = [_act_msg("take mug", content="Taking it.", start=9)]
    calls = _fake_policy(monkeypatch, mod, policy)
    result = await _make_agent(mod, env).run(request=None, body=request)
    assert env.paths() == ["/reset", "/step", "/step", "/step"] and len(calls) == 1
    first = calls[0]["json"]
    assert (first["model"], first["max_tokens"], first["temperature"], first["top_p"]) == ("teacher", 512, 1.0, 1.0)
    assert calls[0]["headers"]["Authorization"] == "Bearer EMPTY" and first["tools"][0]["function"]["name"] == "act"
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    row = takeover.takeover_row(state, 0, result, ALF)
    assert (row["replay_ok"], row["won"], row["done"], row["truncated"], row["turns_total"]) == (
        True,
        True,
        True,
        False,
        3.0,
    )
    assert row["takeover_steps"] == [
        {
            "turn": 2,
            "delib": "Taking it.",
            "command": "take mug",
            "call": {"name": "act", "arguments": json.dumps({"command": "take mug"})},
            "format_ok": True,
            "obs_before": "obs1",
        }
    ]
    assert row["last_observation"] == STEP_WON["observation"]
    assert row["terminal_key"] == common.fingerprint(STEP_WON["observation"])
    assert row["messages"] == [
        *prompt,
        {"role": "user", "content": "obs0"},
        prefix[0],
        {"role": "tool", "tool_call_id": "tc1", "content": "obs1"},
        prefix[1],
        {"role": "tool", "tool_call_id": "tc2", "content": "obs1"},
        policy[0],
        {"role": "tool", "tool_call_id": "tc9", "content": STEP_WON["observation"]},
    ]
    # the env answers the second replayed turn with a format fault's hint: not the dump's observation
    env = FakeEnvServer([STEP_MID, STEP_FORMAT_ERROR, STEP_WON])
    _fake_policy(monkeypatch, mod, policy)
    row = takeover.takeover_row(state, 1, await _make_agent(mod, env).run(request=None, body=request), ALF)
    assert (row["replay_ok"], row["fail"]) == (False, "obs_mismatch@2")
    assert row["messages"][5]["content"].startswith("Nothing happens.")
    # the env ends the episode on the first replayed turn: the policy never acts
    env = FakeEnvServer([STEP_WON])
    _fake_policy(monkeypatch, mod, [])
    row = takeover.takeover_row(state, 2, await _make_agent(mod, env).run(request=None, body=request), ALF)
    assert (row["replay_ok"], row["fail"]) == (False, "prefix_ended@1")


# ---------------------------------------------------------------- stage J: judge.py (C12)

JUDGE = "http://127.0.0.1:13143"
WS_INSTRUCTION = (
    "Find me straight leg men's jeans with color: coal grey, and size: 36w x 32l, and price lower than 80.00 dollars"
)
WS_DESC_PAGE = (
    "You are now at step 3 and your current observation is: Instruction:\nFind me x\n"
    "[button] Back to Search [button_]\nMachine washable cotton.\n"
    "Your admissible actions of the current situation are:\n[\n]."
)
_ASYNC_CLIENT = httpx.AsyncClient


def _ws_page(instruction=WS_INSTRUCTION, *, title="J. Crew Sutton Straight-Fit Chino", price="$59.5"):
    """A text_rich item page at Buy Now as webshop_env renders it (_format.render_page: the step banner, the
    page with [button] / [clicked button] markup, the admissible-action list); REF tests/test_r2s.py:527-535."""
    return (
        "You are now at step 5 and your current observation is: You have clicked coal grey.\n"
        f"You have clicked 36w x 32l.\nInstruction:\n{instruction}\n"
        "[button] Back to Search [button_]\n[button] < Prev [button_]\nsize\n  [button] 34w x 32l [button_]\n"
        "  [clicked button] 36w x 32l [clicked button_]\ncolor\n  [clicked button] coal grey [clicked button_]\n"
        f"  [button] navy [button_]\n{title}\nPrice: {price}\nRating: N.A.\n[button] Buy Now [button_].\n"
        "Your admissible actions of the current situation are:\n[\nclick[buy now]\nclick[navy]\n]."
    )


WS_PAGE = _ws_page()


def _ws_step(turn, command, obs_before, format_ok=True):
    """A teacher step of a WebShop takeover (takeover.takeover_steps' keys)."""
    call = None if command is None else {"name": "act", "arguments": json.dumps({"command": command})}
    return {
        "turn": turn,
        "delib": "",
        "command": command,
        "call": call,
        "format_ok": format_ok,
        "obs_before": obs_before,
    }


def _ws_tau(sample, asin, page=WS_PAGE, *, with_desc=False, state_id="s0", steps=None, **row):
    """A WebShop stage B row whose teacher opened the item (its description too under with_desc) and bought it;
    ``steps`` replaces the teacher steps, ``row`` lays over the flags (replay_ok, done, won)."""
    if steps is None:
        steps = (
            [_ws_step(1, "click[description]", page), _ws_step(2, "click[< prev]", WS_DESC_PAGE)] if with_desc else []
        )
        steps = [*steps, _ws_step(4, "click[buy now]", page)]
    return {
        "state_id": state_id,
        "sample": sample,
        "replay_ok": True,
        "won": True,
        "done": True,
        "truncated": False,
        "takeover_steps": steps,
        "last_observation": "Thank you for shopping with us!",
        "terminal_key": asin,
        **row,
    }


class FakeJudge:
    """POST <judge>/v1/chat/completions stood in through httpx.MockTransport: the reply comes from ``script``
    keyed by the user message's instruction line (a list consumed in request order, or one reply for every
    request; a content string, a (reasoning_content, content) pair as a server with a --reasoning-parser answers,
    or an HTTP status to answer with); ``stop_file`` is written as the first request
    arrives (the preempt sentinel rising mid-stage); every request's URL and JSON body are recorded."""

    def __init__(self, script, stop_file=None):
        self.script = {k: list(v) if isinstance(v, list) else [v] * 3 for k, v in script.items()}
        self.stop_file = stop_file
        self.requests = []

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append((str(request.url), body))
        if self.stop_file:
            open(self.stop_file, "w").close()
        key = body["messages"][1]["content"].split("\n", 1)[0].removeprefix("Instruction: ")
        reply = self.script[key].pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "overloaded"})
        message = {"role": "assistant", "content": reply}
        if isinstance(reply, tuple):
            message = {"role": "assistant", "reasoning_content": reply[0], "content": reply[1]}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    def install(self, monkeypatch):
        transport = httpx.MockTransport(self.handler)
        monkeypatch.setattr(httpx, "AsyncClient", functools.partial(_ASYNC_CLIENT, transport=transport))
        monkeypatch.setattr(judge, "RETRY_BACKOFF_S", 0.0)

    def served(self):
        return collections.Counter(b["messages"][1]["content"].split("\n", 1)[0] for _, b in self.requests)


def test_webshop_purchase_and_match_messages():
    """REF judge.py:72-94, 198-249 byte for byte over a text_rich Buy Now page: the rubric and the parsers pinned,
    the instruction (the Back-to-Search form, then the bare form, whitespace folded), the page without the step
    banner and the admissible-action list (3000 chars), the purchase (asin = the vote key, the [clicked button]
    options, title and price, the instruction read from an earlier page when the buy page lacks it, the
    description/features pages the takeover opened: at most two, 1500 chars each), the messages (MATCH_SYSTEM
    and the user text, no prompt-induced reasoning), and None without a Buy Now step."""
    assert (
        hashlib.sha1(judge.MATCH_SYSTEM.encode()).hexdigest()[:12] == "da25e61205d1" and len(judge.MATCH_SYSTEM) == 577
    )
    assert judge.MATCH_SYSTEM.endswith("Reply with exactly one word: MATCH or MISMATCH.")
    assert judge.WS_INSTRUCTION_RE.pattern == r"Instruction:\s*\n(.*?)\n\s*\[button\] Back to Search"
    assert judge.WS_STEP_PREFIX_RE.pattern == r"^You are now at step \d+ and your current observation is:\s*"
    assert judge.WS_ADMISSIBLE_RE.pattern == r"\n?Your admissible actions of the current situation are:.*$"
    assert judge.WS_PAGE_CHARS == 3000
    # the instruction, from the item page and from the opening page (no Back to Search button there)
    assert judge.webshop_instruction(WS_PAGE) == WS_INSTRUCTION
    assert (
        judge.webshop_instruction("WebShop\nInstruction:\n  a   red \n mug \n[button] Search [button_]\n")
        == "a red mug"
    )
    assert judge.webshop_instruction("Instruction: a mug") == "a mug"
    assert judge.webshop_instruction("no instruction here") == "" and judge.webshop_instruction(None) == ""
    # the page: banner and action list gone, the rest verbatim, capped
    page = judge.webshop_page(WS_PAGE)
    assert page.startswith("You have clicked coal grey.\nYou have clicked 36w x 32l.\nInstruction:\n")
    assert page.endswith("Rating: N.A.\n[button] Buy Now [button_].")
    assert "You are now at step" not in page and "admissible actions" not in page and "click[buy now]" not in page
    assert judge.webshop_page(None) == "" and judge.webshop_page("x" * 4000) == "x" * 3000
    # the purchase of a takeover that opened the description first
    pc = judge.webshop_purchase(_ws_tau(0, "b07gyww3ny", with_desc=True))
    assert pc == {
        "instruction": WS_INSTRUCTION,
        "page": page,
        "extras": [
            ("description", "Instruction:\nFind me x\n[button] Back to Search [button_]\nMachine washable cotton.")
        ],
        "selected": ["36w x 32l", "coal grey"],
        "title": "J. Crew Sutton Straight-Fit Chino",
        "price": "$59.5",
        "asin": "b07gyww3ny",
    }
    plain = judge.webshop_purchase(_ws_tau(1, "b07gyww3ny"))
    assert (plain["extras"], plain["selected"], plain["title"], plain["price"]) == (
        [],
        pc["selected"],
        pc["title"],
        "$59.5",
    )
    nothing_selected = WS_PAGE.replace("[clicked button] coal grey [clicked button_]", "[button] coal grey [button_]")
    assert judge.webshop_purchase(_ws_tau(2, "b07gyww3ny", page=nothing_selected))["selected"] == ["36w x 32l"]
    # the last Buy Now counts, the command is matched case-insensitively, a format-faulted step has no command
    steps = [
        _ws_step(1, "click[buy now]", _ws_page("first try")),
        _ws_step(2, None, "Nothing happens.", format_ok=False),
        _ws_step(3, "click[Buy Now]", WS_PAGE),
    ]
    assert judge.webshop_purchase(_ws_tau(3, "b07gyww3ny", steps=steps))["instruction"] == WS_INSTRUCTION
    # three pages opened: the first two ride, each cut at 1500 chars and named after the button; a buy page
    # without the instruction block takes the instruction from the earliest page that has one
    long_desc = WS_DESC_PAGE.replace("Machine washable cotton.", "Machine washable cotton. " * 100)
    steps = [
        _ws_step(1, "click[features]", WS_PAGE),
        _ws_step(2, "click[< prev]", long_desc),
        _ws_step(3, "click[description]", WS_PAGE),
        _ws_step(4, "click[< prev]", WS_DESC_PAGE),
        _ws_step(5, "click[description]", WS_PAGE),
        _ws_step(6, "click[< prev]", WS_DESC_PAGE),
        _ws_step(7, "click[buy now]", WS_PAGE.replace(f"Instruction:\n{WS_INSTRUCTION}\n", "")),
    ]
    pc3 = judge.webshop_purchase(_ws_tau(4, "b07gyww3ny", steps=steps))
    assert [k for k, _ in pc3["extras"]] == ["features", "description"] and len(pc3["extras"][0][1]) == 1500
    assert pc3["instruction"] == WS_INSTRUCTION and "Instruction:" not in pc3["page"]
    assert judge.webshop_purchase(_ws_tau(5, "b07gyww3ny", steps=[_ws_step(1, "click[navy]", WS_PAGE)])) is None
    assert judge.webshop_purchase({"takeover_steps": []}) is None and judge.webshop_purchase({}) is None
    # the messages: REF's user text to the byte, the description page appended under its button's name
    msgs = judge.match_messages(pc)
    assert msgs[0] == {"role": "system", "content": judge.MATCH_SYSTEM}
    assert msgs[1] == {
        "role": "user",
        "content": (
            f"Instruction: {WS_INSTRUCTION}\n\nProduct page at Buy Now:\n{page}"
            f"\n\nDescription page the agent opened:\n{pc['extras'][0][1]}\n\nMATCH or MISMATCH?"
        ),
    }
    assert judge.match_messages(plain)[1]["content"].endswith(f"{page}\n\nMATCH or MISMATCH?")
    two = judge.match_messages(pc3)[1]["content"]
    assert "\n\nFeatures page the agent opened:\n" in two and two.index("Features page") < two.index(
        "Description page"
    )
    assert "[clicked button] coal grey [clicked button_]" in two and "think" not in two


def test_parse_verdict_last_word():
    """The last verdict word of message.content decides (a thinking judge may reason first; under the server's
    --reasoning-parser the reasoning is not in the content at all): case-insensitive, whole words only, None
    without one; VERDICT_WORDS and the pattern are REF judge.py:87-90, 190-195."""
    assert judge.parse_verdict("MATCH") is True and judge.parse_verdict("MISMATCH") is False
    assert judge.parse_verdict("Both options are selected and $59.5 < 80. Verdict: match.") is True
    assert judge.parse_verdict("It would be a MATCH, but the size is wrong: MISMATCH") is False
    assert judge.parse_verdict("MISMATCH? No, the range starts below the limit. MATCH") is True
    assert judge.parse_verdict("mismatched sizes") is None and judge.parse_verdict("matches") is None
    assert judge.parse_verdict("I cannot tell") is None and judge.parse_verdict("") is None
    assert judge.parse_verdict(None) is None
    assert judge.parse_verdict("I think... SUPPORTED") is True and judge.parse_verdict("USELESS") is False
    assert judge.VERDICT_WORDS == {
        "SUPPORTED": True,
        "UNSUPPORTED": False,
        "USEFUL": True,
        "USELESS": False,
        "MATCH": True,
        "MISMATCH": False,
        "COMPLETED": True,
        "INCOMPLETE": False,
    }


def test_judge_reads_the_reasoning_before_the_content(tmp_path, monkeypatch):
    """The verdict is the last verdict word of reasoning_content + content, the text REF's regex saw on the
    campaign's server (no reasoning parser: the content carried the <think> reasoning, and 4,768 of the campaign's
    6,232 verdicts were the last verdict word of a reasoning cut at the 2048-token budget). Behind a
    --reasoning-parser the cut reasoning arrives as reasoning_content with an empty content: still that verdict,
    not an unparsed call; a closed reasoning's content word wins over the reasoning's; a cut reasoning without a
    verdict word is unparsed; match_text is the tail of the joined text."""
    rows = [
        _ws_tau(0, "b000000001", _ws_page("A")),  # cut at max_tokens, no content: the reasoning's last word
        _ws_tau(1, "b000000001", _ws_page("B")),  # a closed reasoning that argued MISMATCH, then MATCH
        _ws_tau(2, "b000000002", _ws_page("C")),  # cut before any verdict word
    ]
    cut = "8. Size: x-large (Selected: x-large) -> OK, a MATCH so far. 9. Color: navy, not coal grey -> MISMATCH. 10."
    script = {
        "A": (cut, ""),
        "B": ("MISMATCH? No: the range starts below the limit, so MISMATCH is wrong.", "\n\nMATCH"),
        "C": ("I cannot tell from this page; the", None),
    }
    fold = tmp_path / "fold.jsonl"
    common.write_jsonl(str(fold), rows)
    out = tmp_path / "judge.jsonl"
    fake = FakeJudge(script)
    fake.install(monkeypatch)
    base = ["--env", "webshop", "--fold", str(fold), "--judge-url", JUDGE, "--judge-model", "judge", "--out", str(out)]
    assert judge.main(base) == 0
    got = {r["sample"]: r for r in common.read_jsonl(str(out))}
    assert (got[0]["match_ok"], got[0]["n_unparsed"], got[0]["match_text"]) == (False, 0, cut[-200:])
    assert (got[1]["match_ok"], got[1]["n_unparsed"]) == (True, 0)
    assert got[1]["match_text"].endswith("so MISMATCH is wrong.\n\nMATCH")
    assert (got[2]["match_ok"], got[2]["n_unparsed"]) == (False, 1)
    assert got[2]["match_text"] == "I cannot tell from this page; the"
    summary = common.read_summary(f"{out}.summary.json")
    assert (summary["takeovers_judged"], summary["match_ok"], summary["unparsed_calls"]) == (3, 1, 1)


def test_judge_rows_and_summary(tmp_path, monkeypatch, capsys):
    """python -m miles_agent.r2s.judge over a fold: one chat completion per eligible takeover (select.eligible:
    replayed, done, an ASIN, a step; a format-faulted step admitted under the env's allow_invalid) at the judge's
    /v1/chat/completions with REF's payload (temperature 0, max_tokens 2048, enable_thinking true, the served
    name, match_messages of the purchase), three attempts per verdict, rows in REF's schema (match_ok, the reply's
    tail, n_unparsed; "no buy step" without a Buy Now; an error row after the third failure), the summary with
    JUDGE_DONE as key and trailing line, the DONE skip, --force, --no-thinking (16 tokens: the budget sent, which
    the summary reports), --limit, an --env-config override, the stop file before and between verdicts (the
    verdict in flight stays in the file), a --judge-url carrying a path, a non-WebShop env, and select's judge path
    reading the rows."""
    rows = [
        _ws_tau(0, "b000000001", _ws_page("A")),  # MATCH
        _ws_tau(1, "b000000001", _ws_page("B")),  # reasons, then MISMATCH
        _ws_tau(2, "b000000002", _ws_page("C")),  # no verdict word
        _ws_tau(3, "b000000002", _ws_page("D"), with_desc=True),  # a transient 500, then MATCH
        _ws_tau(4, "b000000003", _ws_page("E")),  # 500 three times
        _ws_tau(5, "b000000001", steps=[_ws_step(1, "click[navy]", _ws_page("F"))]),  # done, never bought
        _ws_tau(6, "b000000001", _ws_page("G"), replay_ok=False),  # never replayed
        _ws_tau(7, None, _ws_page("H"), done=False),  # ran out of turns
        _ws_tau(
            8,
            "b000000001",
            steps=[
                _ws_step(1, None, "Nothing happens.", format_ok=False),
                _ws_step(2, "click[buy now]", _ws_page("I")),
            ],
        ),
    ]
    script = {
        "A": "MATCH",
        "B": "The size is selected but the color is navy, not coal grey. MISMATCH",
        "C": "I cannot tell from this page.",
        "D": [500, "MATCH"],
        "E": [500, 500, 500],
        "I": "MATCH",
    }
    fold = tmp_path / "fold.jsonl"
    common.write_jsonl(str(fold), rows)
    out = tmp_path / "judge.jsonl"
    summary_path = f"{out}.summary.json"
    fake = FakeJudge(script)
    fake.install(monkeypatch)
    base = ["--env", "webshop", "--fold", str(fold), "--judge-url", JUDGE, "--judge-model", "judge", "--out", str(out)]
    assert judge.main(base) == 0
    assert fake.served() == {
        "Instruction: A": 1,
        "Instruction: B": 1,
        "Instruction: C": 1,
        "Instruction: D": 2,
        "Instruction: E": 3,
        "Instruction: I": 1,
    }
    for url, body in fake.requests:  # REF's payload (judge.py:572-576), every time
        assert url == JUDGE + "/v1/chat/completions"
        assert (body["model"], body["temperature"], body["max_tokens"]) == ("judge", 0.0, 2048)
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert fake.requests[0][1]["messages"] == judge.match_messages(judge.webshop_purchase(rows[0]))
    assert fake.requests[3][1]["messages"] == judge.match_messages(judge.webshop_purchase(rows[3]))
    got = {r["sample"]: r for r in common.read_jsonl(str(out))}  # appended per verdict, in completion order
    assert sorted(got) == [0, 1, 2, 3, 4, 5, 8] and all(r["state_id"] == "s0" for r in got.values())
    full = ("state_id", "sample", "match_ok", "match_text", "n_unparsed", "asin", "selected")
    assert all(tuple(r) == full for r in got.values() if r["sample"] != 5)
    assert got[0] == {
        "state_id": "s0",
        "sample": 0,
        "match_ok": True,
        "match_text": "MATCH",
        "n_unparsed": 0,
        "asin": "b000000001",
        "selected": ["36w x 32l", "coal grey"],
    }
    assert (got[1]["match_ok"], got[1]["match_text"], got[1]["n_unparsed"]) == (False, script["B"], 0)
    assert (got[2]["match_ok"], got[2]["match_text"], got[2]["n_unparsed"]) == (False, script["C"], 1)
    assert (got[3]["match_ok"], got[3]["asin"], got[3]["n_unparsed"]) == (True, "b000000002", 0)
    assert (got[4]["match_ok"], got[4]["n_unparsed"]) == (False, 1)
    assert got[4]["match_text"].startswith("error: Server error '500 Internal Server Error'")
    assert got[5] == {"state_id": "s0", "sample": 5, "match_ok": False, "match_text": "no buy step", "n_unparsed": 1}
    assert (got[8]["match_ok"], got[8]["asin"]) == (True, "b000000001")
    summary = common.read_summary(summary_path)
    counted = ("mode", "takeovers_judged", "match_ok", "unparsed_calls", "thinking", "judge_model", "max_tokens")
    assert [summary[k] for k in counted] == ["match", 7, 3, 3, True, "judge", 2048]
    assert summary["fold_takeovers"] == 9 and summary["config"] == {**dataclasses.asdict(WS), "goal_keys": ["goal_id"]}
    assert summary["JUDGE_DONE"] is True and open(summary_path).read().endswith("}\nJUDGE_DONE\n")
    assert capsys.readouterr().out.rstrip().endswith("JUDGE_DONE")
    # select's judge path over these rows: the approved takeovers (0, 3, 8) vote by ASIN, the modal item wins
    # and its shortest takeover is tau
    tau_out = tmp_path / "tau.jsonl"
    assert select.main(["--env", "webshop", "--fold", str(fold), "--judge-file", str(out), "--out", str(tau_out)]) == 0
    assert common.read_jsonl(str(tau_out)) == [{"state_id": "s0", "sample": 0, "group_size": 9, "n_eligible": 3}]
    assert common.read_summary(f"{tau_out}.summary.json")["judge_missing"] == 0  # every eligible takeover was judged
    # done: nothing is asked again; --force with --no-thinking and --limit: a verdict-only budget, the first two
    fake2 = FakeJudge(script)
    fake2.install(monkeypatch)
    assert judge.main(base) == 0 and fake2.requests == []
    assert "JUDGE_DONE already in" in capsys.readouterr().out
    assert judge.main([*base, "--force", "--no-thinking", "--limit", "2", "--max-tokens", "999"]) == 0
    assert fake2.served() == {"Instruction: A": 1, "Instruction: B": 1}
    assert all(
        (b["max_tokens"], b["chat_template_kwargs"]) == (16, {"enable_thinking": False}) for _, b in fake2.requests
    )
    assert sorted(r["sample"] for r in common.read_jsonl(str(out))) == [0, 1]
    summary = common.read_summary(summary_path)  # max_tokens is the budget sent (VERDICT_TOKENS), not the flag's 999
    assert [summary[k] for k in ("takeovers_judged", "match_ok", "thinking", "max_tokens", "fold_takeovers")] == [
        2,
        1,
        False,
        16,
        9,
    ]
    # the env's allow_invalid off through --env-config: the format-faulted takeover is not judged
    table = tmp_path / "env.json"
    table.write_text(json.dumps({"webshop": {"allow_invalid": False}}))
    fake3 = FakeJudge(script)
    fake3.install(monkeypatch)
    assert judge.main([*base, "--force", "--env-config", str(table)]) == 0
    assert "Instruction: I" not in fake3.served() and sum(fake3.served().values()) == 8
    summary = common.read_summary(summary_path)
    assert (summary["takeovers_judged"], summary["config"]["allow_invalid"]) == (6, False)
    # the stop file: present before the first verdict -> exit 0, no rows, no summary; raised between verdicts ->
    # the verdict in flight finishes and its row stays in the file, no summary
    stop = tmp_path / "STOP"
    stop.write_text("")
    stopped = tmp_path / "stopped.jsonl"
    argv = [*base, "--out", str(stopped), "--stop-file", str(stop)]
    fake4 = FakeJudge(script)
    fake4.install(monkeypatch)
    assert judge.main(argv) == 0 and fake4.requests == [] and not stopped.exists()
    assert "present before judging; exiting without JUDGE_DONE" in capsys.readouterr().out
    os.unlink(stop)
    fake5 = FakeJudge(script, stop_file=str(stop))
    fake5.install(monkeypatch)
    assert judge.main([*argv, "--concurrency", "1"]) == 0
    assert fake5.served() == {"Instruction: A": 1} and [r["sample"] for r in common.read_jsonl(str(stopped))] == [0]
    assert not os.path.exists(f"{stopped}.summary.json")
    assert "6 takeovers not judged; exiting without JUDGE_DONE" in capsys.readouterr().out
    # a --judge-url that carries a path is refused at the flag, a non-WebShop env before any request
    fake6 = FakeJudge(script)
    fake6.install(monkeypatch)
    with pytest.raises(SystemExit, match="--judge-url .*13143/v1: pass the server's base URL .* /v1/chat/completions"):
        judge.main([*base, "--judge-url", JUDGE + "/v1", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(
        SystemExit, match="--env alfworld groups takeovers by terminal: the match judge reads purchases"
    ):
        judge.main([*base, "--env", "alfworld", "--out", str(tmp_path / "x.jsonl")])
    assert fake6.requests == [] and not (tmp_path / "x.jsonl").exists()
    assert judge.main([*base, "--force", "--judge-url", JUDGE + "/"]) == 0  # a trailing slash is not doubled
    assert {u for u, _ in fake6.requests} == {JUDGE + "/v1/chat/completions"}


def test_judge_appends_rows_and_resumes(tmp_path, monkeypatch, capsys):
    """Rows are appended per verdict (completion order, as stages B and D append theirs) and a rerun skips the
    (state_id, sample) pairs present: two rows pre-written -> only the rest are asked, the file holds all and
    takeovers_judged / match_ok count all (the summary reads the file); --limit applies before the skip; a stop file
    raised after the first verdict leaves that row in the file (no marker, exit 0) and the next run asks for the
    rest only; --force redoes the file."""
    rows = [_ws_tau(s, "b000000001", _ws_page(chr(65 + s))) for s in range(5)]  # instructions A..E
    script = {k: "MATCH" if k in "ACE" else "MISMATCH" for k in "ABCDE"}
    fold = tmp_path / "fold.jsonl"
    common.write_jsonl(str(fold), rows)
    out = tmp_path / "judge.jsonl"
    summary_path = f"{out}.summary.json"
    full = ("state_id", "sample", "match_ok", "match_text", "n_unparsed", "asin", "selected")
    pre = [
        dict(zip(full, ("s0", 0, True, "MATCH", 0, "b000000001", ["36w x 32l", "coal grey"]), strict=True)),
        dict(zip(full, ("s0", 1, False, "MISMATCH", 0, "b000000001", ["36w x 32l", "coal grey"]), strict=True)),
    ]
    common.write_jsonl(str(out), pre)
    fake = FakeJudge(script)
    fake.install(monkeypatch)
    base = ["--env", "webshop", "--fold", str(fold), "--judge-url", JUDGE, "--judge-model", "judge", "--out", str(out)]
    assert judge.main(base) == 0
    assert fake.served() == {"Instruction: C": 1, "Instruction: D": 1, "Instruction: E": 1}
    got = common.read_jsonl(str(out))
    assert got[:2] == pre and sorted(r["sample"] for r in got) == [0, 1, 2, 3, 4]
    assert {r["sample"]: r["match_ok"] for r in got} == {0: True, 1: False, 2: True, 3: False, 4: True}
    summary = common.read_summary(summary_path)
    assert [summary[k] for k in ("takeovers_judged", "match_ok", "unparsed_calls", "max_tokens")] == [5, 3, 0, 2048]
    assert summary["JUDGE_DONE"] is True
    # --limit before the skip: the first two takeovers are in the file, nothing is asked, the file's rows count
    os.unlink(summary_path)
    fake2 = FakeJudge(script)
    fake2.install(monkeypatch)
    assert judge.main([*base, "--limit", "2"]) == 0 and fake2.requests == []
    assert common.read_summary(summary_path)["takeovers_judged"] == 5
    # a stop file raised by the first verdict: that row is in the file, no summary; the next run asks the rest only
    stop = tmp_path / "STOP"
    stopped = tmp_path / "stopped.jsonl"
    argv = [*base, "--out", str(stopped), "--stop-file", str(stop), "--concurrency", "1"]
    fake3 = FakeJudge(script, stop_file=str(stop))
    fake3.install(monkeypatch)
    assert judge.main(argv) == 0 and fake3.served() == {"Instruction: A": 1}
    assert [r["sample"] for r in common.read_jsonl(str(stopped))] == [0]
    assert not os.path.exists(f"{stopped}.summary.json")
    assert "4 takeovers not judged; exiting without JUDGE_DONE" in capsys.readouterr().out
    os.unlink(stop)
    fake4 = FakeJudge(script)
    fake4.install(monkeypatch)
    assert judge.main(argv) == 0
    assert fake4.served() == {f"Instruction: {k}": 1 for k in "BCDE"}
    assert sorted(r["sample"] for r in common.read_jsonl(str(stopped))) == [0, 1, 2, 3, 4]
    assert common.read_summary(f"{stopped}.summary.json")["takeovers_judged"] == 5
    # --force: the file is redone from scratch
    fake5 = FakeJudge(script)
    fake5.install(monkeypatch)
    assert judge.main([*argv, "--force"]) == 0
    assert fake5.served() == {f"Instruction: {k}": 1 for k in "ABCDE"}
    assert sorted(r["sample"] for r in common.read_jsonl(str(stopped))) == [0, 1, 2, 3, 4]
    capsys.readouterr()


# ---------------------------------------------------------------- stage D: label.py (C13)

Q35_KWARGS = '{"enable_thinking": false}'  # the Qwen3.5 rows' launch (MODEL_FAMILIES "Qwen3.5-")
SHORT_HISTORY = [{"role": "system", "content": "sys"}, {"role": "user", "content": OBS0}]


def _campaign_surgery(tokenizer, prompt_tokens, hint):
    """REF label.py:47, 112-128 re-implemented as the oracle of the splice: decode, cut at the last generation
    prompt, re-encode head and tail (None unless they re-tokenize to the prompt), the hint as a ChatML user
    block. The literals live in this oracle only; the port derives both blocks from the template."""
    text = tokenizer.decode(prompt_tokens)
    cut = text.rfind("<|im_start|>assistant\n")
    if cut < 0:
        return None
    head_ids = tokenizer.encode(text[:cut], add_special_tokens=False)
    tail_ids = tokenizer.encode(text[cut:], add_special_tokens=False)
    if head_ids + tail_ids != list(prompt_tokens):
        return None
    return head_ids + tokenizer.encode(f"<|im_start|>user\n{hint}<|im_end|>\n", add_special_tokens=False) + tail_ids


def _prompt_ids(tokenizer, render, history=SHORT_HISTORY, tools=(ACT,), generation_prompt=True):
    """What the engines tokenized for the first policy call: the messages, the tools, the generation prompt."""
    return tokenizer.encode(render(history, list(tools), add_generation_prompt=generation_prompt))


def _token_state(prompt_ids, *, div_turn=1, state_id="d:0:1", goal="g1", n_turns=3, trailing_obs=0, **fields):
    """A stage A row for stage D: ``prompt_ids`` (a real render of the family's template) followed by _episode's
    turns in the unit vocabulary and ``trailing_obs`` observation tokens after the last one; REF's state keys,
    the row keys and the v2 fields (no messages: the splice needs none; ``fields`` lays over)."""
    ep = _episode(n_turns=n_turns)
    response = [*ep.tokens[ep.prompt_length :], *([3] * trailing_obs)]
    return {
        "state_id": state_id,
        "episode_key": state_id.rsplit(":", 1)[0],
        "goal": goal,
        "split": "train",
        "div_turn": div_turn,
        "div_score": 0.5,
        "div_rows": [],
        "tokens": [*prompt_ids, *response],
        "response_length": len(response),
        "loss_mask": [*ep.loss_mask, *([0] * trailing_obs)],
        "turns": [[t.start, t.end, t.delib_end] for t in ep.turns],
        "metadata": {"game_file": goal, "split": "train", "max_episode_steps": 30, "tools": [ACT_FLAT]},
        "reward": 0.0,
        "status": "completed",
        "prompt": None,
        "transcript": None,
        "tito_model": "qwen35",
        "chat_template_kwargs": {"enable_thinking": False},
        **fields,
    }


def _transformers_stub(monkeypatch, tokenizer):
    """transformers.AutoTokenizer stood in: every from_pretrained path recorded, ``tokenizer`` returned."""
    loads = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code=False):
            loads.append(path)
            return tokenizer

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return loads


class FakePrefill:
    """teacher_scoring.prefill stood in: the /generate endpoint asserted, every call recorded, and top-k entries
    that name the scored position ([lp, the token there], [lp, the position], fillers), so a label's tgt_ids
    prove the p + offset read; a stream starting with 999 fails as the teacher would, one starting with 998
    comes back without top entries; ``stop_file`` is written as the first call arrives."""

    def __init__(self, stop_file=None):
        self.calls = []
        self.stop_file = stop_file

    async def __call__(self, url, ids, top_k=0, **kw):
        assert url == TEACHER + "/generate", url
        self.calls.append((tuple(ids), top_k, kw))
        if self.stop_file:
            open(self.stop_file, "w").close()
        await asyncio.sleep(0)
        if ids[0] == 999:
            raise teacher_scoring.TeacherScoringError(f"{url} down")
        n = len(ids)
        if ids[0] == 998 or top_k == 0:
            return {"lps": [None] + [-0.5] * (n - 1), "top": [None] * n}
        fill = [[-2.0 - j, 9000 + j, ""] for j in range(max(0, top_k - 2))]
        top = [None] + [[[-0.5, ids[p], ""], [-1.5, p, ""]][:top_k] + fill for p in range(1, n)]
        return {"lps": [None] + [-0.5] * (n - 1), "top": top}


def test_ref_text_bytes():
    """REF label.py:41-60 verbatim (F4): the header and footer bytes, a step as "{delib}\\nAction: {command}"
    ("Action: ..." alone without deliberation), a step without a command (a format-faulted teacher turn) as its
    stripped deliberation or "(no action)", one newline between steps; the hint has no outer whitespace, so it
    splices as a rendered user message."""
    header = (
        "[Reference walkthrough] The following actions were taken by an expert and successfully completed this "
        "exact task. Use it only as background knowledge when reading the session below; the session is a "
        "separate, independent attempt.\n"
    )
    footer = "\n[End of reference walkthrough]"
    assert (label.REF_HEADER, label.REF_FOOTER) == (header, footer)
    tau = {
        "takeover_steps": [
            {"turn": 2, "delib": " I check the desk. ", "command": "go to desk 1", "format_ok": True},
            {"turn": 3, "delib": "", "command": "take mug 2 from desk 1", "format_ok": True},
            {"turn": 4, "delib": "done?", "command": None, "call": None, "format_ok": False},  # invalid step
            {"turn": 5, "delib": "", "command": None, "call": None, "format_ok": False},
            {"turn": 6, "delib": "I put it down.", "command": "put mug 2 in/on desk 1", "format_ok": True},
        ]
    }
    text = label.ref_text_of(tau)
    assert text == (
        header
        + "I check the desk.\nAction: go to desk 1\nAction: take mug 2 from desk 1\ndone?\n(no action)\n"
        + "I put it down.\nAction: put mug 2 in/on desk 1"
        + footer
    )
    assert text == text.strip() and text.count("Action: ") == 3
    assert label.ref_text_of({"takeover_steps": []}) == header + footer
    assert label.ref_text_of({"takeover_steps": [{"command": "look"}]}) == header + "Action: look" + footer
    assert label.ref_text_of({"takeover_steps": [{"delib": None, "command": None}]}) == header + "(no action)" + footer
    one = label.ref_text_of({"takeover_steps": [{"delib": "go", "command": "go to desk 1"}]})
    assert one.count("Action: go to desk 1") == 1 and one.startswith(header) and one.endswith(footer)


@pytest.mark.parametrize(
    ("tito", "template", "kwargs"),
    [
        ("qwen3", "qwen3_fixed.jinja", ""),  # the family's launch: the template's own thinking default
        ("qwen3", "qwen3_fixed.jinja", Q35_KWARGS),  # thinking off: the generation prompt grows an empty think block
        ("qwen35", "qwen3.5_fixed.jinja", Q35_KWARGS),
    ],
)
@pytest.mark.usefixtures("tito_stub")  # the TITO registry stand-in
def test_privileged_prompt_equals_campaign_surgery(template_render, template_render_chat, tito, template, kwargs):
    """The token splice (D9) against REF label.py:112-128's surgery (decode, rfind the generation prompt,
    re-encode head and tail, the hint as a ChatML user block; _campaign_surgery, the oracle): the same ids on the
    real one-shot ALFWorld prompt plus the reset observation, rendered as the engines render the campaign launch
    (tito_render_chat with the family's kwargs) and under the template's own defaults, with the hints the
    production path builds (ref_text_of, an invalid step included). A hint with outer whitespace is refused before
    the splice: Qwen3.5's template trims a user message's content, so the surgery and a rendered message disagree."""
    tokenizer = FakeTokenizer(template_render)
    bare = template_render_chat(template)
    prompt = prompts.render(prompts.load("alfworld", "r2s_oneshot"), bare, [ACT_FLAT])
    history = [*prompt, {"role": "user", "content": OBS0}]
    taus = [
        {
            "takeover_steps": [
                {"delib": "The mug is on the desk; I take it.", "command": "take mug 2 from desk 1"},
                {"delib": "", "command": "clean mug 2 with sinkbasin 1"},
                {"delib": "done?", "command": None, "format_ok": False},
                {"delib": "", "command": "put mug 2 in/on desk 1"},
            ]
        },
        {"takeover_steps": [{"delib": "", "command": "look"}]},
    ]
    for render in (tito_render_chat(tito, tokenizer, kwargs), bare):
        prompt_ids = _prompt_ids(tokenizer, render, history)
        gen = generation_prompt_ids(render, tokenizer)
        assert prompt_ids[-len(gen) :] == gen and len(prompt_ids) > 2000
        for tau in taus:
            hint = label.ref_text_of(tau)
            ours = label.privileged_prompt_ids(tokenizer, render, prompt_ids, hint)
            oracle = _campaign_surgery(tokenizer, prompt_ids, hint)
            assert oracle is not None and ours == oracle
            block = user_message_ids(render, tokenizer, hint)
            assert ours == prompt_ids[: -len(gen)] + block + gen and hint in tokenizer.decode(block)
        with pytest.raises(ValueError, match="outer whitespace"):
            label.privileged_prompt_ids(tokenizer, render, prompt_ids, " " + hint)
    # why the refusal (the C3 review's binding note): under Qwen3.5 the rendered block of a padded hint is not
    # the surgery's literal block, under Qwen3 it is
    engines = tito_render_chat(tito, tokenizer, kwargs)
    padded = " " + label.ref_text_of(taus[1]) + "\n"
    surgery = _campaign_surgery(tokenizer, prompt_ids, padded)  # the surgery splices a padded hint as it is
    literal = surgery[len(prompt_ids) - len(gen) : len(surgery) - len(gen)]
    assert padded in tokenizer.decode(literal)
    assert (user_message_ids(engines, tokenizer, padded) == literal) == (tito == "qwen3")


@pytest.mark.usefixtures("tito_stub")  # the TITO registry stand-in
def test_privileged_prompt_inserts_hint_before_generation_prompt(template_render):
    """REF tests/test_r2s.py:108-115 on the port: the hint block goes in right before the generation prompt, the
    head is untouched, the tail is the generation prompt itself and the offset is the block's length; a prompt
    that does not end with the generation prompt (none, or cut) is None, the prompt_retok_mismatch drop, never a
    splice at a guessed point."""
    tokenizer = FakeTokenizer(template_render)
    render = tito_render_chat("qwen35", tokenizer, Q35_KWARGS)
    prompt = _prompt_ids(tokenizer, render)
    gen = generation_prompt_ids(render, tokenizer)
    block = user_message_ids(render, tokenizer, "HINT")
    priv = label.privileged_prompt_ids(tokenizer, render, prompt, "HINT")
    ins = len(prompt) - len(gen)
    assert priv is not None and len(priv) == len(prompt) + len(block)
    assert priv[:ins] == prompt[:ins] and priv[ins : ins + len(block)] == block and priv[ins + len(block) :] == gen
    text = tokenizer.decode(priv)
    assert text.index("HINT") < text.rindex(tokenizer.decode(gen)) and text.endswith(tokenizer.decode(gen))
    assert text.startswith(tokenizer.decode(prompt[:ins])) and "HINT" not in tokenizer.decode(prompt)
    assert (
        label.privileged_prompt_ids(tokenizer, render, _prompt_ids(tokenizer, render, generation_prompt=False), "HINT")
        is None
    )
    assert label.privileged_prompt_ids(tokenizer, render, prompt[:-1], "HINT") is None
    assert label.privileged_prompt_ids(tokenizer, render, prompt[-2:], "HINT") is None
    assert label.privileged_prompt_ids(tokenizer, render, [], "HINT") is None
    # another family's render does not splice this prompt: its generation prompt is not this prompt's tail
    assert label.privileged_prompt_ids(tokenizer, tito_render_chat("qwen3", tokenizer, ""), prompt, "HINT") is None
    with pytest.raises(ValueError, match="outer whitespace"):
        label.privileged_prompt_ids(tokenizer, render, prompt, "HINT\n")


@pytest.mark.usefixtures("tito_stub")  # the TITO registry stand-in
async def test_label_positions_offset_and_row_schema(template_render, monkeypatch):
    """label_state: REF label.py:239-295's row ({state_id, split, goal, div_turn, objective suffix, ids,
    positions, meta}) from one prefill of the privileged pass (the spliced prompt, then the plain response to the
    last turn's end, top-k = k, --workers as the transport cap); ids = the plain ids without the trailing
    observation; positions = every token of every turn from the divergence turn on, once, kind hind, dist =
    div_turn - t, the targets read at p + offset of the privileged pass; meta's offset = hint_tokens = the hint
    block's length. build_positions alone (REF :224-236): a turn without a call whole, a position without top
    entries skipped."""
    tokenizer = FakeTokenizer(template_render)
    render = tito_render_chat("qwen35", tokenizer, Q35_KWARGS)
    prompt_ids = _prompt_ids(tokenizer, render)
    state = _token_state(prompt_ids, div_turn=1, trailing_obs=2)
    tau = {
        "sample": 3,
        "won": True,
        "takeover_steps": [{"delib": "go", "command": "go to desk 1"}, {"delib": "", "command": None}],
    }
    fake = FakePrefill()
    monkeypatch.setattr(label.teacher_scoring, "prefill", fake)
    args = argparse.Namespace(k=2, max_ctx=10_000, workers=4)
    counts = collections.Counter()
    row = await label.label_state(state, tau, tokenizer, render, args, counts, TEACHER + "/generate")
    ep = common.episode_from_row(state, "d:0")
    hint = label.ref_text_of(tau)
    offset = len(user_message_ids(render, tokenizer, hint))
    assert list(row) == ["state_id", "split", "goal", "div_turn", "objective", "ids", "positions", "meta"]
    assert [row[k] for k in ("state_id", "split", "goal", "div_turn", "objective")] == [
        "d:0:1",
        "train",
        "g1",
        1,
        "suffix",
    ]
    assert row["ids"] == state["tokens"][: ep.prompt_length + ep.turns[-1].end] == state["tokens"][:-2]
    assert row["meta"] == {"tau_sample": 3, "tau_steps": 2, "tau_won": True, "offset": offset, "hint_tokens": offset}
    assert counts == {} and offset > 0
    # the one prefill: the privileged prompt (the splice) + the plain response, top-k = k, --workers as the cap
    ((ids, top_k, kw),) = fake.calls
    priv_prompt = label.privileged_prompt_ids(tokenizer, render, prompt_ids, hint)
    assert list(ids) == priv_prompt + row["ids"][ep.prompt_length :] and (top_k, kw) == (2, {"concurrency": 4})
    assert len(ids) == len(row["ids"]) + offset and len(priv_prompt) == len(prompt_ids) + offset
    # positions: turns 1 and 2 whole (deliberation and call), turn 0 context only, observations never
    positions = row["positions"]
    assert {x["kind"] for x in positions} == {"hind"} and {x["turn"] for x in positions} == {1, 2}
    assert all(list(x) == ["p", "turn", "dist", "kind", "tgt_ids", "tgt_lps"] for x in positions)
    for t in (1, 2):
        start, _, end = ep.abs_span(ep.turns[t])
        assert [x["p"] for x in positions if x["turn"] == t] == list(range(start, end))
        assert {x["dist"] for x in positions if x["turn"] == t} == {1 - t}
    assert len(positions) == 10 == sum(ep.loss_mask[ep.turns[1].start :])
    assert all(x["p"] >= ep.prompt_length and state["loss_mask"][x["p"] - ep.prompt_length] for x in positions)
    # the targets come from the privileged position p + offset: the student's own token there, the position itself
    for x in positions:
        assert x["tgt_ids"] == [row["ids"][x["p"]], x["p"] + offset] and x["tgt_lps"] == [-0.5, -1.5]
    # k entries per position; a turn without a call is labeled whole; a position without entries is skipped
    args.k = 4
    full = await label.label_state(state, tau, tokenizer, render, args, counts, TEACHER + "/generate")
    assert all(len(x["tgt_ids"]) == 4 and x["tgt_ids"][2:] == [9000, 9001] for x in full["positions"])
    no_call = common.episode_from_row(
        {
            "tokens": [1000, 1001, 1, 1, MARK, 2, 3, 1, 1, 1, 1],
            "response_length": 9,
            "loss_mask": [1, 1, 1, 1, 0, 1, 1, 1, 1],
            "turns": [[0, 4, 2], [5, 9, 9]],
        },
        "d:1",
    )
    n, off = len(no_call.tokens), 3
    priv = {"top": [[[-0.3, 9, ""], [-1.0, 10, ""], [-2.0, 11, ""]]] * (n + off)}
    pos = label.build_positions(no_call, 0, priv, off, 2)
    assert [x["p"] for x in pos] == [no_call.prompt_length + i for i, m in enumerate(no_call.loss_mask) if m]
    assert all(x["tgt_ids"] == [9, 10] and x["tgt_lps"] == [-0.3, -1.0] for x in pos)
    s1, _, e1 = no_call.abs_span(no_call.turns[1])
    assert [x["p"] for x in pos if x["turn"] == 1] == list(range(s1, e1)) and not no_call.turns[1].has_action
    assert [x["dist"] for x in pos] == [0] * 4 + [-1] * 4
    priv["top"][s1 + off] = None
    assert [x["p"] for x in label.build_positions(no_call, 1, priv, off, 2)] == list(range(s1 + 1, e1))
    assert label.build_positions(no_call, 2, priv, off, 2) == []


@pytest.mark.usefixtures("tito_stub")  # the TITO registry stand-in
def test_label_drop_rules_and_gate(template_render, tmp_path, monkeypatch, capsys):
    """python -m miles_agent.r2s.label over states that hit every drop rule (bad_div_turn, prompt_retok_mismatch,
    long_ctx under --max-ctx with the hint, prefill_failed, no_positions) or have no tau (no modal cluster, not in
    the fold): the counts in the summary, labeled_frac under the env's label_frac_gate -> LABEL_FAIL as key and
    trailing line with exit 2; at --min-labeled-frac below it the same labels pass without a new prefill for the
    labeled state (the dropped ones are retried: resume skips labeled state_ids only), LABEL_DONE; --force redoes
    the file; --limit; the DONE skip; a --teacher-url carrying a path refused at the flag."""
    tokenizer = FakeTokenizer(template_render)
    loads = _transformers_stub(monkeypatch, tokenizer)
    render = tito_render_chat("qwen35", tokenizer, Q35_KWARGS)
    prompt_ids = _prompt_ids(tokenizer, render)
    good = _token_state(prompt_ids, state_id="d:0:1")
    late = _token_state(prompt_ids, state_id="d:1:3", div_turn=3)  # three turns: no turn 3
    retok = _token_state(_prompt_ids(tokenizer, render, generation_prompt=False), state_id="d:2:1")
    long = _token_state([5] * 40 + prompt_ids, state_id="d:3:1")
    down = _token_state([999, *prompt_ids[1:]], state_id="d:4:1")
    empty = _token_state([998, *prompt_ids[1:]], state_id="d:5:1")
    split = _token_state(prompt_ids, state_id="d:6:1")  # two clusters of one: no cluster reaches m=2
    orphan = _token_state(prompt_ids, state_id="d:7:1")  # no takeover in the fold
    states = [good, late, retok, long, down, empty, split, orphan]
    fold = []
    for s in states[:6]:
        fold += [_take(0, "A", 3, state_id=s["state_id"]), _take(1, "A", 2, state_id=s["state_id"], won=True)]
    fold += [_take(0, "A", 1, state_id="d:6:1"), _take(1, "B", 1, state_id="d:6:1")]
    states_path, fold_path, out = tmp_path / "states.jsonl", tmp_path / "fold.jsonl", tmp_path / "labels.jsonl"
    common.write_jsonl(str(states_path), states)
    common.write_jsonl(str(fold_path), fold)
    fake = FakePrefill()
    monkeypatch.setattr(label.teacher_scoring, "prefill", fake)
    hint = label.ref_text_of(fold[1])  # tau of every voting state: the 2-step takeover (sample 1)
    cap = len(prompt_ids) + len(user_message_ids(render, tokenizer, hint)) + good["response_length"] + 10
    base = ["--env", "alfworld", "--states", str(states_path), "--fold", str(fold_path), "--out", str(out)]
    base += ["--tokenizer-path", "/models/student", "--tito-model", "qwen35", "--chat-template-kwargs", Q35_KWARGS]
    base += ["--teacher-url", TEACHER, "--k", "2", "--max-ctx", str(cap)]
    assert label.main(base) == 2
    summary_path = f"{out}.summary.json"
    summary = common.read_summary(summary_path)
    counted = ("states", "tau_found", "no_tau", "bad_div_turn", "prompt_retok_mismatch", "long_ctx")
    assert [summary[k] for k in counted] == [8, 6, 2, 1, 1, 1]
    assert [summary[k] for k in ("prefill_failed", "no_positions", "labeled", "resumed")] == [1, 1, 1, 0]
    assert (summary["labeled_frac"], summary["min_labeled_frac"], summary["objective"]) == (0.125, 0.25, "suffix")
    assert summary["select_stats"] == {"source": "inline", "eligible_total": 14} and "check_render" not in summary
    assert (summary["n_positions"], summary["goals"], summary["dist_hist"]) == (10, 1, {"0": 5, "-1": 5})
    assert [summary[k] for k in ("k", "max_ctx", "tito_model")] == [2, cap, "qwen35"]
    assert summary["config"] == {**dataclasses.asdict(ALF), "goal_keys": ["game_file"]}
    assert summary["LABEL_FAIL"] is True and "LABEL_DONE" not in summary
    assert open(summary_path).read().endswith("}\nLABEL_FAIL\n")
    printed = capsys.readouterr().out
    assert printed.rstrip().endswith("LABEL_FAIL") and "prefill failed for d:4:1" in printed
    rows = common.read_jsonl(str(out))
    assert [r["state_id"] for r in rows] == ["d:0:1"] and rows[0]["meta"]["tau_sample"] == 1
    assert rows[0]["meta"]["tau_steps"] == 2 and rows[0]["meta"]["tau_won"] is True
    # the long, the retokenized and the turnless states never reach the server; the others once each
    assert [ids[0] for ids, _, _ in fake.calls] == [prompt_ids[0], 999, 998] and loads == ["/models/student"]
    assert {top_k for _, top_k, _ in fake.calls} == {2} and {kw["concurrency"] for _, _, kw in fake.calls} == {16}
    # the same labels under a lower gate: the labeled state is resumed, the dropped ones retried
    assert label.main([*base, "--min-labeled-frac", "0.1"]) == 0
    summary = common.read_summary(summary_path)
    assert [summary[k] for k in ("labeled", "resumed", "tau_found", "labeled_frac", "min_labeled_frac")] == [
        1,
        1,
        5,
        0.125,
        0.1,
    ]
    assert summary["LABEL_DONE"] is True and "LABEL_FAIL" not in summary
    assert [ids[0] for ids, _, _ in fake.calls[3:]] == [999, 998] and len(common.read_jsonl(str(out))) == 1
    assert capsys.readouterr().out.rstrip().endswith("LABEL_DONE")
    # done: nothing runs; --force redoes the file; --limit 1 takes the first state only
    assert label.main([*base, "--min-labeled-frac", "0.1"]) == 0 and len(fake.calls) == 5
    assert "LABEL_DONE already in" in capsys.readouterr().out
    assert label.main([*base, "--min-labeled-frac", "0.1", "--force"]) == 0
    assert [ids[0] for ids, _, _ in fake.calls[5:]] == [prompt_ids[0], 999, 998]
    assert common.read_jsonl(str(out)) == rows and common.read_summary(summary_path)["resumed"] == 0
    lim = tmp_path / "lim.jsonl"
    assert label.main([*base, "--out", str(lim), "--limit", "1"]) == 0
    summary = common.read_summary(f"{lim}.summary.json")
    assert (summary["states"], summary["labeled"], summary["labeled_frac"]) == (1, 1, 1.0)
    with pytest.raises(SystemExit, match="--teacher-url .*13141/generate: pass the server's base URL"):
        label.main([*base, "--out", str(tmp_path / "x.jsonl"), "--teacher-url", TEACHER + "/generate"])
    assert not (tmp_path / "x.jsonl").exists()


@pytest.mark.usefixtures("tito_stub")  # the TITO registry stand-in
def test_label_takes_tau_file_or_selects_inline(template_render, tmp_path, monkeypatch, capsys):
    """The selection comes from stage C's --tau rows or runs inline (select.select_tau with the env's check):
    both label the same states with the same tau; select_stats names the source; a --tau sample the fold lacks
    is refused; --tau and --judge-file exclude each other; a judge env needs one of them, and with --judge-file
    the approved takeovers vote by ASIN (F9: a missing verdict counted); the stop file before the first state
    and raised between states (the one in flight finishes, no marker)."""
    tokenizer = FakeTokenizer(template_render)
    _transformers_stub(monkeypatch, tokenizer)
    render = tito_render_chat("qwen35", tokenizer, Q35_KWARGS)
    prompt_ids = _prompt_ids(tokenizer, render)
    states = [_token_state(prompt_ids, state_id=f"d:{i}:1", goal=f"g{i}") for i in range(3)]
    fold = [
        _take(0, "A", 3, state_id="d:0:1"),
        _take(1, "A", 2, state_id="d:0:1"),
        _take(2, "B", 1, state_id="d:0:1"),
        _take(0, "C", 2, state_id="d:1:1"),  # one eligible takeover: not enough for ALFWorld's m=2
        _take(1, "C", 2, state_id="d:1:1", replay_ok=False),
        _take(0, "D", 4, state_id="d:2:1"),
        _take(1, "D", 1, state_id="d:2:1", won=True),
        _take(2, "D", 2, state_id="d:2:1"),
    ]
    states_path, fold_path = tmp_path / "states.jsonl", tmp_path / "fold.jsonl"
    common.write_jsonl(str(states_path), states)
    common.write_jsonl(str(fold_path), fold)
    fake = FakePrefill()
    monkeypatch.setattr(label.teacher_scoring, "prefill", fake)
    base = ["--env", "alfworld", "--states", str(states_path), "--fold", str(fold_path)]
    base += ["--tokenizer-path", "/models/student", "--tito-model", "qwen35", "--chat-template-kwargs", Q35_KWARGS]
    base += ["--teacher-url", TEACHER, "--k", "2"]
    inline = tmp_path / "inline.jsonl"
    assert label.main([*base, "--out", str(inline)]) == 0
    rows = common.read_jsonl(str(inline))
    assert [(r["state_id"], r["meta"]["tau_sample"], r["meta"]["tau_steps"], r["meta"]["tau_won"]) for r in rows] == [
        ("d:0:1", 1, 2, False),
        ("d:2:1", 1, 1, True),
    ]
    summary = common.read_summary(f"{inline}.summary.json")
    assert (summary["tau_found"], summary["no_tau"], summary["labeled"], summary["goals"]) == (2, 1, 2, 2)
    assert summary["select_stats"] == {"source": "inline", "eligible_total": 7} and summary["LABEL_DONE"] is True
    # stage C's rows: the same labels, the source named
    tau_path = tmp_path / "tau.jsonl"
    assert select.main(["--env", "alfworld", "--fold", str(fold_path), "--out", str(tau_path)]) == 0
    assert [(r["state_id"], r["sample"]) for r in common.read_jsonl(str(tau_path))] == [("d:0:1", 1), ("d:2:1", 1)]
    from_tau = tmp_path / "from_tau.jsonl"
    assert label.main([*base, "--tau", str(tau_path), "--out", str(from_tau)]) == 0
    assert common.read_jsonl(str(from_tau)) == rows
    summary = common.read_summary(f"{from_tau}.summary.json")
    assert (summary["tau_found"], summary["no_tau"], summary["select_stats"]) == (2, 1, {"source": "tau"})
    assert len(fake.calls) == 4
    # a tau row the fold cannot honour is a mismatch between the files; --tau and --judge-file exclude each other
    bad_tau = tmp_path / "bad_tau.jsonl"
    common.write_jsonl(str(bad_tau), [{"state_id": "d:0:1", "sample": 7, "group_size": 3, "n_eligible": 3}])
    with pytest.raises(SystemExit, match="--tau names sample 7 of d:0:1, which --fold does not hold"):
        label.main([*base, "--tau", str(bad_tau), "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit) as exclusive:
        label.main([*base, "--tau", str(tau_path), "--judge-file", str(tau_path), "--out", str(tmp_path / "x.jsonl")])
    assert exclusive.value.code == 2 and not (tmp_path / "x.jsonl").exists()
    # WebShop: the judge's approval decides (check judge); without --tau or --judge-file nothing is guessed
    ws_states = tmp_path / "ws_states.jsonl"
    common.write_jsonl(str(ws_states), [{**states[0], "state_id": "w:0:1", "episode_key": "w:0", "goal": "goal"}])
    ws_fold = tmp_path / "ws_fold.jsonl"
    common.write_jsonl(
        str(ws_fold),
        [
            _take(0, "b000000001", 3, state_id="w:0:1"),
            _take(1, "b000000001", 2, state_id="w:0:1"),
            _take(2, "b000000002", 1, state_id="w:0:1"),
        ],
    )
    judge_path = tmp_path / "judge.jsonl"
    common.write_jsonl(
        str(judge_path),
        [
            {"state_id": "w:0:1", "sample": 1, "match_ok": True, "match_text": "MATCH"},
            {"state_id": "w:0:1", "sample": 2, "match_ok": True, "match_text": "MATCH"},
        ],
    )
    ws_base = [*base[2:], "--env", "webshop", "--states", str(ws_states), "--fold", str(ws_fold)]
    with pytest.raises(SystemExit, match="--env webshop selects with check 'judge': --tau .* or --judge-file"):
        label.main([*ws_base, "--out", str(tmp_path / "x.jsonl")])
    ws_out = tmp_path / "ws.jsonl"
    assert label.main([*ws_base, "--judge-file", str(judge_path), "--out", str(ws_out)]) == 0
    (ws_row,) = common.read_jsonl(str(ws_out))
    assert (ws_row["meta"]["tau_sample"], ws_row["goal"]) == (2, "goal")  # one vote each, the greater ASIN wins
    summary = common.read_summary(f"{ws_out}.summary.json")
    assert summary["select_stats"] == {"source": "inline", "eligible_total": 3, "judge_missing": 1}
    assert summary["config"]["check"] == "judge" and summary["min_labeled_frac"] == 0.0
    # the preempt sentinel: present before the first state -> no rows, no summary; raised by the first prefill
    # under --workers 1 -> the state in flight finishes, the rest wait for the requeue
    stop = tmp_path / "STOP"
    stop.write_text("")
    stopped = tmp_path / "stopped.jsonl"
    argv = [*base, "--out", str(stopped), "--stop-file", str(stop)]
    n_calls = len(fake.calls)
    assert label.main(argv) == 0 and len(fake.calls) == n_calls and not stopped.exists()
    assert not os.path.exists(f"{stopped}.summary.json")
    assert "2 states not labeled; exiting without LABEL_DONE" in capsys.readouterr().out
    stop2 = tmp_path / "STOP2"
    raised = FakePrefill(stop_file=str(stop2))
    monkeypatch.setattr(label.teacher_scoring, "prefill", raised)
    argv = [*base, "--out", str(stopped), "--stop-file", str(stop2), "--workers", "1"]
    assert label.main(argv) == 0 and len(raised.calls) == 1
    assert [r["state_id"] for r in common.read_jsonl(str(stopped))] == ["d:0:1"]
    assert not os.path.exists(f"{stopped}.summary.json")
    assert "1 states not labeled; exiting without LABEL_DONE" in capsys.readouterr().out


@pytest.mark.usefixtures("stubs", "tito_stub")  # launcher under test_launcher's miles stand-ins; the TITO registry
def test_label_reads_campaign_v1_rows(template_render, tmp_path, monkeypatch, capsys):
    """F8: a campaign v1 state row (no prompt, transcript, tito_model or chat_template_kwargs) splices under
    --tito-model / --chat-template-kwargs, which default to the --tokenizer-path dir's MODEL_FAMILIES row
    (Qwen3.5-4B -> qwen35, thinking off); a v2 row renders under its own family whatever the flags say.
    --check-render counts the full re-render against the splice: render_unavailable on a v1 row, render_equal
    on a v2 row with its messages, render_prompt_differs when the dumped prompt was tokenized otherwise. Under a
    Qwen3 dir the v1 rows do not end with that family's generation prompt and drop as prompt_retok_mismatch while
    the qwen3 v2 row labels; an unlisted dir is refused before any prefill."""
    tokenizer = FakeTokenizer(template_render)
    loads = _transformers_stub(monkeypatch, tokenizer)
    q35 = tito_render_chat("qwen35", tokenizer, Q35_KWARGS)
    q3 = tito_render_chat("qwen3", tokenizer, "")
    prompt = [{"role": "system", "content": "sys"}]
    history = [*prompt, {"role": "user", "content": OBS0}]
    v1_keys = ("prompt", "transcript", "tito_model", "chat_template_kwargs")
    v1a = {
        k: v
        for k, v in _token_state(_prompt_ids(tokenizer, q35, history), state_id="c:0:1").items()
        if k not in v1_keys
    }
    v1b = {
        k: v
        for k, v in _token_state(_prompt_ids(tokenizer, q35, history), state_id="c:1:1").items()
        if k not in v1_keys
    }
    # a Qwen3 launch's v2 row: miles' folded kwargs dict (the family's own), its prompt under its own template
    q3_row = _token_state(
        _prompt_ids(tokenizer, q3, history),
        state_id="c:2:1",
        tito_model="qwen3",
        chat_template_kwargs={"clear_thinking": False},
    )
    transcript = [*history, T0, {"role": "tool", "tool_call_id": "c1", "content": OBS1}]
    v2 = _token_state(_prompt_ids(tokenizer, q35, history), state_id="c:3:1", prompt=prompt, transcript=transcript)
    other_tools = _token_state(
        _prompt_ids(tokenizer, q35, history, tools=()), state_id="c:4:1", prompt=prompt, transcript=transcript
    )
    states = [v1a, v1b, q3_row, v2, other_tools]
    fold = [_take(s, "A", n, state_id=st["state_id"]) for st in states for s, n in ((0, 3), (1, 2))]
    states_path, fold_path, out = tmp_path / "states.jsonl", tmp_path / "fold.jsonl", tmp_path / "labels.jsonl"
    common.write_jsonl(str(states_path), states)
    common.write_jsonl(str(fold_path), fold)
    fake = FakePrefill()
    monkeypatch.setattr(label.teacher_scoring, "prefill", fake)
    base = ["--env", "alfworld", "--states", str(states_path), "--fold", str(fold_path), "--teacher-url", TEACHER]
    base += ["--k", "2", "--out", str(out)]
    assert label.main([*base, "--tokenizer-path", "/models/Qwen3.5-4B", "--check-render"]) == 0
    rows = {r["state_id"]: r for r in common.read_jsonl(str(out))}
    assert sorted(rows) == ["c:0:1", "c:1:1", "c:2:1", "c:3:1", "c:4:1"] and loads == ["/models/Qwen3.5-4B"]
    summary = common.read_summary(f"{out}.summary.json")
    assert (summary["labeled"], summary["prompt_retok_mismatch"], summary["tito_model"]) == (5, 0, "qwen35")
    assert summary["check_render"] == {"render_unavailable": 3, "render_equal": 1, "render_prompt_differs": 1}
    assert summary["LABEL_DONE"] is True
    hint = label.ref_text_of(fold[1])
    for key, render in (("c:0:1", q35), ("c:2:1", q3)):  # each row's hint block is its own family's
        assert rows[key]["meta"]["offset"] == len(user_message_ids(render, tokenizer, hint))
    assert generation_prompt_ids(q3, tokenizer) != generation_prompt_ids(q35, tokenizer)  # what parts the dirs below
    assert len(fake.calls) == 5 and all(top_k == 2 for _, top_k, _ in fake.calls)
    # a Qwen3 dir: the v1 rows take its family and do not end with its generation prompt; the v2 rows keep theirs
    q3_out = tmp_path / "q3.jsonl"
    assert (
        label.main([*base, "--out", str(q3_out), "--tokenizer-path", "/models/Qwen3-4B", "--min-labeled-frac", "0"])
        == 0
    )
    summary = common.read_summary(f"{q3_out}.summary.json")
    assert (summary["labeled"], summary["prompt_retok_mismatch"], summary["tito_model"]) == (3, 2, "qwen3")
    assert sorted(r["state_id"] for r in common.read_jsonl(str(q3_out))) == ["c:2:1", "c:3:1", "c:4:1"]
    assert "check_render" not in summary and loads == ["/models/Qwen3.5-4B", "/models/Qwen3-4B"]
    # explicit flags win over the dir's row; an unlisted dir without them is refused before any prefill
    q3_v1 = tmp_path / "q3_v1.jsonl"
    argv = [*base, "--out", str(q3_v1), "--tokenizer-path", "/models/mystery", "--tito-model", "qwen35"]
    assert label.main([*argv, "--chat-template-kwargs", Q35_KWARGS, "--limit", "2"]) == 0
    assert common.read_summary(f"{q3_v1}.summary.json")["labeled"] == 2
    n_calls = len(fake.calls)
    with pytest.raises(SystemExit, match="--tokenizer-path: no MODEL_FAMILIES entry for 'mystery'"):
        label.main([*base, "--out", str(tmp_path / "x.jsonl"), "--tokenizer-path", "/models/mystery"])
    assert len(fake.calls) == n_calls and not (tmp_path / "x.jsonl").exists()
    capsys.readouterr()


@pytest.mark.usefixtures("tito_stub")  # the TITO registry stand-in
def test_label_counts_an_undiffable_prompt(template_render, tmp_path, monkeypatch, capsys):
    """A row whose template cannot be diffed for its generation prompt (tool_call_markup._appended_ids refuses a
    template that rewrites instead of appending: here a nemotron3 row, whose family keeps the tokenizer's own
    template, over a tokenizer whose template writes no generation prompt) is counted under prompt_retok_mismatch
    with the reason printed, and the stage labels the other rows instead of aborting."""
    flat = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    tokenizer = FakeTokenizer(template_render, chat_template=flat)
    _transformers_stub(monkeypatch, tokenizer)
    q35 = tito_render_chat("qwen35", tokenizer, Q35_KWARGS)
    own = tito_render_chat("nemotron3", tokenizer, "")
    with pytest.raises(ValueError, match="does not append the generation prompt"):
        generation_prompt_ids(own, tokenizer)
    good = _token_state(_prompt_ids(tokenizer, q35), state_id="d:0:1")
    undiffable = _token_state(
        _prompt_ids(tokenizer, own), state_id="d:1:1", tito_model="nemotron3", chat_template_kwargs={}
    )
    states = [good, undiffable]
    fold = [_take(s, "A", n, state_id=st["state_id"]) for st in states for s, n in ((0, 3), (1, 2))]
    states_path, fold_path, out = tmp_path / "states.jsonl", tmp_path / "fold.jsonl", tmp_path / "labels.jsonl"
    common.write_jsonl(str(states_path), states)
    common.write_jsonl(str(fold_path), fold)
    fake = FakePrefill()
    monkeypatch.setattr(label.teacher_scoring, "prefill", fake)
    base = ["--env", "alfworld", "--states", str(states_path), "--fold", str(fold_path), "--out", str(out)]
    base += ["--tokenizer-path", "/models/student", "--tito-model", "qwen35", "--chat-template-kwargs", Q35_KWARGS]
    base += ["--teacher-url", TEACHER, "--k", "2"]
    assert label.main(base) == 0  # one of two labeled: over ALFWorld's 0.25 gate
    summary = common.read_summary(f"{out}.summary.json")
    assert [summary[k] for k in ("states", "tau_found", "labeled", "prompt_retok_mismatch", "labeled_frac")] == [
        2,
        2,
        1,
        1,
        0.5,
    ]
    assert summary["LABEL_DONE"] is True
    assert [r["state_id"] for r in common.read_jsonl(str(out))] == ["d:0:1"] and len(fake.calls) == 1
    printed = capsys.readouterr().out
    assert "no splice for d:1:1: the chat template does not append the generation prompt" in printed


# ---------------------------------------------------------------- stage E: teacher_ft.py (C14)
# The FT logic is REF's; what runs here is everything around the GPU: the goal-disjoint split, the label checks,
# the CLI defaults, the lockstep eval schedule, the export sidecars and the report, and the loss itself on CPU
# tensors where torch is installed.


def _label_row(state_id, goal, kind="hind", objective="suffix"):
    """A stage D row (label.py's schema) with two privileged positions."""
    positions = [
        {"p": 3 + j, "turn": 1, "dist": 0, "kind": kind, "tgt_ids": [1, 2], "tgt_lps": [-0.5, -1.5]} for j in range(2)
    ]
    return {
        "state_id": state_id,
        "split": "train",
        "goal": goal,
        "div_turn": 1,
        "objective": objective,
        "ids": list(range(8)),
        "positions": positions,
        "meta": {},
    }


def test_split_rows_goal_disjoint():
    """md5(goal) mod 100 < val_pct selects the held-out GOALS (REF teacher_ft.py:59-65): every state of a goal lands
    on one side, the fraction follows the percentage, a row without a goal hashes its state_id, 0 and 100 empty a
    side."""
    rows = [_label_row(f"g{g}:{t}:1", f"goal-{g}") for g in range(200) for t in range(3)]
    train, val = teacher_ft.split_rows(rows, 10)
    assert len(train) + len(val) == len(rows) and 0 < len(val) < len(rows)
    assert not ({r["goal"] for r in train} & {r["goal"] for r in val})
    held_out = {r["goal"] for r in rows if int(hashlib.md5(r["goal"].encode()).hexdigest(), 16) % 100 < 10}
    assert {r["goal"] for r in val} == held_out and len(val) == 3 * len(held_out)
    assert abs(len(held_out) / 200 - 0.10) < 0.06  # md5 is uniform: about a tenth of the goals
    _, val25 = teacher_ft.split_rows(rows, 25)
    assert held_out < {r["goal"] for r in val25}  # the threshold moves, the hash per goal does not
    orphans = [{"state_id": f"s{i}", "positions": []} for i in range(50)]  # no goal: the state_id is the key
    assert sum(len(part) for part in teacher_ft.split_rows(orphans, 10)) == 50
    assert teacher_ft.split_rows(rows, 0)[1] == [] and teacher_ft.split_rows(rows, 100)[0] == []


def test_check_labels_and_cli_defaults():
    """teacher_ft trains label.py's objective and nothing else: a file without objective "suffix" or with a plain
    ("anchor") position is refused before any model loads; the CLI's defaults are the campaign's (select.FT_*, the
    design's stage E line) and REF's prefix-objective --anchor-coef is no flag."""
    hind = _label_row("a:1:1", "g")["positions"][0]
    anchor = {**hind, "kind": "anchor"}
    assert teacher_ft.OBJECTIVE == label.OBJECTIVE == "suffix"
    teacher_ft.check_labels([{"objective": "suffix", "positions": [hind, hind]}])  # label.py's rows pass
    refused = [
        [{"positions": [hind, anchor]}],  # prefix labels: no objective field, a plain action target
        [{"scope": "suffix", "positions": [hind]}],  # a suffix file from before the objective field
        [{"objective": "suffix", "positions": [hind, anchor]}],  # a plain target among the privileged ones
        [{"objective": "suffix", "positions": [hind]}, {"positions": [hind, anchor]}],  # a mixed file
    ]
    lacks = r"label rows lack objective 'suffix' and \d+ positions are not privileged"
    for rows in refused:
        with pytest.raises(SystemExit, match=lacks):
            teacher_ft.check_labels(rows)
    with pytest.raises(SystemExit, match="FATAL: 1 label rows lack objective 'suffix' and 1 positions"):
        teacher_ft.check_labels(refused[-1])
    ap = teacher_ft.build_parser()
    args = ap.parse_args(["--data", "l.jsonl", "--model-path", "m", "--out-dir", "o"])
    assert (args.epochs, args.lr, args.grad_accum, args.warmup_steps) == (1, 2e-6, 8, 10)
    assert (args.min_lr_ratio, args.clip, args.val_pct, args.seed) == (0.1, 1.0, 10, 42)
    assert (args.max_ctx, args.skip_gate) == (32768, False)
    with pytest.raises(SystemExit):
        ap.parse_args(["--data", "l.jsonl", "--model-path", "m", "--out-dir", "o", "--anchor-coef", "1.0"])


def test_eval_schedule_keeps_every_rank_in_lockstep():
    """Every rank runs ceil(n/world) forwards (an idle rank would deadlock FSDP's all-gathers); the padding passes
    re-use a real row uncounted, every real row is counted exactly once."""
    sched = [teacher_ft.eval_schedule(2, 4, r) for r in range(4)]
    assert all(len(s) == 1 for s in sched) and [s[0][1] for s in sched] == [True, True, False, False]
    assert sched[2][0][0] in (0, 1) and sched[3][0][0] in (0, 1)
    sched = [teacher_ft.eval_schedule(10, 4, r) for r in range(4)]
    assert all(len(s) == 3 for s in sched)
    assert sorted(i for s in sched for i, counts in s if counts) == list(range(10))
    assert teacher_ft.eval_schedule(3, 1, 0) == [(0, True), (1, True), (2, True)]
    assert teacher_ft.eval_schedule(0, 2, 1) == [(0, False)]  # an empty val set still keeps the ranks in step


def test_export_sidecars_and_report(tmp_path):
    """patch_saved_config serves the base config.json verbatim when it is there, else stamps bfloat16 at both levels
    (a VLM wrapper's text_config too); copy_aux_files copies the tokenizer / processor sidecars that exist, never over
    a file the export wrote; ft_report carries REF's keys with the objective, and write_report creates the dir (an
    FT_FAIL writes it before any export)."""
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    out.mkdir()
    verbatim = '{"dtype": "bfloat16", "text_config": {"rope_theta": 10000000.0}, "num_experts": 8}'
    (base / "config.json").write_text(verbatim)
    (out / "config.json").write_text('{"dtype": "float32", "text_config": {"dtype": "float32"}}')
    teacher_ft.patch_saved_config(str(out), str(base))
    assert (out / "config.json").read_text() == verbatim
    stale = {"torch_dtype": "float32", "text_config": {"dtype": "float32", "torch_dtype": "float32"}, "num_experts": 8}
    (out / "config.json").write_text(json.dumps(stale))
    teacher_ft.patch_saved_config(str(out), str(tmp_path / "absent"))
    cfg = json.loads((out / "config.json").read_text())
    stamped = {"torch_dtype": "bfloat16", "text_config": {"dtype": "bfloat16", "torch_dtype": "bfloat16"}}
    assert cfg == {**stamped, "num_experts": 8}
    (out / "config.json").write_text("{}")
    teacher_ft.patch_saved_config(str(out))
    assert json.loads((out / "config.json").read_text()) == {"torch_dtype": "bfloat16"}  # no dtype key: torch_dtype
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "README.md"):
        (base / name).write_text(name)
    (out / "tokenizer_config.json").write_text("the export's own")
    copied = teacher_ft.copy_aux_files(str(base), str(out))
    assert copied == ["preprocessor_config.json", "tokenizer.json"]  # AUX_FILES order; README is no sidecar
    assert (out / "tokenizer_config.json").read_text() == "the export's own" and not (out / "README.md").exists()
    args = teacher_ft.build_parser().parse_args(["--data", "l", "--model-path", "m", "--out-dir", str(out)])
    report = teacher_ft.ft_report(1.5, 1.2, True, (10, 9, 1), 3, args, copied)
    assert list(report) == [
        "val_hind_ce_before",
        "val_hind_ce_after",
        "passed",
        "objective",
        "rows",
        "train",
        "val",
        "steps",
        "args",
        "copied_aux",
    ]
    assert (report["objective"], report["rows"], report["train"], report["val"]) == ("suffix", 10, 9, 1)
    assert (report["steps"], report["args"]["lr"], report["copied_aux"]) == (3, 2e-6, copied)
    teacher_ft.write_report(str(out / "fresh"), report)
    assert json.loads((out / "fresh" / teacher_ft.REPORT).read_text())["passed"] is True
    assert '"passed": true' in (out / "fresh" / teacher_ft.REPORT).read_text()  # what r2s_teacher_ft.sh greps


@pytest.mark.filterwarnings("ignore:.*CUDA is not available.*:UserWarning")  # row_loss autocasts to cuda
def test_position_loss_and_check_labels():
    """The loss is one renormalised top-k cross-entropy per labeled position, the model's row for token p read at
    p - 1 (REF teacher_ft.py:68-80, 110-121): a uniform model pays log V everywhere; a model's full logits are
    indexed by the positions when it ignored logits_to_keep; the row loss is the plain mean, no second term; and
    the rows it is computed over are label.py's (check_labels) and nothing else."""
    torch = pytest.importorskip("torch")
    positions = [
        {"p": 2, "kind": "hind", "tgt_ids": [0, 1], "tgt_lps": [-0.1, -2.0]},
        {"p": 3, "kind": "hind", "tgt_ids": [2], "tgt_lps": [-0.5]},
        {"p": 5, "kind": "hind", "tgt_ids": [3, 0], "tgt_lps": [-0.7, -0.9]},
    ]
    ce_sum, n = teacher_ft.position_loss(torch.zeros(3, 4), positions, "cpu")
    assert n == 3 and ce_sum.item() == pytest.approx(3 * math.log(4))
    assert teacher_ft.renorm_weights([-0.1, -2.0]).sum().item() == pytest.approx(1.0)  # the top-k mass renormalised
    logits = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(0))

    class Model(torch.nn.Module):
        def forward(self, input_ids, logits_to_keep, use_cache):
            return types.SimpleNamespace(logits=logits)  # full logits: row_loss picks the rows p - 1 itself

    loss, (ce_sum, n) = teacher_ft.row_loss(Model(), {"ids": list(range(6)), "positions": positions}, "cpu")
    logp = torch.log_softmax(logits[0], dim=-1)
    per_position = [
        -(torch.softmax(torch.tensor(x["tgt_lps"]), dim=-1) * logp[x["p"] - 1, x["tgt_ids"]]).sum().item()
        for x in positions
    ]
    assert n == 3 and ce_sum.item() == pytest.approx(sum(per_position), rel=1e-5)
    assert loss.item() == pytest.approx(sum(per_position) / 3, rel=1e-5)
    teacher_ft.check_labels([{"objective": "suffix", "positions": positions}])
    with pytest.raises(SystemExit, match="1 positions are not privileged"):
        teacher_ft.check_labels([{"objective": "suffix", "positions": [{**positions[0], "kind": "anchor"}]}])
