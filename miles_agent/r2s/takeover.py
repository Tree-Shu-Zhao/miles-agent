"""Stage B of the R2S pipeline: teacher takeovers from the divergence states, through the live env.

For every state and sample, the recipe's own env_loop_agent restores the state and the base teacher
finishes the episode: the student's assistant messages before the divergence turn go in the /run body as
``scripted_prefix`` (replayed through /reset and /step of a fresh episode exactly as the policy had
produced them), the teacher's sglang server is the per-request policy (``policy_base_url`` = <teacher>/v1,
design D5) and it acts from the divergence turn until the episode ends or the row's own turn budget runs
out, prefix steps included (D16). Thinking off and the tool-call parser are the served teacher's (the fixed
template and parser the job starts it with, D6), so every teacher turn arrives as the engine parser's
structured message and no chat text is parsed here. REF (the campaign branch's miles_agent/r2s/takeover.py)
put an aiohttp proxy between the agent and the teacher that re-emitted the decoded student turns and checked
every replayed observation inside the episode (400 obs_mismatch@t); here the same check runs on the
transcript the agent returns, after the fact, and the dropped set is the same.

    python -m miles_agent.r2s.takeover --env alfworld --states states.jsonl --gym-url http://127.0.0.1:8200 \\
        --teacher-url http://127.0.0.1:13141 --teacher-model teacher --out fold.jsonl

Row (REF takeover.py:274-317 keys, plus the agent's ``messages``): {state_id, goal, split, div_turn, sample,
replay_ok, [fail,] won, done, truncated, reward, turns_total, invalid_actions, takeover_steps [{turn, delib,
command, call {name, arguments}, format_ok, obs_before}], last_observation, terminal_key, messages}.
replay_ok needs the whole prefix stepped, every observation from the reset one through the one after the
last replayed turn equal to the dump's (whitespace-normalized) and a teacher turn; a state whose transcript
and token spans disagree on the number of turns is dropped as turn_count_mismatch: its K short rows are
written (the resume needs every pair) but never ran, so they leave the replay gate (a sample miles cut at
the per-turn cap, status truncated, holds more transcript turns than spans and is no mismatch). Rows are appended per
state (a restart skips the (state_id, sample) pairs present unless --force), the summary covers the whole
fold (replay_ok_frac over the replayed rows) and ends in TAKEOVER_DONE, or TAKEOVER_FAIL with exit 2 under
--min-replay-ok (F5); --stop-file is honoured between states. --gym-url and --teacher-url are the servers'
base URLs (the launcher's rule): the stage posts to <gym>/run and the agent to <teacher>/v1, and a URL with
a path, query or fragment is refused at the flag.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import itertools
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

import httpx

from miles_agent.gym_bridge import build_run_body, post_json
from miles_agent.r2s import common, select

MARKER = "TAKEOVER_DONE"
FAIL_MARKER = "TAKEOVER_FAIL"
GYM_URL = "http://127.0.0.1:8200"  # the recipe's env_loop_agent inside the pipeline job (REF takeover.py:413)
TEACHER_MODEL = "teacher"  # the job's --served-model-name (REF :412)
CONCURRENCY = 32  # takeovers in flight: the campaign's CONCURRENCY
RUN_TIMEOUT_S = 1800.0  # wall-clock cap of one takeover (REF :421)
MIN_REPLAY_OK = 0.7  # REF :422, never overridden by the campaign (F5)

# WebShop product codes as they appear in admissible actions: click[b000vohh8i] (REF :60-61).
_ASIN_CLICK_RE = re.compile(r"^click\[([a-z0-9]{10})\]$")


# ---- what a teacher turn did ----


def command_from_call(name: str | None, arguments: Any, action_tool: str) -> str | None:
    """The command a tool call carries under the environment's own rule (gym_root/agent_utils/
    session_gym.extract_command; REF takeover.py:153-175): the call must name ``action_tool`` and carry a
    non-empty string ``command`` argument (raw JSON or a parsed dict); anything else is None, the env
    counted a format error and did not step."""
    if isinstance(arguments, str) or arguments is None:
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict) or name != action_tool:
        return None
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    return command.strip()


def first_call(msg: Mapping[str, Any]) -> dict[str, Any] | None:
    """The first tool call of an assistant message as the engine's parser emitted it, {name, arguments}
    with the arguments raw (REF :239-241); None without a call."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0].get("function") or {}
    return {"name": fn.get("name"), "arguments": fn.get("arguments")}


def message_command(msg: Mapping[str, Any], action_tool: str) -> str | None:
    call = first_call(msg)
    return command_from_call(call["name"], call["arguments"], action_tool) if call else None


def purchased_asin(commands: list[str | None]) -> str | None:
    """Vote key for WebShop (paper: the purchased ASIN): the last product-code click of the episode
    (student prefix + takeover), None when nothing was ever opened (REF :87-95)."""
    asin = None
    for cmd in commands:
        m = _ASIN_CLICK_RE.match((cmd or "").strip().lower())
        if m:
            asin = m.group(1)
    return asin


def terminal_key_of(vote_key: str, done: bool, last_obs: str, commands: list[str | None]) -> str | None:
    """How a finished takeover ended, for the vote (REF :98-107 minus Search's answer key): the purchased
    ASIN, or the terminal observation's fingerprint; None for a takeover that did not finish."""
    if not done:
        return None
    if vote_key == "purchased_asin":
        return purchased_asin(commands)
    return common.fingerprint(last_obs) if last_obs else None


# ---- one takeover: the /run body, the replay check, the row ----


def run_body(
    state: Mapping[str, Any], teacher_url: str, teacher_model: str, request_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """The /run body of one takeover: gym_bridge.build_run_body over the row's own prompt, sampling and
    metadata (tools, game_file, max_episode_steps, ... ride to /reset and /step as in training), the teacher
    as the per-request policy (policy_base_url <teacher>/v1, its served model name, the key sglang ignores)
    and the student's turns before the divergence turn as the scripted prefix (D5)."""
    body = build_run_body(teacher_url, state["prompt"], request_kwargs, state["metadata"])
    body["policy_api_key"] = "EMPTY"
    body["scripted_prefix"] = common.policy_turns(state)[: int(state["div_turn"])]
    body["responses_create_params"]["model"] = teacher_model
    return body


def state_fail(state: Mapping[str, Any]) -> str | None:
    """turn_count_mismatch when the transcript's assistant messages and the token-space spans disagree
    (the divergence turn indexes both): no takeover of the state is run. A sample miles cut at the per-turn
    cap (``status`` truncated) keeps its token record to the cut turn while the agent played the episode on
    (env_loop_agent runs to completion), so its transcript holds more turns than its spans; the prefix before
    the divergence turn lies inside both, and that state runs."""
    n_messages, n_spans = len(common.policy_turns(state)), len(state.get("turns") or [])
    if n_messages == n_spans or (state.get("status") == "truncated" and n_messages > n_spans):
        return None
    return "turn_count_mismatch"


def replay_fail(state: Mapping[str, Any], run: Mapping[str, Any]) -> str | None:
    """None when the agent restored the state: every prefix turn stepped (prefix_ended@n otherwise: the env
    ended the episode during the replay), the observations from the reset one through the one after the
    last replayed turn equal to the dump's under norm_text (obs_mismatch@i; REF :210-215's proxy answered
    400 for the same), and at least one teacher turn (no_teacher_turn, REF :292-293)."""
    div_turn = int(state["div_turn"])
    scripted = int(run.get("scripted_steps") or 0)
    if scripted != div_turn:
        return f"prefix_ended@{scripted}"
    want = common.observations_after_prefix(state, div_turn)
    got = common.observations_after_prefix(state, div_turn, run.get("messages") or [])
    for i, (w, g) in enumerate(itertools.zip_longest(want, got)):
        if w is None or g is None or common.norm_text(w) != common.norm_text(g):
            return f"obs_mismatch@{i}"
    if not float(run.get("turns") or 0):
        return "no_teacher_turn"
    return None


def takeover_steps(state: Mapping[str, Any], messages: list[dict[str, Any]], action_tool: str) -> list[dict[str, Any]]:
    """The teacher's turns read from the agent's transcript (REF :236-254 recorded them inside the proxy):
    the assistant messages after the replayed prefix, each with its first tool call as emitted, the command
    under the env's rule, format_ok = the env could step it, and the observation it answered (the last tool
    or user message before it); turn = the policy-turn index, the divergence turn first."""
    steps: list[dict[str, Any]] = []
    obs_before = ""
    turn = 0
    div_turn = int(state["div_turn"])
    for msg in common.episode_messages(state, messages):
        if msg.get("role") != "assistant":
            obs_before = str(msg.get("content") or "")
            continue
        if turn >= div_turn:
            call = first_call(msg)
            command = command_from_call(call["name"], call["arguments"], action_tool) if call else None
            steps.append(
                {
                    "turn": turn,
                    "delib": (msg.get("content") or "").strip(),
                    "command": command,
                    "call": call,
                    "format_ok": command is not None,
                    "obs_before": obs_before,
                }
            )
        turn += 1
    return steps


def _head(state: Mapping[str, Any], sample: int) -> dict[str, Any]:
    return {
        "state_id": state["state_id"],
        "goal": state["goal"],
        "split": state["split"],
        "div_turn": int(state["div_turn"]),
        "sample": sample,
    }


def fail_row(state: Mapping[str, Any], sample: int, fail: str) -> dict[str, Any]:
    """The row of a takeover that never ran or never came back (REF :287-288)."""
    return {**_head(state, sample), "replay_ok": False, "fail": fail}


def takeover_row(
    state: Mapping[str, Any], sample: int, run: Mapping[str, Any], cfg: select.EnvConfig
) -> dict[str, Any]:
    """One takeover's row from the agent's /run reply (REF :274-317 over BASE's fields: ``turns`` counts the
    teacher's turns only and ``scripted_steps`` the replayed ones, so the budget and turns_total add them;
    ``terminated`` is the env's flag). A failed replay keeps the transcript for the diagnosis."""
    row = _head(state, sample)
    fail = replay_fail(state, run)
    if fail:
        return {**row, "replay_ok": False, "fail": fail, "messages": run.get("messages")}
    won = float(run.get("won", 0.0)) >= 1.0 or float(run.get("reward", 0.0)) >= 1.0
    truncated = bool(run.get("env_truncated", 0.0)) or bool(run.get("policy_400_truncated", 0))
    turns_total = float(run.get("scripted_steps", 0.0)) + float(run.get("turns", 0.0))
    budget = int(state["metadata"].get("max_episode_steps") or cfg.max_steps)  # the row's own, D16
    done = won or bool(run.get("terminated", 0.0)) or (not truncated and turns_total < budget)  # REF :297
    done = done and not truncated
    last_obs = str(run.get("last_observation") or "")
    steps = takeover_steps(state, run["messages"], cfg.action_tool)
    prefix = common.policy_turns(state)[: row["div_turn"]]
    commands = [message_command(m, cfg.action_tool) for m in prefix] + [st["command"] for st in steps]
    return {
        **row,
        "replay_ok": True,
        "won": won,
        "done": done,
        "truncated": truncated,
        "reward": float(run.get("reward", 0.0)),
        "turns_total": turns_total,
        "invalid_actions": float(run.get("invalid_actions", 0.0)),
        "takeover_steps": steps,
        "last_observation": last_obs,
        "terminal_key": terminal_key_of(cfg.vote_key, done, last_obs, commands),
        "messages": run["messages"],
    }


# ---- the pass ----


def summarize(rows: list[dict[str, Any]], states_total: int, args: argparse.Namespace, cfg: select.EnvConfig) -> dict:
    """REF :363-387 over every row of the fold (a resumed run's earlier rows included), the knobs and the
    resolved config. replay_ok_frac is over the rows that were replayed: a turn_count_mismatch state never
    ran (its rows are kept for the resume), so it is counted (states) and left out of the gate's denominator;
    with nothing replayed there is no fraction and the fold is refused (TAKEOVER_FAIL: nothing to select from)."""
    counts: collections.Counter = collections.Counter()
    for r in rows:
        counts["replay_ok" if r.get("replay_ok") else f"fail:{str(r.get('fail')).split('@')[0]}"] += 1
    ok = [r for r in rows if r.get("replay_ok")]
    replayed = len(rows) - counts["fail:turn_count_mismatch"]
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in ok:
        by_state[r["state_id"]].append(r)
    votes_ok = sum(
        1
        for rs in by_state.values()
        if max(collections.Counter(r["terminal_key"] for r in rs if r.get("terminal_key")).values(), default=0) >= 2
    )
    n_steps = sum(len(r["takeover_steps"]) for r in ok)
    return {
        **counts,
        "n_takeovers": len(rows),
        "replayed": replayed,
        "turn_count_mismatch": len({r["state_id"] for r in rows if r.get("fail") == "turn_count_mismatch"}),
        "replay_ok_frac": len(ok) / replayed if replayed else None,
        "won_frac": sum(1 for r in ok if r.get("won")) / len(ok) if ok else None,
        "done_frac": sum(1 for r in ok if r.get("done")) / len(ok) if ok else None,
        "states_with_vote_ge2": votes_ok,
        "states_total": states_total,
        "mean_teacher_turns": n_steps / len(ok) if ok else None,
        "format_ok_frac": sum(st["format_ok"] for r in ok for st in r["takeover_steps"]) / max(1, n_steps),
        "samples": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_teacher_tokens": args.max_teacher_tokens,
        "min_replay_ok": args.min_replay_ok,
        "teacher_model": args.teacher_model,
        "config": dataclasses.asdict(cfg),
    }


async def _run(gym_url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any] | str:
    """POST <gym>/run the way gym_bridge.run does (post_json retries transport errors, wait_for bounds the
    whole episode); the failure's name when the takeover never came back (REF :281-288)."""
    try:
        return await asyncio.wait_for(post_json(f"{gym_url}/run", body), timeout=timeout_s)
    except (httpx.HTTPError, ValueError, TimeoutError) as e:
        return f"run_transport:{type(e).__name__}"


async def _main_async(args: argparse.Namespace, cfg: select.EnvConfig, teacher_url: str, gym_url: str) -> int:
    states = common.read_jsonl(args.states)
    if args.limit is not None:
        states = states[: args.limit]
    without = sum(1 for s in states if s.get("transcript") is None)
    if without:
        raise SystemExit(
            f"{without} of {len(states)} states carry no transcript (a campaign v1 dump): stage B replays the "
            "dump's structured transcript and needs a v2 dump (docs/r2s.md section 3)"
        )
    if args.force:
        open(args.out, "w", encoding="utf-8").close()  # every takeover again: the fold is appended to
    done = common.existing_keys(args.out, ("state_id", "sample"))
    request_kwargs = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_teacher_tokens}
    sem = asyncio.Semaphore(args.concurrency)  # takeovers in flight
    # states in flight (their samples fill --concurrency): what a raised stop file lets finish
    slots = asyncio.Semaphore(max(1, math.ceil(args.concurrency / args.samples)))
    stopped: list[str] = []

    async def one(state: dict[str, Any], sample: int) -> dict[str, Any]:
        async with sem:
            run = await _run(
                gym_url, run_body(state, teacher_url, args.teacher_model, request_kwargs), args.run_timeout_s
            )
        return fail_row(state, sample, run) if isinstance(run, str) else takeover_row(state, sample, run, cfg)

    async def one_state(state: dict[str, Any]) -> None:
        todo = [s for s in range(args.samples) if (state["state_id"], s) not in done]
        if not todo:
            return
        fail = state_fail(state)
        if fail:
            rows = [fail_row(state, s, fail) for s in todo]
        else:
            async with slots:
                if common.stop_requested(args.stop_file):
                    stopped.append(state["state_id"])
                    return
                rows = list(await asyncio.gather(*(one(state, s) for s in todo)))
        for row in rows:
            common.append_jsonl(args.out, row)
        print(f"{state['state_id']}: {sum(bool(r.get('replay_ok')) for r in rows)}/{len(rows)} replayed")

    await asyncio.gather(*(one_state(s) for s in states))
    if stopped:
        print(f"stop file {args.stop_file} raised: {len(stopped)} states not taken over; exiting without {MARKER}")
        return 0
    rows = common.read_jsonl(args.out) if os.path.exists(args.out) else []
    summary = summarize(rows, len(states), args, cfg)
    if not summary["replayed"] or summary["replay_ok_frac"] < args.min_replay_ok:
        common.write_summary(args.summary, summary, FAIL_MARKER)
        return 2
    common.write_summary(args.summary, summary, MARKER)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = common.stage_parser(__doc__)
    ap.add_argument("--states", required=True, help="stage A rows")
    ap.add_argument("--gym-url", default=GYM_URL, help="the recipe's env_loop_agent server (POST <url>/run)")
    ap.add_argument(
        "--teacher-url", required=True, help="sglang base URL of the frozen teacher, the per-request policy"
    )
    ap.add_argument("--teacher-model", default=TEACHER_MODEL, help="the teacher server's --served-model-name")
    ap.add_argument("--samples", type=int, default=select.K, help="takeovers per state")
    ap.add_argument("--temperature", type=float, default=select.TAKEOVER_T)
    ap.add_argument("--top-p", type=float, default=select.TOP_P)
    ap.add_argument("--max-teacher-tokens", type=int, default=select.MAX_TEACHER_TOKENS, help="per teacher turn")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY, help="takeovers in flight")
    ap.add_argument("--run-timeout-s", type=float, default=RUN_TIMEOUT_S, help="wall-clock cap of one takeover")
    ap.add_argument(
        "--min-replay-ok",
        type=float,
        default=MIN_REPLAY_OK,
        help=f"{FAIL_MARKER} (exit 2) under this replay_ok fraction",
    )
    args = common.parse_stage_args(ap, argv)
    cfg = select.load_env_config(args.env, args.env_config)
    # the flags are the servers' base URLs (the launcher's --teacher-url rule): build_run_body appends /v1, _run /run
    teacher_url = common.base_url(args.teacher_url, "--teacher-url", "/v1")
    gym_url = common.base_url(args.gym_url, "--gym-url", "/run")
    if common.skip_if_done(args, MARKER):
        return 0
    return asyncio.run(_main_async(args, cfg, teacher_url, gym_url))


if __name__ == "__main__":
    sys.exit(main())
