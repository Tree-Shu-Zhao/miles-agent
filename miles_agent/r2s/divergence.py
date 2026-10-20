"""Stage A of the R2S pipeline: divergence mining over the pilot dump.

For every dumped training episode, prefill the whole token stream on the STUDENT (the base
checkpoint the arm started from) and on the TEACHER, take mean(lp_S - lp_T) over each turn's
deliberation span (>= --min-delib-tokens scored tokens; the paper's divergence score with the
sampled estimator), and keep the argmax turn as the episode's divergence state. No outcome filter
(the paper mines wins and losses alike). Cap --max-per-goal states per goal (highest divergence
first), then the global top --top-n by divergence. The logic is REF's (the campaign branch's
miles_agent/r2s/divergence.py), the defaults are the campaign's (select.py's constants).

Turn spans come from the dump when a row carries them for the requested family. A row without
them, or with spans cut at another family's marker, is re-segmented at the student's own call
marker: the family's fixed template rendered through the --tokenizer-path tokenizer around the
row's action tool (tool_call_markup.marker_token_id, the OPD hook's derivation), never a literal
id. --tito-model and --chat-template-kwargs default to the tokenizer dir's MODEL_FAMILIES row
and stand in for the fields a campaign v1 row lacks (design F8).

    python -m miles_agent.r2s.divergence --env alfworld --dump-glob '<dir>/dump_*.jsonl' \\
        --student-url http://127.0.0.1:13142 --teacher-url http://127.0.0.1:13141 \\
        --tokenizer-path <student HF dir> --out states.jsonl

The URLs are the servers' base URLs (the launcher's --teacher-url rule): the prefill posts to
<url>/generate, derived once at the flags (common.generate_url), which refuse a URL carrying a path.

State row: the dump row's fields (common.Episode) plus {state_id "<episode_key>:<turn>",
episode_key, goal, split, div_turn, div_score, div_rows}. Summary: the counts (episodes_read,
resegmented, long_ctx, prefill_failed, no_eligible_turn, scored), the states' statistics, the
knobs, the resolved EnvConfig and DIVERGENCE_DONE.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import functools
import json
import statistics
import sys
from typing import Any

from miles_agent import tool_call_markup
from miles_agent.opd import teacher_scoring, turns
from miles_agent.r2s import common, select

MARKER = "DIVERGENCE_DONE"
WORKERS = 32  # episodes prefilled at once: the campaign's CONCURRENCY


def turn_divergences(
    ep: common.Episode, lp_s: list[float | None], lp_t: list[float | None], min_delib_tokens: int
) -> list[dict[str, Any]]:
    """Per turn, mean(lp_student - lp_teacher) over the deliberation span (div_delib, None under
    min_delib_tokens scored positions) and over the whole turn (div_all); a position either server
    left unscored is skipped (REF divergence.py:29-49)."""
    rows = []
    for i, turn in enumerate(ep.turns):
        start, delib_end, end = ep.abs_span(turn)

        def span_mean(a: int, b: int) -> tuple[float | None, int]:
            vals = [lp_s[p] - lp_t[p] for p in range(a, b) if lp_s[p] is not None and lp_t[p] is not None]
            return (statistics.mean(vals), len(vals)) if vals else (None, 0)

        d_delib, n_delib = span_mean(start, delib_end)
        d_all, _ = span_mean(start, end)
        rows.append(
            {
                "turn": i,
                "div_delib": d_delib if n_delib >= min_delib_tokens else None,
                "div_all": d_all,
                "n_delib": n_delib,
                "has_action": turn.has_action,
            }
        )
    return rows


def pick_div_turn(
    div_rows: list[dict[str, Any]], min_div_turn: int, require_action: bool = False
) -> dict[str, Any] | None:
    """The eligible turn with the highest deliberation divergence (the first on a tie). With
    require_action, a turn that ended without a tool call is not a deliberation span but a format
    failure: its "deliberation" runs to the end of the turn, and hindsight targets on it teach the
    teacher to end turns without acting (REF divergence.py:52-68: the r2s_a30_17b run mined 266/266 such
    turns and the R2S student collapsed to zero tool calls)."""
    best = None
    for r in div_rows:
        if r["turn"] < min_div_turn or r["div_delib"] is None:
            continue
        if require_action and not r.get("has_action", True):
            continue
        if best is None or r["div_delib"] > best["div_delib"]:
            best = r
    return best


def select_states(states: list[dict[str, Any]], max_per_goal: int, top_n: int) -> list[dict[str, Any]]:
    """The highest-divergence states, at most max_per_goal per goal, then the global top_n of the
    pool (REF divergence.py:71-78)."""
    by_goal: dict[str, list] = collections.defaultdict(list)
    for s in sorted(states, key=lambda s: -s["div_score"]):
        if len(by_goal[s["goal"]]) < max_per_goal:
            by_goal[s["goal"]].append(s)
    pooled = [s for group in by_goal.values() for s in group]
    pooled.sort(key=lambda s: -s["div_score"])
    return pooled[:top_n]


# ---- turn spans: the dump's, or re-segmented at the student's marker ----


def dump_turns(ep: common.Episode, tito_model: str | None) -> list[turns.Turn] | None:
    """The dump's own spans when they serve: the row has them, cut at the requested family's marker
    (a campaign v1 row names no family and is taken to be the requested one, F8). [] when the
    student never spoke (nothing to segment); None asks for a re-segmentation."""
    if ep.turns and (tito_model is None or ep.tito_model in (None, tito_model)):
        return ep.turns
    return [] if not any(ep.loss_mask) else None


def resegment(ep: common.Episode, args: argparse.Namespace) -> list[turns.Turn]:
    """The row's spans cut at the student's call marker: the --tokenizer-path tokenizer under the
    requested family's template around the row's action tool (the OPD hook's derivation,
    opd/generate.py marker_id, for a tokenizer dir instead of a launch)."""
    tools = ep.metadata.get("tools")
    if not tools:
        raise ValueError(f"{ep.key} declares no tools: the call marker comes from the env's action tool")
    marker = marker_id(
        args.tokenizer_path, args.tito_model, args.chat_template_kwargs, json.dumps(tools[0], sort_keys=True)
    )
    return turns.segment_turns(ep.tokens[ep.prompt_length :], ep.loss_mask, marker)


@functools.cache
def marker_id(tokenizer_path: str, tito_model: str, chat_template_kwargs: str, tool: str) -> int:
    """The tokenizer is loaded once per process (as the hook caches it, opd/generate.py:102-111)."""
    from transformers import AutoTokenizer  # the pipeline job's image

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    render = tool_call_markup.tito_render_chat(tito_model, tokenizer, chat_template_kwargs)
    return tool_call_markup.marker_token_id(tokenizer, render, json.loads(tool))


def _with_turns(ep: common.Episode, args: argparse.Namespace, counts: collections.Counter) -> common.Episode:
    own = dump_turns(ep, args.tito_model)
    if own is not None:
        return ep if own is ep.turns else dataclasses.replace(ep, turns=own)
    counts["resegmented"] += 1
    return dataclasses.replace(ep, turns=resegment(ep, args))


# ---- the pass ----


async def _score_episode(
    ep: common.Episode,
    goal: str,
    cfg: select.EnvConfig,
    args: argparse.Namespace,
    counts: collections.Counter,
    student_url: str,
    teacher_url: str,
) -> dict[str, Any] | None:
    """One prefill on each server (``student_url`` / ``teacher_url``: the /generate endpoints main
    derived from the base-URL flags), the per-turn divergences, the argmax turn: the state row, or
    None with the reason counted (REF divergence.py:81-114)."""
    if len(ep.tokens) > args.max_ctx:  # _main_async cut these before the spans were settled; a direct caller's guard
        counts["long_ctx"] += 1
        return None
    if not ep.turns:
        counts["no_eligible_turn"] += 1
        return None
    try:
        student, teacher = await asyncio.gather(
            teacher_scoring.prefill(student_url, ep.tokens, concurrency=args.workers),
            teacher_scoring.prefill(teacher_url, ep.tokens, concurrency=args.workers),
        )
    except teacher_scoring.TeacherScoringError as e:
        counts["prefill_failed"] += 1
        print(f"prefill failed for {ep.key}: {e}")
        return None
    div_rows = turn_divergences(ep, student["lps"], teacher["lps"], args.min_delib_tokens)
    best = pick_div_turn(div_rows, args.min_div_turn, cfg.require_action)
    if best is None:
        counts["no_eligible_turn"] += 1
        return None
    counts["scored"] += 1
    return {
        "state_id": f"{ep.key}:{best['turn']}",
        "episode_key": ep.key,
        "goal": goal,
        "split": ep.split,
        "div_turn": best["turn"],
        "div_score": best["div_delib"],
        "div_rows": div_rows,
        "tokens": ep.tokens,
        "response_length": ep.response_length,
        "loss_mask": ep.loss_mask,
        "turns": [[t.start, t.end, t.delib_end] for t in ep.turns],
        "metadata": ep.metadata,
        "reward": ep.reward,
        "status": ep.status,
        "prompt": ep.prompt,
        "transcript": ep.transcript,
        "tito_model": ep.tito_model,
        "chat_template_kwargs": ep.chat_template_kwargs,
    }


async def _main_async(args: argparse.Namespace, cfg: select.EnvConfig, student_url: str, teacher_url: str) -> int:
    episodes = common.read_dump(args.dump_glob, limit=args.limit)
    episodes = [ep for ep in episodes if ep.status != "aborted" and ep.response_length > 0]
    counts = collections.Counter(
        episodes_read=len(episodes), resegmented=0, long_ctx=0, prefill_failed=0, no_eligible_turn=0, scored=0
    )
    if common.stop_requested(args.stop_file):
        print(f"stop file {args.stop_file} present before mining; exiting without {MARKER}")
        return 0
    # the context cut first: an over-length row is counted and neither re-segmented (a campaign v1 row without
    # metadata["tools"] has nothing to derive the marker from) nor prefilled
    counts["long_ctx"] = sum(len(ep.tokens) > args.max_ctx for ep in episodes)
    episodes = [ep for ep in episodes if len(ep.tokens) <= args.max_ctx]
    if not args.tokenizer_path and any(dump_turns(ep, args.tito_model) is None for ep in episodes):
        raise SystemExit(
            "rows without usable turns (none, or another family's) need --tokenizer-path <student HF dir>"
        )
    goals = [select.goal_of(ep.metadata, cfg) for ep in episodes]  # a dump of another env fails before any prefill
    episodes = [_with_turns(ep, args, counts) for ep in episodes]
    sem = asyncio.Semaphore(args.workers)
    stopped: list[str] = []

    async def guarded(ep: common.Episode, goal: str) -> dict[str, Any] | None:
        async with sem:
            if common.stop_requested(args.stop_file):
                stopped.append(ep.key)
                return None
            return await _score_episode(ep, goal, cfg, args, counts, student_url, teacher_url)

    results = await asyncio.gather(*(guarded(ep, goal) for ep, goal in zip(episodes, goals, strict=True)))
    if stopped:
        print(f"stop file {args.stop_file} raised: {len(stopped)} episodes left unscored; exiting without {MARKER}")
        return 0
    states = select_states([s for s in results if s is not None], args.max_per_goal, args.top_n)
    common.write_jsonl(args.out, states)
    summary = {
        **counts,
        "states_out": len(states),
        "goals_out": len({s["goal"] for s in states}),
        "div_score_min": min((s["div_score"] for s in states), default=None),
        "div_score_median": statistics.median([s["div_score"] for s in states]) if states else None,
        "div_turn_hist": dict(collections.Counter(s["div_turn"] for s in states)),
        "won_frac_in_states": (
            sum(1 for s in states if (s.get("reward") or 0) >= 1.0) / len(states) if states else None
        ),
        "top_n": args.top_n,
        "max_per_goal": args.max_per_goal,
        "min_delib_tokens": args.min_delib_tokens,
        "min_div_turn": args.min_div_turn,
        "max_ctx": args.max_ctx,
        "tito_model": args.tito_model,
        "config": dataclasses.asdict(cfg),
    }
    common.write_summary(args.summary, summary, MARKER)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = common.stage_parser(__doc__)
    ap.add_argument("--dump-glob", required=True, help="the pilot dump files, e.g. '<dir>/dump_*.jsonl'")
    ap.add_argument("--student-url", required=True, help="sglang base URL of the student's base checkpoint")
    ap.add_argument("--teacher-url", required=True, help="sglang base URL of the frozen teacher")
    ap.add_argument(
        "--tokenizer-path", default=None, help="the student's HF dir: re-segments rows without usable turns"
    )
    ap.add_argument("--tito-model", default=None, help="the re-segmentation's TITO family (default: the dir's row)")
    ap.add_argument(
        "--chat-template-kwargs", default=None, help="its --apply-chat-template-kwargs JSON (default: the dir's row)"
    )
    ap.add_argument("--top-n", type=int, default=select.TOP_N, help="states kept after the per-goal cap")
    ap.add_argument("--max-per-goal", type=int, default=select.MAX_PER_GOAL)
    ap.add_argument("--min-delib-tokens", type=int, default=select.MIN_DELIB_TOKENS)
    ap.add_argument("--min-div-turn", type=int, default=select.MIN_DIV_TURN)
    ap.add_argument(
        "--require-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="only a turn ending in a tool call can be the divergence turn (default: the env's)",
    )
    ap.add_argument("--max-ctx", type=int, default=select.MAX_CTX, help="longer episodes are counted and skipped")
    ap.add_argument("--workers", type=int, default=WORKERS, help="episodes prefilled at once")
    args = common.parse_stage_args(ap, argv)
    cfg = select.load_env_config(args.env, args.env_config, require_action=args.require_action)
    # the flags are base URLs; the prefill posts to /generate (teacher_scoring.prefill takes the endpoint,
    # as miles' --opd-teacher-urls carries it): derived once, a URL with a path refused at the flags
    student_url = common.generate_url(args.student_url, "--student-url")
    teacher_url = common.generate_url(args.teacher_url, "--teacher-url")
    if common.skip_if_done(args, MARKER):
        return 0
    common.family_defaults(args)
    return asyncio.run(_main_async(args, cfg, student_url, teacher_url))


if __name__ == "__main__":
    sys.exit(main())
