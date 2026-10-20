"""Stage D of the R2S pipeline: privileged top-k labels on the student's own episode.

For every state with a tau (stage C's --tau row, or select.select_tau inline over the fold with the env's
check and the --judge-file rows), the frozen base teacher scores the student's pilot episode in the privileged
pass: the episode's ids (prompt + every turn to the last turn's end; the plain ids the labels are stored over)
with tau rendered as one more user message right before the generation prompt, so every response position
shifts by one constant offset. Every token of every assistant turn t >= div_turn gets the teacher's top-k at
its shifted position (kind "hind"): deliberation, call and turns without a call alike, each once; the earlier
turns, the prompt and the observations are context. dist = div_turn - t (<= 0). Rows and the summary record
objective "suffix"; teacher_ft refuses any other labels. The logic is REF's (the campaign branch's
miles_agent/r2s/label.py: ref_text_of and build_positions verbatim), the defaults the campaign's.

The privileged prompt is a token splice (design D9), not a re-tokenization: the prompt's ids minus the
generation prompt (tool_call_markup.generation_prompt_ids: the family's fixed template rendered twice and
diffed), the hint as the template writes a user message (user_message_ids), the generation prompt again.
It is byte-equal to REF's decode-rfind-re-encode surgery by construction, names no markup, and needs neither
the transcript nor the reset observation, so it runs on the campaign's own states. The hint is REF's
reference walkthrough (REF_HEADER, one line per teacher step, REF_FOOTER). --tito-model and
--chat-template-kwargs default to the --tokenizer-path dir's MODEL_FAMILIES row and stand in for the fields
a campaign v1 row lacks (F8); a v2 row renders under its own.

    python -m miles_agent.r2s.label --env alfworld --states states.jsonl --fold fold.jsonl \\
        [--tau tau.jsonl | --judge-file judge.jsonl] --tokenizer-path <student HF dir> \\
        --teacher-url http://127.0.0.1:13141 --out labels.jsonl

--teacher-url is the server's base URL (the launcher's rule): the prefill posts to <url>/generate, derived
once at the flag (common.generate_url). Row (REF label.py:273-288): {state_id, split, goal, div_turn,
objective "suffix", ids, positions [{p, turn, dist, kind "hind", tgt_ids, tgt_lps}], meta {tau_sample,
tau_steps, tau_won, offset, hint_tokens}}. A state is dropped, and the reason counted, as bad_div_turn,
prompt_retok_mismatch (the prompt does not end with the generation prompt, or the row's template has none to
diff), long_ctx (over --max-ctx with the hint), prefill_failed or no_positions. Rows are appended per state (a
restart skips the state_ids present unless --force); the summary (the counts, labeled_frac, select_stats, the
knobs, the resolved config) ends in LABEL_DONE, or in LABEL_FAIL with exit 2 under the env's label_frac_gate
(--min-labeled-frac). --check-render re-renders a labeled row's messages with the hint and counts whether the
full render tokenizes to the splice (a diagnostic; never a drop rule). --stop-file is honoured between states.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import itertools
import json
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

from miles_agent import tool_call_markup
from miles_agent.opd import teacher_scoring
from miles_agent.r2s import common, select

MARKER = "LABEL_DONE"
FAIL_MARKER = "LABEL_FAIL"
OBJECTIVE = "suffix"  # recorded on every row and in the summary; teacher_ft refuses any other labels
WORKERS = 16  # states prefilled at once (REF label.py:378)

# REF label.py:41-46, byte for byte: the hint's frame around the teacher's steps.
REF_HEADER = (
    "[Reference walkthrough] The following actions were taken by an expert and successfully "
    "completed this exact task. Use it only as background knowledge when reading the session "
    "below; the session is a separate, independent attempt.\n"
)
REF_FOOTER = "\n[End of reference walkthrough]"


def ref_text_of(tau: dict[str, Any]) -> str:
    """The hint: tau's steps as "{delib}\\nAction: {command}" lines between REF_HEADER and REF_FOOTER (REF
    label.py:50-60 verbatim, design F4); a step without a command (a format-faulted teacher turn here) is its
    deliberation alone, or "(no action)"."""
    lines = []
    for st in tau["takeover_steps"]:
        delib = (st.get("delib") or "").strip()
        command = st.get("command")
        if command is None:
            # a turn without a tool call (Search's final <answer> turn): the text is the action
            lines.append(delib or "(no action)")
            continue
        lines.append(f"{delib}\nAction: {command}" if delib else f"Action: {command}")
    return REF_HEADER + "\n".join(lines) + REF_FOOTER


def privileged_prompt_ids(
    tokenizer, render_chat: tool_call_markup.RenderChat, prompt_tokens: list[int], hint: str
) -> list[int] | None:
    """The prompt's tokens with ``hint`` as one more user message right before the generation prompt, in token
    space (design D9): the generation prompt's ids and the message's ids come from ``render_chat`` (the row's
    family, rendered twice and diffed), the splice point is prompt_length - len(gen_ids). None when the prompt
    does not end with the generation prompt: the constant-offset alignment would not hold, so the state is
    dropped (prompt_retok_mismatch) rather than spliced at a guessed point (REF label.py:112-128 returned None
    when its re-tokenization differed). A hint with outer whitespace is refused: the templates trim a user
    message's content, so its rendered block would not be the block REF spliced; ref_text_of never builds one."""
    if hint != hint.strip():
        raise ValueError("the hint carries outer whitespace, which the chat template trims: no splice for it")
    gen_ids = tool_call_markup.generation_prompt_ids(render_chat, tokenizer)
    ins = len(prompt_tokens) - len(gen_ids)
    if ins < 0 or list(prompt_tokens[ins:]) != gen_ids:
        return None
    hint_ids = tool_call_markup.user_message_ids(render_chat, tokenizer, hint)
    return list(prompt_tokens[:ins]) + hint_ids + list(prompt_tokens[ins:])


def _topk_at(top_entries, k: int) -> tuple[list[int], list[float]]:
    """REF label.py:131-133: the first k [logprob, id, text] entries of a position as (ids, logprobs)."""
    entries = (top_entries or [])[:k]
    return [int(e[1]) for e in entries], [float(e[0]) for e in entries]


def build_positions(ep: common.Episode, div_turn: int, priv: dict, offset: int, k: int) -> list[dict[str, Any]]:
    """Every assistant token from the divergence turn to the end of the episode, once each, with the
    privileged top-k: thinking, tool call and turns without a tool call alike. kind stays "hind"
    (privileged) on every position, as teacher_ft and other label readers expect (REF label.py:224-236)."""
    positions = []
    for t in range(div_turn, len(ep.turns)):
        start, _, end = ep.abs_span(ep.turns[t])
        dist = div_turn - t
        for p in range(start, end):
            ids, lps = _topk_at(priv["top"][p + offset], k)
            if ids:
                positions.append({"p": p, "turn": t, "dist": dist, "kind": "hind", "tgt_ids": ids, "tgt_lps": lps})
    return positions


def render_check(
    state: Mapping[str, Any],
    tokenizer,
    render_chat: tool_call_markup.RenderChat,
    prompt_ids: list[int],
    priv_prompt: list[int],
    hint: str,
) -> str:
    """--check-render: a full re-render of the row's prompt messages, its reset observation and the hint as one
    more user message, encoded, against the spliced prompt (D9's claim as a count). render_equal;
    render_prompt_differs when the plain re-render already differs from the dumped prompt (the engines tokenized
    otherwise: the hint comparison would say nothing); render_hint_differs; render_unavailable on a row without
    the v2 messages (a campaign row)."""
    prompt = state.get("prompt")
    reset = list(itertools.takewhile(lambda m: m.get("role") != "assistant", common.episode_messages(state)))
    if prompt is None or not reset:
        return "render_unavailable"
    history = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    history += [{"role": m["role"], "content": m.get("content")} for m in reset]
    tools = [tool_call_markup.chat_tool(t) for t in (state.get("metadata") or {}).get("tools") or []]

    def ids(messages: list[dict[str, Any]]) -> list[int]:
        return tokenizer.encode(render_chat(messages, tools, add_generation_prompt=True), add_special_tokens=False)

    if ids(history) != list(prompt_ids):
        return "render_prompt_differs"
    hinted = history + [{"role": "user", "content": hint}]
    return "render_equal" if ids(hinted) == priv_prompt else "render_hint_differs"


def row_renderer(tokenizer, args: argparse.Namespace) -> Callable[[Mapping[str, Any]], tool_call_markup.RenderChat]:
    """row -> the engines' render of the row's launch (its tito_model and chat_template_kwargs; --tito-model and
    --chat-template-kwargs for a row that names none, F8), one RenderChat per distinct launch."""
    renders: dict[tuple[str, str], tool_call_markup.RenderChat] = {}

    def render_for(row: Mapping[str, Any]) -> tool_call_markup.RenderChat:
        tito_model = row.get("tito_model") or args.tito_model
        kwargs = row.get("chat_template_kwargs")
        if kwargs is None:
            kwargs = args.chat_template_kwargs
        if not isinstance(kwargs, str):  # a v2 row holds miles' parsed dict; the launcher's field is the JSON
            kwargs = json.dumps(kwargs, sort_keys=True)
        key = (tito_model, kwargs)
        if key not in renders:
            renders[key] = tool_call_markup.tito_render_chat(tito_model, tokenizer, kwargs)
        return renders[key]

    return render_for


async def label_state(
    state: Mapping[str, Any],
    tau: Mapping[str, Any],
    tokenizer,
    render_chat: tool_call_markup.RenderChat,
    args: argparse.Namespace,
    counts: collections.Counter,
    teacher_url: str,
    render_counts: collections.Counter | None = None,
) -> dict[str, Any] | None:
    """One state's row, or None with the drop counted (REF label.py:239-295): the plain ids to the last turn's
    end, the privileged prompt by the token splice (prompt_retok_mismatch when the prompt does not end with the
    generation prompt, or when the row's template cannot be diffed for one: the reason is printed and the stage
    goes on), one prefill of the privileged pass on ``teacher_url`` (the /generate endpoint main derived) with
    the top-k, the positions of every turn from the divergence turn on; ``render_counts`` (--check-render)
    takes the re-render diagnostic of a labeled state."""
    ep = common.episode_from_row(state, state["episode_key"])
    div_turn = int(state["div_turn"])
    if div_turn >= len(ep.turns):
        counts["bad_div_turn"] += 1
        return None
    end_abs = ep.prompt_length + ep.turns[-1].end
    plain_ids = ep.tokens[:end_abs]
    hint = ref_text_of(tau)
    prompt_ids = ep.tokens[: ep.prompt_length]
    try:
        priv_prompt = privileged_prompt_ids(tokenizer, render_chat, prompt_ids, hint)
    except ValueError as e:  # tool_call_markup._appended_ids: the row's template rewrites instead of appending
        print(f"no splice for {state['state_id']}: {e}")
        priv_prompt = None
    if priv_prompt is None:
        counts["prompt_retok_mismatch"] += 1
        return None
    offset = len(priv_prompt) - ep.prompt_length
    priv_ids = priv_prompt + ep.tokens[ep.prompt_length : end_abs]
    if len(priv_ids) > args.max_ctx:
        counts["long_ctx"] += 1
        return None
    try:
        priv = await teacher_scoring.prefill(teacher_url, priv_ids, args.k, concurrency=args.workers)
    except teacher_scoring.TeacherScoringError as e:
        counts["prefill_failed"] += 1
        print(f"prefill failed for {state['state_id']}: {e}")
        return None
    positions = build_positions(ep, div_turn, priv, offset, args.k)
    if not positions:
        counts["no_positions"] += 1
        return None
    if render_counts is not None:
        render_counts[render_check(state, tokenizer, render_chat, prompt_ids, priv_prompt, hint)] += 1
    return {
        "state_id": state["state_id"],
        "split": state["split"],
        "goal": state["goal"],
        "div_turn": div_turn,
        "objective": OBJECTIVE,
        "ids": plain_ids,
        "positions": positions,
        "meta": {
            "tau_sample": tau.get("sample"),
            "tau_steps": len(tau["takeover_steps"]),
            "tau_won": bool(tau.get("won")),
            "offset": offset,
            "hint_tokens": offset,
        },
    }


# ---- the pass ----


def tau_picker(
    args: argparse.Namespace, cfg: select.EnvConfig, by_state: Mapping[str, list[dict[str, Any]]]
) -> Callable[[Mapping[str, Any]], tuple[dict[str, Any] | None, dict[str, Any]]]:
    """state -> (tau, stats): the fold row the --tau file names for the state (stage C's selection; a named
    sample the fold lacks is a mismatch between the files, not a state without tau), or select.select_tau inline
    over the state's takeovers under the env's check with the --judge-file rows (F9)."""
    if args.tau:
        chosen = {r["state_id"]: r["sample"] for r in common.read_jsonl(args.tau)}

        def from_file(state: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            state_id = state["state_id"]
            if state_id not in chosen:
                return None, {}
            tau = next((t for t in by_state.get(state_id, []) if t["sample"] == chosen[state_id]), None)
            if tau is None:
                raise SystemExit(f"--tau names sample {chosen[state_id]} of {state_id}, which --fold does not hold")
            return tau, {}

        return from_file
    judge_rows = (
        {(v["state_id"], v["sample"]): v for v in common.read_jsonl(args.judge_file)} if args.judge_file else None
    )

    def inline(state: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return select.select_tau(by_state.get(state["state_id"], []), cfg, judge_rows)

    return inline


def summarize(
    rows: list[dict[str, Any]],
    counts: collections.Counter,
    select_stats: collections.Counter,
    render_counts: collections.Counter | None,
    args: argparse.Namespace,
    cfg: select.EnvConfig,
) -> dict[str, Any]:
    """REF label.py:342-351 over every row of the file (a resumed run's earlier rows included), the selection
    stats, the knobs and the resolved config."""
    n_states = counts["states"]
    return {
        **counts,
        "labeled": len(rows),
        "objective": OBJECTIVE,
        "labeled_frac": len(rows) / n_states if n_states else None,
        "n_positions": sum(len(r["positions"]) for r in rows),
        "goals": len({r["goal"] for r in rows}),
        "dist_hist": dict(collections.Counter(x["dist"] for r in rows for x in r["positions"])),
        "select_stats": {"source": "tau" if args.tau else "inline", **select_stats},
        **({"check_render": dict(render_counts)} if render_counts is not None else {}),
        "k": args.k,
        "max_ctx": args.max_ctx,
        "tito_model": args.tito_model,
        "min_labeled_frac": cfg.label_frac_gate,
        "config": dataclasses.asdict(cfg),
    }


async def _main_async(args: argparse.Namespace, cfg: select.EnvConfig, teacher_url: str) -> int:
    from transformers import AutoTokenizer  # the pipeline job's image (REF label.py:299-301)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    states = common.read_jsonl(args.states)
    if args.limit is not None:
        states = states[: args.limit]
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in common.read_jsonl(args.fold):
        by_state[row["state_id"]].append(row)
    pick = tau_picker(args, cfg, by_state)
    if args.force:
        open(args.out, "w", encoding="utf-8").close()  # every state again: the labels are appended to
    done = common.existing_keys(args.out, ("state_id",))
    counts = collections.Counter(
        states=len(states),
        resumed=0,
        tau_found=0,
        no_tau=0,
        bad_div_turn=0,
        prompt_retok_mismatch=0,
        long_ctx=0,
        prefill_failed=0,
        no_positions=0,
    )
    select_stats: collections.Counter = collections.Counter()
    render_counts = collections.Counter() if args.check_render else None
    render_for = row_renderer(tokenizer, args)
    sem = asyncio.Semaphore(args.workers)  # states in flight: what a raised stop file lets finish
    stopped: list[str] = []

    async def one(state: dict[str, Any]) -> None:
        if (state["state_id"],) in done:
            counts["resumed"] += 1
            return
        tau, stats = pick(state)
        if "vote_n_eligible" in stats:
            select_stats["eligible_total"] += stats["vote_n_eligible"]
        if "judge_missing" in stats:
            select_stats["judge_missing"] += stats["judge_missing"]
        if tau is None:
            counts["no_tau"] += 1
            return
        counts["tau_found"] += 1
        async with sem:
            if common.stop_requested(args.stop_file):
                stopped.append(state["state_id"])
                return
            row = await label_state(state, tau, tokenizer, render_for(state), args, counts, teacher_url, render_counts)
        if row is not None:
            common.append_jsonl(args.out, row)

    await asyncio.gather(*(one(s) for s in states))
    if stopped:
        print(f"stop file {args.stop_file} raised: {len(stopped)} states not labeled; exiting without {MARKER}")
        return 0
    rows = common.read_jsonl(args.out) if os.path.exists(args.out) else []
    summary = summarize(rows, counts, select_stats, render_counts, args, cfg)
    if states and summary["labeled_frac"] < cfg.label_frac_gate:
        common.write_summary(args.summary, summary, FAIL_MARKER)
        return 2
    common.write_summary(args.summary, summary, MARKER)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = common.stage_parser(__doc__)
    ap.add_argument("--states", required=True, help="stage A rows")
    ap.add_argument("--fold", required=True, help="stage B rows")
    source = ap.add_mutually_exclusive_group()
    source.add_argument(
        "--tau", default=None, help="stage C rows (state_id, sample); without them the rule runs inline"
    )
    source.add_argument("--judge-file", default=None, help="stage J rows for the inline rule under check judge")
    ap.add_argument("--tokenizer-path", required=True, help="the student's HF dir: the tokenizer of the prompt ids")
    ap.add_argument("--teacher-url", required=True, help="sglang base URL of the frozen base teacher")
    ap.add_argument(
        "--tito-model",
        default=None,
        help="the splice's TITO family for a row that names none (default: the dir's row)",
    )
    ap.add_argument(
        "--chat-template-kwargs",
        default=None,
        help="its --apply-chat-template-kwargs JSON for a row that names none (default: the dir's row)",
    )
    ap.add_argument("--k", type=int, default=select.LABEL_K, help="top-k targets per position")
    ap.add_argument(
        "--max-ctx", type=int, default=select.MAX_CTX, help="longer privileged passes are counted and skipped"
    )
    ap.add_argument("--workers", type=int, default=WORKERS, help="states prefilled at once")
    ap.add_argument(
        "--min-labeled-frac",
        type=float,
        default=None,
        help=f"{FAIL_MARKER} (exit 2) under this labeled fraction (default: the env's label_frac_gate)",
    )
    ap.add_argument(
        "--check-render",
        action="store_true",
        help="count whether a full re-render of the messages with the hint equals the splice (diagnostic)",
    )
    args = common.parse_stage_args(ap, argv)
    cfg = select.load_env_config(args.env, args.env_config, label_frac_gate=args.min_labeled_frac)
    # the flag is the server's base URL; the prefill posts to /generate (teacher_scoring.prefill takes the
    # endpoint, as miles' --opd-teacher-urls carries it): derived once, a URL with a path refused at the flag
    teacher_url = common.generate_url(args.teacher_url, "--teacher-url")
    if cfg.check == "judge" and not (args.tau or args.judge_file):
        raise SystemExit(
            f"--env {args.env} selects with check 'judge': --tau (stage C's rows) or --judge-file is required"
        )
    if common.skip_if_done(args, MARKER):
        return 0
    common.family_defaults(args)
    return asyncio.run(_main_async(args, cfg, teacher_url))


if __name__ == "__main__":
    sys.exit(main())
