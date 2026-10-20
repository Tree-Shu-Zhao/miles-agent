"""Stage J of the R2S pipeline: the WebShop judge, one MATCH / MISMATCH verdict per purchase.

WebShop's vote key is the purchased ASIN, and agreeing takeovers often agree on the wrong item or the wrong
options, so the paper's WebShop check is an LLM judge (appendix: Qwen3.5-9B with reasoning mode on, for both
teachers, D15). For every eligible takeover of the fold (select.eligible: replayed, done, an ASIN, at least one
step, format faults admitted under allow_invalid) the judge gets the shopping instruction and the product page
exactly as it looked when the agent clicked Buy Now (selected options carry [clicked button]) plus up to two
description/features pages the takeover opened, and answers with one word; ``match_ok`` admits the takeover to
the vote (select.select_tau with check judge, F9). The prompt, the page parsers (WebShop observation text, the
environment's output) and the verdict parser are REF's byte for byte (the campaign branch's
miles_agent/r2s/judge.py:72-94, 190-249, 567-645). The request is one chat completion at temperature 0 with
``chat_template_kwargs {"enable_thinking": true}``: the served fixed template defaults thinking off and lets an
explicit request value win (D6). The verdict is the last verdict word of the reasoning and the content joined,
in that order: the campaign's teacher server ran without a reasoning parser, so its ``message.content`` carried
the <think> reasoning and REF's regex read the whole reply, and under the 2048-token budget 4,768 of the
campaign's 6,232 WebShop verdicts were the last verdict word of a reasoning cut at max_tokens. This
server's --reasoning-parser moves the reasoning to ``message.reasoning_content``, so the two fields are joined
before the regex (a content-only read left 59 of the smoke's 86 verdicts unparsed). REF's Search graders, the
comparative ``pick``, the ALFWorld/ScienceWorld trace judge and ``--report`` are not ported.

    python -m miles_agent.r2s.judge --env webshop --fold fold.jsonl --judge-url http://127.0.0.1:13143 \\
        --judge-model judge --out judge.jsonl

Row (REF judge.py:636-645): {state_id, sample, match_ok, match_text (the reply's last 200 chars), n_unparsed,
asin, selected}; a takeover without a Buy Now step gets match_ok false and match_text "no buy step". Rows are
appended per verdict in completion order, as stages B and D append theirs (a restart skips the (state_id,
sample) pairs present unless --force; --limit applies before the skip). Summary over the file's rows: {mode
"match", takeovers_judged, match_ok, unparsed_calls, thinking, judge_model, max_tokens (the budget sent:
VERDICT_TOKENS under --no-thinking), fold_takeovers, config} and JUDGE_DONE; --stop-file is honoured between
verdicts and leaves the rows written so far.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

import httpx

from miles_agent.r2s import common, select

MARKER = "JUDGE_DONE"
CHAT_ENDPOINT = "/v1/chat/completions"
TIMEOUT_S = 300.0  # per request (REF judge.py:758)
RETRY_BACKOFF_S = 2.0  # 2 s, 4 s, 6 s between the three attempts (REF :586)
VERDICT_TOKENS = 16  # the budget of a verdict-only reply under --no-thinking (REF :574)

# REF judge.py:72-94, byte for byte: the rubric and the parsers of WebShop's observation text.
MATCH_SYSTEM = (
    "You audit an online-shopping agent. You get the shopping instruction and the product page exactly "
    "as it looked when the agent clicked Buy Now: selected options are marked [clicked button], "
    "unselected ones [button]; a description or features page follows when the agent opened one. "
    "Decide whether this purchase satisfies EVERY requirement of the instruction: the product type, "
    "each listed attribute, each requested option (it must be selected, not merely available), and the "
    "price limit (a price range counts by its lowest price). Reply with exactly one word: MATCH or MISMATCH."
)
VERDICT_WORDS = {
    "SUPPORTED": True,
    "UNSUPPORTED": False,
    "USEFUL": True,
    "USELESS": False,
    "MATCH": True,
    "MISMATCH": False,
    "COMPLETED": True,
    "INCOMPLETE": False,
}
WS_INSTRUCTION_RE = re.compile(r"Instruction:\s*\n(.*?)\n\s*\[button\] Back to Search", re.S)
WS_STEP_PREFIX_RE = re.compile(r"^You are now at step \d+ and your current observation is:\s*", re.S)
WS_ADMISSIBLE_RE = re.compile(r"\n?Your admissible actions of the current situation are:.*$", re.S)
WS_PAGE_CHARS = 3000


def parse_verdict(text: str) -> bool | None:
    """Last verdict word in the reply (thinking models may reason first)."""
    words = re.findall(
        r"\b(MISMATCH|MATCH|SUPPORTED|UNSUPPORTED|USEFUL|USELESS|COMPLETED|INCOMPLETE)\b", (text or "").upper()
    )
    return VERDICT_WORDS[words[-1]] if words else None


def webshop_instruction(obs: str | None) -> str:
    m = WS_INSTRUCTION_RE.search(obs or "")
    if m:
        return " ".join(m.group(1).split())
    m = re.search(r"Instruction:\s*(.*?)(?:\n\[button\]|$)", obs or "", flags=re.S)
    return " ".join(m.group(1).split()) if m else ""


def webshop_page(obs: str | None) -> str:
    """A WebShop observation without the step banner and the admissible-action list."""
    text = WS_ADMISSIBLE_RE.sub("", WS_STEP_PREFIX_RE.sub("", obs or "")).strip()
    return text[:WS_PAGE_CHARS]


def _cmd(step: dict[str, Any]) -> str:
    return str(step.get("command") or "").strip().lower()


def webshop_purchase(tau: dict[str, Any]) -> dict[str, Any] | None:
    """The page at Buy Now plus any description/features page this takeover opened."""
    steps = tau.get("takeover_steps") or []
    buy = next((i for i in range(len(steps) - 1, -1, -1) if _cmd(steps[i]) == "click[buy now]"), None)
    if buy is None:
        return None
    page = webshop_page(steps[buy].get("obs_before"))
    extras = []
    for i, st in enumerate(steps[:buy]):
        if _cmd(st) in ("click[description]", "click[features]") and i + 1 < len(steps):
            extras.append((_cmd(st)[6:-1], webshop_page(steps[i + 1].get("obs_before"))[:1500]))
    instruction = webshop_instruction(steps[buy].get("obs_before")) or next(
        (webshop_instruction(st.get("obs_before")) for st in steps if webshop_instruction(st.get("obs_before"))), ""
    )
    selected = re.findall(r"\[clicked button\]\s*(.*?)\s*\[clicked button_\]", page)
    title_price = re.search(r"\n([^\n]*)\nPrice: ([^\n]*)", page)
    return {
        "instruction": instruction,
        "page": page,
        "extras": extras[:2],
        "selected": selected,
        "title": title_price.group(1).strip() if title_price else "",
        "price": title_price.group(2).strip() if title_price else "",
        "asin": tau.get("terminal_key") or "",
    }


def match_messages(purchase: dict[str, Any]) -> list[dict[str, str]]:
    """REF judge.py:240-249 for the campaign's call (no prompt-induced reasoning: the judge's template
    thinking is on, D15)."""
    extras = "".join(f"\n\n{k.capitalize()} page the agent opened:\n{v}" for k, v in purchase["extras"])
    user = (
        f"Instruction: {purchase['instruction']}\n\nProduct page at Buy Now:\n{purchase['page']}{extras}"
        "\n\nMATCH or MISMATCH?"
    )
    return [{"role": "system", "content": MATCH_SYSTEM}, {"role": "user", "content": user}]


# ---- the requests ----


def match_payload(messages: list[dict[str, str]], judge_model: str, max_tokens: int, thinking: bool) -> dict:
    """The chat completion of one verdict (REF :569-576): temperature 0, the reasoning budget with thinking
    on, a verdict-only budget without; enable_thinking rides as a chat-template kwarg, which the served fixed
    template honours over its own default (D6)."""
    return {
        "model": judge_model,
        "messages": messages,
        "temperature": select.JUDGE_T,
        "max_tokens": max_tokens if thinking else VERDICT_TOKENS,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }


async def ask(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> tuple[bool | None, str]:
    """One verdict, three attempts (REF :571-587): the parsed verdict and the reply's last 200 characters,
    or None with the last error after the third failed attempt (an unparsed call either way). The reply is the
    reasoning (``reasoning_content``, what a --reasoning-parser split off; the campaign's server had none and
    kept it in the content) followed by the content, so a reasoning cut at max_tokens still yields its last
    verdict word, as it did in the campaign."""
    err = ""
    for attempt in range(3):
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            message = r.json()["choices"][0]["message"]
            text = (message.get("reasoning_content") or "") + (message.get("content") or "")
            return parse_verdict(text), text[-200:]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            err = str(e)
            await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
    return None, f"error: {err}"


async def match_one(
    client: httpx.AsyncClient, url: str, tau: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """The row of one takeover (REF :636-645)."""
    purchase = webshop_purchase(tau)
    if purchase is None:
        return {
            "state_id": tau["state_id"],
            "sample": tau["sample"],
            "match_ok": False,
            "match_text": "no buy step",
            "n_unparsed": 1,
        }
    payload = match_payload(match_messages(purchase), args.judge_model, args.max_tokens, args.thinking)
    ok, text = await ask(client, url, payload)
    return {
        "state_id": tau["state_id"],
        "sample": tau["sample"],
        "match_ok": bool(ok),
        "match_text": text,
        "n_unparsed": int(ok is None),
        "asin": purchase["asin"],
        "selected": purchase["selected"],
    }


# ---- the pass ----


async def _main_async(args: argparse.Namespace, cfg: select.EnvConfig, url: str) -> int:
    fold = common.read_jsonl(args.fold)
    taus = [t for t in fold if select.eligible(t, cfg)]  # the takeovers select_tau will consider (REF :711)
    if args.limit is not None:
        taus = taus[: args.limit]
    if common.stop_requested(args.stop_file):
        print(f"stop file {args.stop_file} present before judging; exiting without {MARKER}")
        return 0
    if args.force:
        open(args.out, "w", encoding="utf-8").close()  # every verdict again: the rows are appended to
    done = common.existing_keys(args.out, ("state_id", "sample"))
    todo = [t for t in taus if (t["state_id"], t["sample"]) not in done]
    sem = asyncio.Semaphore(args.concurrency)
    stopped: list[tuple[Any, Any]] = []

    async def guarded(client: httpx.AsyncClient, tau: dict[str, Any]) -> None:
        async with sem:
            if common.stop_requested(args.stop_file):
                stopped.append((tau["state_id"], tau["sample"]))
                return
            row = await match_one(client, url, tau, args)
        common.append_jsonl(args.out, row)  # per verdict: a preempt keeps every verdict already given

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        await asyncio.gather(*(guarded(client, t) for t in todo))
    if stopped:
        print(f"stop file {args.stop_file} raised: {len(stopped)} takeovers not judged; exiting without {MARKER}")
        return 0
    rows = common.read_jsonl(args.out) if os.path.exists(args.out) else []
    summary = {
        "mode": "match",
        "takeovers_judged": len(rows),
        "match_ok": sum(bool(r["match_ok"]) for r in rows),
        "unparsed_calls": sum(r["n_unparsed"] for r in rows),
        "thinking": bool(args.thinking),
        "judge_model": args.judge_model,
        "max_tokens": args.max_tokens if args.thinking else VERDICT_TOKENS,  # the budget match_payload sent
        "fold_takeovers": len(fold),
        "config": dataclasses.asdict(cfg),
    }
    common.write_summary(args.summary, summary, MARKER)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = common.stage_parser(__doc__)
    ap.add_argument("--fold", required=True, help="stage B rows")
    ap.add_argument(
        "--judge-url", required=True, help="sglang base URL of the judge (the teacher's own server, or its own one)"
    )
    ap.add_argument("--judge-model", required=True, help="the judge server's --served-model-name")
    ap.add_argument("--max-tokens", type=int, default=select.JUDGE_MAX_TOKENS, help="per verdict, reasoning included")
    ap.add_argument("--concurrency", type=int, default=select.JUDGE_CONCURRENCY, help="verdicts in flight")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per request")
    ap.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        default=select.JUDGE_THINKING,
        help=f"the verdict word alone, no reasoning ({VERDICT_TOKENS} tokens)",
    )
    args = common.parse_stage_args(ap, argv)
    cfg = select.load_env_config(args.env, args.env_config)
    if cfg.vote_key != "purchased_asin":
        raise SystemExit(f"--env {args.env} groups takeovers by {cfg.vote_key}: the match judge reads purchases")
    # the flag is the server's base URL (the launcher's --teacher-url rule); the stage appends the chat endpoint
    url = common.base_url(args.judge_url, "--judge-url", CHAT_ENDPOINT) + CHAT_ENDPOINT
    if common.skip_if_done(args, MARKER):
        return 0
    return asyncio.run(_main_async(args, cfg, url))


if __name__ == "__main__":
    sys.exit(main())
