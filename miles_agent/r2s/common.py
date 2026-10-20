"""Shared helpers of the R2S stages: dump rows in token space, the structured transcript, jsonl io,
summaries with their DONE marker, the resume primitives and the flags every stage shares.

A dump row (miles_agent/opd/dump.py) carries the merged token stream (tokens, response_length,
loss_mask, turns) and, from v2 on, the chat messages the episode was made of: ``prompt`` (the
row's messages, role/content) and ``transcript`` (what env_loop_agent built: the prompt messages,
the reset observation, then every assistant message with its tool calls and every observation).
The token side feeds stages A and D, the transcript feeds stage B's replay. REF's chat-text
parsers and regexes (the campaign branch's miles_agent/r2s/common.py:96-213) have no successor:
the markup belongs to the model.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from miles_agent.opd.turns import Turn


@dataclass(frozen=True)
class Episode:
    """One dump row (REF common.py:17-40) plus the v2 fields, None on campaign v1 rows."""

    key: str  # "<file basename>:<line>"
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    turns: list[Turn]
    split: str
    metadata: dict[str, Any]
    reward: float | None
    status: str
    prompt: list[dict[str, Any]] | str | None
    transcript: list[dict[str, Any]] | None
    tito_model: str | None
    chat_template_kwargs: str | dict[str, Any] | None

    @property
    def prompt_length(self) -> int:
        return len(self.tokens) - self.response_length

    def abs_span(self, turn: Turn) -> tuple[int, int, int]:
        """(start, delib_end, end) as absolute token positions."""
        return self.prompt_length + turn.start, self.prompt_length + turn.delib_end, self.prompt_length + turn.end


def episode_from_row(row: dict[str, Any], key: str) -> Episode:
    """``turns`` are the dump's [start, end, delib_end] triples, [] when the row has none (stage A
    re-segments such rows with the student tokenizer, design F8); loss_mask defaults to all ones."""
    response_length = int(row["response_length"])
    metadata = row.get("metadata") or {}
    return Episode(
        key=key,
        tokens=list(row["tokens"]),
        response_length=response_length,
        loss_mask=list(row["loss_mask"]) if row.get("loss_mask") is not None else [1] * response_length,
        turns=[Turn(start=s, end=e, delib_end=d) for s, e, d in row.get("turns") or []],
        split=str(metadata.get("split", "train")),
        metadata=metadata,
        reward=row.get("reward"),
        status=str(row.get("status", "completed")),
        prompt=row.get("prompt"),
        transcript=row.get("transcript"),
        tito_model=row.get("tito_model"),
        chat_template_kwargs=row.get("chat_template_kwargs"),
    )


def read_dump(pattern: str, limit: int | None = None) -> list[Episode]:
    """Every row of the dump files matching ``pattern`` (sorted), keyed "<basename>:<line>"; a row
    cut by a killed job or without tokens is skipped (REF common.py:73-91)."""
    episodes: list[Episode] = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a row cut by a killed job
                if not row.get("tokens"):
                    continue
                episodes.append(episode_from_row(row, f"{base}:{i}"))
                if limit is not None and len(episodes) >= limit:
                    return episodes
    return episodes


# ---- the structured transcript ----


def episode_messages(row: dict[str, Any], transcript: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The messages after the row's prompt: the reset observation, then the policy's assistant
    messages, each followed by its observation (a tool reply, or a user message after a turn
    without a call). env_loop_agent re-emits the prompt messages one for one before the reset
    observation (gym_root/agent_utils/chat_dialect.py to_chat_messages), so the split is the
    prompt's length: a one-shot prompt's own demonstration turns stay inside it, and "the first
    assistant message" is not the split. ``transcript`` stands in for the row's own (a takeover's
    /run reply, checked against the same prompt)."""
    messages = transcript if transcript is not None else row.get("transcript")
    prompt = row.get("prompt")
    n_prompt = 1 if isinstance(prompt, str) else len(prompt or [])  # gym_bridge.build_run_body's bare string
    return list(messages or [])[n_prompt:]


def policy_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The assistant messages the policy produced, in order ([] on a row without a transcript):
    stage B's scripted_prefix is a slice of them."""
    return [m for m in episode_messages(row) if m.get("role") == "assistant"]


def observations_after_prefix(
    row: dict[str, Any], n: int, transcript: list[dict[str, Any]] | None = None
) -> list[str]:
    """The observation texts from the reset observation through the one after policy turn n - 1:
    what a replay of the first n turns must reproduce (stage B's replay_ok)."""
    out: list[str] = []
    seen = 0
    for msg in episode_messages(row, transcript):
        if msg.get("role") == "assistant":
            if seen == n:
                break
            seen += 1
        else:
            out.append(str(msg.get("content") or ""))
    return out


def norm_text(text: str) -> str:
    return " ".join(text.split())


def fingerprint(text: str) -> str:
    """The vote key of a terminal observation (REF common.py:189-190)."""
    return hashlib.sha1(norm_text(text).encode()).hexdigest()[:12]


# ---- jsonl, summaries, resume ----


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    """One row appended and closed: stages B and D keep every finished state across a preempt."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str, summary: dict[str, Any], marker: str) -> None:
    """The summary with ``marker: true`` as JSON, then the marker alone on the last line (the job
    greps it); both printed as well (REF common.py:263-268)."""
    text = json.dumps({**summary, marker: True}, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{text}\n{marker}\n")
    print(text)
    print(marker)


def read_summary(path: str) -> dict[str, Any] | None:
    """The summary's JSON, the trailing marker line dropped; None without a file or with a cut one."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text[: text.rfind("}") + 1])
    except json.JSONDecodeError:
        return None


def summary_done(path: str, marker: str) -> bool:
    return bool((read_summary(path) or {}).get(marker))


def existing_keys(path: str, keys: tuple[str, ...]) -> set[tuple[Any, ...]]:
    """The ``keys`` tuples of the rows already in ``path`` (none without the file), so a restarted
    stage skips them. A last line without its newline was cut by a kill whether or not it parses
    (append_jsonl would glue the next row onto a complete one and read_jsonl raise Extra data): the
    file is truncated at that line's byte offset, one syscall and never a rewrite (a kill mid-rewrite
    lost every finished row), and its row is redone; a cut line anywhere else is a corrupt file."""
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    tail = lines.pop()  # b"" when the file ends with a newline
    if tail:
        os.truncate(path, len(data) - len(tail))
    rows = [json.loads(line) for line in lines if line.strip()]
    return {tuple(r[k] for k in keys) for r in rows}


def stop_requested(stop_file: str | None) -> bool:
    """The preempt sentinel is up: finish the row in flight, flush, exit 0 without the marker."""
    return bool(stop_file) and os.path.exists(stop_file)


# ---- the stage CLI surface ----


def positive_int(text: str) -> int:
    """--limit's type: N >= 1. REF's "0 = unlimited" would make a stage run over nothing and write its
    marker (judge sliced taus[:0] under a JUDGE_DONE; read_dump(limit=0) keeps one row)."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"N >= 1, got {text}")
    return value


def stage_parser(doc: str) -> argparse.ArgumentParser:
    """The flags every stage shares (design section 4): --env / --env-config pick the EnvConfig
    (select.load_env_config), --out / --summary the files, --force reruns a stage whose summary
    already carries its marker, --limit is the smoke cap, --seed the stage's random seed and
    --stop-file the preempt sentinel."""
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    ap.add_argument("--env", required=True, help="alfworld | scienceworld | webshop, or an --env-config entry")
    ap.add_argument("--env-config", default=None, help="json table {env: {field: value}} laid over ENV_DEFAULTS")
    ap.add_argument("--out", required=True, help="the stage's jsonl")
    ap.add_argument("--summary", default=None, help="default <out>.summary.json")
    ap.add_argument("--force", action="store_true", help="rerun although --summary already carries the marker")
    ap.add_argument("--limit", type=positive_int, default=None, metavar="N", help="smoke: the first N inputs only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop-file", default="", help="preempt sentinel: stop between rows, exit 0 without the marker")
    return ap


def parse_stage_args(ap: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    args = ap.parse_args(argv)
    args.summary = args.summary or f"{args.out}.summary.json"
    random.seed(args.seed)
    return args


def skip_if_done(args: argparse.Namespace, marker: str) -> bool:
    """True, with a log line, when the summary already carries ``marker`` and --force was not
    given: the job's resume after a requeue skips the stage."""
    if args.force or not summary_done(args.summary, marker):
        return False
    print(f"{marker} already in {args.summary}; nothing to do (--force reruns)")
    return True


def base_url(url: str, flag: str, endpoint: str) -> str:
    """``flag``'s value as the server's base URL (http://host:13141, the launcher's --teacher-url rule), its
    trailing slash dropped so the stage can append ``endpoint``; a URL that carries a path, a query or a
    fragment is refused as launcher._validate_opd refuses it, neither doubled nor posted to as given
    (http://h:13141?x=1 would become ...?x=1/generate)."""
    parts = urlsplit(url)
    if parts.path.strip("/") or parts.query or parts.fragment:
        raise SystemExit(
            f"{flag} {url}: pass the server's base URL (http://host:13141, no path, query or fragment); "
            f"the stage appends {endpoint}"
        )
    return url.rstrip("/")


def generate_url(url: str, flag: str) -> str:
    """The sglang /generate endpoint of ``flag``'s base URL: teacher_scoring.prefill posts to the URL it is
    given, as miles' --opd-teacher-urls carries the endpoint, so stages A and D derive it once here the way
    launcher._algo_args does for the arm."""
    return f"{base_url(url, flag, '/generate')}/generate"


def family_defaults(args: argparse.Namespace) -> None:
    """--tito-model and --chat-template-kwargs default to the --tokenizer-path dir's MODEL_FAMILIES row (design
    F8: stages A and D render the student's family for a row that names none, a campaign v1 row); an unlisted
    dir needs both flags. launcher imports miles, which the job's image has, so the import is local."""
    if not args.tokenizer_path or (args.tito_model is not None and args.chat_template_kwargs is not None):
        return
    from miles_agent import launcher

    try:
        family = launcher.model_family(Path(args.tokenizer_path).name)
    except ValueError as e:
        raise SystemExit(f"--tokenizer-path: {e}") from e
    if args.tito_model is None:
        args.tito_model = family.tito_model
    if args.chat_template_kwargs is None:
        args.chat_template_kwargs = family.chat_template_kwargs
