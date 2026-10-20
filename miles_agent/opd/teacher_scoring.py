"""Score a finished episode on the standalone sglang teacher: the sampled-token log-probs
miles' OPD loss reads (Sample.teacher_log_probs, one per response token).

The student's merged token stream (prompt + every assistant turn + every injected
observation) goes to the teacher's /generate byte-identically as input_ids with
max_new_tokens 0 and return_logprob - miles' own scoring request
(rollout/on_policy_distillation.py) - so the teacher sees exactly the student's context,
the student's own earlier turns included (the paper's separately served teacher).
Observation tokens (loss_mask 0) are not scored: 0.0, which the loss masks out anyway.
A teacher token id that differs from the student's at a scored position is a tokenizer or
template drift and is never trained on.

Endpoint contract (sglang /generate, the one miles' stock OPD reward reads):
  meta_info.input_token_logprobs[i] = [logprob, token_id, text] for input position i
  meta_info.input_top_logprobs[i]   = [[logprob, token_id, text] * k]   (prefill, top_k > 0)
  position 0 has no prediction and is None (some versions omit it entirely).

miles' helpers are imported lazily: they pull torch, which only the actors and the
pipeline's stage servers have.
"""

import asyncio
import logging
import random
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_BACKOFF_S = (2.0, 10.0)
_semaphores: dict[int, asyncio.Semaphore] = {}


class TeacherScoringError(RuntimeError):
    """The teacher could not score this stream: a transport failure, a malformed or misaligned
    reply, or (``ctx_overflow``) its refusal of the request's length."""

    def __init__(self, message: str, *, ctx_overflow: bool = False):
        super().__init__(message)
        self.ctx_overflow = ctx_overflow


def score_payload(input_ids: list[int], top_k: int = 0) -> dict[str, Any]:
    """miles' scoring request (_score_payload): input_ids, max_new_tokens 0, log-probs from
    position 0, top_logprobs_num only when top_k > 0."""
    from miles.rollout.on_policy_distillation import _score_payload

    return _score_payload(list(input_ids), top_k)


def _semaphore(concurrency: int) -> asyncio.Semaphore:
    key = id(asyncio.get_running_loop())
    if key not in _semaphores:
        _semaphores[key] = asyncio.Semaphore(concurrency)
    return _semaphores[key]


async def post_json(
    url: str, payload: dict[str, Any], *, timeout_s: float = 900.0, attempts: int = 4, concurrency: int = 32
) -> dict[str, Any]:
    """miles' _post_json under a per-loop semaphore of ``concurrency`` calls in flight. Transport
    failures, timeouts and 5xx are retried ``attempts`` times with a 2-10 s backoff; a 4xx is
    final. sglang answers 400 to a /generate whose input exceeds its context window: with this
    fixed, well-formed payload that is the only 400, hence ``ctx_overflow``."""
    from miles.rollout.on_policy_distillation import _post_json

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            async with _semaphore(concurrency):
                return await _post_json(url, payload, timeout_secs=timeout_s)
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                raise TeacherScoringError(
                    f"teacher {url} answered {e.status}: {e.message}", ctx_overflow=e.status == 400
                ) from e
            last = e
        except (aiohttp.ClientError, TimeoutError, ValueError) as e:
            last = e
        if attempt < attempts - 1:
            await asyncio.sleep(random.uniform(*_BACKOFF_S))
    raise TeacherScoringError(f"teacher {url} unreachable after {attempts} attempts: {last!r}") from last


def _align_input_field(values: list[Any] | None, field: str, total_length: int) -> list[Any]:
    """One entry per input position (position 0 = None), whatever the server's convention
    (REF miles_agent/opd/teacher_scoring.py:90-98)."""
    if values is None:
        raise TeacherScoringError(f"teacher reply lacks meta_info.{field}")
    if len(values) == total_length:
        return values
    if len(values) == total_length - 1:
        return [None, *values]
    raise TeacherScoringError(f"meta_info.{field} has {len(values)} entries for {total_length} input tokens")


def sampled_logprobs(
    reply: dict[str, Any], tokens: list[int], response_length: int, loss_mask: list[int] | None
) -> list[float]:
    """The teacher's log-prob of every response token: 0.0 where loss_mask is 0 or at input
    position 0 (REF parse_topk's zero-fill), the reply's value elsewhere, after the token-id
    check. The scored values are miles' _trim_input_field(...)[r][0]."""
    meta = reply.get("meta_info") or {}
    total = len(tokens)
    prompt_length = total - response_length
    token_lps = _align_input_field(meta.get("input_token_logprobs"), "input_token_logprobs", total)
    mask = loss_mask if loss_mask is not None else [1] * response_length
    out: list[float] = []
    for r in range(response_length):
        pos = prompt_length + r
        if not mask[r] or pos == 0:
            out.append(0.0)
            continue
        entry = token_lps[pos]
        if entry is None or int(entry[1]) != int(tokens[pos]):
            got = None if entry is None else int(entry[1])
            raise TeacherScoringError(f"teacher token id {got} != student token {tokens[pos]} at position {pos}")
        out.append(float(entry[0]))
    return out


async def score_sample(args, sample) -> dict[str, float]:
    """One prefill of ``sample.tokens`` on the launch's teacher (--opd-teacher-urls default=...,
    the launcher's --teacher-url) into sample.teacher_log_probs; the metrics for env_metrics.
    Raises TeacherScoringError with the sample left unscored."""
    from miles.rollout.on_policy_distillation import parse_teacher_urls

    url = parse_teacher_urls(args.opd_teacher_urls)["default"]
    t0 = time.monotonic()
    reply = await post_json(
        url,
        score_payload(sample.tokens),
        timeout_s=args.opd_score_timeout_s,
        attempts=args.opd_score_attempts,
        concurrency=args.opd_score_concurrency,
    )
    log_probs = sampled_logprobs(reply, sample.tokens, sample.response_length, sample.loss_mask)
    assert len(log_probs) == sample.response_length  # what Sample.validate and the loss hub check
    sample.teacher_log_probs = log_probs
    return {"opd_teacher_scoring_failed": 0.0, "opd_teacher_latency_s": time.monotonic() - t0}


async def prefill(url: str, ids: list[int], top_k: int = 0, **post_kwargs) -> dict[str, list]:
    """The server's log-prob of every token of ``ids`` under its own prefix, for the R2S stages
    (A: student and teacher per episode, D: the top-k targets): {"lps": [None|float]*len(ids),
    "top": [None|[[lp, id, text]...]]*len(ids)}, position 0 having no prediction. The server's
    token id must equal ours at every position >= 1 (REF miles_agent/r2s/common.py:219-245)."""
    reply = await post_json(url, score_payload(ids, top_k), **post_kwargs)
    meta = reply.get("meta_info") or {}
    total = len(ids)
    token_lps = _align_input_field(meta.get("input_token_logprobs"), "input_token_logprobs", total)
    top = (
        _align_input_field(meta.get("input_top_logprobs"), "input_top_logprobs", total)
        if top_k > 0
        else [None] * total
    )
    lps: list[float | None] = [None] * total
    for pos in range(1, total):
        entry = token_lps[pos]
        if entry is None:
            raise TeacherScoringError(f"no logprob at position {pos}")
        if int(entry[1]) != int(ids[pos]):
            raise TeacherScoringError(f"server token {int(entry[1])} != ours {ids[pos]} at position {pos}")
        lps[pos] = float(entry[0])
    return {"lps": lps, "top": top}
