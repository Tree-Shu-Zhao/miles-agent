"""Turn segmentation of a merged multi-turn sample (token space, no tokenizer needed).

miles assembles one contiguous Sample per episode: prompt tokens, then the response =
assistant turn 1, injected observation 1 (loss_mask 0), assistant turn 2, ... Every
maximal run of loss_mask 1 is therefore one assistant turn. Within a turn, the tokens
before the marker that opens a tool call are the model's deliberation (the R2S paper's
"reasoning span"); the rest is the action span (the call's markup and arguments). The
marker is the model's own (tool_call_markup.marker_token_id), never a literal id.

Both counts are protocol metrics: the paper filters R2S mining on a minimum deliberation
length and reports how often the student deliberates at all.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    start: int  # response-relative start (inclusive)
    end: int  # response-relative end (exclusive)
    delib_end: int  # response-relative index of the first marker token, or `end`

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def delib_tokens(self) -> int:
        return self.delib_end - self.start

    @property
    def has_action(self) -> bool:
        return self.delib_end < self.end


def segment_turns(response_tokens: list[int], loss_mask: list[int] | None, marker_id: int) -> list[Turn]:
    """Split the response into assistant turns (loss_mask runs of 1) with deliberation spans
    (REF miles_agent/opd/turns.py:66-89, the marker handed in instead of a token-id constant)."""
    n = len(response_tokens)
    mask = loss_mask if loss_mask is not None else [1] * n
    assert len(mask) == n, f"loss_mask length {len(mask)} != response length {n}"
    turns: list[Turn] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i
        delib_end = end
        for j in range(start, end):
            if response_tokens[j] == marker_id:
                delib_end = j
                break
        turns.append(Turn(start=start, end=end, delib_end=delib_end))
    return turns


def turn_metrics(turns: list[Turn]) -> dict[str, float]:
    """Numeric per-episode metrics for sample.metadata["env_metrics"] (-> wandb rollout/env/opd_*)."""
    if not turns:
        return {
            "opd_turns": 0.0,
            "opd_delib_tokens_mean": 0.0,
            "opd_turns_with_delib_frac": 0.0,
            "opd_turns_with_action_frac": 0.0,
        }
    delib = [t.delib_tokens for t in turns]
    return {
        "opd_turns": float(len(turns)),
        "opd_delib_tokens_mean": sum(delib) / len(delib),
        "opd_turns_with_delib_frac": sum(1 for d in delib if d > 0) / len(delib),
        "opd_turns_with_action_frac": sum(1 for t in turns if t.has_action) / len(turns),
    }
