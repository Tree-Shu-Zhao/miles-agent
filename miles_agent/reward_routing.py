"""Trainer-side reward routing: the env's verify score vs the training reward.

Env-agnostic by contract. The resources server's /verify returns `reward`
under semantics IT owns — exact match, F1, LLM-as-a-judge, anything — and
this hook never interprets it. The agent function stashed that score in
sample.metadata["reward"]; here it fans out into two keys:

    Sample.reward = {"env": score, "train": shape(score, sample)}

run.py pins --reward-key train (what the optimizer sees) and
--eval-reward-key env (what eval/* logs), so eval curves always report the
environment's own score no matter what shaping is active.
metadata["raw_reward"] = score additionally routes the env score over the
*training* distribution to rollout/episode_raw_reward, giving shaped and
unshaped curves side by side.

Shaping is a per-recipe plugin, not library code: REWARD_SHAPING_PATH names
a `f(env_score: float, sample: Sample) -> float` in module.attr form,
resolved like miles custom hooks (e.g. search_r1 sets "reward.shape" —
its recipes/search_r1/reward.py). Unset -> train == env score exactly.

Episodes whose /run failed have no metadata["reward"] and score 0.0
(counted separately in env_metrics.run_transport_failed).
"""

import os

from miles.utils.misc import load_function
from miles.utils.types import Sample

_SHAPING_PATH = os.getenv("REWARD_SHAPING_PATH")
_shape = load_function(_SHAPING_PATH) if _SHAPING_PATH else None


def _reward_dict(sample: Sample) -> dict[str, float]:
    score = float(sample.metadata.get("reward", 0.0))
    sample.metadata["raw_reward"] = score
    return {"env": score, "train": _shape(score, sample) if _shape else score}


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> dict | list[dict]:
    if isinstance(samples, list):
        return [_reward_dict(s) for s in samples]
    return _reward_dict(samples)
