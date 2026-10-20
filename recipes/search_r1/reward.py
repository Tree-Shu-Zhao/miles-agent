"""search_r1 reward shaping: format + budget terms on top of the env score.

Plugged into miles_agent/reward_routing.py via REWARD_SHAPING_PATH=reward.shape
(set by run.py). This file owns every wiki_search-specific assumption (the
<answer> tag convention the task prompt asks for); the library stays
env-agnostic. The tool-call markup is the model's (its chat template and the
engine's parser), so the budget term counts the calls tool_call_agent reports
(env_metrics.tool_call_count), never markup in the response text.

Coefficients (both default 0 -> shape is the identity and runs are
byte-identical to the unshaped campaign arms):

    REWARD_FORMAT_COEF   + coef if the response carries a parseable
                         <answer>...</answer> tag (format adherence)
    REWARD_BUDGET_COEF   - coef per tool call the model issued (search budget)
"""

import os
import re

_FORMAT_COEF = float(os.getenv("REWARD_FORMAT_COEF", "0"))
_BUDGET_COEF = float(os.getenv("REWARD_BUDGET_COEF", "0"))

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def shape(env_score: float, sample) -> float:
    """sample is a miles Sample; only .response and .metadata are read."""
    response = sample.response or ""
    shaped = env_score
    if _FORMAT_COEF:
        shaped += _FORMAT_COEF * float(_ANSWER_RE.search(response) is not None)
    if _BUDGET_COEF:
        shaped -= _BUDGET_COEF * float(sample.metadata.get("env_metrics", {}).get("tool_call_count", 0.0))
    return shaped
