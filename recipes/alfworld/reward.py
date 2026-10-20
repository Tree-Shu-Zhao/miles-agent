"""alfworld reward shaping: format + step-budget terms on top of the env score.

Plugged into miles_agent/reward_routing.py via REWARD_SHAPING_PATH=reward.shape
(set by run.py). This file owns every alfworld-specific assumption (the
env_metrics counters the alfworld_env server reports per episode:
step_count, action_format_errors); the library stays env-agnostic. Both terms
read those structured counters, never the response text: the tool-call markup
belongs to the model (its chat template and the engine's parser), and the
merged response also carries the env's observations — a markup regex over it
counted the env's own retry hint as a well-formed call.

Coefficients (both default 0 -> shape is the identity and runs are
byte-identical to the unshaped campaign arms):

    ALFWORLD_FORMAT_COEF   + coef x the share of env steps whose reply carried
                           a usable act tool call (format adherence)
    ALFWORLD_STEP_PENALTY  - coef per env step taken in the episode
                           (efficiency budget)
"""

import os

_FORMAT_COEF = float(os.getenv("ALFWORLD_FORMAT_COEF", "0"))
_STEP_PENALTY = float(os.getenv("ALFWORLD_STEP_PENALTY", "0"))


def shape(env_score: float, sample) -> float:
    """sample is a miles Sample; only .metadata is read."""
    env_metrics = sample.metadata.get("env_metrics", {})
    steps = float(env_metrics.get("step_count", 0.0))
    shaped = env_score
    # A format fault still spends a step (session_gym), so errors <= steps.
    if _FORMAT_COEF and steps > 0:
        shaped += _FORMAT_COEF * (1.0 - float(env_metrics.get("action_format_errors", 0.0)) / steps)
    if _STEP_PENALTY:
        shaped -= _STEP_PENALTY * steps
    return shaped
