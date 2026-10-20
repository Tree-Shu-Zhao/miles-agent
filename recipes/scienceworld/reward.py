"""scienceworld reward shaping: identity (the trained reward is the env score).

Plugged into miles_agent/reward_routing.py via REWARD_SHAPING_PATH=reward.shape
(set by run.py); this file exists so the path resolves for this recipe too, and
it owns the one scienceworld-specific fact about the score. The env score is
the episode's dense reward sum: the scienceworld_env server pays the running-max
simulator score delta / 100 per step, so the sum is TCOD's task score in [0, 1]
(best score / 100) - what eval/* reports and what GRPO trains on. Success is a
separate flag: the simulator's completion (``env.server.getCompleted()``),
reported as env/won in env_metrics - the R2S paper's metric. On ScienceWorld
the two differ (partial credit is common), so success rates are read from won,
never from the score. Keep this the identity: a shaping term would move the
trained objective away from TCOD's reward.
"""


def shape(env_score: float, sample) -> float:
    """sample is a miles Sample; unused - the env score is the trained reward."""
    return env_score
