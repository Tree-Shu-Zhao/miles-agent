"""webshop reward shaping: map the env-truth score to the training reward.

Plugged into miles_agent/reward_routing.py via REWARD_SHAPING_PATH=reward.shape
(set by run.py). The env's /step reward (and thus env_score here) is the RAW
WebShop score in [0,1] at the buy step - the neutral env truth that eval reads
(--eval-reward-key env). Unlike alfworld (whose env score already IS its binary
training reward), WebShop's env-truth score and its RL reward genuinely differ,
so this shape is NOT the identity even at defaults; it gives the standard
WebShop training reward (10 iff perfect purchase).

    train = SCALE * base - penalties
      base = task_score          if WEBSHOP_DENSE_REWARD=1  (Turn-PPO, 2512.17008)
           = won (0/1)           otherwise  (10-iff-perfect, the default)
      SCALE = WEBSHOP_REWARD_SCALE                      (default 10.0)
      penalties = WEBSHOP_INVALID_PENALTY  * invalid_actions
                + WEBSHOP_FORMAT_PENALTY   * action_format_errors
                + WEBSHOP_STEP_PENALTY     * step_count

won / task_score / the counters ride in sample.metadata["env_metrics"]
(env_loop_agent merged the terminal /step info; gym_bridge forwarded it).
WEBSHOP_FORMAT_PENALTY defaults to 0.1: on the same canonical held-out 500,
wstrain4 (0.1) beat wstrain3 (0) 0.820 vs 0.794 won and drove
action_format_errors from ~9/episode to 0 by step ~40, where the unpenalized
run still made ~1/episode at step 250. Set it to 0 for the unpenalized arm.
The other coefficients default 0; e.g. set WEBSHOP_INVALID_PENALTY=0.1 to
penalize inadmissible actions as well. The WEBSHOP_*
knobs reach this module through the ray runtime env (run.py forwards every
WEBSHOP_* var from the job environment); the resolved values are logged once
at import so a penalty arm is visible in the driver log.
"""

import logging
import os

logger = logging.getLogger(__name__)

_SCALE = float(os.getenv("WEBSHOP_REWARD_SCALE", "10"))
_DENSE = os.getenv("WEBSHOP_DENSE_REWARD", "0") == "1"
_INVALID_PENALTY = float(os.getenv("WEBSHOP_INVALID_PENALTY", "0"))
_FORMAT_PENALTY = float(os.getenv("WEBSHOP_FORMAT_PENALTY", "0.1"))
_STEP_PENALTY = float(os.getenv("WEBSHOP_STEP_PENALTY", "0"))

logger.info(
    "webshop reward shaping: scale=%s dense=%s invalid_penalty=%s format_penalty=%s step_penalty=%s",
    _SCALE,
    _DENSE,
    _INVALID_PENALTY,
    _FORMAT_PENALTY,
    _STEP_PENALTY,
)


def shape(env_score: float, sample) -> float:
    """sample is a miles Sample; env_score is the raw WebShop task_score."""
    em = getattr(sample, "metadata", {}).get("env_metrics", {})
    won = float(em.get("won", 0.0))
    task_score = float(em.get("task_score", env_score))
    base = task_score if _DENSE else won
    shaped = _SCALE * base
    if _INVALID_PENALTY:
        shaped -= _INVALID_PENALTY * float(em.get("invalid_actions", 0.0))
    if _FORMAT_PENALTY:
        shaped -= _FORMAT_PENALTY * float(em.get("action_format_errors", 0.0))
    if _STEP_PENALTY:
        shaped -= _STEP_PENALTY * float(em.get("step_count", 0.0))
    return shaped
