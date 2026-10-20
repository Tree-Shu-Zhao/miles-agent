"""On-policy distillation of the env recipes: the R2S paper's student arm and its pilot dump.

The loss is miles' own sampled OPD (--use-opd, launcher algo opd); this package is the
per-sample glue around it:
- generate:        the generate hook over miles' agentic episode: turn metrics, the pilot
                   dump row, the teacher scoring into Sample.teacher_log_probs.
- teacher_scoring: one prefill of the student's token stream on the sglang teacher, aligned
                   and zero-filled to the response; prefill() for the R2S stages.
- turns:           turn segmentation of a merged multi-turn sample in token space.
- dump:            per-process jsonl of training episodes (the R2S pipeline's input).
zero_shape below is the REWARD_SHAPING_PATH plugin of the opd arm.
"""


def zero_shape(env_score: float, sample) -> float:
    """REWARD_SHAPING_PATH of the opd arm (launcher.build_extra_env_vars): the student trains on
    the teacher KL alone, so the training reward is 0 while reward_routing keeps the env score
    in metadata["raw_reward"], rollout/episode_raw_reward and every eval/* metric."""
    return 0.0
