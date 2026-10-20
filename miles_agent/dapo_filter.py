"""Dynamic-sampling filters for the env recipes (--dynamic-sampling-filter-path).

Wraps the stock miles filters instead of passing check_reward_nonzero_std
directly because these recipes' Sample.reward is a dict ({"env", "train"};
resolved via --reward-key) and an unscored sample can carry reward None - the
stock filter would raise inside torch.tensor and, under fully-async, kill the
rollout worker loop. Dropped groups are replaced with fresh prompts by the
rollout (true DAPO dynamic sampling), not recycled.

Transport failures (check_no_aborted_no_transport_failure*): a gym /run that
failed for infrastructure reasons - env engine worker crash/respawn/timeout,
adapter 5xx, a gym-side exception of any kind - leaves the sample with
metadata["env_metrics"]["run_transport_failed"] truthy (miles_agent.gym_bridge)
and no env reward, so reward_routing scores it 0.0. That 0.0 is NOT environment
feedback (the policy never received a verdict), so the whole group is dropped
and re-sampled instead of training as a failed episode. Genuine env feedback -
invalid actions, no-op clicks, bad purchases, low scores - is never filtered.
"""

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.filter_hub.dynamic_sampling_filters import (
    check_no_aborted,
    check_reward_nonzero_std,
)


def _flatten(samples):
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def _has_transport_failure(sample) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    return bool((metadata.get("env_metrics") or {}).get("run_transport_failed"))


def _nonzero_std(args, samples, **kwargs):
    if any(s.get_reward_value(args) is None for s in _flatten(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_none_reward")
    return check_reward_nonzero_std(args, samples, **kwargs)


def check_no_aborted_nonzero_std(args, samples, **kwargs):
    """DAPO: drop aborted groups, None-reward groups and zero-reward-std groups."""
    out = check_no_aborted(args, samples, **kwargs)
    if not out.keep:
        return out
    return _nonzero_std(args, samples, **kwargs)


def check_no_aborted_no_transport_failure(args, samples, **kwargs):
    """GRPO/GSPO/PPO: drop aborted groups and groups with an infrastructure failure."""
    out = check_no_aborted(args, samples, **kwargs)
    if not out.keep:
        return out
    if any(_has_transport_failure(s) for s in _flatten(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_transport_failure")
    return DynamicFilterOutput(keep=True)


def check_no_aborted_no_transport_failure_nonzero_std(args, samples, **kwargs):
    """DAPO on top of the transport-failure drop."""
    out = check_no_aborted_no_transport_failure(args, samples, **kwargs)
    if not out.keep:
        return out
    return _nonzero_std(args, samples, **kwargs)
