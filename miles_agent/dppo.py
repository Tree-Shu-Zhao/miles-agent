"""TMax's DPPO with the binary total-variation trust region, as miles' TIS/RS hook.

TMax (arXiv 2606.23321; open-instruct grpo_utils.py @6d3d606c, DPPO arXiv 2602.04879) takes the rollout engine's
log-probs as the old policy (mu) and drops PPO's clip: a token's loss is -A * r, r = pi / mu. A token is masked out
where the sampled token's probability moved more than DELTA between mu and pi and the update would push it further:
(A > 0 and r > 1) or (A < 0 and r < 1). A masked token stays in the denominator (the token count), as miles keeps it:
the hook's masks change the numerator alone (losses.py, rollout_mask_sums).

    --use-rollout-logprobs --eps-clip 1e6 --eps-clip-high 1e6 \\
    --get-mismatch-metrics --custom-tis-function-path miles_agent.dppo.dppo_binary_tv

and neither --use-tis nor --eps-clip-c. The hook gets the per-token pg_loss, not the advantages: with the clip off
pg_loss = -A * r and r > 0, so A's sign is -pg_loss's (0 where A is, which has no gradient to mask). The probabilities
are of the trainer's (detached) and the engine's log-probs of the sampled token, each clamped to [-30, 0] in log space
as TMax clamps them.
"""

from typing import Any

import torch

# TMax's threshold on |mu(y) - pi(y)| (qwen35_4b.sh --dppo_divergence_threshold 0.1)
DELTA = 0.1
# eps_clip at least this: PPO's clip, which DPPO drops, is off
NO_CLIP = 1e3


def dppo_binary_tv(
    args: Any,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    if min(args.eps_clip, args.eps_clip_high) < NO_CLIP or getattr(args, "eps_clip_c", None) is not None:
        raise ValueError("dppo_binary_tv needs PPO's clip off: --eps-clip 1e6 --eps-clip-high 1e6, no --eps-clip-c")
    if not args.use_rollout_logprobs:
        raise ValueError("dppo_binary_tv needs --use-rollout-logprobs: mu is the rollout engine's policy")
    pi_log = torch.cat(train_log_probs, dim=0).detach().float()
    mu_log = torch.cat(rollout_log_probs, dim=0).to(pi_log.device).float()
    if pi_log.shape != pg_loss.shape or mu_log.shape != pg_loss.shape:
        raise ValueError(f"dppo_binary_tv: log-probs {pi_log.shape} / {mu_log.shape} vs pg_loss {pg_loss.shape}")
    tv = (pi_log.clamp(-30, 0).exp() - mu_log.clamp(-30, 0).exp()).abs()
    ratio = (pi_log - mu_log).clamp(-20, 20).exp()
    loss = pg_loss.detach()
    away = ((loss < 0) & (ratio > 1)) | ((loss > 0) & (ratio < 1))
    masked = (tv > DELTA) & away
    metrics = {
        "dppo_masked_frac": masked.float(),
        "dppo_tv": tv,
        "dppo_tv_over_frac": (tv > DELTA).float(),
    }
    return pg_loss * (~masked).to(pg_loss.dtype), loss_masks, metrics
