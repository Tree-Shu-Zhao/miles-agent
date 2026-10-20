"""miles_agent.dppo: TMax's binary-TV DPPO mask, called as miles' losses.py calls its TIS/RS hook (pg_loss of PPO's
loss with the clip off, the trainer's and the engine's log-probs per sample)."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from miles_agent.dppo import DELTA, dppo_binary_tv  # noqa: E402

ARGS = SimpleNamespace(eps_clip=1e6, eps_clip_high=1e6, eps_clip_c=None, use_rollout_logprobs=True)


def pg_loss_of(advantages, pi, mu):
    """miles' compute_policy_loss with the clip off: -A * pi / mu, with the gradient through pi."""
    ratio = (pi - mu).exp()
    return -ratio * advantages


def call(advantages, pi_probs, mu_probs, args=ARGS):
    pi = torch.tensor([math.log(p) for p in pi_probs], requires_grad=True)
    mu = torch.tensor([math.log(p) for p in mu_probs])
    advantages = torch.tensor(advantages)
    pg_loss = pg_loss_of(advantages, pi, mu)
    split = [1, len(pi_probs) - 1]  # two samples, as the hook gets them
    masks = [torch.ones(n) for n in split]
    out, out_masks, metrics = dppo_binary_tv(
        args,
        pg_loss=pg_loss,
        train_log_probs=list(pi.detach().split(split)),
        rollout_log_probs=list(mu.split(split)),
        loss_masks=masks,
    )
    return pi, pg_loss, out, out_masks, masks, metrics


def test_a_token_far_from_mu_is_masked_only_when_the_update_pushes_it_further():
    # each token: (A, pi(y), mu(y)); |pi - mu| > 0.1 for all but the last two
    cases = [
        (1.0, 0.5, 0.3),  # A > 0, r > 1: masked
        (1.0, 0.3, 0.5),  # A > 0, r < 1: kept (the update pulls it back)
        (-1.0, 0.3, 0.5),  # A < 0, r < 1: masked
        (-1.0, 0.5, 0.3),  # A < 0, r > 1: kept
        (0.0, 0.5, 0.3),  # A = 0: no gradient, kept
        (1.0, 0.35, 0.3),  # within DELTA: kept
        (-1.0, 0.25, 0.3),  # within DELTA: kept
    ]
    advantages, pi_probs, mu_probs = zip(*cases, strict=True)
    pi, pg_loss, out, out_masks, masks, metrics = call(advantages, pi_probs, mu_probs)
    keep = torch.tensor([0.0, 1, 0, 1, 1, 1, 1])
    assert torch.equal(out.detach(), pg_loss.detach() * keep)
    # the masks (the denominators') are untouched: a masked token still counts
    assert out_masks is masks
    out.sum().backward()
    assert torch.equal(pi.grad == 0, torch.tensor([True, False, True, False, True, False, False]))
    assert torch.equal(metrics["dppo_masked_frac"], 1 - keep)
    assert torch.allclose(metrics["dppo_tv"], torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05]), atol=1e-6)
    assert torch.equal(metrics["dppo_tv_over_frac"], torch.tensor([1.0, 1, 1, 1, 1, 0, 0]))


def test_the_threshold_is_tmaxs():
    assert DELTA == 0.1


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(eps_clip=0.2, eps_clip_high=0.28, eps_clip_c=None, use_rollout_logprobs=True),
        SimpleNamespace(eps_clip=1e6, eps_clip_high=1e6, eps_clip_c=3.0, use_rollout_logprobs=True),
        SimpleNamespace(eps_clip=1e6, eps_clip_high=1e6, eps_clip_c=None, use_rollout_logprobs=False),
    ],
)
def test_it_refuses_a_clipped_loss_or_the_trainers_old_policy(args):
    with pytest.raises(ValueError):
        call([1.0, 1.0], [0.5, 0.5], [0.3, 0.3], args=args)
