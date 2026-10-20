# pyright: reportMissingImports=false
"""scienceworld OPD preset: the R2S paper's student arm (agentic on-policy distillation) on the
scienceworld recipe.

Layered on run.py, whose environment, data paths, eval slices (scienceworld_seen /
scienceworld_unseen), group filters and /run timeout it keeps; ``OpdArgs`` pins only the paper's
protocol (docs/r2s.md): --algo opd under the teacher topology, 150 steps of 16 episodes (one
rollout per prompt, 30-step episodes), rollout and eval at temperature 0.4, 512 tokens a turn in
training / 4096 in eval, context and packing budget 32768, Adam lr 1e-6 constant with no warmup
and weight decay 0.1 (betas 0.9/0.98, clip 1.0 as every recipe), weights pushed every step, eval
and save every 25 steps, seed 42, thinking off for every model family, the teacher KL at beta 1
as the whole objective (miles_agent/launcher.py ``_algo_args``; the train reward is zeroed on
the actors, eval keeps the env's shaped score, with success as env/won). The teacher is the
job's business: a ``*_teacher`` topology serves it with sglang on the allocation's last node,
outside ray, and passes its base URL as --teacher-url.

The OPD arms' prompt is PROMPT=r2s_oneshot (prompts/scienceworld/r2s_oneshot.py: the paper's
one-shot rows); the job defaults to the recipe's plain prompt otherwise. The paper's vanilla and
anchor cohort (seed 43, --topology async_2n_teacher) is documented in docs/r2s.md, not defaulted
here.

Usage (RUN_SCRIPT=opd.py selects this script on the training entry):
    RUN_SCRIPT=opd.py PROMPT=r2s_oneshot ... scripts/run_local.sh --recipe scienceworld --run-tag <tag> --topology async_4n_teacher --num-nodes 4
    # the R2S pilot dump: the base policy at the arm's caps, no teacher, nothing trains or saves
    RUN_SCRIPT=opd.py MODE=debug_rollout_only TOPOLOGY=colocated_1n \\
        EXTRA_ARGS="--algo grpo --opd-dump --rollout-temperature 1.0"
    # decodes 2 and 3 of the final checkpoint (miles_agent/eval_ckpt.py):
    RUN_SCRIPT=eval_ckpt.py TOPOLOGY=colocated_1n EXTRA_ARGS="--recipe-script opd.py --eval-load-tag <tag> --seed 43"
"""

from dataclasses import dataclass
from typing import Literal

import miles.utils.external_utils.command_utils as U
import run
import typer


@dataclass
class OpdArgs(run.ScriptArgs):
    """The paper's protocol over run's environment: only what differs from the GRPO recipe."""

    # The R2S cohort's topology: async_3n's ray split plus the teacher node.
    topology: Literal["colocated_1n", "async_3n", "fully_async_3n", "async_4n_teacher", "async_2n_teacher"] = (
        "async_4n_teacher"
    )
    algo: Literal["grpo", "gspo", "ppo", "dapo", "opd"] = "opd"

    # 150 steps of 16 episodes, one rollout per prompt: the teacher scores each episode once.
    num_rollout: int = 150
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 1
    global_batch_size: int = 16
    # T 0.4 in training and in eval; 512 tokens a turn in training, 4096 in eval. The context
    # and packing budget are pinned here so that the GRPO recipe can retune its own without
    # moving the paper's arm.
    rollout_temperature: float = 0.4
    eval_temperature: float = 0.4
    rollout_max_response_len: int = 512
    eval_max_response_len: int = 4096
    rollout_max_context_len: int = 32768
    train_max_tokens_per_gpu: int = 32768
    # Adam lr 1e-6 constant, no warmup, weight decay 0.1.
    lr: float = 1e-6
    lr_warmup_iters: int = 0
    weight_decay: float = 0.1
    # Eval and save every 25 steps; weights pushed every step (16 episodes a step, nothing to
    # gain from staleness); seed 42.
    eval_interval: int = 25
    save_interval: int = 25
    update_weights_interval: int = 1
    seed: int = 42
    # Thinking off for every family, the Qwen3 hybrids included (their template thinks by default).
    chat_template_kwargs: str = '{"enable_thinking": false}'
    # The teacher KL at beta 1 is the whole objective; the pilot dump is a job line (--opd-dump
    # under --mode debug_rollout_only), never an arm's default.
    opd_kl_coef: float = 1.0
    opd_dump: bool = False


@U.dataclass_cli
def main(args: OpdArgs):
    run.cleanup()
    if not args.skip_prepare:
        run.prepare(args)
    run.execute(args)


if __name__ == "__main__":
    typer.run(main)
