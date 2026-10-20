"""TMax's terminal-agent RL on miles-agent, matched to TMax-4B.

TMax (AllenAI, arXiv 2606.23321; hamishivi/tmax @6d3d606c, training/open-instruct/scripts/tmax/RL/qwen35_4b.sh)
trains Qwen3.5-4B with outcome-only RL on TMax-15K, 14,601 synthetic terminal tasks each in its own prebuilt image,
and lifts TB-Lite from 31.8 to 42.6 (TB 2.1 14.2 -> 19.9, TB 2.0 16.6 -> 18.9; released checkpoint: step 200). This
recipe keeps its harness, tasks, budgets and loss: the tasks are Harbor tasks (recipes/tmax/prepare.py), run by
miles_harbor's trial server with TMax's own harness, vanillux (miles_harbor.vanillux), beside
the sandbox.

    TMax-4B (qwen35_4b.sh)                          this recipe
    hamishivi/Qwen3.5-4B (Qwen3.5-4B's weights,      Qwen3.5-4B, thinking on (enable_thinking), each turn
      a template keeping every turn's <think>)        appended to the session's tokens, so every turn's
                                                      thinking stays in context
    8 prompts x 32 samples, 1 optimizer step,        --rollout-batch-size 8 --n-samples-per-prompt 32
      500 steps                                        --global-batch-size 256 --num-rollout 500
    groups of one reward dropped and refilled        miles_agent.dapo_filter (also drops groups with a
      (active sampling)                                transport failure: a trial without a reward)
    advantage r - mean(group), no std                 --disable-grpo-std-normalization
    token-level loss, masked tokens in the count     --calculate-per-token-loss
    DPPO, binary TV 0.1 against the engine's         miles_agent.dppo (--use-rollout-logprobs, the clip off)
      log-probs, no clip, no TIS, no KL
    lr 1e-6 constant, AdamW wd 0, grad clip 1        same
    T 1.0, top-p 1.0                                  same
    every call of an episode seeded: vLLM, seed      every call samples anew; seeded calls (tmax4b4) cut
      42 + j for sample j of a prompt, sent with       the repetition and runaway replies, but they rule out
      each call, a generator seeded anew per call      MTP and scored 2.2 points lower on TB 2.1
    16,384 tokens a turn, 67,584 in all (2,048       vanillux's budgets (max_tokens, max_response_tokens,
      prompt + 65,536 replies and tool output),        max_context_tokens; each bash call of a reply a step);
      64 tool calls, 120 s a command                   the engines and the trainer take CONTEXT_MARGIN more
                                                      for its estimate of tool output
    the tests run by the submit's command, in its    Harbor's verifier once the agent has submitted, the
      120 s (past them: no reward, the episode         task's 600 s (task.toml)
      goes on)
    no clock on an episode                            the agent's 3,600 s (HARBOR_AGENT_TIMEOUT_S): past it
                                                      reward 0, trained (staleness drops most such groups)
    a sandbox out of memory: reward 0, the end       the memory guard kills its largest process, the
                                                      episode goes on
    fully async, 4 steps of staleness, in-flight     --fully-async, no staleness cap by default (in-flight
      weight updates (~7 min a step: ~30 min)          requests retract onto the new weights; at our ~2 min a
                                                      step a 4-version cap dropped the slowest groups)
    FP32 LM head (trainer and vLLM)                   the engines' (--sglang-enable-fp32-lm-head) and the
                                                      trainer's (miles --fp32-lm-head: fp32 logits computed
                                                      in log_probs_chunk_size chunks)
    vLLM engines                                      sglang engines, attention by trtllm_mha in 64-token pages
                                                      (attention_backend; sglang's pick, triton, decodes 1.5-1.7x
                                                      slower on the engines used)
    no submit: reward 0, tests not run                HARBOR_REWARD_REQUIRES_SUBMIT=1 (trial server)
    sandboxes 4 GB, no CPU limit, internet on         4 GB (task.toml), 2 pinned CPUs, no network while the
                                                      agent works (the user's call: no internet then;
                                                      vanillux calls the model from beside the sandbox); the
                                                      tests keep the task's network
    16 learners (2 nodes, ZeRO-3 + SP 4) and 16      --train-nodes / --rollout-nodes of the job's 8-GPU
      vLLM TP1 engines (2 nodes)                       nodes: TP 1, the data-parallel width the rest

The trainer is at TP 1: Qwen3.5-4B fits one 178 GiB GPU with a 69,632-token micro-batch (full recompute, the log-probs in
--log-probs-chunk-size chunks, the allocator's expandable segments: TRAIN_ENV_VARS), so every training GPU is a
data-parallel rank and --skip-actor-forward-only (one optimizer step a batch: the trainer's own log-probs are the
step's) drops the separate forward pass.

The launch platform (checkpoint conversion, cleanup, checkpoints with resume and the preemption sentinel, session
servers and router, the engines' parser, the actors' environment, the submission) is miles_agent.launcher's, shared
with every recipe; this module holds TMax's science.

Usage (bash scripts/run_local.sh --recipe tmax ... with NEMO_GYM_URL set to the trial server(s); the driver flags:
    NEMO_GYM_URL=http://<env host>:8200 python run.py --run-tag tmax1 --topology fully_async --train-nodes 1 \\
        --rollout-nodes 3 --num-gpus-per-node 8 --base-dir $MILES_AGENT_MODELS/torch_dist_v5 \\
        --hf-checkpoint $MILES_AGENT_MODELS/Qwen3.5-4B --ref-load $MILES_AGENT_MODELS/torch_dist_v5/Qwen3.5-4B_torch_dist \\
        --save-dir $MILES_AGENT_RESULTS/tmax1/ --prompt-data $MILES_AGENT_DATA/harbor/tmax_rows/train.jsonl \\
        --eval-prompt-data $MILES_AGENT_DATA/harbor/tmax_rows/train.jsonl
"""

import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import miles.utils.external_utils.command_utils as U
import typer

SCRIPT_DIR = Path(__file__).resolve().parent
# miles_agent (the launcher, the dotted hook paths below) lives at the repo root; the ray
# actors get it through PYTHONPATH (launcher.build_extra_env_vars), this driver through sys.path.
_REPO_ROOT = str(SCRIPT_DIR.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from miles_agent import launcher  # noqa: E402
from miles_agent.launcher import NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER, LaunchArgs, RecipeSpec  # noqa: E402

# vanillux's budget (its max_context_tokens): 2,048 prompt tokens and 65,536 of replies and tool output
AGENT_CONTEXT = 67584
# what the engines and the trainer take past it: vanillux counts a tool output's tokens by an estimate
CONTEXT_MARGIN = 2048
# The trainer's allocator. Without fp32_lm_head, a full micro-batch's log-probs hold its bf16 logits and, chunk by
# chunk for the backward, their fp32 softmax (~97 GiB at 69,632 tokens and Qwen3.5's 248,320-token vocabulary), then
# take the logits' 32 GiB bf16 gradient in one piece. With the default allocator, 48 GiB of the GPU's 178 sat
# reserved in blocks too small for it and tmax4b1's step 0 ran out of memory there; expandable segments grow one
# mapping instead.
TRAIN_ENV_VARS = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@dataclass
class ScriptArgs(LaunchArgs):
    topology: Literal["colocated_1n", "fully_async"] = "fully_async"
    # fully_async: the job's training and rollout nodes
    train_nodes: int = 1
    rollout_nodes: int = 3
    eval_only: bool = False
    run_tag: str = ""  # wandb group; one RUN_TAG per run
    # NEMO_GYM_RUN_TIMEOUT: a backstop over one /run (the trial server bounds the agent and the trial)
    run_timeout_s: int = 9000
    # TMax's thinking: on, every turn's kept in the session's tokens
    chat_template_kwargs: str = '{"enable_thinking": true}'

    # Under MILES_AGENT_RESULTS / MILES_AGENT_DATA (launcher.results_dir / data_path); the job passes both.
    save_dir: str = field(default_factory=lambda: launcher.results_dir("tmax"))
    prompt_data: str = field(default_factory=lambda: launcher.data_path("tmax_train.jsonl"))
    eval_prompt_data: str = ""

    num_rollout: int = 500
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 32
    # colocated_1n's DAPO refill; fully_async keeps async_max_concurrent_samples in flight instead
    over_sampling_batch_size: int = 16
    global_batch_size: int = 256
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_top_k: int = -1
    max_tokens_per_turn: int = 16384
    context_len: int = AGENT_CONTEXT + CONTEXT_MARGIN
    # log-probs and their loss in chunks of this many tokens (fp32 logits of a 69k micro-batch whole: ~69 GB)
    log_probs_chunk_size: int = 4096
    # 0 = never: TMax picks its checkpoint by TB-Lite, evaluated apart (Harbor, Vanillux2Agent's sampling)
    eval_interval: int = 0
    # TMax: every 20 steps; a link that dies loses at most save_interval - 1 steps (the preemption sentinel saves)
    save_interval: int = 20
    seed: int = 42
    # the rows in miles' seeded shuffle (seed); off for rows already in an order of their own (prepare.py rows --order:
    # rows that grow with the image store, resumed without meeting a task twice)
    rollout_shuffle: bool = True
    # fully_async: trajectories in flight (TMax: a pool of 512 sandboxes); the entry sets it from the
    # trial servers' capacity (HARBOR_MAX_CONCURRENT each)
    async_max_concurrent_samples: int = 512
    # Drop a group whose oldest tokens are more than this many weight versions old when the trainer takes it; 0 =
    # never. TMax's cap of 4 steps spans ~30 min at its ~7 min a step; at our ~2 min a step it spanned ~8-9 min and
    # dropped the slowest ~10% of the mixed groups, which held a third of the context overflows (tmax4b2).
    max_weight_staleness: int = 0
    # TMax's FP32 LM head on the trainer's side as well: the log-probs from fp32 logits computed log_probs_chunk_size
    # tokens at a time (miles --fp32-lm-head, patches/miles/miles-fp32-lm-head.patch), and the policy's entropy
    # logged (--observe-training-entropy: train/entropy_loss). The engines' fp32 head is always on.
    fp32_lm_head: bool = True
    # The engines' attention: sglang picks triton for Qwen3.5 on the engines used, where trtllm_mha (64-token pages) decodes a
    # session 1.5-1.7x as fast at 20-32 sessions and 25k-token contexts, with the same log-probs
    # (measured with the sglang bench). "" leaves sglang's pick.
    attention_backend: str = "trtllm_mha"
    # Speculative decoding with Qwen3.5's own MTP layer as the draft (sglang's NEXTN, top-1: rejection sampling that
    # keeps the target's distribution, and the target's log-probs come back): draft steps a verify, 0 off. The draft
    # is not trained (the trainer holds no MTP layer), so its acceptance falls as the policy moves (rollout/spec_*).
    mtp_steps: int = 0


SPEC = RecipeSpec(name="tmax", run_file=__file__, reward_shaping_path=None)

cleanup = launcher.cleanup
prepare = launcher.prepare


def _rollout_args(args: ScriptArgs) -> str:
    return (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        f"{'--rollout-shuffle ' if args.rollout_shuffle else ''}"
        f"--seed {args.seed} "
        f"--rollout-seed {args.seed} "
        f"--num-rollout {0 if args.eval_only else args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-top-p {args.rollout_top_p} "
        f"--rollout-top-k {args.rollout_top_k} "
        f"--rollout-max-response-len {args.max_tokens_per_turn} "
        f"--max-seq-len {args.context_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )


def _eval_args(args: ScriptArgs) -> str:
    if args.eval_interval <= 0:
        return ""
    return (
        f"--eval-prompt-data tmax {args.eval_prompt_data} "
        f"--eval-interval {args.eval_interval} "
        "--n-samples-per-eval-prompt 1 "
        # Vanillux2Agent's (TMax's evaluation agent) sampling
        "--eval-temperature 0.7 "
        "--eval-top-p 0.95 "
        f"--eval-max-response-len {args.max_tokens_per_turn} "
    )


# TMax's loss (module docstring): GRPO's centered advantage, DPPO's mask against the engine's log-probs, no clip
_DPPO_ARGS = (
    "--advantage-estimator grpo "
    "--disable-grpo-std-normalization "
    "--entropy-coef 0.0 "
    "--eps-clip 1e6 "
    "--eps-clip-high 1e6 "
    "--use-rollout-logprobs "
    "--get-mismatch-metrics "
    "--custom-tis-function-path miles_agent.dppo.dppo_binary_tv "
    "--calculate-per-token-loss "
    # one optimizer step a batch: the trainer's log-probs of the step are pi's, no separate forward pass
    "--skip-actor-forward-only "
)


def _optimizer_args(args: ScriptArgs) -> str:
    return (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--clip-grad 1.0 "
        # 0 train iters (eval only) would give Megatron's LR scheduler 0 decay steps.
        f"{'--lr-decay-iters 1 ' if args.eval_only else ''}"
    )


def _perf_args(args: ScriptArgs) -> str:
    return (
        f"{launcher.parallel_args(args)}"
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.context_len} "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
        f"--train-env-vars {shlex.quote(json.dumps(TRAIN_ENV_VARS))} "
    )


def _engine_args(args: ScriptArgs) -> str:
    out = ""
    if args.attention_backend:
        # trtllm_mha's kernels read the KV cache in pages (sglang's default is 1 token): 64, as benchmarked
        pages = "--sglang-page-size 64 " if args.attention_backend == "trtllm_mha" else ""
        out += f"--sglang-attention-backend {args.attention_backend} {pages}"
    if args.mtp_steps > 0:
        out += (
            "--sglang-speculative-algorithm NEXTN "
            f"--sglang-speculative-num-steps {args.mtp_steps} "
            "--sglang-speculative-eagle-topk 1 "
            f"--sglang-speculative-num-draft-tokens {args.mtp_steps + 1} "
        )
    return out


def _topology_args(args: ScriptArgs) -> str:
    if args.topology == "colocated_1n":
        return f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
    # The RolloutManager and its session servers on the ray head, where the agents' calls arrive.
    return (
        f"--actor-num-nodes {args.train_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_nodes * args.num_gpus_per_node} "
        "--pin-rollout-manager-to-head "
        "--fully-async "
        f"{f'--max-weight-staleness {args.max_weight_staleness} ' if args.max_weight_staleness > 0 else ''}"
        f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
    )


def build_train_args(args: ScriptArgs) -> str:
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    return (
        f"{launcher.checkpoint_args(args, skip_eval_on_resume=True)}"
        f"{_rollout_args(args)}"
        f"{_eval_args(args)}"
        f"{_DPPO_ARGS}"
        f"{_optimizer_args(args)}"
        f"{_perf_args(args)}"
        f"{launcher.sglang_args(args, context_length=args.context_len)}"
        # TMax's FP32 LM head: the engines' and (fp32_lm_head) the trainer's
        "--sglang-enable-fp32-lm-head "
        f"{'--fp32-lm-head --observe-training-entropy ' if args.fp32_lm_head else ''}"
        f"{_engine_args(args)}"
        f"{launcher.session_args(args, filter_path=NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER)}"
        f"{launcher.REWARD_ARGS}"
        f"{_topology_args(args)}"
        f"{launcher.misc_args(args)}"
        f"{launcher.wandb_args(args, SPEC, run_id=args.run_tag or None)}"
        f"{debug_args}"
        f"{args.extra_train_args} "
    )


def execute(args: ScriptArgs):
    if args.eval_only and args.eval_interval <= 0:
        raise ValueError("--eval-only needs --eval-interval > 0")
    if args.eval_interval > 0 and not args.eval_prompt_data:
        raise ValueError("--eval-interval needs --eval-prompt-data")
    env = {"NEMO_GYM_RUN_TIMEOUT": str(args.run_timeout_s)}
    launcher.launch(args, SPEC, build_train_args(args), extra_env_vars=env)


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
