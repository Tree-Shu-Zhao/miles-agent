# Local patches on top of upstream pins

Patch-form copies of every local change carried on the 3rd-party pins.

The `3rdparty/miles` gitlink pins the UPSTREAM base commit (`a22fd8241`, the
`v0.1.2` release tag, which anyone can fetch from radixark/miles). The patched
commits exist only in local working trees, so the live source is always
**base + these patches applied with `git am`**, in this order:

    cd 3rdparty/miles
    git checkout v0.1.2             # the pinned upstream base, a22fd8241
    git am ../../patches/miles/miles-env-metrics.patch
    git am ../../patches/miles/miles-reward-key-none-safe.patch
    git am ../../patches/miles/miles-save-while-resident.patch
    git am ../../patches/miles/miles-session-text-parts.patch
    git am ../../patches/miles/miles-preempt-save.patch
    git am ../../patches/miles/miles-fp32-lm-head.patch
    git am ../../patches/miles/miles-fp32-lm-head-hybrid.patch
    git am ../../patches/miles/miles-mamba-packed-seq-idx.patch
    git am ../../patches/miles/miles-qwen3.5-2B-model-args.patch
    # conflicts: git apply --3way <patch> then resolve + commit

Verified 2026-10-17: on a fresh `v0.1.2` checkout this sequence applies with
plain `git am` and gives tree `901563c5a4` (only the commit SHAs differ
between checkouts; `254a2eb05f` without the last patch, `1751a6d0aa` without
the last two, `e19020e605` without the last three, `4cfed940e7` without the
last four). `patch_ids.txt` pins each patch's `git patch-id --stable`;
`tests/test_miles_patches_applied.py` checks both the ids and the working
tree. The bump from `c7d81d478` (840 upstream commits) dropped
`miles-weight-version-spans.patch` and half of the former
`miles-session-client-view.patch`, which upstream now covers, and
`miles-qwen25-tito.patch` went with Qwen2.5 itself; the sections below say
what each remaining patch needed. Consequences:

- `git submodule status` shows `+` for 3rdparty/miles by design; `.gitmodules`
  sets `ignore = all` so `git status` / `git add -A` / `git add -u` leave it
  alone. `git commit -a` does NOT honor that and would re-pin the gitlink to a
  local SHA nobody can fetch: never use `git commit -a` here, never stage
  `3rdparty/miles` unless you are bumping the upstream base.
- `scripts/check_patches.sh` refuses a 3rdparty/miles checkout that lacks any
  of these patches (subject check; `--apply` applies the missing ones in
  `series` order), and `tests/test_miles_patches_applied.py` checks the same
  locally, plus that every patch's id matches `patch_ids.txt`.
- Upstream bump = move the gitlink to the new base, `git checkout <base>`,
  re-`git am` the patches, re-export them with `git format-patch` if they
  needed conflict resolution, and update the SHA above.

## Dropped on the v0.1.2 bump: miles-qwen25-tito.patch

The qwen25 TITO tokenizer entry for Qwen2.5 (stock TITO has none: the engine
stops at `<|im_end|>` while the chat template renders `<|im_end|>\n`, so the
default splice drops one token per ChatML turn boundary). Qwen2.5 is retired:
the recipes run the Qwen3.5 / Qwen3 families, whose TITO entries are upstream
(`qwen35`, `qwen3`; miles_agent/launcher.py MODEL_FAMILIES), so nothing
selects qwen25 any more and `--tito-model qwen25` is now an invalid choice.
Branches that still run Qwen2.5 keep the patch on their own pins.

## Dropped on the v0.1.2 bump: miles-weight-version-spans.patch

It fixed the `weight_version/mixed_version_ratio` blind spot: miles read
only the trailing scalar `weight_version`, so a request whose decode
straddled a weight update looked version-pure. Upstream now does the same
(#1891 and #1892): `Sample.weight_versions` holds the per-call token spans
from `meta_info["weight_versions"]` on the native path and the session
path, and `mixed_version_ratio` counts versions over all spans. The patch's
other half, `weight_version/retracted_sample_ratio` and
`retractions_per_sample` from `meta_info["num_retractions"]`, was dropped
with it: it was the fallback signal for when spans are missing, and both
read ~0 on our runs (docs/agentic_rl_metrics.md).

## miles-env-metrics.patch

Environment-error counters as wandb metrics (applies cleanly, no upstream
equivalent).
`_compute_metrics_from_samples` means every numeric under
`sample.metadata["env_metrics"]` over the full sample set into `env/*`
(train `rollout/env/*` and, because eval passes samples too, per-dataset
`eval/<ds>/env/*`). Producers live in this repo, not the pin:
gym_root tool_call_agent counts policy_400_truncated + tool_errors
(invalid tool args, plus any tool reply the resources server flags with
the generic X-Env-Tool-Error header) into the /run response;
miles_agent/gym_bridge.py forwards them and stamps
run_transport_failed=1 on /run transport failure (previously a silent
0.0 reward indistinguishable from a wrong answer).
CAVEAT: verified on session-server v1 (metadata overlay driver-side,
agentic_tool_call.py:129). v2's default postprocessor only extracts
agent_metadata["reward"] — recheck the merge before relying on env/*
under --use-session-server v2.

## miles-reward-key-none-safe.patch

Dict-reward support fix (applies cleanly).
Aborted/unscored samples keep
reward=None and never meet the reward hook, so with the miles_agent
reward-routing contract active ({"env", "train"} + metadata raw_reward):
(a) key extraction crashed on None["key"] in both eval collection twins and
get_reward_value instead of flowing into none_reward_ratio accounting, and
(b) the raw_reward metadata paths (metrics + train-data conversion) gated on
samples[0] then crashed on the first unscored sample. Pass None through and
fall back per sample. Found by mrsmoke16/18 (eval slices with aborted
episodes — agent /run 500s under load are a pre-existing flake at this
scale: passed smokes carried ~174 of them).

## miles-save-while-resident.patch

Zero-weight checkpoint fix (ported by hand:
the drivers now call `rollout_executor.save()` instead of
`rollout_manager.save.remote()`, and upstream moved the eval-before-train
block next to the old `save()`). v0.1.2 still has the bug: `--offload-train`
still forces the backup off (arguments.py:3679), `wake_up()` only resumes
torch_memory_saver, `save_model()` saves the model as it is, and both
`use_critic` branches still offload before the save gate. Upstream's opt-in
`--rematerialize-param-from-master-weight` rebuilds the actor's params only
(it asserts `num_critic_only_steps == 0`), so it does not cover this path.
With --offload-train miles
force-disables the param-buffer CPU backup (arguments.py:3679), so
torch_memory_saver.pause() DISCARDS bf16 param contents and wake_up()
remaps fresh zero pages; only the next train step's weights_backuper
restore refills them. Both drivers' use_critic branches offloaded the
just-trained model BEFORE the save gate, then save's onload()+save_model
serialized all-zero bf16 weights while the separately backed-up optimizer
state stayed real. Silent: the converter, the save log, and in-loop evals
(sglang holds real weights) all look healthy. Hit every one of
colo1n_ppo_r1's 10 checkpoints (sync colocated PPO); GRPO and criticless
async runs were safe only because their branch saves before offloading.
Fix: decide the save before training and save each model inline while
resident (the proven criticless pattern). Weights for the affected run
were recovered offline from the distributed optimizer's fp32 main params
(an offline recovery script, validated bitwise on a good ckpt).
Worth reporting upstream together with the finding below.

## miles-session-text-parts.patch

Text-only content-part lists for black-box agents.
`flatten_text_part_messages` turns a text-only part list (the
OpenAI list form ProRL's OpenHands fork sends) into SGLang's space-joined
string. `resolve_request_args_by_config` applies it to the forwarded body
with the other server rules, and both session versions apply it to the
messages they match history against and render input_ids from: the TITO
templates read string content only. Qwen3's template concatenates a system
message's content as a string, so a list there fails the render, and other
list content renders as ''. The first version of this patch flattened only
the forwarded body; its test checked that body (the mock backend renders
`messages`, not `input_ids`), so it passed while every OpenHands turn got a
500 (a smoke run). The test now decodes the input_ids, for v1 and
v2: `test_text_part_content_renders_as_sglang_string` in
tests/fast/router/test_session_v1_v2_parity.py.

Until the v0.1.2 bump this was `miles-session-client-view.patch`, whose
other half kept the forced `logprobs` / `meta_info` out of client replies
(LiteLLM 1.71 rejected SGLang's logprob entries). Upstream does
that now: #3877 drops `meta_info` from client replies and #3716 sends
`logprobs` only when the client asked, with the SessionRecord keeping both.

## miles-preempt-save.patch

Save and stop before a preemption or time-limit kill (no upstream
equivalent: `--save-trigger-sentinel` only saves at the
next step boundary and never stops a rollout). `--preempt-sentinel <path>`:
a scheduler wrapper touches `<path>` on the preemption or time-limit signal
(docs/cluster.md, Contract C); both drivers poll it every 2 s and stop at the
earliest safe point:

- In a rollout or eval: `RolloutExecutor.abort_rollout()` cancels the train
  rollout coroutine, so no more prompts are drawn while the data source is
  saved, and calls the agent abort hook. `InferenceController.halt_generation()`
  then pauses every engine with `pause_generation(mode="abort")`: running
  requests are aborted and new ones held, never resumed. That makes the
  following `release_memory_occupation` safe; SGLang asserts the scheduler is
  idle there, and env_loop_agent episodes keep calling the engines after
  their client is gone (they run to completion on disconnect).
- The last trained step is saved unless it is already on disk: inline when
  the signal lands during a train step (the actor is resident); otherwise
  the colocated engines offload, the actor wakes, and `save_model` restores
  the trained weights from the host backup first (`_resident_weights_stale`),
  because `--offload-train` wakes to fresh pages (see
  miles-save-while-resident.patch).
- `<path>.done` holds `{"rollout_id": N}`, the step the run is durable up to.
  The wrapper then requeues (preemption) or exits (time limit), and the
  resumed run `--load`s that checkpoint.

Limits: the aborted rollout's prompts are skipped on resume (their draws are
in the saved data-source state); a PPO critic is only saved inside its train
step, so a stop there warns and reports the last periodic save; an eval in
flight is dropped, so that point is missing from wandb. Tests:
tests/fast/utils/test_preempt.py and `TestPreemptSentinel` in
tests/fast/test_train.py and test_train_async.py (container only: train.py
imports sglang).

## miles-fp32-lm-head.patch

The trainer scores the policy with fp32 logits (no upstream equivalent).
`--sglang-enable-fp32-lm-head` already makes
the engines sample from fp32 logits, but Megatron's output layer rounds the
trainer's logits to bf16 (a spacing of 0.125 for logits in [16, 32)), so the
log-probs the loss and its importance ratios use come from a different
distribution than the one sampled. `--fp32-lm-head` passes Megatron's
`output_processor` hook to the model's forward (training and forward-only),
which hands back the final hidden states and the output weight instead of
the logits; `get_log_probs_and_entropy` then computes each response chunk's
logits as bf16 x bf16 with fp32 accumulation and output, their logsumexp, the
sampled tokens' log-probs and, detached, the entropy (`loss/hub/fp32_lm_head.py`).
The backward recomputes each chunk with TF32 matmuls and accumulates the
weight gradient in fp32, so no `[tokens, vocab]` tensor of the micro-batch
exists: it also frees the bf16 logits and their gradient (35 GB each for a
69,632-token micro-batch over Qwen3.5's 248,320-token vocabulary).

Limits (`validate_fp32_lm_head_args`, from miles_validate_args): Megatron
backend, tensor and context parallel 1, no `--true-on-policy-mode`, no
sampling-support replay (`--rollout-top-p` below 1 or `--rollout-top-k`: their
masks of the logits; checked on the flags, since miles derives
`use_sampling_support_replay` later), entropy coefficient 0 (the entropy is
observed, never differentiated). Tests:
tests/fast/backends/training_utils/loss/test_fp32_lm_head.py (log-probs,
entropy and gradients equal fp32 autograd on the full logits).

## miles-fp32-lm-head-hybrid.patch

`--fp32-lm-head` for a model whose forward has no `output_processor` hook
(after miles-fp32-lm-head.patch). Megatron's
`HybridModel`, which Megatron-Bridge builds for Nemotron-H (Mamba-2 + MoE +
attention), does not take the hook, so the first patch's forward failed on it.
`fp32_lm_head.hidden_states_out(model)` now wraps both forward calls: a model
whose forward takes the hook (`GPTModel`) gets it as before; any other has its
output layer return its input for the call, so the model's own
`logits.transpose(0, 1)` hands back the final hidden states in the hook's
`[b, s, h]` layout and the output weight goes to the same stash. A model with
MTP layers but no hook is refused (their loss runs through the output layer
before the logits do): Nemotron-H trains without its MTP layer
(`MILES_NEMOTRONH_KEEP_MTP` unset). Tests: the `hidden_states_out` cases in
tests/fast/backends/training_utils/loss/test_fp32_lm_head.py.

## miles-mamba-packed-seq-idx.patch

Packed sequences kept apart in Mamba layers (after
miles-fp32-lm-head-hybrid.patch). Megatron's Mamba mixer separates the
sequences of a packed (`thd`) micro-batch only through
`PackedSeqParams.seq_idx`, each token's sequence index, which its fused conv +
scan kernel restarts at; `PackedSeqParams` derives it only when given
`total_tokens`, and miles' `get_packed_seq_params` gave neither. In a
micro-batch of several samples every sample after the first so started from the
previous one's conv and SSM states instead of zero (attention and MoE layers
were unaffected). Found on Nemotron-3.5-Lightning (a smoke run: train vs
rollout log-prob diff 0.08-0.12 and 3% of tokens outside the DPPO trust region,
where Qwen3.5-4B shows 0.009). `get_packed_seq_params` now sets `seq_idx` from
the packed boundaries, sized by the host's total so that building it does not
sync the device. Tests: tests/fast/backends/megatron_utils/test_packed_seq_idx.py.

## miles-qwen3.5-2B-model-args.patch

`scripts/models/qwen3.5-2B.py` (after miles-mamba-packed-seq-idx.patch). v0.1.2
ships model-arg scripts for Qwen3.5
0.8B / 4B / 9B / 27B / 35B-A3B but not 2B, and `--megatron-model-type` is
resolved only against the pin's scripts/models (model_args_utils.py), so a
Qwen3.5-2B policy (the R2S paper's 2B students)
could neither be converted nor trained. The script is the 4B one with the 2B
dimensions from Qwen/Qwen3.5-2B config.json: 24 layers, hidden 2048, ffn
6144, 8 heads / 2 kv groups, tied embeddings (no
`--untie-embeddings-and-output-weights`, like 0.8B and 4B). Upstreamable as is.

## Finding: offload-train x distributed weight push (no patch)

The async-PPO crash investigated on 2026-08-30 (CUDA illegal memory access in
the pp process group at the first weight push) is NOT keep-old-actor or the
critic - it is --offload-train (torch_memory_saver) x
update_weight_from_distributed. Isolation matrix: koa+async no-offload PASS,
critic colocated+offload PASS, critic async+offload CRASH, grpo async+offload
no-critic CRASH, ppo async no-offload PASS. Async PPO arms therefore pass
--no-offload-train (both 7B models resident; fits a 288 GB GPU).

A fifth patch (miles-uncorrected-async-ppo.patch, an env-var escape hatch
around validate_async_off_policy_correction) was carried briefly for the
algorithm campaign and removed on 2026-08-31 after the fa3n_ppo_r1
post-mortem confirmed the validator's rationale: uncorrected async PPO
runs with ratio==1 (clip inert) and collapses. Standard practice now:
don't run it; async PPO uses --keep-old-actor (+ optional --use-tis).
