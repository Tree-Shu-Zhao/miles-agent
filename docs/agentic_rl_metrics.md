# Agentic RL monitoring metrics: a field guide

For agentic RL training on miles + NeMo Gym. Every metric gets four things: **why it matters / normal range /
alarm threshold / diagnosis and fix**. Numbers marked (measured) come from this repository's search_r1 campaigns
(Qwen2.5-7B, the colocated / async / fully_async topologies and a nine-arm deployment eval); the rest are rules
of thumb, marked (rule of thumb). File names refer to the miles pin the guide was written against (before the
v0.1.2 bump) and may have moved.

**First, why wandb looks messy** — three sources of confusion; once they are understood, 80 % of the "clutter"
disappears:

1. **The `rollout/` prefix is shared by two processes**: the rollout manager's per-sample statistics
   (metrics.py) and the trainer-side gathered batch means (log_utils.py) land under the same prefix on the same
   step axis. `rollout/response_len/mean` vs `rollout/response_lengths`, and `rollout/truncated_ratio` vs
   `rollout/truncated`, are near-duplicate pairs emitted by the two processes.
2. **Two eval key separators**: the aggregate keys use a dash (`eval/<ds>-truncated_ratio`, `eval/<ds>-pass@k`),
   the per-sample twins use a slash (`eval/<ds>/response_len/mean`). A dashboard glob has to spell both.
3. **The fixed four-statistic family**: `compute_statistics` always produces `{mean,median,max,min}`. In nearly
   every case only the mean matters (occasionally the max); ignore median/min outright and the metric count
   halves.

---

## Tier 0 — core learning curves (any algorithm, any topology; read daily)

### `rollout/episode_raw_reward`
- **Why it matters**: the unshaped environment score on the training distribution (the per-rollout mean of
  `metadata['raw_reward']`); the first signal of whether the policy is learning, and immune to reward shaping.
- **Normal range** (measured): cold start 0.31–0.38 (the smokes and the first rollout of all three topologies
  land in this band); 0.52–0.58 after convergence at 500 steps. The whole curve rises slowly and monotonically;
  ±0.05 jitter is allowed.
- **Alarm threshold**:
  - **A single-step cliff**: losing more than half within one step = policy degeneracy, not noise. Measured
    sample: two nemo-rl fully-async arms fell from 0.50 to 0.02 at step 254 simultaneously.
  - Flat for the first 50 steps and clearly below 0.3: most likely a structural zero reward, not slow learning.
- **Diagnosis and fix**:
  - A cliff -> check the off-policy correction (the whole of Tier 1); the nemo-rl case's root cause was an
    unbounded token-level IS weight. Roll back to the pre-cliff checkpoint, add a correction / tighten
    staleness, resume.
  - A structural zero -> check the agent loop before the algorithm: the measured mrsmoke3 EM≈0 was the engines
    not bound to the right tool-call parser (back then `--sglang-tool-call-parser` had to be passed by hand; now
    the launcher binds it from the TITO family of `MODEL_FAMILIES`, so on this symptom first check that the
    family row matches the model), the engine returning the tool-call markup as bare text and the episode
    ending after one turn. Criterion: median response_len ~60 and
    `multi_turn/multi_turn_metric/round_number_mean` ≈ 1.
  - Diverging from `rollout/rewards` (the post-processed reward the optimizer actually consumes) -> check the
    shaping logic itself.

### `eval/<ds>` (the per-dataset held-out score)
- **Why it matters**: in-loop validation during training. Note that it takes `--eval-reward-key` (here = the
  unshaped env score), so it may not share a source with `eval/<ds>/episode_raw_reward`.
- **Normal range** (measured, the 893-row 7-source macro-EM): step 0 = 0.220; mid-run 0.43–0.45; final
  0.45–0.47. Healthy arms sit within ≤2 pt of each other.
- **Alarm threshold and noise discipline**:
  - **At n = 128 per source the 95 % CI is ≈ ±8.6 pt** — the 893-row small eval **cannot** rank close arms or
    settle a checkpoint choice. Measured lesson: the TIS arm's "edge" of .4534@349 on the 893 set did not
    replicate on the 51,713-row full set (.4354, tied with the baseline). Use the small eval for trends and
    for catching collapses only.
  - A high training-side eval but poor deployment: measured nemo-rl as2n self-eval .4623 vs deployment .1342
    — the model lost the `<answer>` tag format and its own eval path was more lenient. **Conclusion:
    re-evaluate periodically through an independent deployment harness** (the deployment path itself carries a
    +0.8 pt systematic reading bias).
- **Fix**: compare close arms on the full set, always; a flat curve (the RLP arm at .33–.36 throughout) ->
  check Tier 1's logprob quality first, then the algorithm.

### `rollout/response_len/mean` (+ `rollout/episode_response_length/mean`)
- **Why it matters**: in agentic tasks, length is the shape of behaviour. Multi-turn tool use lengthens
  responses as it starts; a length collapse = the model giving up on tools; exploding length + a falling score
  = a degenerate basin. The former is the per-sample count of effective (trainable) tokens, the latter the
  per-rollout sum over turns; either works for trends — pick one and stick with it.
- **Normal range** (measured): the smoke grew from 71 to 1111 (multi-turn kicking in); a trained healthy
  fully-async arm ~1450 tokens.
- **Alarm threshold**:
  - Stable at ~900 with flat EM: a degeneration signal (measured on the RLP arm: the engine-logprob denominator
    hurt the policy).
  - >1400 **and** eval falling at the same time: a degenerate basin (the signature of the nemo-rl collapsed
    arms). Length alone is no alarm; "long + falling" is.
  - Median ~60: a broken loop (parser / tool-result backfill failure), not a learning problem.
- **Fix**: a collapse -> check the corrections and staleness; an explosion -> read it with `truncated_ratio`
  and ppo_kl to judge whether it is out of control; consider tightening staleness / adding TIS / rolling back.

### `rollout/truncated_ratio`
- **Why it matters**: the truncation rate is the most sensitive behavioural indicator of a staleness pathology
  — in this campaign it reflected the off-policy mismatch (GRPO length inflation) earlier and more
  monotonically than the reward did.
- **Normal range** (measured, final values): colocated 1.4 %; drain-async 8 %; fully-async + TIS ≤5 %. The
  inflation grows monotonically with the staleness dose.
- **Alarm threshold**: **≥0.15 alarms**; the measured pathological band is 0.18–0.46 (uncorrected fully-async
  ended at 43 %; the RLP arm spiked twice to 0.18–0.21). Note: the raw reward may look normal (measured: raw
  still .53 at 43 % truncation) — **never watch the reward alone**.
- **Diagnosis and fix**: length inflation under an uncorrected off-policy denominator + high staleness. In
  order: turn on TIS (measured: brought fa3n from 43 % down to ≤5 %) > lower the staleness dose (max-age /
  update interval) > switch to drain mode. Engine logprobs as the bare denominator (the rollout-logprobs
  route) measured harmful: do not use.
- The trainer-side twin `rollout/truncated` has the same values; ignore one of the two.

### `train/grad_norm`
- **Why it matters**: the catch-all for numerical training health; collapse precursors often show here before
  the loss.
- **Normal range** (rule of thumb): small fluctuations inside a steady band; the absolute value depends on the
  configuration, watch the relative change.
- **Alarm threshold** (rule of thumb): sustained >5–10x the baseline, or NaN/inf; accompanied by `train/loss`
  spikes.
- **Fix**: check the data side first (over-long samples, None rewards mixed in), then lr / clip; together with
  a staleness spike, handle it as Tier 1.

### `multi_turn/multi_turn_metric/round_number_mean` (needs `--log-multi-turn`)
- **Why it matters**: agentic-specific — the number of tool-call rounds shows directly whether the model is
  using the tools. A broken parser or a session fault hits this first.
- **Normal range** (measured): a healthy search_r1 task is multi-round (>1, drifting slightly with training).
- **Alarm threshold**: ≈1 = a broken loop (the mrsmoke3 signature); a sudden drop in rounds = the tool path or
  the session server is in trouble.
- **Fix**: check `--sglang-tool-call-parser`, the gym /run error rate (Tier 2), session server liveness.

---

## Tier 1 — off-policy / staleness health (async topologies)

A must-read for every async topology: this is where agentic asynchronous training dies most easily.

### `train/ppo_kl` + `train/pg_clipfrac`
- **Why it matters**: the divergence between the new and the old policy and the clip trigger rate — direct
  readings of the off-policy strength.
- **Normal range** (rule of thumb): |ppo_kl| ≲ 0.01–0.03 per step; clipfrac in the low single-digit percent.
- **Alarm threshold**: ppo_kl persistently >0.1, or a long one-sided drift of its sign; clipfrac >0.2.
- **Pitfall** (measured): **under GSPO these two keys silently become sequence-level quantities** — the ratio
  is the per-sequence geometric mean broadcast to the tokens, and clipfrac becomes "the token share of the
  clipped sequences". A GSPO arm's values cannot be compared across to GRPO / PPO arms; and GSPO needs a
  sequence-level eps-clip (this repository uses 4e-4; keeping 0.2 silently switches clipping off).
- **Fix**: high KL -> lower lr / tighten clip / lower staleness; a one-sided drift together with rising
  truncation = the length-inflation pathology, follow Tier 0's truncated_ratio fix.

### `train/train_rollout_kl` + `train/train_rollout_logprob_abs_diff`
- **Why it matters**: the divergence between the rollout engine's logprobs and the trainer's recomputation
  (k3, clamped ±10). The direct quantification of "can the engine's numerics be trusted" — it decides whether a
  correction such as TIS / RLP has a foundation to stand on. Emitted whenever the batch carries rollout
  logprobs.
- **Normal range**: abs_diff ≤0.03 (the upstream CI assertion band; measured healthy runs sit far below it).
- **Alarm threshold**: abs_diff persistently >0.05, or a step-like rise of the KL.
- **Diagnosis and fix**: engine / trainer precision mismatch, KV reuse across weight versions, sglang version
  issues. When the mismatch is large: **never** use the engine logprobs as the bare denominator (measured on the
  RLP arm: EM flat at .33–.36 throughout, responses collapsed to ~900 tokens); clamped, as a TIS correction
  term, they are safe. nemo-rl's counterpart GenKLErr, measured: the drain arm's healthy band .1–.19, the fatal
  collapse .62.

### `train/tis` + `train/tis_clipfrac` (with `--use-tis`)
- **Why it matters**: the TIS weight mean and its clip rate = the correction pressure gauge. Low pressure means
  a gentle correction (a stability insurance), high pressure means the distribution has drifted far.
- **Normal range** (measured, tis_clip 2.0): weight mean 0.971, clipfrac 0.005, mean |log w| 0.095 — at a mild
  dose TIS is "almost free insurance".
- **Alarm threshold** (rule of thumb): clipfrac >0.05, or the weight mean off 1 by more than ±0.15: the
  staleness dose has left the range TIS corrects gently.
- **Fix**: lower staleness first (interval / max-age); TIS only underwrites stability. Measured conclusion: at
  a mild dose TIS brings no EM gain (tied on the full set); its value is preventing stability pathologies such
  as the truncation explosion.

### `rollout/weight_version/mixed_version_ratio`
- **Why it matters**: the share of requests whose decode crossed a weight update mid-way (continued after a
  retract) — the real "in-body mixed-version" dose. From miles v0.1.2 upstream counts it over per-token version
  spans (`Sample.weight_versions`; #1891/#1892); an earlier pin needed the weight-version-spans patch, and
  without the spans mixed_version_ratio is **structurally 0 (false health)**. The patch's companions
  `retracted_sample_ratio` / `retractions_per_sample` left with it (measured ≈ 0 throughout).
- **Normal range** (measured): steady state 0–3 % per rollout; isolated spikes .072 (the tis arm) – .168 (the
  rlp arm).
- **Alarm threshold**: a curve identically 0 -> suspect a blind spot before believing health (are the spans
  being emitted?); persistently >10 % -> the mixed-version dose is too high.
- **Fix**: identically 0 -> check the patch / the engine-side spans; a high dose -> lengthen the engine's
  weight-update interval, or accept it and make sure TIS is on.

### `rollout/fully_async/avg_staleness` / `max_staleness` / `queue_size`
- **Why it matters**: fully-async's direct staleness reading (the consumer group's current_version −
  oldest_weight_version) and the buffer depth.
- **Normal range** (measured): sample age 1–2 steps; queue_size steady, not growing monotonically.
- **Alarm threshold**: avg_staleness persistently >3–4 (rule of thumb; a dose range this campaign never
  measured — try it small before entering it); queue_size climbing monotonically = consumption cannot keep up
  with production.
- **Fix**: high staleness -> tighten the `--rollout-max-staleness`-type budget or raise the training
  throughput; a climbing queue -> check whether the trainer slowed down (see the perf panel).

---

## Tier 2 — environment and infrastructure (agentic-specific; env/* needs the env-metrics patch)

Aggregation semantics first: every `env/*` is a **whole-batch mean, a missing value counted as 0**.
`run_transport_failed` / `policy_400_truncated` are 0/1 -> read them as rates; `tool_errors_*` are per-episode
counts -> the mean can exceed 1.

### `rollout/env/run_transport_failed`
- **Why it matters**: before the fix, a gym /run transport failure showed up as a **silent 0.0 reward**,
  indistinguishable from a wrong answer, polluting the training signal directly.
- **Normal range** (measured): there is a steady baseline — ~150–180 /run 500s per run (a passing smoke
  carries ~174 too), a rate in the per-mille range. Plus a benign transient burst: once dynamic sampling has
  filled the batch it cancels the over-issued requests, measured 165 within 2 seconds, harmless.
- **Alarm threshold**: the rate stepping up for good (several times the baseline) or runs of consecutive
  failures.
- **Fix**: check the gym agent load, the session server, the retriever chain (below). The extreme case of
  100 % None rewards (measured: mrsmoke1, 128/128) was miles' `ray stop --force` preamble killing the gym's
  private ray — infrastructure killing infrastructure, not a model problem.

### `rollout/env/policy_400_truncated`
- **Why it matters**: the policy server answers 400 when the accumulated transcript exceeds max_model_len.
  Before the fix it **killed the whole /run** — measured: it took out two full-set evals (a long-output
  checkpoint). After the fix the episode is wrapped up as truncated and counted.
- **Normal range**: ≈0; a small share on long-output checkpoints (high-truncation arms).
- **Alarm threshold**: persistently >1–2 % -> a downstream symptom of the model's output length running away.
- **Fix**: look upstream at Tier 0's response_len / truncated_ratio; this is the effect, not the cause.

### `rollout/env/tool_errors_model_bad_args` + `tool_errors_env_<cat>`
- **Why it matters**: separates "the model's fault" (illegal tool arguments = a capability / format degradation
  signal) from "the environment's fault" (retriever timeouts and the like, keyed dynamically by exception
  category).
- **Normal range** (rule of thumb): model_bad_args low and falling with training; the env categories near 0.
- **Alarm threshold**: a rising trend of model_bad_args = the tool-call format degrading (the same kind of
  precursor as as2n losing its `<answer>` tags); a burst in an env category = an environment-side incident.
- **Fix**: model side -> inspect the latest checkpoint's behaviour, consider rolling back; environment side ->
  the retriever (a remote retriever behind a bridge: when it saturates the /search proxy hangs and sockets pile
  up), the FD limit, the bridge process.

### `rollout/env/tool_call_count` (tool_call_agent)
- **What it is**: per episode, the number of calls the model emitted that the engine's parser turned into
  structured tool_calls (illegal-argument ones included). search_r1's budget term `REWARD_BUDGET_COEF` reads
  this instead of counting markup in the response text — the markup format varies with the model family (a
  JSON-style or an XML-style parser), the count does not.
- **How to read it**: a per-episode count, the mean can exceed 1; read it together with
  `tool_errors_model_bad_args` to tell "many calls" from "bad calls".

### `eval/<ds>-none_reward_ratio`
- **Why it matters**: the share of unscored samples (aborted / transport failures), the precondition for
  trusting an eval number. None is coerced to 0.0 inside `eval/<ds>` — a high none rate pulls the eval score
  down systematically.
- **Normal range** (measured): ~0–2 % (the /run 500 baseline).
- **Alarm threshold**: >5 % and that eval number is unusable; =100 % is infrastructure death (the mrsmoke1
  kind).
- **Fix**: repair the infrastructure before reading scores; when comparing arms' evals, confirm that the none
  rates are of the same order.

### `rollout/tito_session_mismatch_rate/v1|v2` (+ 4 sub-classes)
- **Why it matters**: TITO token-alignment quality — the mismatch rate between the engine's output and the
  session replay, i.e. "are the tokens trained the tokens generated".
- **Normal range** (measured): smoke ≈0; over long multi-turn episodes the benign `assistant_text` sub-class
  (tool-call JSON re-serialization differences) can reach 0.22; **the strict sub-classes stay 0 throughout**.
- **Alarm threshold**: any strict sub-class >0 is worth stopping for — that is real alignment breakage.
- **Fix**: check the chat-template boundary concatenation (a chat-template boundary patch of an earlier pin
  was exactly this kind of problem), the tokenizer version, the session server version.

### `perf/step_time` + `perf/wait_time_ratio` + `perf/rollout_time`
- **Why it matters**: throughput and bottleneck location (a high trainer wait share = a rollout bottleneck).
- **Normal range** (measured): colocated steady at 47–48 s/step; drain-async 59→79 s (episodes lengthening);
  fully-async steady state down to 14 s; generation wait share 0.52–0.73.
- **Alarm threshold**: step_time climbing monotonically with no episode-length explanation; a stall monitor on
  the log's mtime (> 30 min) is the last line of defence.
- **Fix**: a high wait share -> add rollout capacity (measured: DAPO's 4–7x generation overhead cost only 18 %
  on this topology because the rollout fleet had idle capacity); a slow trainer -> check the offload /
  parallelism configuration.

### `eval/skipped_<reason>`
- **Why it matters**: explicit bookkeeping of silently missing eval points (busy / export_failed / crashed /
  ckpt_missing / unhealthy / pin_violation). When a point is missing from the curve, look here first; do not
  treat it as if nothing had happened.
- **Alarm threshold**: any reason other than busy is worth investigating.

---

## Tier 3 — data efficiency and algorithm-specific

### `rollout/zero_std/all_zero_percentage` + `all_one_percentage`
- **Why it matters**: zero-variance groups (all right / all wrong) contribute no gradient — under a binary EM
  reward this is the core data-efficiency reading and the basis for "should we switch to DAPO".
- **Normal range** (measured): late in training all_zero .37–.48 + all_one .31–.42, in total **75–85 %** of the
  groups have zero gradient -> DAPO's dynamic sampling needs ~4–7x the generation per kept batch.
- **Alarm threshold**: a total >90 % = the curriculum is nearly exhausted; the marginal return of further
  training is very low.
- **Pitfall** (measured): **with the DAPO filter on, this metric is computed on the filtered batch and reads
  ~0 throughout** — the real drop volume moves to `rollout/dynamic_filter/drop_zero_std_0.0/1.0` (window counts,
  not percentages; measured ~240 groups dropped per 64 kept ≈ 79 %).
- **Fix**: saturation -> a harder curriculum / other data / accept convergence; estimate DAPO's cost as the
  drop count ÷ (drop + kept).

### `passrate/pass@<k>` (needs `--log-passrate`)
- **Why it matters**: the within-group pass-rate distribution tells "can, but unstable" from "cannot at all"
  better than the mean reward; the rollout-side version groups by group_index and is robust to filtering.
- **Pitfall**: the eval twin `eval/<ds>-pass@<k>` (dash) reshapes with the global n_samples_per_eval_prompt,
  which mis-groups when a dataset overrides it — trust only the rollout-side one.

### PPO only: `train/critic-value_loss` + `train/critic-grad_norm` + `rollout/values`
- **Why it matters**: whether the critic has learned decides the advantage quality directly. The value head is
  randomly initialised, so a critic-only warm-up is a must (one step here).
- **Normal range** (measured, a run in progress): value_loss starts at ~0.26 and falls monotonically.
- **Alarm threshold** (rule of thumb): value_loss plateauing high or rising again; `rollout/values` out of
  scale with the actual returns.
- **Fix**: confirm `--critic-lr` (the default falls back to the actor lr, which is wrong; set 1e-5 explicitly)
  and the warm-up step; **a resume must pass `--critic-load <save>_critic`** (the default falls back to
  args.load and pours the actor's weights into the critic). Async PPO must pass `--no-offload-train` (offload x
  distributed weight push = CUDA illegal memory access; isolation matrix in patches/miles/README.md).

### `train/ess_ratio`
- **Why it matters**: the effective-sample-size share, a scalar summary of importance-sampling degeneration.
- **Normal / alarm** (rule of thumb): healthy ≳0.9; <0.7 means the weight is concentrated on a few samples and
  the gradient variance is high — locate it with tis_clipfrac and ppo_kl.

### `train/kl_loss` (needs `--use-kl-loss`)
- **Why it matters**: the anchoring strength to the ref model (a different divergence from ppo_kl; do not
  confuse the two).
- **Rule of thumb**: small and steady; sustained growth = the policy drifting away from ref, acceptable or not
  depending on the task; the KL coefficient this repository uses for PPO is 0.001 low_var_kl.

---

## Recommended dashboard: 12 panels

| # | panel | key (wandb glob) | what to see at a glance |
|---|---|---|---|
| 1 | training reward | `rollout/episode_raw_reward` | rising monotonically; a cliff = collapse |
| 2 | held-out | `eval/*` (the dataset main keys only) | the trend; flat / falling back |
| 3 | response length | `rollout/response_len/mean` | a collapse, or "explosion + score drop" |
| 4 | truncation rate | `rollout/truncated_ratio` | ≥0.15 alarms |
| 5 | policy divergence | `train/ppo_kl` + `train/pg_clipfrac` | drift and clip pressure |
| 6 | engine mismatch | `train/train_rollout_kl` + `..._abs_diff` | abs_diff ≤0.03 |
| 7 | mixed versions | `rollout/weight_version/mixed_version_ratio` | the 0–3 % band; identically 0 = blind spot |
| 8 | staleness | `rollout/fully_async/avg_staleness` + `queue_size` | age 1–2; the queue not climbing |
| 9 | environment faults | `rollout/env/run_transport_failed` + `policy_400_truncated` + `tool_errors_*` | a step = an incident |
| 10 | data efficiency | `rollout/zero_std/all_*_percentage` (under DAPO `rollout/dynamic_filter/drop_*` instead) | the share of zero-gradient groups |
| 11 | throughput | `perf/step_time` + `perf/wait_time_ratio` | an unexplained climb |
| 12 | numerical health | `train/grad_norm` (PPO adds `train/critic-value_loss`) | spikes / NaN / rising again |

Plus two disciplines: (a) a small eval (the 893-row class) is for trends only; a ±8.6 pt per-source CI supports no
ranking — verdicts come from the full set; (b) a good training-side eval ≠ good deployment; run an independent
deployment harness periodically (the lesson of as2n's 33 pt gap).

## Ignore list (hide from the dashboard)

- `correct_length/*`, `correct_entropy` — computed but never sent to wandb at the pin this guide was written
  against (the insertion point sits after the gather); skip them when a tutorial mentions them.
- `eval/<ds>/num_training_samples` (just the row count), `eval/<ds>/zero_std/*` (eval samples carry no
  group_index, the whole batch is one group: meaningless).
- `rollout/truncated` (= the redundant twin of `truncated_ratio`), `rollout/response_lengths` (the trainer-side
  twin; keep one of it and `response_len/mean`).
- The `median/min` of every statistics family (keep the mean; length-type keys may keep the max).
- `train/entropy_loss` is identically 0 unless `--entropy-coef` / `--observe-training-entropy` is on — not "the
  entropy is gone", it is not being computed. Since 2026-08-31 run.py turns observe on by default, so later
  runs carry a real value; for earlier runs use `-rollout/log_probs` as a proxy (measured healthy shape: starts
  at ~0.4 and eases down to a ~0.1 plateau; alarm on the shape, not the direction: a cliff down to <0.05 with EM
  stalled = collapse, a rebound / doubling = divergence — the PPO arm's late 0.47→1.02 was of this kind).
- The whole `<adapter>/*` family (multi-LoRA only); the `perf/non_generation_time/*` breakdown (the total
  throughput panel suffices).
- Treat the dynamic key families as patterns, not literals: `zero_std/count_<r>`, `dynamic_filter/drop_<reason>`,
  `env/tool_errors_env_<cat>`, `eval/skipped_<reason>`, `train/lr-pg_<i>`.

## Silent kill switches (when a metric disappears, look here first)

`--custom-rollout-log-function-path` / `--custom-eval-rollout-log-function-path` returning a truthy value
suppresses the whole family; `--load-debug-rollout-data` skips all rollout-side logging;
`--custom-async-data-buffer-path` replaces `rollout/fully_async/*` wholesale; `--wandb-always-use-train-step`
re-keys the x axis of rollout / eval (when step counts do not line up across runs, check it first). Log mining:
the metric-dict lines of a run log are hard-wrapped at ~1027 characters, so a number may be cut at its decimal
point (measured: 2wiki .453 misread as 0).
