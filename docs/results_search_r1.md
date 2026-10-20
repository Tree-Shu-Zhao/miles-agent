# search_r1 results: the full test set, the async fixes and the algorithm arms

The search_r1 recipe (`wiki_search`, Qwen2.5-7B; the colocated / async / fully_async topologies) trained in late
August 2026; this page keeps the measured numbers and the decisions those runs produced. Arm names are run tags
(`colo1n` / `colo`, `async3n` / `a3n`, `fasync3n` / `fa3n`, `tis`, `rlp`, `gspo`, `dapo`, `ppo`); the metric
vocabulary is `docs/agentic_rl_metrics.md`.

## 1. Full test-set evals (2026-08-29)

51,713 rows, the miles eval-only path (`recipes/search_r1/eval_full.py`: `--num-rollout 0` plus a pinned-tracker
load view; the weights are pushed to the engines before the eval), temperature 1.0, n = 1. Each arm's checkpoint
is its best on the 893-row validation mix (macro-EM): `colo1n` @399, `async3n` @499, `fasync3n` @449; ~75 min
per arm on one node.

| source (n) | colo399 | a3n499 | fa3n449 |
|---|---|---|---|
| nq (3,610) | .4543 | .4443 | .4463 |
| triviaqa (11,313) | .6339 | .6354 | .6328 |
| popqa (14,267) | .4422 | .4472 | .4437 |
| hotpotqa (7,405) | .4307 | .4320 | .4492 |
| 2wikimultihopqa (12,576) | .3830 | .4307 | .4300 |
| musique (2,417) | .1833 | .1895 | .1849 |
| bamboogle (125) | .4160 | .4000 | .4400 |
| **macro** | .4205 | .4256 | **.4324** |
| **micro** | .4568 | .4698 | **.4706** |

The full-set ranking inverts the 893-row one (893: colo .4848 > a3n .4713 > fa3n .4590). The two async arms win on
multi-hop (2wiki +4.7 pt at n = 12.6k, CI ±0.9 pt — real). Lesson: the 893-row set is n = 128 per source, a
per-source 95 % CI of ±8.6 pt, so ranking close arms on it is noise; async ≥ sync on the full test set.

## 2. The miles audit and its fixes (2026-08-29)

- sglang's miles fork already emits `weight_versions` spans in `meta_info` (token ranges per weight version,
  recorded for the in-flight and waiting requests at every version change) and `num_retractions`; the
  pause-retract path counts too. No sglang patch was needed.
- miles read only the scalar `weight_version`, so `rollout/weight_version/mixed_version_ratio` was blind to
  within-request mixing after a retract. Fixed in the pin (consume the spans and `num_retractions`; new metrics
  `weight_version/retracted_sample_ratio`, `retractions_per_sample`) and carried as a pin patch until the v0.1.2
  bump, which counts per-token spans upstream (`Sample.weight_versions`, #1891/#1892); the companion metrics left
  with the patch.
- Verified live in the fix arms: `mixed_version_ratio` became non-zero (spikes .168 on `rlp`, .072 on `tis`;
  identically 0 before). `retractions_per_sample` = 0 so far was genuine (engines near-idle at the update
  instants under the early buffer back-pressure; the mixing so far happened at turn boundaries).

| fix | what it changed | patch |
|---|---|---|
| per-token weight-version spans | `mixed_version_ratio` counts within-request retract mixing; `retracted_sample_ratio`, `retractions_per_sample` | `miles-weight-version-spans.patch` — retired: upstream in miles v0.1.2 |
| the Qwen2.5 TITO template boundary | the chat-template boundary of the TITO session server for Qwen2.5 | `miles-qwen25-tito.patch` — retired: not in `patches/miles/series` |
| save while resident | both drivers save each model inline while resident (section 6) | `miles-save-while-resident.patch` — in `patches/miles/series` |
| uncorrected async PPO | the `MILES_ALLOW_UNCORRECTED_ASYNC_PPO` escape hatch (section 5) | `miles-uncorrected-async-ppo.patch` — removed 2026-08-31: uncorrected async PPO is not run |

## 3. The async-fix arms: TIS vs RLP (2026-08-29)

`recipes/search_r1/run.py` grew the `use_rollout_logprobs` / `use_tis` fields (miles asserts the two are
mutually exclusive, so the requested combination arm became the TIS-only arm; `tis_clip` default 2.0). Two arms on
`fully_async_3n`, 500 steps: `fa3n_rlp` (the rollout-logprob denominator) and `fa3n_tis` (the clamped IS weight).

| step | rlp (893 macro-EM) | tis (893 macro-EM) |
|---|---|---|
| 199 | .3524 | .4187 |
| 249 | .3348 | .4142 |
| 299 | .3582 | .4086 |
| 349 | .3514 | **.4534** |

The uncorrected `fa3n` baseline's best on the same eval was .4324 @449. TIS led it by step 349 under gentle
correction pressure (mean weight .971, clipfrac .005, mean |log w| .095); RLP hurt — flat at .33–.36, responses
much shorter (~900 vs ~1450 tokens): the engine-vs-trainer logprob mismatch feeds straight into the gradient. The
staleness dose was mild throughout: `mixed_version_ratio` 0–3 % per rollout, zero retractions, sample age 1–2
steps. The 893-row edge did not survive the full-set deployment eval of the same checkpoint (.4354 on the
51,713-row set, tied with the baseline; `docs/agentic_rl_metrics.md`, `eval/<ds>`); what did survive is that TIS
removes the truncation instability (the rlp arm and the uncorrected baseline spiked to .18–.46 truncation, tis
stayed ≤ .05) at no EM cost, and that RLP hurts on every axis.

## 4. The algorithm arms on `fully_async_3n` (2026-08-30): design decisions

PPO, GSPO and DAPO, three nodes fully async, no TIS; all three shipped as `--algo` arms of
`recipes/search_r1/run.py`.

- **gspo**: miles implements the true sequence-level geometric-mean ratio but reuses the shared `--eps-clip`;
  token-style values such as 0.2 never trigger on a sequence ratio (silently). The GSPO recipes' configuration was
  adopted: eps-clip 4e-4 symmetric, KL off. Canonical GSPO, the deviation documented.
- **dapo**: both ingredients — clip-higher (the protocol already ran 0.28) and zero-reward-std group filtering with
  refill (true dynamic sampling: fully-async drops are replaced by new prompts, not recycled).
  `miles_agent/dapo_filter.py` wraps the stock filters (dict / None-reward safe; the stock one would kill the
  fully-async worker on a None reward). Measured cost from the earlier runs: all_zero + all_one ≈ 0.75–0.85 of the
  groups late in training -> ~4–7x generation per kept batch; the in-flight budget was raised to 128.
- **ppo**: the critic mirrors the actor's topology, both initialised from `--ref-load`, the value head random (one
  critic-only warm-up step), critic lr 1e-5. The async drivers require a behaviour-policy correction — TIS excluded
  by the directive, engine logprobs proven harmful (section 3) — so the arm ran with keep-old-actor first, then
  with the deliberately uncorrected denominator (section 5); there was no precedent for PPO x fully-async in the
  repository.

## 5. PPO post-mortem: why the uncorrected async arm collapsed (2026-08-30 / 31)

The run completed 500 steps; 893 macro-EM: .292 @0 -> **.408 @149 (peak)** -> .398 @199 -> .381 @249 ->
**.259 @299 (crash)** -> .233–.249 flat to 449. Controlled against `fasync3n` (GRPO, same topology, same uncorrected
denominator, ended at .459): the async conditions were identical or worse for the control (average staleness ~2.4
on both; GRPO absorbed 44 mixed-version steps > 0.1 and a max-staleness-14 excursion against PPO's single burst
and max 5). The algorithm, not the async plumbing, is the differentiator.

Mechanism (source-verified on the pin of the time):

- With no correction, `old_log_probs` is the current actor's own pre-step recompute, so the **ratio ≡ 1 at every
  step** (measured: ppo_kl = 0, pg_clipfrac = 0, ess_ratio = 1 for all 499 steps). eps-clip 0.2 / 0.28 was dead
  weight; "PPO" ran as an unclipped policy gradient on GAE advantages — exactly the failure mode the
  `validate_async_off_policy_correction` docstring names.
- PPO's absolute, critic-based advantages transmit off-policy shocks that GRPO's same-batch group-mean baseline
  cancels structurally (GRPO advantages log at a ~1e-9 mean).
- The critic regresses at weight 1 on GAE returns built from stale trajectories (no IS anywhere in the value
  path), and truncated samples are terminal-0 with no bootstrap, so truncation bursts poison the value targets;
  the biased V feeds back into the actor's advantages.

Collapse timeline: ref-KL drift from ~190 (kl_loss 2x the control by 250, matching the mild eval slip at 199–249)
-> an engine/trainer KL spike 0.02 -> 0.14 plus an entropy jump at 262 (the first sharp event) -> a mixed-version
burst at 274–284 (ratio .89 peak) -> a length / truncation blow-up at 300–307 (569-token means, truncation .377)
-> the training reward cliff .354 -> .008 in 9 steps (306–312) = the eval crash @299 -> degenerate 12–66-token
outputs afterwards, the entropy proxy rising monotonically to 1.02 by 499. **The critic metrics never led**
(value_loss flat at .19–.40 until 306; values tracked the returns all run): the critic was the amplifier by
construction, not the igniter.

Would TIS have helped? Partially, and probably enough to prevent this ignition: the collapse started actor-side
from exactly the engine/trainer mismatch TIS clamps (TIS held the same pathology to ≤ 5 % truncation on GRPO). But
`--use-tis` multiplies pg_loss only; the GAE targets and the whole critic regression stay uncorrected, and the clip
machinery stays inert. A principled async-PPO retry is `--keep-old-actor` (the true behaviour denominator at
interval 1; viable with `--no-offload-train`, `patches/miles/README.md`) plus `--use-tis` for the residual
engine-vs-trainer numerics; the critic's stale-state regression remains an open gap (V-trace territory, not in
miles). PPO's best checkpoint is 149, not the final one.

Follow-up (2026-08-31): the `MILES_ALLOW_UNCORRECTED_ASYNC_PPO` escape hatch was removed everywhere (the pin
commit dropped, the patch file and the recipe field deleted): uncorrected async PPO is not run.

## 6. The zero-weight checkpoint bug under `--offload-train` (2026-08-31)

The sync colocated PPO baseline's (`colo1n_ppo`) deployment evals first failed their pilot gate with pure
gibberish. The chain: every exported HF tensor was zero (339 / 339) -> the converter's logs clean and identical to
good runs -> the raw bytes at the `.metadata`-recorded chunk offsets showed the bf16 weight payloads zero on disk
in every one of the run's 10 checkpoints, while the optimizer state was real data.

Root cause (source-verified): with `--offload-train` miles force-disables the param-buffer CPU backup, so
`torch_memory_saver.pause()` discards the bf16 parameter contents and `wake_up()` maps fresh zero pages — only the
next train step's `weights_backuper` restore refills them. Both drivers' `use_critic` branches offload the
just-trained model before the save gate, so the save's `onload()` + `save_model` serialized zeros. Everything
downstream looked healthy: the in-loop evals use sglang's separately maintained weights, and the optimizer's fp32
main params have their own backup. GRPO and critic-less async runs save while resident and were never affected —
which is why only the sync colocated PPO arm was hit.

Fixes: (a) the pin patch `miles-save-while-resident.patch` — both drivers now save each model inline while
resident (`patches/miles/README.md`); (b) the weights were recovered without retraining by locating every
parameter inside the distributed optimizer's flat fp32 main-param buffer by value search, validated bitwise on a
known-good checkpoint before applying; the recovered exports passed the pilot gate.
