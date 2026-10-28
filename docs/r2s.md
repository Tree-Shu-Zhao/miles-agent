# R2S: the OPD student arm, the pilot dump and the teacher repair pipeline

The paper's method in main: the student arm (§2), the pilot dump (§3), the repair pipeline (§4), the job
chain and its resume (§5), the fidelity notes (§6), what stayed on the branches (§7), the pitfalls (§8) and
the reproduction ledger (§9). Paths are relative to the repo root; `$MILES_AGENT_MODELS`, `$MILES_AGENT_DATA` and
`$MILES_AGENT_RESULTS` are the roots of `docs/local_setup.md`; every entry line below is a `scripts/` entry
(`run_local.sh`, `eval_ckpt.sh`, `r2s_teacher_ft.sh`) on one machine (a scheduler wrapper rides the same knobs:
`docs/cluster.md`).

Reproduction: the release tag `r2s` pins the code the paper's main-table cells were run with — `git checkout r2s` —
because later commits on `main` may delete or refactor any of this. The trained teachers and students are on the
Hugging Face Hub (24 repositories under https://huggingface.co/TreezzZ): each student as
`TreezzZ/r2s-<env>-<student>-student` and the repaired teacher that distilled it as
`TreezzZ/r2s-<env>-<teacher>-teacher-for-<student>`, its base model in the name.

## 1. What the method does

Agentic on-policy distillation (OPD) trains a student on its own multi-turn episodes against a frozen
teacher: the student's whole episode, observations included, is scored byte-identically in a separately
served teacher context, and the sampled reverse KL to the teacher's per-token log-probabilities is the
whole loss; the task reward is logged, never trained on. R2S repairs the teacher first. A pilot dump of the
untrained student's episodes (§3) is mined for the states where the student's deliberation diverges most
from the teacher's (stage A); the teacher takes each mined state over K = 8 times through the live
environment at T 1.0 (stage B); one takeover is selected per state (stage C); the teacher's own top-16
distribution with the selected takeover in context becomes the target on every assistant token from the
mined turn to the episode's end (stage D); the teacher is fine-tuned on those labels and promoted only if
the held-out privileged cross-entropy decreases (stage E). The selection rule is the paper's: any check
that tells a finishing takeover from a failing one will do; a state is kept when at least m of its
takeovers pass the check, the passing takeovers are grouped by how they end, and the shortest takeover of
the largest group is selected. For the main table the check is a consensus vote with m = 2 on ALFWorld
(over the takeovers that end within the turn budget) and m = 1 on ScienceWorld (over those that end with
the environment's completion message, `Task completed.`); on WebShop an LLM judge (Qwen3.5-9B with reasoning
on) checks each purchase against the instruction, m = 1, and the approved takeovers are grouped by the item
bought. The student is then distilled from the repaired teacher under the unchanged protocol of §2.

## 2. The student arm (`recipes/<env>/opd.py`, `--algo opd`)

### 2.1 Flags (`miles_agent/launcher.py`)

| flag | default | role |
|---|---|---|
| `--algo opd` | `grpo` | miles' native sampled OPD: the launcher emits `--advantage-estimator grpo --entropy-coef 0.0 --eps-clip 0.2 --use-opd --opd-type sglang --opd-kl-coef <beta> --opd-log-prob-top-k 0` and, with a teacher, `--opd-teacher-urls default=<teacher>/generate`; never `--use-kl-loss`, `--eps-clip-high` or `--rm-url` |
| `--teacher-url http://host:13141` | `""` | the teacher's base URL; the job's teacher gate passes it (§2.3); an eval-only run (`--num-rollout 0`) needs none |
| `--opd-kl-coef` | `1.0` | beta: with the train reward zeroed, the per-token advantage is `-beta * (student log-prob - teacher log-prob)` inside the PPO surrogate (clip 0.2 inert at ratio 1) |
| `--lr`, `--lr-warmup-iters`, `--weight-decay` | `1e-6`, `142`, `0.01` | the optimizer knobs (the presets set 1e-6 / 0 / 0.1); `--algo opd` adds `--use-checkpoint-opt-param-scheduler`, so a resumed link keeps the scheduler state |
| `--topology async_4n_teacher` / `async_2n_teacher` | `colocated_1n` | `async_3n`'s ray split (1 actor node + 2 rollout nodes) / one ray node split 2 + 2 GPUs, plus the teacher node (§2.3) |
| `--opd-dump` | off | the pilot dump (§3): exports `MILES_OPD_DUMP_DIR=<save dir>/opd_dump` to the actors |
| `--opd-score-concurrency`, `--opd-score-timeout-s`, `--opd-score-attempts` | 32, 900, 4 | the generate hook's scoring calls (train args, through `--extra-train-args`) |

`_validate_opd` refuses the flags rather than the allocation: `--algo opd` needs `--teacher-url` unless
`--num-rollout 0`; `--teacher-url` and the `*_teacher` topologies need `--algo opd`; `--rm-url` /
`--custom-rm-path` in `--extra-train-args`, `--use-tis` and `--use-rollout-logprobs` are refused; thinking not
disabled in `--chat-template-kwargs` is a WARN (the paper trained every family with thinking off).

How the loss comes about: `miles_agent/opd/generate.py` (`--custom-generate-function-path`) wraps miles'
agentic episode. For every training sample it segments the merged token stream into turns at the model's
own tool-call marker (`miles_agent/opd/turns.py`; the marker is derived from the chat template, never a
literal), posts the whole stream to the teacher's `/generate` with `max_new_tokens 0` and stores the sampled
log-probs in `Sample.teacher_log_probs` (`miles_agent/opd/teacher_scoring.py`; observation tokens score 0, a
token-id mismatch is a tokenizer drift error and is never trained on). miles' `apply_opd_kl_to_advantages`
turns them into the per-token advantage above; `train/opd_reverse_kl` is miles' own metric of it. A sample
the teacher could not score is ABORTED and its group dropped by the recipes' no-aborted filter: no step
trains on a half-scored sample (`rollout/env/opd_teacher_scoring_failed`, `OPD_TEACHER_SCORING_FAILED` in
the log). Pure distillation is the existing shaping plugin: under `--algo opd` the launcher sets
`REWARD_SHAPING_PATH=miles_agent.opd.zero_shape`, so the train reward is 0 while the env score still reaches
`rollout/episode_raw_reward` and every `eval/*` metric. Evaluation samples are never scored or dumped.

### 2.2 Presets (`recipes/{alfworld,webshop,scienceworld}/opd.py`, selected with `RUN_SCRIPT=opd.py`)

`OpdArgs(run.ScriptArgs)` pins the paper's protocol over each recipe's environment; data paths, eval slices,
group filters and the `/run` timeout stay run.py's. `PROMPT=r2s_oneshot` is the OPD arms' prompt (the paper's
one-shot rows, `prompts/<env>/r2s_oneshot.py`, rendered in the policy model's own markup).

| field | alfworld | webshop | scienceworld |
|---|---|---|---|
| algo / topology | opd / async_4n_teacher | same | same |
| num_rollout, rollout_batch_size, n_samples_per_prompt, global_batch_size | 150, 16, 1, 16 | 150, 16, 8, 128 | 150, 16, 1, 16 |
| rollout_temperature / eval_temperature | 0.4 / 0.4 | same | same |
| rollout_max_response_len / eval_max_response_len (per turn) | 512 / 4096 | same | same |
| rollout_max_context_len / train_max_tokens_per_gpu | 32768 / 32768 | same | same |
| lr / lr_warmup_iters / weight_decay (betas 0.9/0.98, clip 1.0 as every recipe) | 1e-6 / 0 / 0.1 | same | same |
| eval_interval / save_interval / update_weights_interval / seed | 25 / 25 / 1 / 42 | same | same |
| chat_template_kwargs | `{"enable_thinking": false}` | same | same |
| opd_kl_coef / opd_dump | 1.0 / off | same | same |

The paper's models: Qwen3.5-9B teaches Qwen3.5-2B and 4B, Qwen3-30B-A3B-Instruct-2507 teaches Qwen3-4B and 8B.
A repaired teacher's directory is named `r2s-<env>-<teacher>-teacher-for-<student>` (e.g.
`r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b`, the Hub names of the release) and resolves its model family by
the longest `MODEL_FAMILIES` key in its name. The paper's vanilla and anchor
cohort ran seed 43 on ALFWorld and ScienceWorld (42 on WebShop) under `async_2n_teacher`; the presets default
to seed 42 and `async_4n_teacher`, so a vanilla twin passes `EXTRA_ARGS="--seed 43"`.

### 2.3 The teacher: the last node of a `*_teacher` topology, or `TEACHER_URL`

`scripts/run_local.sh` accepts `TOPOLOGY=async_4n_teacher` (`--num-nodes 4`) and `async_2n_teacher` (`--num-nodes 2`); both
need `TEACHER_MODEL=<dir name under $MILES_AGENT_MODELS>` (`TEACHER_HF=<dir>` overrides the path) or
`TEACHER_URL=http://host:port`, an already serving teacher, accepted with any topology. With `TEACHER_URL` a
`*_teacher` topology takes one node fewer, 3 for `async_4n_teacher` and 1 for `async_2n_teacher`: the
ray side is unchanged (`async_3n`'s split, or the one node split 2 + 2 GPUs), every node is handed to ray, and
the old count is refused (`FATAL: TOPOLOGY=async_4n_teacher needs NNODES=3, got 4`). The node count is the
entry's `--num-nodes` (its environment name is `NNODES`, Contract A's under a wrapper; a wrapper's `-N`): the entry
FATALs `TOPOLOGY=async_2n_teacher needs NNODES=2, got 1` otherwise.

On the teacher node `scripts/teacher_node.sh` (over `scripts/lib/sglang.sh`; both public, no scheduler call)
starts sglang on `:13141` with TP 1 x `TEACHER_DP` (4), `TEACHER_MEM_FRAC` 0.75, chunked prefill 4096,
`TEACHER_CONTEXT_LEN` 33792 (the arm's 32768 context + 1024, so an episode that fills the student's context
still scores), no cuda graphs, no chat template and no parser (scoring only); waits for `/health_generate` up to
`TEACHER_WAIT_TRIES` (480) x 5 s; publishes `$SYNC_DIR/TEACHER_URL` whole (`TEACHER_UP` in the log) or touches
`TEACHER_FAILED`; and serves until the head's cleanup drops `$SYNC_DIR/ENDED` (`TEACHER_ENDED`). TERM and USR1
are logged and served through (`TEACHER_SIGNAL`), so a preemption's save on the head is never cut short. The
head's `teacher_gate` runs once the gym is up: `TEACHER_URL` wins, else it waits for the file; `curl
/health_generate`; the tokenizer probe `python3 -m miles_agent.launcher tokenizer-probe <student> <teacher>`
(the vocabulary size and the ids of an ASCII + CJK + tool-call probe must agree: `TOKENIZER_PROBE_OK`;
`TEACHER_TOKENIZER_CHECK=0` skips it); then `TEACHER_GATE_OK` and `--teacher-url <url>` on the run command. The
teacher's log is `$RUNS_DIR/<RUN_TAG>_teacher_<JOB_ID>.log`.

Entry lines (`scripts/run_local.sh`; the knobs are environment variables, the topology and node count are flags):

```bash
# the R2S arm: ALFWorld, Qwen3.5-9B's repaired teacher -> Qwen3.5-4B, the paper's protocol (seed 42)
RUN_SCRIPT=opd.py PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B TEACHER_MODEL=r2s-alfworld-qwen3.5-4b-teacher \
  scripts/run_local.sh --recipe alfworld --run-tag <tag> --topology async_4n_teacher --num-nodes 4
# the vanilla twin: the base teacher and the paper's vanilla-cohort seed
RUN_SCRIPT=opd.py PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B TEACHER_MODEL=Qwen3.5-9B EXTRA_ARGS="--seed 43" \
  scripts/run_local.sh --recipe alfworld --run-tag <tag> --topology async_4n_teacher --num-nodes 4
# the paper's vanilla / anchor cohort topology: one ray node split 2 + 2 GPUs, plus the teacher node
RUN_SCRIPT=opd.py PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B TEACHER_MODEL=Qwen3.5-9B EXTRA_ARGS="--seed 43" \
  scripts/run_local.sh --recipe alfworld --run-tag <tag> --topology async_2n_teacher --num-nodes 2
# an already serving teacher: one node fewer, every node to ray
RUN_SCRIPT=opd.py PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B TEACHER_URL=http://<host>:13141 \
  scripts/run_local.sh --recipe alfworld --run-tag <tag> --topology async_4n_teacher --num-nodes 3
```

The campaign branches' job knobs are refused with their replacement the moment the head starts
(`scripts/lib/retired_knobs.sh`): `TEACHER2_MODEL`, `ONESHOT_PROTOCOL`, `ALFWORLD_OPD_PROMPT`, `SCIW_OPD_PROMPT`,
`OPD_TOOL_CALL_FORMAT`, `TEACHER_TOOL_PARSER`, `WEBSHOP_TOOL_CALL_STYLE`, `*_OPD_REFS`, `WEBSHOP_OPD_REF_FILE`,
`TOOLS_FILE`, `VOTE_KEY`, `TAU_SELECT`.

### 2.4 Evaluation: the final checkpoint, three decodes

A cell is its final checkpoint decoded three times: the in-training eval after step 150 (the training seed 42,
T 0.4, 4096 tokens a turn) and two eval-only passes over the same weights with seeds 43 and 44, two
`miles_agent/eval_ckpt.py` reloads through the shim `recipes/<env>/eval_ckpt.py`. The paper's own cells used
42 / 1001 / 1002; the seeds only name independent decodes. Save ids are 0-based: the final checkpoint of a
150-step arm is `iter_0000149`, and the in-training decode is the eval at rollout 149, step 150 of 150. The
tool builds its load view only once the arm's tracker names the iteration (`--eval-load-iter`; the default -1
takes what the tracker names now, an iteration beyond `num_rollout - 1` is refused), so the decode runs can be
queued before the arm finishes. A decode run carries the arm's `PROMPT=r2s_oneshot`: `prepare_data` renders the
run's PROMPT, so an eval-only run without it would evaluate the weights under the recipe's default prompt.

```bash
# decodes 2 and 3: one run each with its own RUN_TAG, one node, the arm's prompt
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B scripts/eval_ckpt.sh alfworld --recipe-script opd.py \
  --eval-load-tag <arm> --eval-load-iter 149 --seed 43 --run-tag <arm>_d43
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B scripts/eval_ckpt.sh alfworld --recipe-script opd.py \
  --eval-load-tag <arm> --eval-load-iter 149 --seed 44 --run-tag <arm>_d44
# anchors (the teacher alone, the student alone): the HF weights, rollout-only, seeds 42 / 43 / 44
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-9B scripts/eval_ckpt.sh alfworld --recipe-script opd.py --eval-base --seed 42 --run-tag <tag>
```

The reported number is the mean +- std over the three decodes of `eval/<slice>/env/won`: WebShop's `won`
(never `purchased` or the raw score), ScienceWorld's `env/won` (its top-level eval key is the dense task
score). The paper's table entry averages the seen and unseen splits per decode on ALFWorld and ScienceWorld
and takes the 500 canonical goals (the `webshop` slice) on WebShop; `tools/r2s_decodes.py` prints both (§9).

### 2.5 wandb

The arms log to the recipe's project (`miles-<env>_run`, group = RUN_TAG) next to the GRPO arms.
`python tools/wandb_core_view.py --recipe alfworld webshop scienceworld --update` adds the pinned section
"Tier 4 · OPD arm (RUN_SCRIPT=opd.py)", one metric a panel: `train/opd_reverse_kl` (the loss; it falls over
training), `rollout/episode_raw_reward` (the env score, zeroed for the loss),
`rollout/env/opd_teacher_scoring_failed` (alarm > 0), `rollout/env/opd_turns`, `rollout/env/opd_delib_tokens_mean`,
`rollout/env/opd_turns_with_action_frac`, `rollout/env/opd_teacher_latency_s`, and `eval/<slice>/env/won` per
slice (scienceworld has no recorded view yet: the first run without `--update` prints the URL to record in
`tools/wandb_views.local.json`). The generate hook's per-episode counters surface as `rollout/env/<key>`, means over all samples:
`rollout/env/opd_teacher_ctx_overflow` separates a context overflow at the teacher from a transport failure,
`rollout/env/opd_empty_response` counts episodes without a response, `rollout/env/opd_turns_with_delib_frac`
the turns with any deliberation.

## 3. The pilot dump (`--opd-dump` under `--mode debug_rollout_only`)

The untrained student plays the arm's protocol at T 1.0 with no teacher and nothing training or saving: the
arm's prompt and caps, thinking off, 16 episodes a step (WebShop 16 x 8), seed 42, 150 steps = 2,400 rows on
ALFWorld and ScienceWorld (the pipeline requires >= 2,000) and 19,200 on WebShop (stage A's cap of 3 states
per goal absorbs the duplication). `--opd-dump` is independent of the algo: `--algo grpo` keeps the teacher
out and `TOPOLOGY=colocated_1n` the node count at one (the preset's `async_4n_teacher` is refused under grpo).
Under the pilot the launcher emits `--eval-interval 1000000 --save-interval 1000000 --skip-eval-before-train`
whatever the preset says: no eval before the first step and none within the run; the eval after the last step
is miles' own and stays (its save is a no-op under `--debug-rollout-only`).

```bash
RUN_SCRIPT=opd.py MODE=debug_rollout_only PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B \
EXTRA_ARGS="--algo grpo --opd-dump --rollout-temperature 1.0" \
  scripts/run_local.sh --recipe alfworld --run-tag <pilot> --topology colocated_1n
```

Rows go to `$MILES_AGENT_RESULTS/<RUN_TAG>/opd_dump/dump_<host>_<pid>.jsonl` (one file per actor process, appended and
flushed per row). Row v2 (`miles_agent/opd/dump.py`), the campaign's keys first, the additive fields after:

    rollout_id, index, status, reward (metadata["reward"], the env score), metadata (jsonable, minus
    env_metrics / agent_metrics / reward / raw_reward / transcript / session_server*; "tools" kept),
    env_metrics, tokens (prompt + response), response_length, loss_mask,
    turns [[start, end, delib_end], ...] (response-relative)
    v: 2, prompt (the row's chat messages), transcript (the structured chat env_loop_agent produced: the
    prompt messages, the reset observation, assistant messages with their tool calls, tool / user
    observations), tito_model, chat_template_kwargs

The transcript rides `sample.metadata` for one hop (gym_bridge forwards the agent's messages only when
`MILES_OPD_DUMP_DIR` is set) and is popped from every sample before anything else sees it; rows are written
before scoring and carry no teacher log-probs. Stage A reads tokens / loss_mask / turns, stage B replays the
transcript's assistant messages as a `scripted_prefix`, stage D splices into the prompt tokens: no stage
parses chat text. Campaign-era rows (no transcript) stay readable by stages A and D; stage B refuses them.

## 4. The repair pipeline (`miles_agent/r2s/`)

Five stages and a judge, each `python -m miles_agent.r2s.<stage>` over plain jsonl: A `divergence` mines the
dump, B `takeover` lets the teacher finish the mined states through the live env, J `judge` (WebShop) reads
each purchase against the instruction, C `select` picks one takeover per state, D `label` writes the
privileged top-k targets, E `teacher_ft` fine-tunes the teacher on them. `MODE=r2s_pipeline` runs A..D on one
node (§4.4); E is its own job (§4.5).

### 4.1 The per-environment table (`miles_agent/r2s/select.py`) and `--env-config`

`--env alfworld | scienceworld | webshop` selects an `EnvConfig` from `ENV_DEFAULTS`, the campaign's values:

| field | alfworld | scienceworld | webshop | read by |
|---|---|---|---|---|
| `goal_keys` (the per-goal key, joined with `/`) | `game_file` | `task_name`, `var_num` | `goal_id` | A's cap, E's split |
| `action_tool` | `act` | `act` | `act` | B's command rule |
| `vote_key` (how takeovers are grouped) | `terminal` | `terminal` | `purchased_asin` | B, C |
| `check` | `vote` | `vote` | `judge` | C (J inserted for `judge`) |
| `m` (votes the largest group needs) | 2 | 1 | 1 | C |
| `done_regex` (the terminal observation must match) | `""` | `Task completed.` | `""` | C |
| `label_frac_gate` (`LABEL_FAIL` below it) | 0.25 | 0.25 | 0.0 | D |
| `allow_invalid` / `require_action` | true / true | same | same | C / A |
| `max_steps` (fallback when a row lacks `metadata["max_episode_steps"]`) | 30 | 30 | 15 | B |

`vote` admits the takeovers that replayed, ended inside the turn budget with a terminal key and at least one
step, format-faulted steps included (`allow_invalid`), and whose last observation matches `done_regex`, and
groups them by the fingerprint of the last observation; `judge` admits the ones stage J approved
(`match_ok`) and groups them by the item bought. `"Task completed."` is the campaign's exact export, dot
unescaped (the simulator's failure line is `Task finished.`, so the selected set is the same). Shared constants
in the same file: `TOP_N` 2400, `MAX_PER_GOAL` 3, `MIN_DELIB_TOKENS` 3, `MIN_DIV_TURN` 0, `K` 8, `TAKEOVER_T` 1.0,
`TOP_P` 1.0, `MAX_TEACHER_TOKENS` 512, `LABEL_K` 16, `MAX_CTX` 32768, `MIN_STATES` 100, the judge's thinking on /
2048 tokens / concurrency 64 / T 0, the FT's epochs 1 / lr 2e-6 / grad_accum 8 / warmup 10 / min_lr_ratio 0.1 /
clip 1.0 / val_pct 10 / seed 42.

An override is a flag or `--env-config file.json`, a table `{"<env>": {"<field>": value}}` laid over
`ENV_DEFAULTS` (an env the file alone defines needs `goal_keys`; a field no `EnvConfig` has is an error), never
a new default: a selection run alone is a config change. The per-field flags (`--check`, `--m`, `--done-regex`
on `select`; `--require-action` / `--no-require-action` on `divergence`; `--min-labeled-frac` on `label`) win over
the table.

### 4.2 The shared CLI surface and the file contracts

Stages A-D and J take `--env`, `--env-config`, `--out <jsonl>`, `--summary` (default `<out>.summary.json`),
`--force` (rerun although the summary already carries the stage's marker), `--limit N` (a smoke: the first N
inputs), `--seed 42` and `--stop-file <path>` (the preempt sentinel: stop between rows, exit 0 without the
marker); E (`teacher_ft`, a torchrun worker) has `--seed` alone of that set. A summary is the stage's JSON (its
counts, knobs and the resolved `EnvConfig` under `config`) with the marker as a key (`"DIVERGENCE_DONE": true`)
and again as the file's last line, which the job greps. Stages B, J and D append and flush per state (J per
verdict) and, restarted, skip the keys already present, `(state_id, sample)` for B and J and `state_id` for D,
unless `--force`; A and C write their rows once at the end. Every server flag is the server's base URL
(`http://host:13141`, the launcher's `--teacher-url` rule): A and D append `/generate`, B `/v1` to
`--teacher-url` and `/run` to `--gym-url`, J `/v1/chat/completions`; a URL carrying a path, a query or a
fragment is refused at the flag. The row schemas are REF's (the campaign branch's), so the campaign's files
and these are interchangeable; new fields are additive.

**Stage A, divergence** (one prefill per episode on the student's base checkpoint and on the teacher; the
turn with the largest mean `lp_student - lp_teacher` over its deliberation span, `>= 3` scored tokens, is the
episode's state; wins and losses alike; 3 states per goal, then the global top 2400):

```bash
python -m miles_agent.r2s.divergence --env alfworld --dump-glob '<dir>/dump_*.jsonl' \
    --student-url http://127.0.0.1:13142 --teacher-url http://127.0.0.1:13141 --out states.jsonl \
    [--tokenizer-path <student HF dir> --top-n 2400 --max-per-goal 3 --min-delib-tokens 3 --min-div-turn 0 \
     --require-action --max-ctx 32768 --workers 32 --tito-model <family> --chat-template-kwargs '<json>']
```

Turn spans come from the dump when the row carries them for the requested family; a campaign v1 row, or a
row cut at another family's marker, is re-segmented at the student's own marker derived from the
`--tokenizer-path` dir's fixed template (`--tito-model` / `--chat-template-kwargs` default to that dir's
`MODEL_FAMILIES` row); the flag is required only for such rows, and the job always passes it. State row = the
dump row's fields plus `state_id "<episode_key>:<turn>", episode_key, goal, split, div_turn, div_score,
div_rows`. Summary: `episodes_read, resegmented, long_ctx, prefill_failed, no_eligible_turn, scored, states_out,
goals_out, div_score_min, div_score_median, div_turn_hist, won_frac_in_states`, the knobs, `config`,
`DIVERGENCE_DONE`.

**Stage B, takeover** (per state and `sample < K`: the student's assistant messages before `div_turn` go to the
recipe's `env_loop_agent` as `scripted_prefix`, replayed through a fresh episode, and the teacher's sglang
server is the per-request policy until the episode ends or the row's own turn budget, prefix steps included,
runs out; thinking off and the tool-call parser are the served teacher's):

```bash
python -m miles_agent.r2s.takeover --env alfworld --states states.jsonl --gym-url http://127.0.0.1:8200 \
    --teacher-url http://127.0.0.1:13141 --teacher-model teacher --out fold.jsonl \
    [--samples 8 --temperature 1.0 --top-p 1.0 --max-teacher-tokens 512 --concurrency 32 \
     --run-timeout-s 1800 --min-replay-ok 0.7]
```

Row: `state_id, goal, split, div_turn, sample, replay_ok, [fail,] won, done, truncated, reward, turns_total,
invalid_actions, takeover_steps [{turn, delib, command, call {name, arguments}, format_ok, obs_before}],
last_observation, terminal_key, messages`. `replay_ok` needs the whole prefix stepped, every observation from
the reset one through the one after the last replayed turn equal to the dump's (whitespace-normalized) and a
teacher turn; `fail` is `prefix_ended@<n>`, `obs_mismatch@<i>`, `no_teacher_turn`, `run_transport:<Exc>` or
`turn_count_mismatch` (a state whose transcript and token spans disagree: K failure rows, no run; they stay in
the fold but leave the gate; a sample miles cut at the per-turn cap, `status` truncated, keeps its spans to the cut
turn while the agent played on, so its longer transcript is no disagreement and it runs). `done` = won, terminated,
or ended under the budget, and never truncated;
`terminal_key` is the fingerprint of the last observation (`terminal`) or the last `click[<asin>]` over prefix
and teacher commands (`purchased_asin`).
Campaign v1 rows (no transcript) are refused up front. Summary: `n_takeovers, replayed, turn_count_mismatch,
replay_ok_frac` (over the replayed rows; nothing replayed: no fraction and `TAKEOVER_FAIL`), `replay_ok, fail:<reason>,
won_frac, done_frac, states_with_vote_ge2, states_total, mean_teacher_turns, format_ok_frac`, the knobs,
`config`, `TAKEOVER_DONE`; a `replay_ok_frac` under `--min-replay-ok` (0.7, REF's default) ends the summary in
`TAKEOVER_FAIL` and the stage exits 2.

**Stage J, the WebShop judge** (the paper's check for WebShop: Qwen3.5-9B with reasoning on reads the shopping
instruction and the product page as it looked when the agent clicked Buy Now, plus up to two description /
features pages the takeover opened, and answers MATCH or MISMATCH; one chat completion at T 0 with
`enable_thinking` true in the request, which wins over the served template's default; the verdict is the
last verdict word of the reasoning and the content joined, the text REF's regex read on the campaign's server,
which had no reasoning parser: under the 2048-token budget 4,768 of the campaign's 6,232 verdicts were the
last verdict word of a reasoning cut at max_tokens, and a content-only read leaves those unparsed):

```bash
python -m miles_agent.r2s.judge --env webshop --fold fold.jsonl --judge-url http://127.0.0.1:13143 \
    --judge-model judge --out judge.jsonl [--max-tokens 2048 --concurrency 64 --timeout 300 --no-thinking]
```

Only the eligible takeovers are judged (the same set `select` considers). Row: `state_id, sample, match_ok,
match_text, n_unparsed, asin, selected` (a takeover without a Buy Now step: `match_ok` false, `"no buy step"`).
Rows are appended per verdict: a restarted run skips the `(state_id, sample)` already present unless `--force`
(`--limit` applies before the skip), and a stop keeps every verdict given. Summary, over the file's rows: `mode
"match", takeovers_judged, match_ok, unparsed_calls, thinking, judge_model, max_tokens` (the budget sent),
`fold_takeovers, config`, `JUDGE_DONE`. An env whose `vote_key` is not `purchased_asin` is refused at the flags.

**Stage C, select** (the rule of §1 over the fold; no gate, the paper gated at D):

```bash
python -m miles_agent.r2s.select --env alfworld --fold fold.jsonl --out tau.jsonl [--check vote --m 2 --done-regex '']
python -m miles_agent.r2s.select --env webshop --fold fold.jsonl --judge-file judge.jsonl --out tau.jsonl
```

Rows `{state_id, sample, group_size, n_eligible}`; summary `{n_states, n_selected, coverage, n_takeovers,
eligible_total, judge_missing, config}`, `SELECT_DONE`. Ties between groups go to the greater key; inside the
largest group the shortest takeover, then the lowest sample, is tau. A `judge` env needs `--judge-file`; an
eligible takeover the judge file lacks is not approved and is counted in `judge_missing`.

**Stage D, label** (per selected state, the frozen base teacher scores the student's own episode with tau
spliced in as one more user message right before the generation prompt: every token of every assistant turn
`t >= div_turn` gets the teacher's top-k at its shifted position, deliberation, call and call-less turns
alike; the earlier turns, the prompt and the observations are context):

```bash
python -m miles_agent.r2s.label --env alfworld --states states.jsonl --fold fold.jsonl \
    [--tau tau.jsonl | --judge-file judge.jsonl] --tokenizer-path <student HF dir> \
    --teacher-url http://127.0.0.1:13141 --out labels.jsonl \
    [--k 16 --max-ctx 32768 --workers 16 --min-labeled-frac 0.25 --check-render \
     --tito-model <family> --chat-template-kwargs '<json>']
```

Without `--tau` the rule of stage C runs inline over the fold (a `judge` env then needs `--judge-file`). The
privileged prompt is a token splice: the prompt's ids minus the generation prompt, the hint as the template
writes a user message, the generation prompt again (`miles_agent/tool_call_markup.py` renders the family's
fixed template twice and diffs; nothing is re-tokenized and no markup is spelled). The hint is REF's reference
walkthrough: a header, one line per teacher step (`<delib>\nAction: <command>`; a step without a command, a
format-faulted teacher turn, is its deliberation alone, or `(no action)` when that is empty), a footer. Row:
`{state_id, split, goal, div_turn, objective "suffix", ids, positions [{p, turn, dist, kind "hind", tgt_ids,
tgt_lps}], meta {tau_sample, tau_steps, tau_won, offset, hint_tokens}}` (`ids` are the plain episode's; `p`
indexes them). A state is dropped and counted as
`bad_div_turn`, `prompt_retok_mismatch` (the prompt does not end with the generation prompt), `long_ctx`
(over `--max-ctx` with the hint), `prefill_failed` or `no_positions`. Summary: the counts, `labeled`,
`labeled_frac`, `n_positions`, `goals`, `dist_hist`, `select_stats`, the knobs, `config`; `LABEL_DONE`, or
`LABEL_FAIL` with exit 2 when `labeled_frac` is under the env's `label_frac_gate` (`--min-labeled-frac`).
`--check-render` counts whether a full re-render of the messages with the hint equals the splice (a
diagnostic, never a drop).

**Stage E, teacher FT** (full-parameter FSDP under torchrun; the loss is the renormalised top-16
cross-entropy on the labeled positions, AdamW betas 0.9/0.95 wd 0, linear warmup then cosine to 0.1x, a
goal-disjoint 10 % validation split; label files with any other objective are refused):

```bash
torchrun --standalone --nproc-per-node 4 -m miles_agent.r2s.teacher_ft \
    --data labels.jsonl --model-path $MILES_AGENT_MODELS/Qwen3.5-9B --out-dir $MILES_AGENT_MODELS/r2s-alfworld-qwen3.5-4b-teacher.tmp \
    [--epochs 1 --lr 2e-6 --grad-accum 8 --warmup-steps 10 --min-lr-ratio 0.1 --clip 1.0 --val-pct 10 \
     --seed 42 --max-ctx 32768 --skip-gate]
```

The promotion gate: the held-out privileged CE on the labeled positions must drop from before to after
training. `FT_PASS` exports bf16 HF weights with the base `config.json` served verbatim and the processor
sidecars copied (a Qwen3.5 checkpoint keeps its frozen vision tower, so sglang serves the result like the
base) and writes `r2s_ft_report.json` (`passed`, `val_hind_ce_before` / `_after`, `objective "suffix"`, the
args) beside them. `FT_FAIL` writes the report with `"passed": false`, exports nothing and the worker exits 2;
`--skip-gate` exports anyway (`FT_GATE_SKIPPED`, `passed` false in the report). torchrun itself exits 1
whatever the worker's code, so the verdict is the report: `scripts/r2s_teacher_ft.sh` trains into
`<models>/<TAG>.tmp`, moves it to `<models>/<TAG>` on `R2S_FT_PASS`, and on a report saying `passed` false
prints `R2S_FT_FAIL` and exits 2 with the `.tmp` holding only the report (nothing promoted); any other
nonzero exit is `FATAL: torchrun exited <rc> before the gate verdict`. An existing `<TAG>` is refused, an
earlier `.tmp` is set aside as `<TAG>.failed-<job>`. The output name is `r2s-<env>-<teacher>-teacher-for-<student>`,
the student arm's `TEACHER_MODEL`; `MODEL_FAMILIES` resolves its family by the longest key in the name.

### 4.3 WebShop

Nothing in the stages is WebShop-specific beyond the table: `check judge` makes the job insert J before C,
`vote_key purchased_asin` groups the approved takeovers by the item bought, `label_frac_gate` 0 means every
approved state counts and nothing gates D, `max_steps` 15 is the fallback turn budget. The judge is the base
teacher by default (`JUDGE_MODEL` Qwen3.5-9B for both teachers, the paper's appendix): when its dir is the
teacher's, the teacher's server answers the judge's requests (their `enable_thinking` true wins over the
served template's default); when it is another dir, the job starts a judge server on `:13143` after stage B.

### 4.4 The run: `MODE=r2s_pipeline` (`scripts/r2s_pipeline.sh`, sourced by `scripts/run_local.sh`)

One `colocated_1n` node: the recipe's gym stack (ScienceWorld's JVMs, WebShop's engine) and data prep are the
entry's as for a training run (a wrapper adds its own staging, preempt sentinel and watchdog: `docs/cluster.md`);
once `GYM_UP` is printed the entry sources the script and runs `r2s_pipeline_main` in place of a driver. The
script makes no scheduler call: a plain machine with the recipe's `env_loop_agent` on `:8200` sources it after
`scripts/lib/sglang.sh` and calls `r2s_pipeline_main`.

| knob | default | role |
|---|---|---|
| `TAG` | required | names the outputs: `$MILES_AGENT_DATA/r2s/<TAG>/{states,fold,judge,tau,labels}.jsonl`, each with `.summary.json`; the served templates `teacher_template.jinja` / `judge_template.jinja` beside them |
| `RECIPE` | required | `alfworld` / `scienceworld` / `webshop`: the stages' `--env` and the gym the entry boots |
| `DUMP_GLOB` | required for A | the pilot dump, `$MILES_AGENT_RESULTS/<pilot>/opd_dump/dump_*.jsonl`; `STATES_IN` / `FOLD_IN` start from a finished stage's file instead (`STAGES` then defaults to `B,C,D` / `C,D`) |
| `STAGES` | `A,B,C,D` | any subset in that order; `J` goes before `C` when the env's check is `judge` and `C` runs |
| `TEACHER_MODEL` / `TEACHER_HF` | required | the base teacher being repaired (a name under `$MILES_AGENT_MODELS`, or a dir) |
| `JUDGE_MODEL` / `JUDGE_HF` | `Qwen3.5-9B` | the judge (§4.3) |
| `POLICY_MODEL` / `POLICY_HF` | `Qwen3.5-4B` | the student's base checkpoint: the tokenizer of the rows and stage A's student server |
| `R2S_ARGS_A` / `_B` / `_J` / `_C` / `_D` | `""` | more flags for one stage, verbatim and last (`--force` reruns a finished stage, `--limit N` is a smoke, `--env-config` a table): a knob change never edits the script. The value is word-split without quote re-parsing, so a JSON value is one word without spaces; the one case that needs it is a Qwen3 hybrid student whose campaign v1 rows were rendered thinking off (`MODEL_FAMILIES["Qwen3-"]` keeps the template's default, thinking on, so stage D would drop every such state as `prompt_retok_mismatch`): `R2S_ARGS_A='--chat-template-kwargs {"enable_thinking":false}'` and the same `R2S_ARGS_D` |
| `TOP_N`, `MAX_PER_GOAL` | 2400, 3 | stage A |
| `SAMPLES`, `TAKEOVER_TEMPERATURE`, `MIN_REPLAY_OK` | 8, 1.0, 0.7 | stage B |
| `K` | 16 | stage D |
| `CONCURRENCY` | 32 | A's and B's requests in flight |
| `MIN_STATES` | 100 | the states B needs (counted off the states file after A, or under `STATES_IN`) |
| `STOP_AFTER_FOLD` | 0 | 1: the job ends after B with `R2S_PIPELINE_PASS fold_only` (B must be in `STAGES`: FATAL before any server boots otherwise) |
| `TEACHER_DP`, `TEACHER_WAIT_TRIES` | 2, 480 | the teacher server's data parallelism; x 5 s per server boot, the preempt sentinel polled too |

Servers (`start_sglang`, logs `$RUNS_DIR/<RUN_TAG>_<teacher|student|judge>_<JOB_ID>.log`): the teacher on
`:13141` over GPUs `0..TEACHER_DP-1` with the family's fixed template (thinking defaulted off), its tool-call
and reasoning parsers and `--served-model-name teacher`, alive from the first stage through D (stage D's
labels are that teacher's own privileged pass, so it is never stopped before `LABEL_DONE`); the student's
base checkpoint on `:13142` over the next GPU for stage A alone (stopped after A); the judge on `:13143` over
the GPUs after the teacher's, only for J and only when `JUDGE_MODEL`'s dir is not the teacher's (a 4-GPU node:
teacher 0,1 / student 2 / judge 2,3 DP 2). A selection alone (`STAGES=C`) starts no server. Every server's wait
(`TEACHER_WAIT_TRIES` x 5 s) polls the preempt sentinel first: raised while a server boots, the entry writes
`<sentinel>.done` (`{"stage": "boot", "rows": 0}`), prints `R2S_PIPELINE_PREEMPT_STOP stage=boot` and exits 0
for the requeue (§5); a server that dies or never answers is FATAL with its log's tail. Gates and markers:
`DIVERGENCE_DONE` and `>= MIN_STATES` states -> `R2S_DIVERGENCE_PASS`; `TAKEOVER_DONE` -> `R2S_TAKEOVER_PASS`;
`JUDGE_DONE` -> `R2S_JUDGE_PASS`; `SELECT_DONE` -> `R2S_SELECT_PASS`; `LABEL_DONE` -> `R2S_LABEL_PASS`;
`R2S_PIPELINE_PASS`, `RUN_INNER_PASS` (and a wrapper's `JOB_PASS_miles_run_<RUN_TAG>`). A stage's gate refusal (exit 2:
`TAKEOVER_FAIL`, `LABEL_FAIL`) and a missing marker are FATAL. A stage whose summary already carries its marker
says so and does nothing (`R2S_ARGS_<S>=--force` reruns it).

```bash
# ALFWorld / ScienceWorld: A -> B -> C -> D in one run (~6 h; B dominates: 2,400 x 8 takeovers on DP 2)
MODE=r2s_pipeline TAG=<tag> TEACHER_MODEL=Qwen3.5-9B POLICY_MODEL=Qwen3.5-4B \
DUMP_GLOB="$MILES_AGENT_RESULTS/<pilot>/opd_dump/dump_*.jsonl" scripts/run_local.sh --recipe alfworld --run-tag <tag>
# WebShop: J is inserted before C. The campaign's split, the fold first and the judge later, is
# STOP_AFTER_FOLD=1 and a second run with the same TAG and STAGES=J,C,D
MODE=r2s_pipeline TAG=<tag> TEACHER_MODEL=Qwen3.5-9B POLICY_MODEL=Qwen3.5-4B STOP_AFTER_FOLD=1 \
DUMP_GLOB="$MILES_AGENT_RESULTS/<pilot>/opd_dump/dump_*.jsonl" scripts/run_local.sh --recipe webshop --run-tag <tag>
MODE=r2s_pipeline TAG=<tag> STAGES=J,C,D TEACHER_MODEL=Qwen3.5-9B POLICY_MODEL=Qwen3.5-4B \
  scripts/run_local.sh --recipe webshop --run-tag <tag>_jcd
# a smoke: 100 states, 20 of them taken over
MODE=r2s_pipeline TAG=<tag> TEACHER_MODEL=Qwen3.5-9B POLICY_MODEL=Qwen3.5-4B TOP_N=100 MIN_STATES=20 R2S_ARGS_B="--limit 20" \
DUMP_GLOB="$MILES_AGENT_RESULTS/<pilot>/opd_dump/dump_*.jsonl" scripts/run_local.sh --recipe alfworld --run-tag <tag>
```

### 4.5 The FT stage (`scripts/r2s_teacher_ft.sh`)

One GPU machine (~1-2 h for a 9B teacher on the campaign's labels); `torchrun --nproc-per-node <the machine's GPUs>`. Knobs: `TAG`
(the output name, `r2s-<env>-<teacher>-teacher-for-<student>`), `LABELS` (stage D's `labels.jsonl`, `LABEL_DONE`),
`BASE_MODEL` (`Qwen3.5-9B`), `LR` / `EPOCHS` / `GRAD_ACCUM` (2e-6 / 1 / 8), `FT_EXTRA` (more `teacher_ft` flags,
verbatim). Exit 2 = the gate refused the teacher (`R2S_FT_FAIL`, the report in `<TAG>.tmp`); exit 0 =
`R2S_FT_PASS out=<MILES_AGENT_MODELS>/<TAG>`.

```bash
TAG=r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b LABELS=$MILES_AGENT_DATA/r2s/<tag>/labels.jsonl BASE_MODEL=Qwen3.5-9B \
  scripts/r2s_teacher_ft.sh
```

## 5. Job chain and resume

- The chain of one cell: pilot (`colocated_1n`, ~2 h) -> pipeline A..D (~6 h) -> FT (~1-2 h) -> the student
  arm (`async_4n_teacher`, 150 steps, ~10-12 h) -> decodes 2 and 3 (one node each). Between steps only paths
  move: `$MILES_AGENT_RESULTS/<pilot>/opd_dump`, `$MILES_AGENT_DATA/r2s/<TAG>`, `$MILES_AGENT_MODELS/<TAG>`;
  inside a run every server is `127.0.0.1`. On a scheduler every link is a dependent job of the previous one
  (`docs/cluster.md`); on one machine the steps run in sequence. One RUN_TAG per training run, never reused (it is
  the wandb group and the save dir `$MILES_AGENT_RESULTS/<RUN_TAG>/`; the save dir's `wandb_run_id` file, written
  by the first link, makes every later link resume the same wandb run).
- A preemption or the time limit (Contract C, `docs/cluster.md`: the wrapper raises the preempt sentinel) makes
  the driver save the last trained step and the wrapper requeue or exit 75 with `JOB_STOPPED_TO_RESUME_<name>`
  for the next link; the teacher node serves through the signal (`TEACHER_SIGNAL`) and exits with
  the head (`TEACHER_ENDED`). A resumed link loads the tracker's save with the scheduler state
  (`--use-checkpoint-opt-param-scheduler`), serves the teacher again on its own last node and re-runs the gate;
  the rollout in flight is lost, nothing else.
- The pipeline job resumes per stage and per row: every stage gets `--stop-file $PREEMPT_SENTINEL` and every
  server wait polls the same file, so a raised sentinel ends the stage between rows or the wait while a server
  boots; the script writes `<sentinel>.done` (`{"stage", "rows"}`, `boot` / 0 for a wait), prints
  `R2S_PIPELINE_PREEMPT_STOP` and exits 0 for the wrapper to requeue; the requeued run skips the stages
  whose summaries carry their marker and the rows B, J and D already wrote. The FT job holds no state worth
  resuming: a stopped one leaves `<TAG>.tmp`, which the next run sets aside as `<TAG>.failed-<job>`.
- Decodes 2 and 3 wait on the arm's tracker (`--eval-load-iter 149`), so they can be queued behind the arm's
  last chain link.
- A wrapper's idle watchdog must take the job-wide maximum over nodes, so a quiet teacher node never trips it
  while the ray nodes train (`docs/cluster.md`).

## 6. Fidelity notes

Byte-equal to the campaign branch: the dump's `tokens` / `loss_mask` / `turns`; the state,
fold, judge, tau and label rows (REF's keys and values, additive fields after them); the per-env defaults (the
appendix table); `select_tau`'s ordering (the `(count, key)` tie-break, the shortest then the lowest sample);
the hint text; the privileged prompt ids (the token splice equals REF's decode-and-re-encode surgery:
`tests/test_r2s.py::test_privileged_prompt_equals_campaign_surgery`, and the byte check on the paper's ALFWorld
fold of §9); `teacher_ft`'s inputs, objective and report; the miles OPD flags the launcher emits.

Equivalent, not identical:
- State restore in stage B is `env_loop_agent`'s `scripted_prefix` with a post-hoc `replay_ok`; REF put an
  aiohttp proxy between the agent and the teacher that re-emitted the decoded student turns and refused the
  first mismatching observation inside the episode (400 `obs_mismatch@t`). The dropped set is the same, a few
  more episodes are wasted, the reset observation is compared too (REF's agent did not return it), and
  `turns_total` counts the replayed turns as REF's `turns` did.
- The teacher and judge servers render with the family's fixed template, thinking defaulted off (the
  training engines' template); REF served the HF template with a per-request flag. The same bytes for these
  conversations (the prompt-home change verified that the fixed template renders the paper's rows
  byte-identically; the pipeline smoke of §9 re-checks 20 takeover transcripts fixed-vs-HF).
- Seeds: the presets' 42 where the paper's vanilla cohort ran 43 on ALFWorld and ScienceWorld (§2.2, passed
  as `--seed 43`); decodes 2 and 3 under 43 / 44 where the paper's cells used 1001 / 1002 (§2.4).
- The teacher of an arm scores at context length 33792 (the arm's 32768 + 1024) where the campaign served
  32768: scoring only, so an episode that fills the student's context still scores instead of overflowing.
- BASE's format-fault hint to the policy is markup-neutral; REF's named the family's tool-call syntax.
- The judge shares the teacher's server when `JUDGE_MODEL` is the teacher's dir, and the teacher stays up
  through stage D; REF ran the judge as its own job on a fresh server. Stage A's `--workers` is also the
  prefill transport cap (REF had a separate env knob at the same value); stage B refuses campaign v1 rows
  rather than replaying them from decoded text; `--min-labeled-frac` defaults to the table's gate where REF's
  CLI said 0.5 and the campaign passed 0.25 / 0; `teacher_ft` writes its report on `FT_FAIL` too (REF wrote
  none), and the shell maps that report to exit 2.
- The pilot is `--algo grpo --opd-dump` under `debug_rollout_only` (no teacher); the campaign ran its opd
  recipe at lr 0 with a top-k flag that never touched the rows.

## 7. Not in main

The paper's campaign code is a private branch of the author's repository ("the campaign branch", REF in the module
docstrings); this repository at the release tag `r2s` is the public reproduction, and the port's row schemas read the
campaign's files unchanged. Left on the branch: the baselines of the main table (TCOD-F2B, FutureBridge, Guided-OPD) and
their modules, patches and submit scripts; the earlier OPD / R2S workflow with the four-node teacher topology; the
top-k KL variants (tab:rs) and the flags `--opd-kl-type` / `--opd-topk` / `--opd-teacher-alpha`, never emitted by the
launcher; stage B's `PolicyProxy`; the judge's grade / derive / pick / trace modes, the Search graders and `--report`;
the label stage's `gap_*` readout (tab:drift) and `--tau-select gt` (the ground-truth selector of the selection
ablation); the self-distillation, scope and label-SFT controls; the campaign orchestration, the per-env `*_opd` run
scripts, the Search recipe's OPD variant and the campaign analyses; and the prefix-objective teachers of the earlier
campaign (privileged thinking targets up to the mined turn): `teacher_ft` refuses their label files.

## 8. Pitfalls

- The pilot needs `--algo grpo` and `TOPOLOGY=colocated_1n`: the preset defaults to `--algo opd` under
  `async_4n_teacher`, and `_validate_opd` refuses an OPD arm without a teacher.
- The node count is the entry's `--num-nodes` (a wrapper's `-N`): the entry FATALs `<topology> needs <n> nodes`
  otherwise. With `TEACHER_URL` a `*_teacher` topology takes one node fewer (3 / 1), and the old count is
  refused as well.
- A decode or anchor run without `PROMPT=r2s_oneshot` evaluates the weights under the recipe's default prompt.
- The retired knobs of §2.3 are FATAL on purpose; the message names the replacement.
- `address already in use` in a server's log: a process outside the run's cleanup (an earlier run's leftover on
  the machine) holds `13141`, `13142` or `13143`. The driver probes the three before any server starts and moves a
  server up to the next free port, each above the one before (`R2S_PORT_SHIFT <SERVER> <from> -> <to>`); the
  stages take the shifted URLs, nothing else needs the number. A FATAL names the span when none of 20 is free.
- ScienceWorld's top-level `eval/<slice>` is the dense task score (partial credit is common): read
  `eval/<slice>/env/won`. WebShop's success is `won`, never `purchased` or `task_score`.
- An arm trained with a non-default `--num-rollout` needs the same flag on its decode lines (the save-id
  bound is read from the decode's own invocation) and on `tools/r2s_decodes.py`.
- Never point `--load` at a bare `iter_` directory: it appears before the save completes; `eval_ckpt.py`
  waits for the tracker instead.
- `rollout/env/opd_teacher_scoring_failed > 0`: those samples were ABORTED and their groups dropped, so the step
  trained on fewer episodes. Read the teacher's log and `rollout/env/opd_teacher_ctx_overflow` (raise
  `TEACHER_CONTEXT_LEN` only if the arm's context grew).
- The tokenizer probe fails for a teacher of another family or vocabulary, which cannot score the student's
  token ids; `TEACHER_TOKENIZER_CHECK=0` is for a deliberate experiment, not the protocol.
- A Qwen3 hybrid thinks by default; the presets pin thinking off, and `_validate_opd` WARNs when an override
  drops it. Its campaign v1 dump rows were rendered thinking off as well, while `MODEL_FAMILIES["Qwen3-"]` keeps
  the template's default: stages A and D read them under
  `R2S_ARGS_A='--chat-template-kwargs {"enable_thinking":false}'` and the same `R2S_ARGS_D` (the value is
  word-split without quote re-parsing, so a JSON value is one word without spaces), or stage D drops every such
  state as `prompt_retok_mismatch` and `LABEL_FAIL` follows.
- The stage CLIs import `miles_agent` from the first entry of `sys.path`, which `python -m` makes the current
  directory: run them with the repo root on `PYTHONPATH` (the job does) or from it. `teacher_ft --help` needs
  torch (the container has it; the Mac suite tests its parser directly).
- Every server flag is a base URL, `--gym-url` included; `http://host:13141/generate`, `/v1`, a `?query` or a
  `#fragment` is refused at the flag.
- Campaign v1 dump rows (no transcript) are read by stages A and D (`--tito-model` / `--chat-template-kwargs`
  stand in for the missing fields) and refused by stage B: a v1 dump cannot be replayed.
- `TAKEOVER_FAIL` is not resumed past: a rerun with every `(state_id, sample)` present posts nothing and
  fails again unless `R2S_ARGS_B` lowers `--min-replay-ok` or adds `--force`. Replay failures keep their
  transcripts in the fold, so a low `replay_ok_frac` is diagnosed from the fold alone.
- `TEACHER_DP` equal to the node's GPU count leaves no GPU for the student or judge server: the job is FATAL
  before any server boots. `MIN_STATES` is counted off the states file, also when B starts from `STATES_IN`.
- The judge is refused for an env whose `vote_key` is not `purchased_asin`. Its regex runs over
  `reasoning_content` + `content`, so the server may run with or without a `--reasoning-parser` (the job
  derives the family's); a content-only read left 59 of the WebShop smoke's 86 verdicts unparsed (§9),
  since most 2048-token replies end inside the reasoning.
- `tools/r2s_decodes.py` prints `unparsed=<n>` per run; a nonzero count means the decode shown may be an
  earlier eval than the run's last (§9).

## 9. Reproduction ledger

The decode summary (sample std, n - 1; `--ddof 0` for the population std):

```bash
python3 tools/r2s_decodes.py --runs <arm> <arm>_d43 <arm>_d44        # one RUN_TAG per decode; RUNS_DIR set
python3 tools/r2s_decodes.py --logs run_<job>.log run_<job>.log run_<job>.log   # one job log per decode
```

It reads each run's last `eval <rollout_id>: {...}` line from its job logs (the links of a resumed arm oldest
first; `--runs` finds them through `<RUN_TAG>_gym_<jobid>.log` under `--runs-dir`, default `$RUNS_DIR` else
`$MILES_AGENT_RESULTS/runs`, as `run_<jobid>.log` or, when the wrapper set its own log name, `miles_run_<jobid>.log`)
and prints, per slice, the mean +- std over the decodes with the values, plus the seen/unseen mean
per decode on ALFWorld and ScienceWorld. The job log carries that several-KB line in pieces with other
records between them; the tool joins them back, and each run's header line ends in `unparsed=<n>`, the eval
records of its logs that never closed. A nonzero `n` means the decode printed may be an earlier eval than the
run's last: read the log before entering it. An arm whose last eval is not rollout `--num-rollout - 1`
(default 150, the presets') is flagged `WARN`: a cut-short chain is not a cell.

Reproduction status (the port's own runs, 2026-10-18; every run from one pinned source tree, the markers and
summaries read from the logs):

- The eval tool on the paper's ScienceWorld students, 4 HF students x 3 decodes (seeds 42 / 1001 / 1002, before the
  42 / 43 / 44 default): each mean within 2 paper sd (1.0-point floor) of its paper row (`env/won`) and above the
  paper's vanilla OPD in all four — 9B -> 2B 39.2 +- 1.8 vs the paper's 37.9 +- 0.6, 9B -> 4B 48.2 +- 2.5 vs
  47.8 +- 3.0, 30B-A3B -> Qwen3-4B 44.4 +- 1.8 vs 45.8 +- 1.4, 30B-A3B -> Qwen3-8B 52.5 +- 1.7 vs 50.5 +- 1.4. PASS.
- The arm's pre-flight smoke on `async_2n_teacher` (`--num-rollout 2`): the arm boots end to end, `TEACHER_UP` ->
  `TOKENIZER_PROBE_OK` -> `TEACHER_GATE_OK` -> two rollouts, `train/opd_reverse_kl` 0.076 / 0.082,
  `rollout/rewards` 0.0 under `zero_shape`, `rollout/env/opd_teacher_scoring_failed` 0, teacher latency
  0.2-0.35 s, `opd_turns` ~22, the pass marker in 19 min (a first attempt with the wrong node count was refused
  by the entry's FATAL, §2.3). PASS.
- The ALFWorld Qwen3.5-9B -> 4B cell: the R2S arm (`TEACHER_MODEL=r2s-alfworld-qwen3.5-4b-teacher`, seed 42) and
  the vanilla twin (`Qwen3.5-9B`, `--seed 43`), 150 steps each, decodes = the in-train step-149 eval + seeds
  1001 / 1002: R2S 84.2 +- 1.1 vs the paper's 81.9 +- 2.4, vanilla 62.7 +- 2.7 vs 65.8 +- 2.6, R2S - vanilla
  21.5; `train/opd_reverse_kl` 0.08 -> 0.02 and never zero, `opd_teacher_scoring_failed` 0. PASS.
- `select` + `label` on the paper's ALFWorld fold against the campaign's suffix-objective labels: the identical
  summary (1,715 / 2,399 labeled, `labeled_frac` 0.7149, 684 `no_tau`, 2,129,896 positions) and every
  deterministic field (`tau_sample`, `ids`, `positions[*].p`, `offset`, `hint_tokens`) identical on 1,715 /
  1,715 states; the teacher's top-16: top-1 identical on 99.50 % of positions, 16-set identical on 82.5 %,
  probability-weighted abs. delta log p mean 0.009 / p99 0.064 (another SGLang build's noise); `teacher_ft` on
  those labels: 1 epoch / 192 steps, `val_hind_ce` 0.5564 -> 0.5389, `R2S_FT_PASS`. PASS.
- The pipeline on ALFWorld from the pilot dump (one 311 MB dump). The smoke on 20 states: A 20 states, B 160
  takeovers all replayed, `replay_ok_frac` 1.0, won 0.494, C 12 / 20, D 12 labeled, 10,374 positions; the §4.4
  smoke line with `R2S_ARGS_B="--limit 20"` ends in `LABEL_FAIL` 0.10 < 0.25 by configuration (B's `--limit`
  shrinks the labeled fraction, whose denominator is all of A's states). The full budget, 3 h 22 min: A 2,396
  states, `won_frac_in_states` 0.514; B 19,168 takeovers, all replayed, `replay_ok_frac` 1.0, won 0.550,
  `format_ok_frac` 0.994, 1,683 states with a vote >= 2; C 1,683 / 2,396; D 1,683 labeled, `labeled_frac` 0.702,
  2,061,591 positions (the campaign: 2,399 states, 0.519; 19,192, all replayed, 0.567, 0.981, 1,715; 1,715 /
  2,399, 0.715, 2,129,896); `scripts/r2s_teacher_ft.sh`: 1 epoch / 186 steps, `val_hind_ce` 0.5781 -> 0.5628,
  `R2S_FT_PASS`. PASS.
- The per-env paths, markers and summaries only, no number claimed: WebShop `STOP_AFTER_FOLD=1` on 50 states
  (B 400 takeovers, `replay_ok_frac` 1.0), then `STAGES=J,C,D`: J 100 judged, `unparsed_calls` 0, 74 % of the
  replies end inside the reasoning (campaign 76.5 %), `match_ok` 29, p(won | match_ok) 0.724 against
  p(won | all judged) 0.630 (campaign 0.77 / 0.677), C 10 / 50, D 10 labeled (an earlier judge that read the
  content alone left 59 of 86 verdicts unparsed, §8); ScienceWorld 50 states with the `Task completed.` regex:
  B all 400 replayed, `replay_ok_frac` 0.7675, C 32 / 50, D 32 labeled. PASS.
- Not run: the preempt drill on a smoke (save, exit-75 requeue, `TEACHER_ENDED`, the requeued link resuming with
  the scheduler state) and a 5-step `async_2n_teacher` smoke (the pre-flight ran two rollouts on that topology).

The cells (the paper's entries in parentheses: `tab:main`, Qwen3.5-9B -> Qwen3.5-4B, ALFWorld):

| cell | arm RUN_TAG / jobs | decodes 43 / 44 | seen | unseen | entry (paper) | when (PDT) |
|---|---|---|---|---|---|---|
| ALFWorld Qwen3.5-9B -> 4B, R2S | | | | | (81.9 +- 2.4) | |
| ALFWorld Qwen3.5-9B -> 4B, vanilla OPD | | | | | (65.8 +- 2.6) | |
| ALFWorld anchor: teacher alone (Qwen3.5-9B) | | 42 / 43 / 44 | | | (63.1 +- 1.7) | |
| ALFWorld anchor: student alone (Qwen3.5-4B) | | 42 / 43 / 44 | | | (43.4 +- 1.0) | |
