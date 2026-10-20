# ScienceWorld integration (scienceworld_env + recipes/scienceworld)

Fourth environment of the campaign (the R2S paper's fourth), ported from
the author's earlier environment code onto
the shared miles-agent stack. Everything env-agnostic is reused; only the
simulator mechanics and presentation are new.

## What is reused

| Piece | Reused from | Role here |
|---|---|---|
| `agent_utils.session_gym.SessionGymnasiumServer` | alfworld_env / webshop_env | cookie sessions, `/reset` `/step` `/close`, first-`act`-tool-call transport, format-fault hint, step budget, TTL sweeper |
| `agent_utils.procslots.ProcSlots` | alfworld worker / webshop engine tier | spawn'd worker processes, per-slot executors, timeout -> SIGKILL + respawn, generation counters, parked-slot revival |
| `responses_api_agents/env_loop_agent` | alfworld / webshop | the episode driver (chat completions against the per-request policy URL, tool replies, `scripted_prefix` replay, `last_observation` for the R2S takeover stage) |
| `miles_agent.launcher` + `RecipeSpec` | all recipes | Miles launch surface, eval slices, run timeout |
| `recipes/alfworld` shape | alfworld | run.py campaign protocol (64 prompts x 8 samples per step, 500 steps, eval every 50), prompt selection (`PROMPT=<name>`), the reward plugin |
| `miles_agent/prompts.py` | alfworld / webshop | the prompt rendered in the policy model's own tool-call markup |
| the training entry's gym ladder (`scripts/lib/gym_ladder.sh`) | all recipes | one `scienceworld)` arm + three functions (data prep, venv probe, pre-gym runtime check) |

## What is new

- `gym_root/resources_servers/scienceworld_env/`
  - `task_splits.py` — TCOD task sets (verbatim `TASK_VARIATIONS`, 17 train
    task types x first 50% variations = 2294; `train_heldout` = the other half;
    `test` = the frozen 1308-row TCOD taskset). Rows address a task by
    `task_name` + `var_num`; the split name on a row is validated for membership
    (no positional index invariant, unlike the earlier code).
  - `_worker.py` — `EpisodeHost` (one persistent `ScienceWorldEnv` = one JVM,
    `env.load(task, var, "easy")` per episode, running-max score, raw
    `getCompleted()`), the spawn target, and `WorkerPool` = `ProcSlots` plus
    **slot admission**: a JVM hosts one episode, so `/reset` waits for a free
    worker (`slot_wait_timeout_s`, then 503) and every release returns the slot.
    The JDK goes on PATH inside the worker (`java_home`), heap bounded via
    `JAVA_TOOL_OPTIONS` (`java_opts`, default `-Xmx2g`).
  - `app.py` — reset/step presentation in TCOD wording (task description,
    observation, compact action templates, current objects — re-listed every
    step), dense reward = running-max score delta / 100 (episode sum = TCOD
    task score in [0, 1]), `won` = raw simulator completion, `task_score` in
    every info; "No known action matches that input." counts as
    `invalid_actions` (no penalty). `env_workers=0` = in-process mode (tests).
- `recipes/scienceworld/` — `prepare_data.py` (stdlib + the server's
  `task_splits.py`; the prompt from `prompts/scienceworld/<PROMPT>.py` rendered
  for the policy model; the frozen taskset `scienceworld_test1308.jsonl` is
  not shipped: `scripts/fetch_data.sh scienceworld` rebuilds it, sha256
  `b2e1024f…2739`, from the campaign file `9632fe4f…ebf8`, `data/README.md`), `run.py`
  (the alfworld campaign config on this env), `reward.py` (identity). Eval
  file order: 140 `valid_seen` (seed-42 sample of train_heldout) then 134
  `valid_unseen` (first rows of the frozen test set); `scienceworld_test1308.jsonl`
  is the full TCOD eval set.
- `prompts/scienceworld/` — `default` (the paper's plain system prompt: TCOD's
  wording of the lists, brief deliberation, then one `act` call) and
  `r2s_oneshot` (the paper's one-shot prompt: the same system message plus one
  worked example as conversation turns); `prompts/README.md`.

## Scores (two of them)

- The **env score** is the episode's dense reward sum = the simulator's best
  score / 100, TCOD's task score in [0, 1]. It is what GRPO trains on
  (`reward.py` is the identity) and what the `eval/*` top-level numbers report.
- **Success** is the simulator's completion flag (`getCompleted()`), reported
  as `won` in `env_metrics` (`env/won` in wandb, per eval slice too). This is
  the R2S paper's metric. On ScienceWorld the two differ — partial credit is
  common — so success rates are read from `won`, never from the score
  (`docs/agentic_rl_metrics.md`).

## Runtime

- Venv: `scripts/build_gym_venvs.sh` builds
  `$GYM_VENV_ROOT/gym_venvs/resources_servers/scienceworld_env/.venv`
  (scienceworld 1.2.3 + py4j 0.10.9.9, pure-python wheels) and, under
  `--verify`, runs a real-JVM episode through the server with the pinned jar
  and JDK. Build it once for this env.
- Simulator jar: `$MILES_AGENT_DATA/scienceworld/scienceworld.jar` when that
  file exists — the TCOD repro campaign's pinned jar (sha256 `f4e1fdce…f0d7`,
  the scienceworld 1.2.3 build, copied from the author's earlier environment
  code). `SCIW_JAR` (alias of `SCIENCEWORLD_JAR`) overrides; `""` (no file, no
  knob) falls back to the wheel's bundled jar.
- JDK: Java 11+ — `SCIW_JAVA_HOME=<a JDK dir>`, else the first `java` on PATH (the author's runs used a JDK 17; the same floor as WebShop's runtime, `env_servers/webshop/setup.sh:70` `openjdk=11`, and the README's Install line).
  `SCIENCEWORLD_JAVA_HOME` and `JAVA_HOME` are the same knob under its other
  names (read in that order). py4j execs the first `java` on PATH; the worker
  processes put `$SCIW_JAVA_HOME/bin` there themselves. The training entry's
  `pre_gym_scienceworld` checks both exist before the gym servers start and
  hands jar / JDK / workers to the env server as gym overrides.
- Workers: `SCIW_ENV_WORKERS` (default 128, ~350 MB RSS each) = warm JVMs =
  concurrent episodes. The campaign's 512-rollout training wave (64 prompts x
  8 samples) and the 274-row eval wave queue on `/reset`'s slot wait
  (`slot_wait_timeout_s` 600 s; a timeout is a 503 = transport failure, the
  group is dropped and re-sampled, never trained as reward 0). Raise it, memory
  permitting, if the gym log shows 503s.
- Run: `[PROMPT=r2s_oneshot] [POLICY_MODEL=<hf dir name>] scripts/run_local.sh --recipe scienceworld --run-tag <tag>`
  (the rows are rebuilt per launch under `$MILES_AGENT_DATA/runs/$RUN_TAG`;
  `MILES_DATA_DIR` overrides). On a scheduler the same knobs ride the wrapper
  (`docs/cluster.md`).

## First GPU run of the env (2026-09-12)

Eval-only anchor with Qwen3.5-9B served as the policy (T 0.4, 30 turns; the
paper's campaign): eval metric = env score (dense TCOD task
score), `won` = completion.

| slice | rows | score | won | invalid actions / episode | truncated | transport failures |
|---|---|---|---|---|---|---|
| scienceworld_seen (train types, held-out variations) | 140 | 0.589 | 0.429 | 1.59 | 0 | 0 |
| scienceworld_unseen (frozen TCOD test, first 134) | 134 | 0.523 | 0.306 | – | 0 | 0 |

Mean episode response length ~1.2k (seen) / 1.5k (unseen) tokens; gym log clean
(no worker respawns, no 503s at 128 workers). An earlier pair of smoke runs
scored 0 everywhere with 85 % truncation: that was the engines serving released
weights under `--debug-skip-weight-update`, not the env.

## Protocol deviations (disclosed)

- Actions travel as `act` tool calls (`{"command": ...}`), not TCOD's
  `<action>` tags; the observation text and action/object rendering are TCOD's.
- Full-history episodes (every turn in context) instead of TCOD's 2-step
  window; same as the other three envs.
- `valid_seen` (train task types, held-out variations) is our addition; TCOD
  reports only unseen task types.

## Not in this recipe

The R2S / agentic on-policy distillation arms (the paper's method: a teacher
scoring the student's episodes) follow in the R2S port and will have their own
document; the paper's baseline-only machinery (gold action paths, the B2F
curriculum, FutureBridge) is not ported.
