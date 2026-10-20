# WebShop environment integration

WebShop (Yao et al. 2022, the princeton-nlp text env) wired into miles-agent as
a third environment alongside `wiki_search` and `alfworld_env`, reusing every
miles-agent chassis component. The fork's text-env code is vendored and
committed under `env_servers/webshop/webshop`, so miles-agent is self-contained.
An engine-worker pool (below) adds the RL-scale concurrency the environment
needs.

## Why two tiers (the Python-version boundary)

nemo-gym resources servers require **Python 3.13.14** (`3rdparty/Gym`
`requires-python`). The WebShop engine only preserves byte-exact reward/search
semantics under its **Python 3.10** pins — `pyserini==0.17.0` (JVM/Lucene BM25
search), `spacy==3.7.2` (reward attribute matching), `gym==0.24.0`,
`Flask==2.1.2`. Bumping any of them to a py3.13-compatible release changes the
NLP or search path and breaks reproduction; the old pins have no py3.13
wheels. The two cannot share one venv.

So WebShop is split across that process boundary, and the nemo-gym side proxies
across it the way the in-repo `wiki_search` server proxies to its retriever:

```
miles train ─custom-generate─▶ agentic_tool_call.generate
                                     │  per-episode session URL
                                     ▼
                     miles_agent.gym_bridge.run  (unchanged)
                                     │  POST /run
                                     ▼
        env_loop_agent  (py3.13, UNCHANGED — the alfworld agent)
                                     │  Gymnasium /reset + /step, act tool call
                                     ▼
        webshop_env resources server (py3.13, agent_utils.session_gym.SessionGymnasiumServer)
                                     │  thin proxy: HTTP /reset /step /close
                                     ▼
        webshop engine server (py3.10, conda env + JDK) at 127.0.0.1:6200
                                     │  K engine workers (agent_utils.procslots):
                                     │  catalog + Lucene index + spaCy each
                                     ▼
                            vendored princeton-nlp/WebShop
```

Division of labor mirrors the rest of miles-agent: **Miles owns tokens**
(the policy model family's TITO), **env_loop_agent owns the episode protocol** (unchanged),
the **webshop_env adapter owns training-semantics presentation** (prompt
observation format, admissible-action rendering, reward routing), and the
**py3.10 engine owns env mechanics** (search, render, reward computation).
The session plumbing both Gymnasium envs need (session registry, per-session
lock + liveness re-check, done-replay guard, TTL sweeper, `/close`, act-tool
command extraction) lives once in `gym_root/agent_utils/session_gym.py`;
alfworld_env and webshop_env are subclasses that only implement
`_do_reset` / `_do_step` / `_release_state`.

## Action transport: tool call, not `<action>` tags

Common WebShop RL setups parse `<action>...</action>` text tags. miles-agent's
standing convention (and the whole `env_loop_agent` design) is that environment
actions travel as a structured **tool call** — the dataset row declares an
`act` tool, the model calls it with `{"command": "..."}`, and the env reads
the first tool call. WebShop reuses that verbatim (same shape as alfworld's
`act`). The `command` string is a raw WebShop action, e.g. `search[red mug]`
or `click[b09xyz]`, copied from the admissible-actions list. No text parsing
anywhere. The TCOD-protocol prompt (`PROMPT=think`, the default, `prompts/webshop/think.py`) asks
for a `<think></think>` reasoning block before the tool call, exactly as TCOD's
template does before its `<action>` tag; the adapter still reads only the tool
call (`none` = tool call only, the wstrain1/wstrain2 prompt). The prompt's
example call is rendered by the policy model's own chat template
(`miles_agent/tool_call_markup.py`, `prepare_data.py --model`): the wording
names no markup, so the rows belong to one model family and the job rebuilds
them for every launch.

The default family runs with the template's own thinking off (`MODEL_FAMILIES`:
Qwen3.5 and Nemotron take `enable_thinking=false`), so under `think` the
reasoning block is text the model writes before its call, as TCOD had it, not
the template's hidden thinking. The trainer sees the engine's token ids (TITO),
so the template re-rendering an earlier turn without its `<think>` block changes
nothing the model trains on. `PROMPT=none` drops the block, for a
family whose template keeps thinking on (hybrid Qwen3 at its default) or a
protocol other than TCOD's.

## Failure semantics (frozen — user decision)

**Environment feedback is never filtered or reshaped.** Invalid actions,
no-op clicks, bad purchases and low scores are legitimate signal and train
as-is. A non-admissible command spends its turn and is counted in
`invalid_actions` (for the optional penalty in `recipes/webshop/reward.py`);
what the model then reads is the adapter's `invalid_action_feedback` knob
(`WEBSHOP_INVALID_FEEDBACK`): **`error`** (default, TCOD protocol) does not
step the engine and shows TCOD's error sentence ("Incorrect action format,
make sure you have the correct button name that is within the current page.
The buttons you can click now are: [...]" etc.) with the unchanged admissible
list; **`noop`** (wstrain1/wstrain2) forwards the command and WebShop's own
silent no-op (page unchanged, reward 0) is the feedback.

**Infrastructure failures never enter training as a reward-0 episode.** The
mechanism, tier by tier:

1. **Engine.** A worker crash / op timeout / respawn drops the episodes
   resident in that worker; their next `/step` answers **410**. Other backend
   errors answer **500**. An unknown `goal_id` (or, on the legacy path, a bad
   goal index / non-int `task_id`) on `/reset` answers **400**, which the
   adapter never retries: a dataset/engine mismatch, not infrastructure. Past
   the adapter the two are indistinguishable (env_loop_agent raises on any
   non-2xx `/reset`, so the sample still lands in `run_transport_failed`);
   the engine/adapter log lines are what tell them apart.
2. **Adapter** (`webshop_env`). Any `EngineError` on `/step` — 410, 500, a
   connection failure after the client's bounded retries, or a malformed
   engine response — raises HTTP **502** and evicts the session (a failed step
   frees the episode instead of lingering to TTL); on 410 the engine has
   already dropped the episode, so the adapter marks its client handle closed
   (`EngineSession.forget()`) and skips the `/close` round trip. `/reset`
   regenerates the
   engine `session_id` for every retry attempt, so a stale earlier attempt can
   only create an orphan the engine TTL-evicts, never overwrite the live
   session.
3. **Agent + bridge.** `env_loop_agent` calls `raise_for_status` on the
   adapter's `/reset` and `/step`, so an adapter 5xx fails the whole gym
   `/run`; `miles_agent.gym_bridge.run` catches that (and `/run` timeouts /
   cancellations) and returns `env_metrics.run_transport_failed = 1.0` with
   no `reward` key.
4. **Trainer filter.** `recipes/webshop/run.py` selects
   `miles_agent.dapo_filter.check_no_aborted_no_transport_failure` (GRPO) or
   `check_no_aborted_no_transport_failure_nonzero_std` (DAPO) as the
   dynamic-sampling filter: a group containing ANY sample with
   `run_transport_failed` is rejected and re-sampled, so it never reaches the
   optimizer — at every topology. Each drop is counted in wandb as
   `rollout/dynamic_filter/drop_group_has_transport_failure`.

`rollout/env/run_transport_failed` in wandb therefore counts infrastructure
incidents, not learning signal (a dataset/engine goal-set mismatch that slipped
past the digest gate would land there too: grep the engine log for `is not in
the engine goal list`): a non-zero plateau means engine / adapter /
gym trouble, not a model regression.

### Reserved search tokens

`<a>`, `<c>`, `<q>`, `<r>` as the FIRST token of a search query crash the
upstream engine code. They are treated as a normal invalid action, never as an
engine error: the engine glue (`env_servers/webshop/_webshop_engine.step_env`)
replaces such an action with the no-op string `"invalid"` before stepping
(upstream: an action that parses as neither search nor click returns the
unchanged page, reward 0, done False), and the adapter counts it in
`invalid_actions` (`_format.is_reserved_search`) while still forwarding the
command unchanged. The vendored code under `env_servers/webshop/webshop/` is
not modified.

A related presentation detail: the search-bar submit button `click[search]`,
which upstream explicitly no-ops, is no longer advertised in the admissible
list (`_format.available_commands` drops it when the page has a search bar).

### Timeout and TTL coupling

Single source of truth for the shell side: `env_servers/webshop/defaults.sh`
(Python-side defaults exist only for manual runs and must equal it).

| layer | knob | value | rule |
|---|---|---|---|
| gym `/run` | `NEMO_GYM_RUN_TIMEOUT` (`recipes/webshop/run.py`) | 1200 s | 15 model calls under saturation exceed the 600 s default |
| adapter | `session_ttl_s` (`configs/webshop.yaml`) | 1800 s | > run timeout: a live episode is never evicted mid-run |
| engine | `--session-ttl` (`WEBSHOP_ENGINE_SESSION_TTL`) | 2400 s | > adapter TTL: the engine never evicts a session the adapter still considers live |
| engine | `--engine-call-timeout` (`WEBSHOP_ENGINE_CALL_TIMEOUT`) | 120 s | per IPC op; exceeding it SIGKILLs + respawns the worker |
| adapter | `engine_timeout_s` (`configs/webshop.yaml`) | 240 s | > engine call timeout + respawn margin (kill/join <= 5 s + replacement-engine boot): the adapter then receives the engine's definitive 410/500 rather than a client-side ReadTimeout. Holds while a replacement boots within ~115 s (`--small`); see the respawn-margin note below for `--full` |

Other stability facts:

- **The respawn margin is not bounded below 240 s.** A caller queued on a
  wedged slot waits, inline on that slot's executor, for the 120 s op timeout,
  the kill/join (<= 5 s) AND the replacement engine's boot, which is bounded
  only by `SPAWN_READY_TIMEOUT_S` (300 s, `_engine_worker.py`). With the
  `--small` catalog an engine boots in tens of seconds, so 240 s dominates;
  with `--full` raise `engine_timeout_s` (or lower `--engine-call-timeout`) if
  you want the definitive status. The training outcome is identical either
  way: a ReadTimeout is `EngineError(status=None)` -> adapter 502 ->
  `run_transport_failed` -> group dropped; only the log line and one harmless
  `/close` round trip (instead of `forget()`) differ.
- Both tiers run TTL sweepers (60 s) for sessions abandoned by aborted
  episodes (miles oversampling / weight-sync retracts leave orphans by design).
  One failing close does not stop a sweep.
- `/step` is non-idempotent (a WebShop click mutates engine session state), so
  the adapter's engine client retries `/step` **only** on pre-send (connect)
  failures — a blind retry could apply an action twice. `/reset` 5xx is
  retried with a fresh `session_id` per attempt; `/close` retries once on a
  connect failure only; 4xx is never retried.
- The vendored env auto-resets onto a random goal on `done`; the returned
  terminal observation precedes that reset. The engine drops a done episode
  itself and the adapter marks its client handle closed without a redundant
  `/close` round trip (`EngineSession.forget()`), so the auto-reset junk is
  never stepped.
- Per-episode engine session keys are prefixed unique (`<session>:`) so GRPO
  groups (8 samples of the SAME goal) never collide on the engine's
  `user_sessions` dict.

## Concurrency (the RL-throughput lever)

The recipe runs **128 concurrent episodes** per rollout step (16 prompts x 8
samples, `recipes/webshop/run.py`), each stepping the engine up to 15 times.
WebShop steps are cheap HTML renders except the Lucene search (a JVM call),
but the vendored stack is single-threaded by construction (global Flask app
contexts, module-level `random` reseeded per env, one shared `user_sessions`
dict), so ONE engine can only be touched serially. A single shared engine
behind one global lock (`engine_workers=0`) makes every search of every
concurrent episode queue behind every other — the same shape of hazard that
livelocked the alfworld trainer (2026-08-31) before its worker pool. (No
per-op latency figures are quoted here; measure your own wave via `/health`
and the engine log before tuning K.)

Two modes, selected by `--engine-workers` (mirrors alfworld's `env_workers`):

- **`0`** — one shared `SimServer` behind an `RLock`. Test / fallback /
  tiny-catalog mode.
- **`K>0`** (production default 8) — K worker **processes**, each owning its
  own full `SimServer` engine and hosting many lightweight per-episode
  `WebAgentTextEnv(server=<worker engine>)` sessions, placed least-loaded
  across workers. Each worker is single-threaded, so its engine access is
  naturally serialized while K workers run in true parallel.

The pool is `gym_root/agent_utils/procslots.py` (`ProcSlots`), shared with
alfworld's `_worker.py`:

- **Per-slot FIFO executor.** Each worker's pipe is owned by a one-thread
  executor; callers queue in arrival order on the slot they were placed on, and
  a wedged slot blocks only the callers queued on it, never the other K-1.
- **Timeout -> respawn.** An op exceeding `--engine-call-timeout` (120 s) or a
  dead pipe SIGKILLs and respawns the worker. Its resident episodes are gone; a
  generation counter makes their next op answer 410 instead of touching a
  stranger's env after the respawn.
- **Dead-slot parking + revival.** A slot is parked for the duration of a
  respawn (no new placements) and stays parked when the replacement fails to
  boot (no per-op respawn storm); a background thread retries only the failed
  ones every 60 s, re-checking on the slot's own executor so it never kills a
  replacement that already booted. `dead_workers` counts both kinds of parked
  slot. A stale generation, a failed slot or a closing pool is rejected at
  submit time, so no caller queues behind a booting replacement.
- **`/health`** exposes it all: `{"status", "sessions", "workers",
  "dead_workers", "respawns", "goals", "goals_digest", "sweeper_alive"}`.
  The training entry boot-gates on it and then refuses to start
  unless `goals_digest` (and `goals`) equal the dataset's `protocol.json`
  values ("Data & task feeding").

Memory is K x (catalog + Lucene index + JVM + spaCy). For the default
**small** 1000-product catalog (6910 synthetic goals) a worker is a few
hundred MB, so K in the tens is cheap on a node with a few hundred GB of RAM. The **full** catalog
(~5 GB download per engine) forces small K (or `engine_workers=0`).

## Reward routing

The engine reports the raw WebShop score in `[0,1]` (`info.task_score`,
computed at `click[buy now]`) and `won = task_score == 1.0`. The adapter's
Gymnasium `/step` reward is the raw `task_score` at the terminal step (0
otherwise) — the neutral **env truth**. It also forwards `won`,
`task_score`, `invalid_actions`, `action_format_errors`, `step_count` as
env metrics.

miles routing then fans out (`miles_agent/reward_routing.py`, unchanged):
`Sample.reward = {"env": task_score, "train": shape(task_score)}`.
`--eval-reward-key env` → eval curves report raw score;
`--reward-key train` → the optimizer sees the shaped reward.

`recipes/webshop/reward.py` owns the WebShop-specific shape. Unlike alfworld
(whose env score already IS its binary training reward), WebShop's env-truth
score and its RL reward genuinely differ, so this shape is **not** the
identity even at defaults:

```
train = SCALE * base − penalties
  base   = task_score        if WEBSHOP_DENSE_REWARD=1   (Turn-PPO 2512.17008)
         = won  (0/1)        otherwise  (default: 10-iff-perfect)
  SCALE  = WEBSHOP_REWARD_SCALE           (default 10.0)
  penalties = WEBSHOP_INVALID_PENALTY  · invalid_actions
            + WEBSHOP_FORMAT_PENALTY   · action_format_errors
            + WEBSHOP_STEP_PENALTY     · step_count
```

`WEBSHOP_FORMAT_PENALTY` defaults to 0.1 (a format fault = a turn with no
parsable `act` tool call; the turn is spent, the engine is not stepped): on the
same canonical held-out 500, `wstrain4` (0.1) beat `wstrain3` (0) 0.820 vs 0.794
won and drove `action_format_errors` from ~9/episode to 0 by step ~40, where
the unpenalized run still made ~1/episode at step 250. Set it to 0 for the
unpenalized arm. The other coefficients default to 0; e.g. set
`WEBSHOP_INVALID_PENALTY=0.1` to penalize inadmissible actions as well. These are
**trainer** knobs: `recipes/webshop/run.py` forwards every `WEBSHOP_*` variable
in its environment into the ray runtime env (the only guaranteed path to the
actors) and `reward.py` logs the resolved coefficients once at import, so a
penalty arm is visible in the driver log. They never reach the engine.

### Eval metric

The eval file (`webshop_eval.jsonl`) is read as two index slices with
`--eval-reward-key env`, one sample per goal at `--eval-temperature 0.4`
(verl-agent's and TCOD's validation temperature; the recipe default), per-turn
cap `--eval-max-response-len 4096` (TCOD Table 5; training keeps 512), every
`--eval-interval 50` steps (the launcher default), seed 42. The slice layout follows
`WEBSHOP_PROTOCOL` (see "TCOD protocol" below):

| protocol | `eval/webshop` | second slice |
|---|---|---|
| `canonical` (default) | rows 0..499 = WebShop paper test set, goals-file positions 0..499, fixed and never trained | `eval/webshop_pool` rows 500..755 = 256 goals sampled from the whole pool (constant seed over file positions), trained goals included (verl-agent/GiGPO's validation distribution) |
| `tcod` | rows 0..99 = TCOD's test set, goals-file positions 4096..4195 | `eval/webshop_heldout` rows 100..499 = positions 4196..4595, never trained (lower-noise companion) |

`canonical` is the default because it is the only layout every mainstream
WebShop evaluation shares: the original paper, ReAct, Reflexion, LATS and
AgentBench all score the fixed test prefix (idx 0..499 of the seed-233
shuffle, on their data); verl-agent/GiGPO nominally hold out idx < 500 too, but
re-seed per worker, so their validation is a pool sample (our
`eval/webshop_pool`). Numbers from this repo are comparable across runs exactly
when the runs used the same frozen goals file
(`webshop_goals.jsonl`, digest `455c9eaa96191605`, "Data & task feeding"): rows name
goals by content id and the digest gate refuses an engine whose goal set
differs from that file, so two runs that both started scored the same
questions — nobody has to remember which catalog, goal code or
`WEBSHOP_GOAL_SEED` a run had; a drift in any of them fails the gate instead of
silently relabelling the curve. wstrain1/wstrain2 predate the frozen file and
ran on the seed-0 list positions 0..499, a different question set (same
distribution, not the same questions).

Each `eval/<slice>` is the mean raw `task_score`; the success rate is
**`env/won`** = perfect purchase (`task_score == 1`). Read both: a rising
score with flat `won` means partial-credit purchases. WebShop's `done` flag
(buy-now reached at any score) is deliberately not exported: TCOD's workflows
log it as `env_done` and the paper reports that as WebShop "success rate",
which we treat as an error of the paper, not a metric to adopt.

### Context budget (why 32768)

WebShop trains on the **full multi-turn transcript**. miles assembles ONE
contiguous sample per episode by prefix extension (TITO), so a
verl-agent-style 2-step sliding window — a single-turn re-render of the current
page plus a short history, not a transcript — is not portable to this
assembly. Each WebShop step costs roughly 1.0-1.4k tokens of page + admissible
list plus the reply, so 15 steps need ~17-22k tokens; at 16384, long episodes
hit the policy 400 around step 11-14 and trained as reward-0 truncations. The
recipe therefore sets `rollout_max_context_len = train_max_tokens_per_gpu =
32768` (32768 fits the GPUs used; 65536 OOMs), the same as alfworld.

## Layout

```
gym_root/agent_utils/                        # shared by both Gymnasium envs
  session_gym.py    SessionGymnasiumServer: session registry, lock + liveness re-check,
                    done-replay guard, TTL sweeper, /close, act-tool command extraction
  procslots.py      ProcSlots: worker-process slots with per-slot FIFO executor,
                    timeout -> respawn, dead-slot parking + revival (stdlib; py3.10-compatible
                    because the py3.10 engine tier imports it too)
  chat_dialect.py   chat-completions dialect helpers shared by the episode-driver agents
  webshop_goals.py  goal_id / goals_set_digest: content identity of WebShop goals (stdlib; shared by
                    the py3.10 engine, the adapter tests and recipes/webshop/prepare_data.py)

gym_root/resources_servers/webshop_env/     # py3.13 nemo-gym adapter
  app.py            WebshopResourcesServer(SessionGymnasiumServer): /reset /step proxy,
                    invalid-action + reserved-token counters, 502 + eviction on engine failure
  _engine_client.py EngineSession: bounded-retry HTTP client (fresh session_id per /reset
                    attempt, connect-only retry on /step, malformed reply -> EngineError,
                    forget(), EngineError.status)
  _format.py        obs/admissible rendering (drops the no-op click[search] submit button),
                    is_reserved_search; act-tool command extraction is session_gym's
  configs/webshop.yaml   webshop_env instance + env_loop_agent (webshop_agent);
                    engine_base_url is overridden per run by the training entry
  pyproject.toml    nemo-gym[dev] + miles-agent-utils (py3.13)
  tests/            test_app.py (adapter vs FakeEngine), test_engine_client.py (MockTransport
                    wire tests), test_contract.py (adapter <-> real session_server over ASGI)

env_servers/webshop/                         # py3.10 engine tier
  defaults.sh       the ONE place shell defaults live (host/port, workers, timeouts, layout,
                    coupling rules); sourced by every script here and by the training entry
  session_server.py generic reset/step/close/health framework + TTL sweeper
  server.py         WebshopSharedBackend (engine_workers=0) + main()
  _engine_worker.py WebshopEngineWorkerPool over agent_utils.procslots
  _webshop_engine.py vendored-stack glue; reserved-token neutralization
  webshop/          vendored WebShop fork — text-env CODE (committed, never edited)
  vendor.sh         maintainer-only: refresh webshop/ from upstream
  setup.sh          build the self-contained $WEBSHOP_HOME (env + data + index)
  requirements.txt  the py3.10 pins
  launch_engine.sh  start the engine server; every knob an explicit flag from defaults.sh
  export_goals.py   GET /health + /goals -> the frozen webshop_goals.jsonl (WEBSHOP_GOALS_FILE; not committed)
                    (+ .meta.json); refuses to write if the recomputed digest differs from /health
  tests/            fake_engine_factory.py + test_server.py (pool, HTTP, sweeper, gated real engine)

miles_agent/
  gym_bridge.py     /run client; sets env_metrics.run_transport_failed
  reward_routing.py env vs train reward keys
  dapo_filter.py    dynamic-sampling filters incl. check_no_aborted_no_transport_failure
  launcher.py       shared recipe launcher (train-arg / env-var builders, why-comments)

recipes/webshop/
  prepare_data.py   content-addressed rows (goal_id, from the goals export in file order)
                    + ACT_TOOL + system prompt; protocol.json carries goals_digest
  data/             README.md: provenance + digest of the frozen goal export (rebuilt by scripts/fetch_data.sh webshop --regenerate-goals, never committed)
  run.py            recipe launcher: episode knobs, context 32768, filter selection,
                    WEBSHOP_* forwarding
  reward.py         binary/dense + invalid/format/step shaping (logs coefficients at import)

scripts/
  run_local.sh      RECIPE=webshop: engine bring-up, /health protocol + goal-set digest gate,
                    engine_base_url gym override
```

## Data & task feeding

Rows address goals by **content**, not by position. A WebShop "question" is a
goal dict the engine builds at boot from the catalog (instruction text, target
asin, required attributes / options, price cap = the rubric `get_reward`
scores against). The engine keeps those goals in a seed-shuffled list, and the
previous scheme — `metadata.task_id` = a list position — was an identity only
as long as nothing moved: change the goal seed, the catalog file or the
vendored goal code and position 37 is another question, silently, with every
eval curve still labelled the same. Now:

- **`goal_id`** (`gym_root/agent_utils/webshop_goals.py`, stdlib, shared by
  the py3.10 engine, the py3.13 adapter tests and `prepare_data.py`) =
  `sha1(canonical JSON of the scoring-relevant fields)[:16]`: asin, name,
  category, query, product_category, instruction_text, attributes,
  goal_options, price_upper — everything `get_reward` reads plus the question
  text (the sampling `weight` is not identity). Same content, same id,
  everywhere.
- **`goals_set_digest`** = `sha1("\n".join(sorted(goal_ids)))[:16]`:
  order-free, so it names a goal SET regardless of how the engine ordered it.
- **The frozen goals file** `webshop_goals.jsonl` (+ `.meta.json`; rebuilt by
  `scripts/fetch_data.sh webshop --regenerate-goals`, digest `455c9eaa96191605`, provenance in `recipes/webshop/data/README.md`)
  is the dataset's source of truth: one row per goal, in the seed-233 engine
  order at export time, each carrying its `goal_id` and content fields.
  `recipes/webshop/prepare_data.py --protocol tcod|canonical` reads it IN FILE
  ORDER (positions in the FILE define the splits; the canonical pool sample is
  drawn with the constant `POOL_SAMPLE_SEED = 233` over file positions),
  verifies every stored `goal_id` against the row content, collapses repeated
  content to its first row (9 twin rows on the small catalog: same question,
  never a second held-out goal or a back door into training; the layouts run
  over the 6901 distinct questions in file order, so `tcod` positions past the
  first twin (1367) sit up to 9 places before TCOD's raw `task_id`)
  and short files (usage error, exit 2), and emits `webshop_train.jsonl` +
  `webshop_eval.jsonl` rows with `metadata = {tools, goal_id,
  instruction_text, split, max_episode_steps, data_source}` — no `task_id` —
  plus a `protocol.json` sidecar with `goals_digest` (digest of ALL ids in the
  file), `goals` (count) and `goals_file_sha256`. There is no `--goal-seed` /
  `--num-goals`: the dataset is a pure function of the goals file.
- **The engine resolves ids.** `/reset` takes `goal_id` (the legacy int
  `task_id` still works for manual use); an id not in its goal list is a
  **400** — `goal_id ... is not in the engine goal list (goals_digest ...)` —
  never retried. `/health` reports `goals` and `goals_digest`; `GET /goals`
  returns the list (`goal_id`, `goal_idx`, content fields) in list order.
- **The digest gate.** The training entry regenerates the data on
  every run (a few seconds: the policy model's tokenizer renders the example
  call; a stale file from another protocol, prompt or model is never reused),
  then refuses to start unless
  `protocol.json.goals_digest == /health.goals_digest` (the row / distinct
  counts are printed, not gated): `FATAL: engine goal set (digest X, N rows /
  U unique) != dataset (digest Y, M rows / V unique): catalog / goal code /
  seed changed since the goals file was exported (scripts/fetch_data.sh webshop --regenerate-goals)`.
  The `/health.protocol`
  parity gate (seed, observation mode, goal source, catalog vs the job's
  `WEBSHOP_*` knobs) stays. What the gate guarantees: every row the trainer
  will feed names a goal the engine holds under exactly that rubric, and two
  runs that both passed the gate on the same goals file scored the same
  questions.

What is still "environment" rather than "dataset": the catalog JSON
(`items_shuffle_1000.json` / `items_ins_v2_1000.json` under `WEBSHOP_FULL=0`)
and the vendored goal + scoring code (`web_agent_site/engine/goal.py`). The
goal seed is an input of the engine build, not a dataset parameter: our engine
seeds the whole build and `get_synthetic_goals` draws each goal's price cap
from that RNG, so a different `WEBSHOP_GOAL_SEED` yields different price
clauses — different goal CONTENT, a different digest, a refused start — where
the old scheme would have trained and evaluated on other questions under the
same labels. 233 remains the export order (the original WebShop / TCOD
constant, so file position *i* is TCOD's `task_id` *i* on the same catalog and
goal code), but nothing downstream depends on that order anymore.

Re-exporting (after a deliberate catalog / goal-code / seed change, or to
build a second goals file for another arm): start the engine with the intended
knobs (`env_servers/webshop/launch_engine.sh`), then
`python3 env_servers/webshop/export_goals.py --engine-url
http://127.0.0.1:6200 --output recipes/webshop/data/webshop_goals.jsonl`. It
reads `/health` + `/goals`, writes the JSONL in `goal_idx` order, recomputes
the set digest and refuses to write if it differs from `/health`, and writes
the `.meta.json` sidecar (`goals`, `goals_digest`, `protocol`, `exported_at`,
`engine_url`). Fill in `recipes/webshop/data/README.md` and commit both files;
every run from then on scores the new set and says so through the digest. The
adapter's `/reset` info echoes `goal_id` and `goal_idx` as **strings** on
purpose: `env_loop_agent` forwards numeric info values into wandb `env/*`, and
an id is a label, not a metric.

## TCOD protocol

Sources: the TCOD paper (arXiv 2604.24005v3, Tables 1/4/5, App. E.3) and the
TCOD code in the Trinity-RFT repository (its WebShop example configs and the
WebShop `utils` / `OPD_workflow` modules). Their WebShop protocol, and where this recipe stands
(defaults = the aligned value except the split, which defaults to `canonical`;
the knob that moves each row in parentheses):

| Dimension | TCOD | This recipe | Status |
|---|---|---|---|
| Environment | princeton-nlp WebShop, `WebAgentTextEnv-v0` | vendored fork of the same env (verl-agent lineage, seed-parametrized) | aligned |
| Goal shuffle seed | 233 (original constant) | 233 (`WEBSHOP_GOAL_SEED`): orders the engine's list and draws the price caps; the export order of the frozen goals file. Rows carry content ids, the digest gate verifies the resulting goal set | aligned |
| Catalog | 1000-product small catalog: `num_products=None` on a stock princeton-nlp clone whose `web_agent_site/utils.py` defaults to `items_shuffle_1000.json` / `items_ins_v2_1000.json`; README installs with `setup.sh -d small` (`-d all` listed as optional). TCOD did not use the full 1.18 M catalog. | 1000-product small catalog (`WEBSHOP_FULL=0`) | aligned |
| Goals | code passes `human_goals=True`; on the small catalog that yields 13 goals (13 of the 1000 asins have `items_human_ins.json` entries) while `SimServer.receive` indexes `goals[task_id]` without wrap-around up to 4195, so the released code cannot serve its own task ids on this catalog; the goal source actually run is not recoverable from repo or paper | **synthetic** (`WEBSHOP_HUMAN_GOALS=0`, 6910 goals), the only goal source with >= 4596 goals on this catalog | index layout aligned; goal text unverifiable |
| Split | train `task_id` 0..4095, test 4096..4195 (`get_webshop_data.py`); train contains the canonical test set | default `canonical` (goals-file positions 0..499 held out); `WEBSHOP_PROTOCOL=tcod` reproduces TCOD's ranges as file positions | **deliberately not the default**: TCOD's layout is not a mainstream evaluation |
| Page rendering | `text_rich` | `text_rich` (`WEBSHOP_OBS_MODE`) | aligned |
| Admissible actions | listed per turn, `search[<query>]` / `click[x]` | same wording | aligned (we hide the no-op `click[search]`) |
| Invalid action | not stepped; error sentence as observation; no penalty | same (`WEBSHOP_INVALID_FEEDBACK=error`) | aligned |
| Max steps | 15 | 15 | aligned |
| Reasoning | `<think>` block before the action | `<think>` block before the tool call (`PROMPT=think`) | aligned |
| Action transport | `<action>` tags | a tool call to the `act` tool, in the policy model's own markup (hermes JSON for Qwen3, qwen3_coder XML for Qwen3.5) | **kept by convention** |
| History | 2-step sliding window, re-rendered prompt | full multi-turn transcript, context 32768 | **kept by convention** (miles assembles one sample per episode) |
| Train temperature / response cap | 1.0 / 512 | 1.0 / 512 | aligned |
| Eval temperature / top-p / cap | 0.4 / 1.0 / 4096 | 0.4 / 1.0 / 4096 | aligned |
| Eval set & cadence | 10 tasks sampled from the 100-task test set, every 5 steps | the whole 100-task test set + 400 held-out, every 50 steps | superset (lower noise); sparser cadence |
| Success metric | `env_done` (buy-now reached, any score) | `env/won` (score == 1) | **deliberately not aligned**: `env_done` is not success |
| Seed | 42 | 42 (`--seed`) | aligned |
| Training | OPD (KL to a Qwen3-30B-A3B teacher), batch 16, 150–250 steps | GRPO 16×8, 250 steps (a *teacher* recipe; TCOD has no GRPO WebShop teacher) | n/a |

Two caveats on comparability. TCOD's train range (idx 0..4095) contains the
WebShop paper's test set (idx 0..499), so a `tcod`-protocol model is not
comparable to leaderboard numbers — use `WEBSHOP_PROTOCOL=canonical` for that.
And the original env seeds only the goal shuffle: product prices and the
"price lower than X" clause of human goals are drawn from the process RNG, so
even TCOD's own workers see slightly different price constraints for the same
`task_id`; our engine seeds the whole build, which fixes them per seed but
cannot reproduce theirs.

To reproduce the pre-TCOD arms (wstrain1/wstrain2): `WEBSHOP_PROTOCOL=canonical
WEBSHOP_GOAL_SEED=0 WEBSHOP_OBS_MODE=text WEBSHOP_INVALID_FEEDBACK=noop
PROMPT=none` plus `--eval-temperature 1.0 --eval-interval 50
--eval-max-response-len 0 --seed 1234` (wstrain1 additionally
`WEBSHOP_FORMAT_PENALTY=0`; wstrain2: `--eval-temperature 0.4` and the
now-default penalty 0.1). A seed-0 engine builds a different goal set
(price caps), so this also needs a goals file exported from that engine
(`export_goals.py` against it) named by `WEBSHOP_GOALS_FILE`; with the frozen
seed-233 export the
digest gate refuses a seed-0 engine, by design.

## Bring-up

Self-contained, no other repo required. One-time prep: `scripts/fetch_data.sh
webshop --from <dir>` stages the catalog, then `env_servers/webshop/setup.sh`
builds the whole py3.10 runtime — conda/mamba env (py3.10 + JDK 11 + the pins),
a writable copy of the committed fork, the catalog, and the built Lucene index —
under `$WEBSHOP_HOME` (default `$MILES_AGENT_DATA/webshop`). Run it once; on a
scheduler, as a CPU job.

`RECIPE=webshop` on the training entry then sources
`env_servers/webshop/defaults.sh`, starts the engine server (K workers) via
`launch_engine.sh` with every knob as an explicit flag, health-gates it on
`http://127.0.0.1:6200/health`, digest-gates the data's goal set against
`goals_digest`, passes `+webshop_env.resources_servers.webshop_env.engine_base_url=`
to gym so the adapter dials the same port, and runs the trainer exactly like
the alfworld branch. `WEBSHOP_*` set in the environment passes through
unchanged. `scripts/build_gym_venvs.sh` carries the `webshop_env` py3.13 venv
(nemo-gym[dev] + `agent_utils` — thin proxy, no engine deps). See
`env_servers/webshop/README.md` for the operator view (knob table, HTTP
contract, prep details).
