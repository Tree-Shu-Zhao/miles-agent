# WebShop engine tier (py3.10)

The heavy WebShop engine — product catalog, Lucene/JVM BM25 search, spaCy
reward model — lives here, OUTSIDE the nemo-gym py3.13 world. The py3.13
`webshop_env` resources server (`gym_root/resources_servers/webshop_env/`)
proxies to it over loopback HTTP. Why the process boundary exists, the failure
semantics, the timeout/TTL coupling rules and the concurrency design are in
`docs/webshop_integration.md`; this file is the operator's view of the tier.

## Files

| file | role |
|---|---|
| `defaults.sh` | the ONE place shell-side defaults live (host/port, workers, timeouts, `$WEBSHOP_HOME` layout, coupling rules); sourced by every script below and by the training entry (`scripts/run_local.sh`) |
| `session_server.py` | generic reset/step/close/health HTTP framework + TTL sweeper (`create_app(backend, session_ttl, max_concurrency, sweep_interval)`) |
| `_webshop_engine.py` | vendored-stack glue (build engine, make/step/drop episode); neutralizes reserved-token searches |
| `server.py` | `WebshopSharedBackend` (engine_workers=0) + `main()` argparse/dispatch |
| `_engine_worker.py` | `WebshopEngineWorkerPool` — K engine processes over `agent_utils.procslots.ProcSlots` (the concurrency lever) |
| `gym_root/agent_utils/procslots.py` | shared process-slot pool (per-worker FIFO executor, generation counters, dead-slot parking + revival, shutdown) that `_engine_worker.py` and alfworld's `_worker.py` build on; the py3.10 tier imports it via a `sys.path` fallback to `<checkout>/gym_root` |
| `webshop/` | vendored WebShop fork — text-env CODE, committed (the source of truth; never edited by our glue) |
| `vendor.sh` | maintainer-only: refresh `webshop/` from an upstream checkout |
| `setup.sh` | build the self-contained runtime under `$WEBSHOP_HOME` |
| `launch_engine.sh` | start the engine server; every knob from `defaults.sh` is passed as an explicit flag |
| `requirements.txt` | the py3.10 pins (load-bearing for reward/search parity) |
| `tests/` | `fake_engine_factory.py` + `test_server.py`: pool, HTTP mapping, sweeper, wedged/dead-slot behaviour; real-engine tests gated on a built runtime |

miles-agent is **self-contained**: the fork's code is committed under
`webshop/` (minus the 16 MB Selenium chromedriver and the downloaded
data/indexes, which `setup.sh` rebuilds), so there is no dependency on any
other repo.

## Endpoint and knobs

Loopback only, **`127.0.0.1:6200`**: the adapter is the sole client and shares
the node. 6200 sits below 9000 — on nodes whose ephemeral port range starts
at 9000, anything above can be squatted
between check and bind. The adapter dials `WEBSHOP_ENGINE_BASE_URL`
(`http://127.0.0.1:$WEBSHOP_SERVER_PORT`), which the training entry passes to
gym as `+webshop_env.resources_servers.webshop_env.engine_base_url=...`, so one
knob moves both ends.

| env var (`defaults.sh`) | `server.py` flag | default | rule |
|---|---|---|---|
| `WEBSHOP_SERVER_HOST` / `WEBSHOP_SERVER_PORT` | `--host` / `--port` | `127.0.0.1` / `6200` | loopback; sub-9000 |
| `WEBSHOP_ENGINE_WORKERS` | `--engine-workers` | `8` | `0` = one shared engine behind a lock |
| `WEBSHOP_ENGINE_CALL_TIMEOUT` | `--engine-call-timeout` | `120` | adapter `engine_timeout_s` 240 > this + respawn margin while a replacement engine boots within ~115 s (`--small`; `--full` may need `engine_timeout_s` raised). A ReadTimeout is still 502 -> group dropped; see `docs/webshop_integration.md` |
| `WEBSHOP_ENGINE_SESSION_TTL` | `--session-ttl` | `2400` | > adapter `session_ttl_s` 1800 > `NEMO_GYM_RUN_TIMEOUT` 1200 |
| `WEBSHOP_FULL` | `--full` / `--small` | `0` | must match the index `setup.sh` built |
| `WEBSHOP_VENDOR` / `WEBSHOP_DATA_DIR` | `--vendored-path` / `--data-dir` | `$WEBSHOP_HOME/webshop` / `.../data` | |
| — | `--max-concurrency` / `--sweep-interval` | `256` / `60` | pass extra args through `launch_engine.sh` |

`server.py` consults the environment only for `WEBSHOP_VENDOR` /
`WEBSHOP_DATA_DIR` (manual runs); its argparse defaults must equal
`defaults.sh` and never decide anything under the launcher.

## HTTP contract

```
POST /reset  {"session_id", "goal_id", "task_id", "split"}  -> {"obs", "info"}
             goal_id (content id) wins when present, task_id (list index) is the legacy path;
             400 unknown goal_id / bad index (the adapter never retries 4xx); 500 other backend error
POST /step   {"session_id", "action"}            -> {"obs", "reward", "done", "info"}
             410 unknown / evicted session (worker respawn, TTL, done); 500 backend error
POST /close  {"session_id"}                      -> {}   (backend errors swallowed)
GET  /health -> {"status", "sessions", "workers", "dead_workers", "respawns", "goals", "goals_digest",
                 "protocol", "sweeper_alive"}
GET  /goals  -> [{"goal_id", "goal_idx", ...GOAL_FIELDS}]  (the export scripts/fetch_data.sh webshop --regenerate-goals freezes)
```

The training entry boot-gates on `/health` and then refuses to start unless
`goals_digest` equals the frozen dataset's (`455c9eaa96191605` for the paper's export;
`gym_root/agent_utils/webshop_goals.py`).

## Concurrency

`--engine-workers K` runs K worker processes, each with its own `SimServer`
engine hosting many lightweight per-episode envs: K parallel Lucene searches
with no shared lock. `--engine-workers 0` is the single-shared-engine + RLock
fallback (correct but throughput-capped). Tune K to the rollout wave — the
recipe runs 128 concurrent episodes (16 prompts x 8 samples); the small
catalog makes tens of workers cheap, the full catalog (each engine loads the
~5 GB catalog) forces a small K.

Each slot's pipe is owned by a one-thread executor (FIFO; a wedged slot blocks
only the callers queued on it). An op exceeding `--engine-call-timeout`
SIGKILLs and respawns the worker (its resident episodes 410 on their next op);
a slot whose respawn fails is parked and retried every 60 s. `/health`
reports `dead_workers` and `respawns`. Details: `docs/webshop_integration.md`.

## Bring-up (one-time, self-contained)

`setup.sh` builds the entire py3.10 runtime under `$WEBSHOP_HOME` from the
committed fork + gdown downloads — no other repo required:

```
$WEBSHOP_HOME/env/                            conda/mamba env (py3.10 + JDK 11 + pins)
$WEBSHOP_HOME/webshop/                        writable copy of the committed fork
$WEBSHOP_HOME/webshop/data/                   catalog (items_*.json)
$WEBSHOP_HOME/webshop/search_engine/indexes/  built Lucene index (-> indexes_<mode>)
```

Nothing is written back into the committed tree. Run it once (on a scheduler,
as a CPU prep job); `WEBSHOP_HOME` defaults to `$MILES_AGENT_DATA/webshop` when
that root is set, else `/data/webshop`:

```
WEBSHOP_HOME=$MILES_AGENT_DATA/webshop bash env_servers/webshop/setup.sh          # --small (default)
WEBSHOP_HOME=$MILES_AGENT_DATA/webshop bash env_servers/webshop/launch_engine.sh  # manual start; RECIPE=webshop does this for you
```

`RECIPE=webshop` on the training entry sources `defaults.sh`,
starts the engine via `launch_engine.sh`, gates on `/health` + the goal-set digest,
and points the adapter at it. Any `WEBSHOP_*` you set in the environment passes
through unchanged. Building `pyserini`/`spaCy`/`gym==0.24.0` needs a
conda/mamba/micromamba on the prep node; the conda env is a prefix env, so its
shebangs resolve unchanged inside the container.
