# Running this repo under a job scheduler

A job scheduler never runs this repo directly: a *wrapper* (a batch script, or a job framework) exports the
variables of Contract A below, starts one *entry* of `scripts/` per node, relays a preemption or time-limit
signal, and reads the entry's log markers. The entries know nothing about the scheduler: they read the exported
names, never a scheduler variable, and the same entries run on a plain machine with none of them set (local
mode: docs/local_setup.md). Two such wrappers exist today:
`examples/slurm/miles_run.example.sbatch` (a minimal Slurm wrapper) and a site's own wrapper repository.
What a wrapper does, in order:

1. exports Contract A (section A; the names, one per line, are `tests/contract_vars.txt`);
2. runs the entry on every node, `exec bash scripts/run_local.sh` for training (rank 0 starts the services and
   the driver, the other ranks join the compute cluster and idle until the head is done), or one of the other hook
   entries of section B: `scripts/serve_model.sh`, `scripts/smoke_tests.sh`, `scripts/build_gym_venvs.sh <root>`,
   `scripts/export_hf.sh`;
3. relays a preemption or time-limit signal as section C says, and reads the markers of section D (the entry's
   `RUN_INNER_PASS`, or `RUN_INNER_PREEMPT_STOP rc= durable=` after a save; the wrapper prints the job-level
   `JOB_PASS_<job>_<tag>` itself).

The two repositories meet only through environment variables, files under `SYNC_DIR`, log markers and the
wrapper's project adapter, never through a Python import across repositories. This page and the wrapper
repository's `runbooks/contract.md` hold the same A, C and D blocks; `tests/test_contract_doc.py` diffs them
whenever that checkout is present (`CLUSTER_JOBS_DIR`).

## A. wrapper -> entry

Exported by the wrapper before the entry starts. The wrapper repository's `lib/job_common.sh` exports every
required name and passes the optional ones (`NEMO_GYM_URL TEACHER_URL RETRIEVER_URL ENTRY
SESSION_SERVER_EXTERNAL_HOST SANDBOX_TRANSPORT`, the secrets) to the step only when they are set; the entries
treat an empty optional as unset either way. The example wrapper below exports the subset the smallest job
needs. The `today` column names where the old in-repo wrapper set each one.

| variable | value | today |
|---|---|---|
| `CONTRACT_VERSION` | 1; entry FATALs on an unknown major; unset = local mode | new |
| `RUN_TAG` | `[A-Za-z0-9._-]+`, collision-guarded by the stage launcher | `miles_run.sbatch:72` |
| `JOB_ID JOB_RESTART JOB_NAME` | `SLURM_JOB_ID`, `SLURM_RESTART_COUNT` (0), `SLURM_JOB_NAME`; the entry never reads `SLURM_*` | `:84` |
| `NNODES NODE_RANK HEAD_IP NODE_IP` | node count, rank, head and own IPv4; entry validates `TOPOLOGY` vs `NNODES` itself (ex `:86-90`) | `:106-108`; `miles_run_inner.sh:9,11-13` |
| `GPUS_PER_NODE` | `SLURM_GPUS_ON_NODE` else profile `GPUS_PER_NODE`; entry measures when unset | `miles_run_inner.sh:46` |
| `PROJECT_DIR` + `MODELS_DIR DATA_DIR RESULTS_DIR RUNS_DIR TOOLS_DIR IMAGES_DIR STAGE_TARS_DIR` | `$PROJECT_DIR/{models,data,results,runs,tools,images,stage_tars}` (= `$P` today) | `:64`; `miles_run_inner.sh:351-357` |
| `CACHE_DIR HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_EGRESS=1` | hub cache + offline; entry never downloads under `NO_EGRESS` | `miles_run_inner.sh:39-40` |
| `NODE_LOCAL` | `$LOCAL_BASE/<word>-$JOB_ID`, removed by the wrapper at exit | `:80`; `stale_sweep.sh:8-9` |
| `TMPDIR TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TRITON_PTXAS_PATH NCCL_*` | from profile `HW_ENV`; entry never sets them | `miles_run_inner.sh:33-38` |
| `STAGE_DIR STAGE_SHA INFRA_SHA` | `$PROJECT_DIR/jobs/<stage>`, project tar sha (EXPECT_SHA-verified `:94-99`), cluster-jobs sha | `sbatch_stage.sh:24` |
| `SYNC_DIR` | per job+restart; files `ENDED preempt preempt.done TEACHER_URL TEACHER_FAILED env_lost` | `:84`; `miles_run_inner.sh:52,62,69` |
| `PREEMPT_SENTINEL` | `$SYNC_DIR/preempt` (D16) | `preempt.sh:26-27` |
| `WANDB_API_KEY HF_TOKEN` | from `PROJECT_SECRETS="wandb_key:WANDB_API_KEY hf_token:HF_TOKEN"` via `jc_secret` under `set +x`; unset when `WANDB_ENABLE=0` | `miles_run_inner.sh:544-549` |
| `NEMO_GYM_URL` (opt) | preset => external agent server(s), entry starts no gym | `swe_run_inner.sh:370-372`; `gym_bridge.py:17-19` |
| `TEACHER_URL` (opt), `RETRIEVER_URL` (opt) | external teacher; the wrapper's tailnet bridge sets `RETRIEVER_URL=http://127.0.0.1:6100` | `miles_run_inner.sh:206-266` |
| `GYM_VENV_ROOT` (opt) | `/opt` on the cluster (D6); unset => `uv run` online path | `miles_run.sbatch:138-142` |
| `ENTRY` (opt, submitter) | overrides `PROJECT_ENTRY` (`--env ENTRY=scripts/r2s_teacher_ft.sh`) | new |
| `SESSION_SERVER_EXTERNAL_HOST` (opt) | the training node's address the sandboxed agents dial back to; set by `jobs/run_sandboxed_inner.sh` (S4) | `swe_run_inner.sh` (S4) |
| `SANDBOX_TRANSPORT` (opt) | `cluster` or `tailnet`: the Contract E transport the sandbox-env job was reached by (S4) | `swe_env_client.sh:1-25`; `swe_env_ctl.py` |

Not in A on purpose: the wrapper's own roots and mount points, the QOS / partition / account (the wrapper's
profile), in-container paths (the adapter's mounts). `NCCL_*` is a family the wrapper's hardware profile sets,
not enumerated in `tests/contract_vars.txt`; `TRITON_PTXAS_PATH` is exported only where the profile sets it. The
entries derive the rest from A (`scripts/lib/common.sh`): `MILES_AGENT_MODELS` / `MILES_AGENT_DATA` /
`MILES_AGENT_RESULTS` fall back to `MODELS_DIR` / `DATA_DIR` / `RESULTS_DIR`, `GPUS_PER_NODE` is measured when
unset, `NODE_IP` is resolved when unset, `JOB_ID` defaults to the shell's pid, and the per-run sentinel defaults to
`$MILES_AGENT_RESULTS/<tag>/stop` when no `PREEMPT_SENTINEL` is exported.

## B. the project's entries

Every entry is bash with `set -euo pipefail`, sources `scripts/lib/common.sh` first, takes `--dry-run` (prints
every command as `DRY_RUN <label>: <argv>` and exits 0 before any side effect; `DRY_RUN=1` in the environment does
the same), prints `FATAL: <text>` before any non-zero exit, prints its PASS marker last, and removes nothing
outside `$NODE_LOCAL` / `$SYNC_DIR`. The wrapper runs one entry per job kind and knows no other name:

| job kind (the adapter's hook) | entry | ends with |
|---|---|---|
| training (`PROJECT_ENTRY`; `ENTRY=` overrides it per job) | `scripts/run_local.sh`, rank-aware | `RUN_INNER_PASS`, or `RUN_INNER_PREEMPT_STOP rc= durable=` after a save on a signal |
| serving (`PROJECT_SERVE_CMD`) | `scripts/serve_model.sh` | `SERVE_PARSER`, `SERVE_PREFLIGHT`, `SERVE_UP url=`, then it stays up |
| environment smoke (`PROJECT_SMOKE_CMD`) | `scripts/smoke_tests.sh` | `CONTRACT_TESTS_OK` |
| venv build (`PROJECT_VENV_BUILD_CMD <root>`) | `scripts/build_gym_venvs.sh <root>` | `GYM_VENVS_OK root= servers=` |
| checkpoint conversion (`PROJECT_CONVERT_CMD`) | `scripts/export_hf.sh`, the hook form (`ITER_DIR`, `OUT_DIR` from the environment) | `EXPORT_HF_OK out= iter=` |

The rest of `scripts/` is for people and for the entries themselves: `start_gym.sh`, `ray_cluster.sh`,
`eval_ckpt.sh`, `fetch_data.sh`, `check_patches.sh`, `stop_all.sh`, `setup.sh`, `contract_selftest.sh`,
`harbor_trial_server.sh`, `harbor_run.sh`, `r2s_teacher_ft.sh`, the sourced `teacher_node.sh` and
`r2s_pipeline.sh`, and the libs under `scripts/lib/`. `<script> --help` prints each one's header.

## C. preemption / time limit

Wrapper `#SBATCH --signal=B:USR1@$PREEMPT_LEAD_S` (600) -> `touch $PREEMPT_SENTINEL`, kill watchdog, wait
(`preempt.sh:26-69`) -> entry passed `--preempt-sentinel`, refuses to start if it pre-exists
(`miles_run_inner.sh:563-567`), every rank traps TERM/USR1 -> touch (`:22-31`) -> driver
(`patches/miles/miles-preempt-save.patch`) finishes the step, saves, writes `<sentinel>.done={"rollout_id":N}` ->
entry `RUN_INNER_PREEMPT_STOP rc= durable=` (`:590`) -> wrapper `PREEMPT_REQUEUE=auto|always|never`:
`scontrol requeue` on `PreemptTime` else exit 75 `JOB_STOPPED_TO_RESUME_<job> durable=` (`preempt.sh:72-105`,
f7c57e8). Student: `touch results/<tag>/stop`, rerun the tag to resume. The trap lives in the ENTRY (a trap in
`run_inner.sh` dies at `exec`, S3). A step that failed without a signal and without a save is a crash, not a
preemption: the wrapper prints `FATAL: the main step failed` and exits with the step's code (f7c57e8;
`durable=none` with a non-zero step rc is the tell).

## D. markers

Grammar: one line, `UPPER_SNAKE[_<name>] key=value ...`; `FATAL: <text>` precedes every non-zero exit; `WARN:`
never changes an exit code; a PASS marker is the step's last line and is printed only after rc 0; no marker
contains a pipe (the launcher's queue parsing splits on it); names are greppable without alternation (a log grep may be BRE).

| emitter | markers | today |
|---|---|---|
| wrapper (job) | `JOB_PASS_<PROJECT_JOB_NAME>_<RUN_TAG>`, `JOB_REQUEUE_<name>`, `JOB_STOPPED_TO_RESUME_<name> durable=`, `PREEMPT_SIGNAL`, `PREEMPT_STOP step_rc= durable=`, `STALE_SWEEP <node> ...`, `NODE_LOCAL_REMOVED <dir> in <s> s`, `IDLE_WATCHDOG_CANCEL job=`, `ENDPOINT_UP` / `ENDPOINT_WITHDRAWN`, `SANDBOX_ENV_UP` / `STOPPING` / `DRAIN` / `DONE` / `CLIENTS` / `REACH` (ex `SWE_ENV_*`), `STAGE_SHA_OK`, `JC_CLUSTER <name>`, `THREAD_BUDGET pids_max=`, `PIDS_HIGH current=` | `miles_run.sbatch:165`; `preempt.sh:18,96,103`; `idle_watchdog.sh:57`; `model_serve.sbatch:110`; `swe_env_inner.sh:38-41,214-229` |
| entry (project) | `RUN_INNER_PASS`, `RUN_INNER_PREEMPT_STOP`, `RAY_WORKER_JOINED` / `RAY_WORKER_DONE node_rank=`, `GYM_UP agent=`, `WEBSHOP_ENGINE_UP`, `SCIW_RUNTIME`, `TEACHER_UP` / `TEACHER_GATE_OK` / `TEACHER_FAILED` / `TEACHER_ENDED`, `SERVE_PARSER` / `SERVE_PREFLIGHT` / `SERVE_UP`, `HARBOR_TRIAL_SERVER_UP url=`, `HARBOR_SUMMARY` / `HARBOR_AGENT_STATS` / `HARBOR_EXCEPTION` / `HARBOR_NOT_PASSED` / `HARBOR_RESULTS`, `R2S_*_PASS`, `FT_PASS` / `FT_FAIL`, `CONTRACT_TESTS_OK` | `miles_run_inner.sh:594,590,61,70,542`; PORT `model_serve_inner.sh:31,84`; `harbor_env.sh:221`; `harbor_run_inner.sh:145-157`; `job_env_smoke.sbatch:100` |

`JOB_PASS` is never printed by an entry: only the wrapper knows the whole job passed. Chain rule: a link's
`afterany` successor resumes on `JOB_STOPPED_TO_RESUME_*`, treats `JOB_PASS_*` as done and a cancelled job as
ordinary (D5 keeps every existing name; `RUN_LOCAL_PASS`/`SCRIPT_PASS` renames were rejected).

Entry markers added with the public `scripts/` (D5: new names, one emitter each, outside the table the wrapper
repository copies):

- `HARBOR_TRIAL_SERVER_STOP rc=0` — `scripts/harbor_trial_server.sh`: the trial server exited by itself (a non-zero exit is a FATAL instead).
- `RAY_JOINED head=<ip>:6379 ip=<own ip> gpus=<n>` — `scripts/ray_cluster.sh join`: this machine joined the head.
- `RAY_STOPPED hosts=<n>` — `scripts/ray_cluster.sh stop`: `ray stop --force` ran here and over ssh on the `<n>` hosts given.

## The example wrapper: `examples/slurm/miles_run.example.sbatch`

The smallest Slurm wrapper: the four portable headers (`#SBATCH -N 1`, `-t 04:00:00`, `-J miles_run`,
`--signal=B:USR1@600`), the site's account / partition / QOS / gres / log lines as `##SBATCH` comments (pass them
as `sbatch` flags: flags beat headers), `RUN_TAG RECIPE TOPOLOGY` from the environment
(`sbatch --export=ALL,RUN_TAG=alf1,RECIPE=alfworld examples/slurm/miles_run.example.sbatch`, submitted from the
checkout), `PROJECT_DIR=${PROJECT_DIR:-$HOME/miles-agent-runs}` and the seven dirs under it, `HEAD_IP` from
`scontrol show hostnames` + `getent`, `SYNC_DIR` per job and restart, the Contract A exports, a USR1 trap that
touches the sentinel, one `srun --ntasks-per-node=1 --kill-on-bad-exit=1` task per node that execs
`scripts/run_local.sh` with `NODE_RANK=$SLURM_PROCID`, and the two job markers: exit 75 with
`JOB_STOPPED_TO_RESUME_miles_run_<tag> durable=` when the sentinel was raised (submit again with the same tag to
resume), else `JOB_PASS_miles_run_<tag>`. It carries no requeue, no idle watchdog, no stale-file sweep, no
secrets and no container flags: those are site policy, which a site's own wrapper adds around the same entry (a
commented `srun --container-image=... --container-mounts=...` line shows the container form).
`scripts/contract_selftest.sh` checks that the file parses and that none of `--account --partition --qos
--container-image` is active in it.

## The self-test and the pin

`scripts/contract_selftest.sh` is the no-cluster self-test of this contract. Under `tests/fake_env.sh` (a complete
fake Contract A under `FAKE_ENV_ROOT`, addresses from the documentation range 192.0.2.x, every optional exported
empty) it parses every public script and the example wrappers, lists each Contract A name as set or unset, checks
the contract version, runs `scripts/run_local.sh --dry-run --recipe alfworld --run-tag selftest` and verifies that
the driver line carries the wrapper's `--preempt-sentinel` and `--save-dir`, greps the example for active site
flags, and ends with `CONTRACT_SELFTEST_OK vars=43/43`:

    . tests/fake_env.sh && bash scripts/contract_selftest.sh

A wrapper repository pins the commit of this repo its wrapper was last checked against by running exactly this
under its own fake environment and recording the SHA (the `PIN` of its project adapter); a bump of the pin is
that run again. The same script in a bare shell is the local-mode check: `CONTRACT_SELFTEST_OK vars=<n>/43` with
whatever names the shell happens to have.

## Containers without a scheduler plugin

The training entries run in the miles image (docs/containers.md: the tag `3rdparty/miles` pins,
`radixark/miles:v0.1.2` today, with the checkout mounted at its own path and the patched `3rdparty/miles` at
`/root/miles`). Under a scheduler's container plugin the wrapper passes the image and the mounts as step flags
(the commented line of the example). Without one, enroot alone does the same on a node:

    enroot import -o miles.sqsh docker://radixark/miles:v0.1.2
    enroot create --name miles miles.sqsh
    enroot start --rw --mount "$REPO:$REPO" --mount "$REPO/3rdparty/miles:/root/miles" --mount "$PROJECT_DIR:$PROJECT_DIR" \
      miles bash -c 'cd "$REPO" && exec bash scripts/run_local.sh'

The Contract A exports of the wrapper's shell have to reach the container (`enroot start --env NAME` per name, or
an `--rc` script that exports them); the container sees the same paths as the host, so every marker and every
`DRY_RUN` line stays valid on both sides.
