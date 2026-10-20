# Local setup: a plain machine, no scheduler

Everything a job wrapper does on a cluster (docs/cluster.md) is optional here: the scripts under `scripts/` read
their knobs from the environment, then from `local.env`, then default everything under the checkout, and the same
`scripts/run_local.sh` a wrapper hands a job to runs from a terminal. Training needs the miles image or a machine
with miles' dependencies installed (docs/containers.md); the setup, the data preparation, the tests, the self-test
and every `--dry-run` need only bash and uv.

## scripts/setup.sh

`git clone --recurse-submodules <url> miles-agent && bash miles-agent/scripts/setup.sh` (from anywhere: the
script finds its checkout). Its steps, each skipped when already done:

1. `uv` on PATH, the one tool it needs (else `FATAL:` with the install hint,
   https://docs.astral.sh/uv/getting-started/installation/).
2. `git submodule update --init --recursive` for whichever of `3rdparty/miles`, `3rdparty/Gym` is not initialised
   yet, or is checked out at a commit that is neither the recorded gitlink nor a descendant of it (the pin moved
   under the checkout, after a pull). A submodule at the gitlink or ahead of it is left alone: `3rdparty/miles` is
   ahead by the patch commits the next step applies, and a bare `git submodule update` on it would silently drop
   them.
3. `scripts/check_patches.sh --apply`: the patches of `patches/miles/series` onto the miles pin, in order, each
   skipped when its subject is already in the checkout's log (`patches/miles/README.md` says what each one
   does). `scripts/check_patches.sh` alone is the check: `PATCHES_OK applied=<n>/<n> newly_applied=0`.
4. The test venv: `uv venv --python 3.13.14 .venv`, then `uv pip install --group test` from `pyproject.toml`
   (the repo itself is not a package: the tests and the data preparation import it from the checkout).
   `.venv/bin/python -m pytest -q tests` runs this repo's tests.
5. `local.env` from `local.env.example` when there is none yet.

It ends with the next steps and `SETUP_OK venv=... local_env=...`; `scripts/setup.sh --dry-run` prints each
step's command instead. On Linux the scripts want bash 4 or newer. macOS's bash 3.2 runs the dry runs, the
self-test and the tests (those paths avoid bash-4 syntax), not training.

## local.env

`scripts/lib/common.sh` reads `local.env` line by line: only `KEY=value` lines count, no shell syntax is run,
`$REPO` (the checkout) and `$HOME` are expanded as whole words (`$REPOSITORY` and `$HOMEBREW_PREFIX` stay as they
are), a trailing ` # note` is dropped, the last line of a key wins (as it would when sourced), and a variable
already set in the environment wins over the file (an `export FOO=...` in the shell beats it). `local.env.example`
documents every knob with its default. `LOCAL_ENV=<file>` points the scripts at another file and
`LOCAL_ENV=/dev/null` at none, which is what a job wrapper exports so that a run never reads a machine's
`local.env`. The roots:

| knob | default | holds |
|---|---|---|
| `MILES_AGENT_MODELS` | `$REPO/models` | the HF model dirs (`$MILES_AGENT_MODELS/$POLICY_MODEL` is the policy) and their torch_dist twins (`MILES_TORCH_DIST_DIR`, default `torch_dist_v5/` under it) |
| `MILES_AGENT_DATA` | `$REPO/data` | the per-environment data (`scripts/fetch_data.sh`, docs/data.md) and the rows of each run under `runs/<tag>/` |
| `MILES_AGENT_RESULTS` | `$REPO/results` | the checkpoints, logs and the stop sentinel of each run under `<tag>/` |
| `GYM_VENV_ROOT` | empty | empty: the gym servers start through `uv run` from `3rdparty/Gym`; a path: the prebuilt venvs of `scripts/build_gym_venvs.sh` |

`GYM_VENV_ROOT` must equal the path the venvs were built at: uv venvs are not relocatable (shebangs,
`activate`, `pyvenv.cfg` and the editable `.pth` files bake absolute paths), so `scripts/build_gym_venvs.sh
[<root>]` builds at the root the runs will use, and the knob names that root, nothing else.

## Harness packs for the coding and terminal recipes

`gym_root/miles_harbor/packs/harbor_packs.sh <name> <version>` installs a harness pack (mini-swe-agent so far)
under `HARBOR_PACK_ROOT`, and `miles_harbor.harnesses` (the trial server of `scripts/harbor_trial_server.sh`, the
agents of `scripts/harbor_run.sh`) reads the packs from `HARBOR_PACK_ROOT` too. The installer and the runtime share
one default, `$HOME/.cache/miles-harness`; set `HARBOR_PACK_ROOT` to put the packs elsewhere (a container mounts
them read-only at the same path), and keep it exported for the scripts:

    export HARBOR_PACK_ROOT=$HOME/.cache/miles-harness
    bash gym_root/miles_harbor/packs/harbor_packs.sh mini-swe-agent 2.4.6

`scripts/harbor_trial_server.sh` exports that same default itself, resolved from the account's `HOME` before it
gives the server a `HOME` under its work dir.

## Sandboxes on an OpenSandbox server

The Harbor recipes can run their sandboxes on an [OpenSandbox](https://github.com/opensandbox-group/OpenSandbox)
server instead of the local docker daemon: the trial server and `harbor run` stay where they are, each trial's
sandbox is created on the server, and every command reaches it through the server's proxy, over the endpoint's
scheme (https).

    export HARBOR_ENVIRONMENT_IMPORT_PATH=miles_harbor.opensandbox_env:OpenSandboxEnvironment
    export OPENSANDBOX_ENDPOINT=https://<server> OPENSANDBOX_KEY=<key>   # from your shell, never a committed file
    export OPENSANDBOX_CA_FILE=<server.pem>        # a self-signed server: its certificate
    python -m miles_harbor.opensandbox_env check    # OPENSANDBOX_CHECK_OK: a sandbox created, a command run, killed
    HARBOR_DATASET=terminal-bench@2.0 HARBOR_N_TASKS=8 scripts/harbor_run.sh --out results/tb2-oracle

The server and its key are read from the environment only (the scripts print the backend's kwargs with the job's
config); `OPENSANDBOX_PROXY` names a proxy to reach the server through. What the tasks and the harnesses need: a
prebuilt image per task (`environment.docker_image`: no Dockerfile build, no compose), checked as the trial server
starts; one network for the whole trial (the agent keeps the task's: `HARBOR_AGENT_NETWORK=endpoint` is not
enforced, which the trial server says as it starts); a harness that runs beside the sandbox (vanillux) needs
nothing more, one inside it must reach its model endpoint from the server's network. Each sandbox carries the run's
`RUN_TAG` as a label and expires on the server after 6 hours; `python -m miles_harbor.opensandbox_env sweep --run
<RUN_TAG>` kills a run's leftovers at once.

## Ports: one run per machine

A run and its services use fixed ports and hand each other `http://127.0.0.1:<port>` URLs, so the scripts assume
one run per machine: a second run on the same box would collide. `scripts/run_local.sh` pre-checks
`PROJECT_PORTS` with `ss -tln` before it starts anything and ends with `FATAL:` naming the holder (never a kill;
`scripts/stop_all.sh` stops what a crashed run left behind, in the launcher's own cleanup order). A port held by
one of the run's own external services (`RETRIEVER_URL`, `NEMO_GYM_URL`, `WEBSHOP_ENGINE_BASE_URL`,
`TEACHER_URL`) is skipped.

| port(s) | holder | knob |
|---|---|---|
| 6100 | the retriever relay of search_r1 under a wrapper (locally `RETRIEVER_URL` names the retriever itself) | `RETRIEVER_URL` |
| 6200 | the WebShop engine | `WEBSHOP_SERVER_PORT` |
| 6379 | ray's GCS (the head) | fixed |
| 7100-7131 | the session servers of the rollout engines | fixed |
| 7200 | the router of the rollout engines (`--sglang-router-port`) | fixed |
| 8100 | the gym head server | fixed |
| 8200 | the gym agent server a run starts itself | `NEMO_GYM_PORT`; an external one: `NEMO_GYM_URL` |
| 8265 | ray's dashboard: cannot move, miles submits its training job to `http://127.0.0.1:8265` | fixed |
| 8300-8999 | the gym resource and model servers | fixed range |
| 13141-13143 | the OPD teacher (13141); the R2S pipeline's base-student and judge servers (13142, 13143) | fixed |
| 2000-2999 | ray's worker ports (open between the machines of a multi-node run) | fixed |

`PROJECT_PORTS` (default `6100 6200 6379 7100-7131 7200 8100 8200 8265 13141-13143`) is the list the pre-check
walks; `scripts/serve_model.sh` (`SERVE_PORT`) and a second `scripts/ray_cluster.sh` head are the only things
meant to share a box with a run, on ports of their own.

## Stop and resume

`touch results/<tag>/stop` (the run's `PREEMPT_SENTINEL`, under `MILES_AGENT_RESULTS`): the driver finishes the
step it is in, saves, writes `results/<tag>/stop.done` (`{"rollout_id": N}`), and the entry ends with
`RUN_INNER_PREEMPT_STOP rc=0 durable=...`. A TERM or USR1 to `scripts/run_local.sh` does the same: the trap
raises the sentinel instead of killing the run. Resume is the same command with the same `--run-tag`, after the
sentinel is gone: an entry started while `results/<tag>/stop` still exists prints `PREEMPT before launch: not
starting the driver` and exits 0 without training. The driver continues from the last save under
`results/<tag>/`; a different tag is a new run.

## Multi-node without a scheduler

`scripts/ray_cluster.sh head` on the head machine, `scripts/ray_cluster.sh join <head ip>` on each other one (or
`ssh-join <head ip> <host>...` from the head), then on the head `scripts/run_local.sh --topology <t> --num-nodes
N ...`, which reuses the running head and waits for the workers' GPUs. Requirements: the same checkout, model
and data paths on every machine (a shared filesystem, or an rsync); the ports 6379, 8265 and 2000-2999 open
between them; the gym on the head only (`scripts/run_local.sh` starts it there, so every session URL stays on
the box that runs the gym). `scripts/ray_cluster.sh stop [<host>...]` ends the cluster. In containers: the same
container on every machine (docs/containers.md).

## Checks that need no GPU

- `scripts/run_local.sh --dry-run --recipe <recipe> --run-tag <tag>`: every command of a run, nothing started
  (a missing model or data file is a `WARN:` there, a `FATAL:` in a real run).
- `scripts/contract_selftest.sh`: parses every script, lists the wrapper contract's variables, runs the dry run
  above and checks the example wrapper (docs/cluster.md); `. tests/fake_env.sh` first for the complete fake
  contract.
- `scripts/check_patches.sh`: the patches are applied to `3rdparty/miles`.
- `scripts/smoke_tests.sh` (where miles imports: the image, or a machine with its dependencies): this repo's
  tests and the miles fast tests that cover the patches, `CONTRACT_TESTS_OK` last.
