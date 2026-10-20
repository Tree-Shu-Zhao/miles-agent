# miles-agent

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/)
[![miles](https://img.shields.io/badge/miles-0.1.2-blue.svg)](https://github.com/radixark/miles)
[![NeMo Gym](https://img.shields.io/badge/NeMo%20Gym-0.7.0-76b900.svg)](https://github.com/NVIDIA-NeMo/Gym)

> **Personal research code**, not a product: this repository may be heavily deleted or refactored at any time.
> Each paper's reproduction is pinned by a release tag named after the paper ([Papers](#-papers)); `main` moves on.

## 📖 Overview

miles-agent sits between **miles**, the RL trainer, and the environments an agent acts in: trainer-side adapters
plus one recipe per experiment, modifying neither side beyond `patches/miles/`. The environments are **NeMo Gym**
HTTP servers for ALFWorld, ScienceWorld, WebShop and wiki search, and **Harbor** trials (a containerised task, a
harness and its verifier) for coding and terminal agents. miles owns the tokens, the environment owns the episode,
this repository owns only the adapters between them and each experiment's knobs. Everything here is meant to be
read, swapped or copied; [`docs/architecture.md`](docs/architecture.md) shows how an episode runs.

### 🔑 What is here

- **Adapters** (`miles_agent/`): the launcher, the episode bridge, reward routing, tool-call markup, `opd/`, `r2s/`.
- **Environments** (`gym_root/`, `env_servers/`): Gym servers and agents, the Harbor trial server, the WebShop engine.
- **Recipes and prompts** (`recipes/`, `prompts/`): one directory per experiment, one file per prompt.
- **Patches and scripts** (`patches/miles/`, `scripts/`): the miles patch series; every entry script takes `--dry-run`.

## 📄 Papers

| paper | tag | document |
|---|---|---|
| R2S: Rollout-to-Scorer Distillation (teacher repair for agentic on-policy distillation) | `r2s` | `docs/r2s.md` |

`git checkout r2s` ([browse the tag](https://github.com/Tree-Shu-Zhao/miles-agent/tree/r2s)) gives the paper's code
with its own README; the trained teachers and students are on the [Hugging Face Hub](https://huggingface.co/TreezzZ).

## 🔧 Installation

Requirements: Linux, bash 4+, `git`, `curl`, [`uv`](https://docs.astral.sh/uv/), docker with GPU support
(recommended), one multi-GPU machine or several reachable over ssh; Java 11+ for WebShop and ScienceWorld.

```bash
git clone --recurse-submodules https://github.com/Tree-Shu-Zhao/miles-agent && cd miles-agent
bash scripts/setup.sh
```

- `setup.sh` initialises the submodules `3rdparty/miles` and `3rdparty/Gym` (pinned to upstream releases) and
  applies the patches of `patches/miles/` to the miles pin (`scripts/check_patches.sh --apply`).
- Training runs in the miles image (the patched submodule mounted over its copy, `docs/containers.md`) or on bare metal.
- Configuration is one file, `local.env` (`setup.sh` copies `local.env.example`): the roots `MILES_AGENT_MODELS`,
  `MILES_AGENT_DATA` and `MILES_AGENT_RESULTS` (default `models/`, `data/`, `results/` under the checkout), the venv
  root `GYM_VENV_ROOT`, wandb and Hub tokens. A variable set in the environment always wins over the file.

## 🚀 Quick start

From the checkout root, inside the miles container (`REPO=$(pwd)`, then the `docker run` line of `docs/containers.md`):

```bash
uvx --from "huggingface_hub>=1.0" hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
export GYM_VENV_ROOT=$(pwd)/.gym_venvs                                 # absolute; or set it in local.env
bash scripts/build_gym_venvs.sh --verify alfworld_env env_loop_agent   # builds the venvs at $GYM_VENV_ROOT
bash scripts/fetch_data.sh alfworld                                    # the game files, through that venv
bash scripts/run_local.sh --dry-run --recipe alfworld --run-tag alf1   # prints every command, starts nothing
bash scripts/run_local.sh --recipe alfworld --run-tag alf1             # trains on this machine's GPUs
bash scripts/eval_ckpt.sh alfworld --eval-load-tag alf1 --seed 1001
bash scripts/export_hf.sh alf1 latest models/alf1-hf                   # <run tag> <save id or latest> <output dir>
bash scripts/stop_all.sh
```

`touch results/alf1/stop` stops the run after the step it is in (it saves first); remove the file, then rerun the
same run tag to resume from that save.

Several machines without a scheduler: `scripts/ray_cluster.sh head` on one,
`scripts/ray_cluster.sh join <head ip>` on the others, then `run_local.sh --topology async_3n --num-nodes 3` on the
head (`--help` lists the topologies; `docs/local_setup.md` covers both).

Under a scheduler, any wrapper that exports
the contract of `docs/cluster.md` and runs `scripts/run_local.sh` works; `examples/slurm/` holds the minimal one.

## 📊 Data

Nothing is redistributed: every dataset belongs to its upstream project. `scripts/fetch_data.sh <recipe>` downloads
or imports the upstream files under `$MILES_AGENT_DATA`, checks the WebShop catalog's md5 and pins the derived
ScienceWorld task list and WebShop goal set by digest; `docs/data.md` lists the sources, commands and licences.

## 🧪 Recipes

| recipe | environment | protocol | notes |
|---|---|---|---|
| `alfworld` | `alfworld_env` | env_loop_agent | `run.py`, `eval_ckpt.py`, `opd.py` (on-policy distillation) |
| `scienceworld` | `scienceworld_env` | env_loop_agent | same shape; a frozen task split |
| `webshop` | `webshop_env` + the `env_servers/webshop` engine tier | env_loop_agent | `docs/webshop_integration.md` |
| `search_r1` | `wiki_search` | tool_call_agent | a retriever at `RETRIEVER_URL`; `eval_full.py` for the full test set |
| `tmax` | Harbor trials (`miles_harbor`) | trial server | terminal agents (TMax-15K) on Harbor's docker backend; the trial server also serves coding harnesses |

`RECIPE`, `RUN_TAG`, `TOPOLOGY`, `PROMPT` and `RUN_SCRIPT` select a run (environment variables or `local.env` keys;
the first three are also `run_local.sh` flags); the ALFWorld, ScienceWorld and WebShop prompts live in
`prompts/<recipe>/<name>.py`; `miles_agent/launcher.py` holds every shared flag once, with its reason.

## 📚 Documentation

| page | contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | how an episode runs: the flow, the division of labor, the three episode protocols, tool-call markup |
| [`docs/local_setup.md`](docs/local_setup.md) | a plain machine: `setup.sh`, the roots, dry runs, several machines over ray |
| [`docs/containers.md`](docs/containers.md) | the miles image and the `docker run` line |
| [`docs/data.md`](docs/data.md) | the sources, licences and pinned digests of the data |
| [`docs/cluster.md`](docs/cluster.md) | the contract between a scheduler wrapper and the entry scripts |
| [`docs/agentic_rl_metrics.md`](docs/agentic_rl_metrics.md) | the wandb metrics: what matters, normal ranges, alarms, fixes |
| [`docs/webshop_integration.md`](docs/webshop_integration.md) | the WebShop environment and its engine tier |
| [`docs/scienceworld_integration.md`](docs/scienceworld_integration.md) | the ScienceWorld environment |
| [`docs/results_search_r1.md`](docs/results_search_r1.md) | the search_r1 campaign: numbers and decisions |
| [`docs/r2s.md`](docs/r2s.md) | the R2S paper: student arm, pilot dump, repair pipeline, reproduction ledger |

## 📁 Repository layout

```
miles-agent/
├── miles_agent/     # launcher, gym_bridge, rewards, filters, tool-call markup; opd/ distillation, r2s/ teacher repair
├── gym_root/        # the NeMo Gym side: resources_servers/, responses_api_agents/, agent_utils/, miles_harbor/
├── env_servers/     # webshop/: the WebShop engine tier (vendored WebShop)
├── recipes/         # alfworld, scienceworld, webshop, search_r1, tmax
├── prompts/         # <env>/<name>.py: the ALFWorld, ScienceWorld and WebShop prompts
├── patches/         # miles/: the patches on the miles pin, in series order
├── scripts/         # every public command; lib/ the shared bash
├── tools/           # checkpoint conversion, the wandb view, the R2S decode summary
├── examples/        # slurm/: the minimal scheduler wrapper
├── docs/            # the pages of the table above
└── tests/           # the coupling-surface contracts; `pytest -q tests` needs no GPU
```

## ⚖️ License

Apache-2.0 (`LICENSE`). `env_servers/webshop/webshop/` is a vendored fork of WebShop under its own `LICENSE.md`;
third-party notices in `NOTICE`. Cite the software with `CITATION.cff`, and a paper with the README at its tag.
