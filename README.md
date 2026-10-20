# miles-agent

**Personal research code**, not a product: this repository may be heavily deleted or refactored at any time.
Each paper's reproduction is pinned by a release tag named after the paper (see [Papers](#papers)); `main` moves on.

miles-agent wires **miles** (the RL trainer and its TITO session server) to agentic environments without modifying
either side: **NeMo Gym** HTTP environments (`wiki_search`, `alfworld_env`, `webshop_env`, `scienceworld_env`) and
**Harbor** trials (the task format the coding and terminal papers use). The overlay owns only the adapters between
trainer and environment and the recipes that set each experiment's knobs: miles owns the tokens, the environment
owns the episode. Everything here is meant to be read, swapped or copied.

## How an episode runs

```
miles train  ──custom-generate──▶ miles.rollout.generate_hub.agentic_tool_call.generate
                                        │  mints per-episode session URL
                                        ▼
                     miles_agent/gym_bridge.run
                                        │  POST /run {responses_create_params,
                                        │             metadata (tools, task, budgets),
                                        ▼             policy_base_url=session URL}
                     an episode server (a gym agent, or miles_harbor's trial server)
                        │ episode loop over POST {session}/v1/chat/completions
                        │ tool_calls → the env's routes (/search, /step, /sandbox/exec ...)
                        │ done       → the env's verdict → reward
                        ▼
                     Miles session server records every turn (TITO: token ids,
                     logprobs, loss masks; obs spans masked 0) → Sample(s)
                     reward rides sample.metadata["reward"] → reward hook
```

Division of labor: **Miles owns tokens** (fixed per-family TITO template,
append-only, drift asserted not healed), **the environment owns the episode**
(tool dispatch, sandboxes, verification), the overlay owns only the thin
adapters above.

Three episode protocols share this chassis; all speak OpenAI tool calling on the policy side and differ only in how
the episode runs:

- `tool_call_agent` — OpenAI tool-calling protocol: tools declared per row,
  each tool_call POSTs to the resources-server route of the same name, the
  MODEL ends the episode by not calling tools, reward comes from POST /verify.
  (wiki_search / search_r1.)
- `env_loop_agent` — Gymnasium protocol: the ENV speaks first (POST /reset),
  every assistant turn is one action attempt carried as a call to the row's
  action tool (POST /step), the ENV declares terminated/truncated, reward is
  the sum of step rewards; no /verify. (alfworld_env / alfworld, webshop_env /
  webshop, scienceworld_env / scienceworld.) The three env servers subclass
  `gym_root/agent_utils/session_gym.py` and share `procslots.py`, the
  worker-process slot pool. A /run body may carry `scripted_prefix`,
  assistant turns replayed through /step before the policy's first call so
  the policy resumes a recorded episode mid-trajectory; the /run response
  carries `last_observation`, the observation after the final action (the
  R2S takeover stage votes on its fingerprint).
- `miles_harbor.trial_server` — Harbor trials: POST /run runs one Harbor trial (the task's environment is Harbor's
  docker environment, or the backend named by `HARBOR_ENVIRONMENT_IMPORT_PATH`; a harness from
  `miles_harbor.harnesses`, installed from a pack under `HARBOR_PACK_ROOT` built by
  `gym_root/miles_harbor/packs/harbor_packs.sh`; the task's verifier) with the agent pointed at the rollout's
  session server; reward is the verifier's. (`recipes/tmax`; `scripts/harbor_trial_server.sh`,
  `scripts/harbor_run.sh`.) With Harbor's docker environment the pack must be visible inside every trial container:
  build it, then mount it into the containers yourself — the public tree has no mount mechanism for it yet
  (`docs/data.md`).

Tool-call markup belongs to the model, never to the overlay: the policy
model's chat template renders the markup a prompt example shows
(`miles_agent/tool_call_markup.py`), miles' TITO family parses it back, and
envs, agents and rewards see structured calls only. `POLICY_MODEL` selects the
model (default Qwen3.5-4B); `launcher.MODEL_FAMILIES` maps its name prefix to
the TITO template, the engines' tool-call parser and the chat-template
kwargs, so a new family is one table entry (or the explicit `--tito-model` /
`--chat-template-kwargs` / `--megatron-model-type` flags). A mixture of
experts (a family marked `moe`, or a Qwen checkpoint named by its active
parameters, `-A3B`) trains with its experts over a node's GPUs and the
engines' routing replayed.

## Install

Requirements: Linux, bash 4+, `git`, `curl`, [`uv`](https://docs.astral.sh/uv/), docker (recommended; see
`docs/containers.md`), one or more machines with GPUs that can reach each other over ssh. WebShop and ScienceWorld
need Java 11+ (micromamba installs it without root; `scripts/fetch_data.sh` tells you how).

    git clone --recurse-submodules https://github.com/Tree-Shu-Zhao/miles-agent && cd miles-agent
    bash scripts/setup.sh          # uv venv for the tests, submodules, the miles patches (scripts/check_patches.sh), local.env

- `3rdparty/miles` and `3rdparty/Gym` are submodules pinned to upstream releases; the trainer runs as that pin plus
  the patches under `patches/miles/` (`patches/miles/README.md` documents each one; `scripts/check_patches.sh --apply`
  applies them, `setup.sh` calls it).
- Training runs inside the public miles image with the patched submodule mounted over the image's copy:
  `docs/containers.md` has the `docker run` line. Bare metal works too: install miles as its own docs say, then
  use the same scripts.
- Gym environment venvs: `bash scripts/build_gym_venvs.sh --verify [server ...]` builds them under `GYM_VENV_ROOT`
  (default `.gym_venvs/`). The root must be the path they will run from (uv venvs are not relocatable), and on a
  shared home directory put it on local disk.
- Configuration is one file, `local.env` (copied from `local.env.example` by `setup.sh`): the three roots
  `MILES_AGENT_MODELS`, `MILES_AGENT_DATA`, `MILES_AGENT_RESULTS` (default `models/ data/ results/` in the checkout),
  `GYM_VENV_ROOT`, `WANDB_API_KEY`/`WANDB_ENTITY`/`WANDB_ENABLE`, `HF_TOKEN`. The environment always wins over the file.
  Every script takes `--dry-run` and prints the commands it would run; `scripts/run_local.sh --dry-run --recipe
  <recipe> --run-tag <tag>` prints every command of a run without starting anything.

## Data

Every dataset belongs to its upstream project; nothing is redistributed here. `scripts/fetch_data.sh <env>`
downloads or imports it into `$MILES_AGENT_DATA/<env>` and verifies digests; `docs/data.md` lists sources and licences.
Nothing derived from upstream data is committed: the WebShop goal export and the ScienceWorld eval list are rebuilt
by the same script and checked against pinned digests (`455c9eaa96191605`, `b2e1024f…2739`).

| env | command | what you get |
|---|---|---|
| ALFWorld | `bash scripts/fetch_data.sh alfworld` | the `json_2.1.1` game files via `alfworld-download` |
| ScienceWorld | `bash scripts/fetch_data.sh scienceworld [--from <test.jsonl> \| --synthesize]` | checks Java and the simulator jar; rebuilds the frozen evaluation list (sha256 `b2e1024f82a09e63…`) from the TCOD campaign file, or synthesizes the same tasks in sorted order |
| WebShop | `bash scripts/fetch_data.sh webshop --from <dir>` / `--regenerate-goals` | imports the WebShop authors' catalog files (md5 table) and regenerates the frozen goal set (digest `455c9eaa96191605`) |
| Search-R1 | `bash scripts/fetch_data.sh search_r1` | NQ + HotpotQA train, the 893-row validation mix and the test set from FlashRAG; the retriever is your own service (`RETRIEVER_URL`) |

## Quick start (one machine)

    hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
    # everything below runs inside the miles container (docs/containers.md)
    bash scripts/build_gym_venvs.sh --verify alfworld_env env_loop_agent
    bash scripts/fetch_data.sh alfworld
    bash scripts/run_local.sh --recipe alfworld --run-tag alf1 --dry-run   # prints every command
    bash scripts/run_local.sh --recipe alfworld --run-tag alf1             # GPUs are measured, one node
    touch results/alf1/stop                           # stop at the next checkpoint; rerun the same tag to resume
    bash scripts/eval_ckpt.sh alfworld --eval-load-tag alf1 --seed 1001
    bash scripts/export_hf.sh alf1 latest models/alf1-hf && bash scripts/serve_model.sh models/alf1-hf
    bash scripts/stop_all.sh

Multi-node without a scheduler: `scripts/ray_cluster.sh head` on the head, `scripts/ray_cluster.sh join <head ip>`
on the others, then `run_local.sh --topology <t> --num-nodes <n>` on the head (`docs/local_setup.md`). Under a
scheduler, any wrapper that exports the small environment contract of `docs/cluster.md` and runs
`scripts/run_local.sh` works; `examples/slurm/` holds the minimal one.

## Recipes

| recipe | environment | protocol | notes |
|---|---|---|---|
| `alfworld` | `alfworld_env` | env_loop_agent | `prepare_data.py`, `run.py`, `eval_ckpt.py`, `opd.py` (the distillation arm) |
| `scienceworld` | `scienceworld_env` | env_loop_agent | same shape; frozen TCOD task split |
| `webshop` | `webshop_env` + `env_servers/webshop` engine tier | env_loop_agent | `docs/webshop_integration.md` |
| `search_r1` | `wiki_search` | tool_call_agent | needs a retriever at `RETRIEVER_URL`; `eval_full.py` for the full test set |
| `tmax` | Harbor trials (`miles_harbor`) | trial server | terminal-agent RL on Harbor's docker backend |

`RECIPE`, `RUN_TAG`, `TOPOLOGY`, `PROMPT` and `RUN_SCRIPT` select the run; `prompts/<env>/<name>.py` holds the words of
every prompt (`prompts/README.md`); `miles_agent/launcher.py` holds every shared science flag once, with its reason.

## Papers

| paper | tag | document |
|---|---|---|
| R2S (teacher repair for agentic on-policy distillation) | `r2s` | `docs/r2s.md` |

`git checkout r2s` gives the code the paper's main-table cells were run with. The paper's checkpoints are not
published yet.

## Layout

- `miles_agent/` — the trainer-side adapters: `launcher` (shared flags, `MODEL_FAMILIES`, topologies), `gym_bridge`
  (transport, env_metrics), `reward_routing`, `dapo_filter`, `dppo`, `tool_call_markup`, `prompts`, `eval_ckpt`,
  `opd/` and `r2s/` (the distillation arm and the repair pipeline).
- `gym_root/` — the NeMo Gym overlay (`NEMO_GYM_EXTRA_ROOTS`): resources servers, the two protocol agents,
  `agent_utils/`, `miles_harbor/` (the Harbor trial server, its harness packs and TMax's vanillux harness).
- `env_servers/webshop/` — the py3.10 WebShop engine tier (vendored WebShop, MIT).
- `recipes/`, `prompts/`, `patches/miles/`, `scripts/` (every public command; `scripts/lib/` the shared bash),
  `tools/` (`convert_ckpt_to_hf.py`, `wandb_core_view.py --entity <you>`, `r2s_decodes.py`), `examples/slurm/`.
- `docs/` — `local_setup.md`, `containers.md`, `data.md`, `cluster.md` (the wrapper contract),
  `agentic_rl_metrics.md` (the wandb metric guide), `webshop_integration.md`, `scienceworld_integration.md`,
  `results_search_r1.md`, `r2s.md`.
- `tests/` — the coupling-surface contracts; `pytest -q tests` needs no GPU.

## License

Apache-2.0 (`LICENSE`). `env_servers/webshop/webshop/` is a vendored fork of WebShop under its own `LICENSE.md`;
third-party notices in `NOTICE`.
