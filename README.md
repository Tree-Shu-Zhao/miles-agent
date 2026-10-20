# R2S: Rollout-to-Scorer Distillation

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-checkpoints-yellow.svg)](https://huggingface.co/TreezzZ)
[![Code](https://img.shields.io/badge/code-tag%20r2s-lightgrey.svg)](https://github.com/Tree-Shu-Zhao/miles-agent/tree/r2s)

*Personal research code frozen at the tag `r2s` for the paper; the repository's `main` may be heavily deleted or refactored at any time.*

## 📖 Overview

This repository holds the code of **R2S** (Rollout-to-Scorer Distillation), a teacher-repair step for agentic on-policy distillation, and the trainer wiring it runs on: [miles](https://github.com/radixark/miles) as the RL trainer (submodule `3rdparty/miles`, pinned plus the patches under `patches/miles/`) connected to NeMo Gym HTTP environments (submodule `3rdparty/Gym`) for ALFWorld, ScienceWorld and WebShop.

Agentic on-policy distillation (OPD) trains a student on its own multi-turn episodes against a frozen teacher: the teacher scores every token the student wrote, the loss is the sampled reverse KL to those per-token log-probabilities, and the task reward is logged, never trained on. R2S repairs the teacher before that distillation starts, from a pilot dump of the untrained student's episodes:

- **A, mine** the states where the student's deliberation diverges most from the teacher's; **B, take over**: the teacher continues each mined state K = 8 times through the live environment at temperature 1.0
- **C / J, select** one takeover per state: a consensus vote on how the takeovers end (ALFWorld, ScienceWorld) or an LLM judge of the purchase (WebShop)
- **D, label** every assistant token from the mined turn to the end of the episode with the teacher's own top-16 distribution given the selected takeover; **E, fine-tune** the teacher on those labels, promoted only if its held-out privileged cross-entropy drops

The student is then distilled from the repaired teacher under the unchanged OPD protocol.

### 🔑 Key Features

- 🛠️ **Teacher repair that needs no reward model**: stages A-E run on the student's own episodes, the live environment and the teacher's own distributions
- 🗳️ **Selection by consensus or by a judge**: a takeover is kept when enough of its siblings agree on how the episode ends (ALFWorld, ScienceWorld) or when an LLM judge approves the purchase (WebShop)
- 🎯 **Pure on-policy distillation**: the student arm is the sampled reverse KL on the student's own episodes, with the task reward only logged
- 🌍 **Three environments**: ALFWorld, ScienceWorld and WebShop, each as a NeMo Gym server behind one shared episode protocol
- 🤗 **24 released checkpoints**: the repaired teacher and the distilled student of every cell of the main table

| paper | tag | document |
|---|---|---|
| R2S: Rollout-to-Scorer Distillation | `r2s` | `docs/r2s.md` |

## 📋 Table of Contents

- [Installation](#-installation)
- [Data and Models](#-data-and-models)
- [Released Checkpoints](#-released-checkpoints)
- [Reproducing the Main Table](#-reproducing-the-main-table)
  - [1. Pilot Dump](#1-pilot-dump)
  - [2. Teacher Repair Pipeline](#2-teacher-repair-pipeline)
  - [3. Teacher Fine-Tuning](#3-teacher-fine-tuning)
  - [4. Student Arm](#4-student-arm)
  - [5. Decodes and the Table](#5-decodes-and-the-table)
  - [6. Evaluating a Released Checkpoint](#6-evaluating-a-released-checkpoint)
- [Repository Layout](#-repository-layout)
- [Citation](#-citation)

## 🔧 Installation

### Prerequisites

- Linux, bash 4+, `git`, docker with GPU support (the training side runs in the miles image; `docs/containers.md`)
- [uv](https://docs.astral.sh/uv/)
- Machines of at least four GPUs (the topologies count four per node) with the same checkout, model and data paths and the ports 6379, 8265, 2000-2999 and 13141 open between them (`docs/local_setup.md`): one for the pilot, the pipeline, the fine-tuning and each decode; four for a student arm
- One GPU must hold the whole teacher: it is served tensor-parallel 1 and data-parallel `TEACHER_DP` (4 in a student arm, 2 in the pipeline), every replica within 75 % of its GPU's memory (`TEACHER_MEM_FRAC`; the pipeline's teacher takes 80 %), weights in bf16
- Java 11 or newer for WebShop and ScienceWorld
- Disk for the six base models (2B to 30B parameters) and the checkpoints you train (a 150-step arm saves every 25 steps)

### Setup

```bash
git clone --recurse-submodules -b r2s https://github.com/Tree-Shu-Zhao/miles-agent.git
cd miles-agent
bash scripts/setup.sh    # submodules, the miles patches, the test venv, local.env
```

Training runs inside the public miles image with the patched submodule mounted over the image's copy; `docs/containers.md` has the `docker run` line. Inside the container, build the venvs of the six gym servers a run starts (the three environments, their agent, the model server and the relay server); a run starts its servers from these venvs and installs nothing, so none of the six may be missing (`--verify` import-probes each venv and plays one ScienceWorld episode through a real JVM):

```bash
bash scripts/build_gym_venvs.sh --verify alfworld_env webshop_env scienceworld_env env_loop_agent openai_model legacy_agent
```

Configuration is one file, `local.env` (copied from `local.env.example`), which the scripts read themselves: the roots `MILES_AGENT_MODELS`, `MILES_AGENT_DATA`, `MILES_AGENT_RESULTS` (default `models/`, `data/`, `results/` under the checkout) and `GYM_VENV_ROOT`, which you set to the build root above, `GYM_VENV_ROOT=$REPO/.gym_venvs` (`$REPO` is expanded by the file; the knob ships empty, which means `uv run` from `3rdparty/Gym` and leaves the built venvs unused). The command lines below expand the three roots in your shell, so export them there as well (`export MILES_AGENT_MODELS=$(pwd)/models MILES_AGENT_DATA=$(pwd)/data MILES_AGENT_RESULTS=$(pwd)/results`), or write the paths out. Every entry takes `--dry-run`: `scripts/run_local.sh --dry-run --recipe alfworld --run-tag <tag>` prints every command of a run without starting anything (`docs/local_setup.md`).

## 📊 Data and Models

**Datasets.** Every dataset belongs to its upstream project and is downloaded by you; `scripts/fetch_data.sh` imports it under `$MILES_AGENT_DATA/<env>` and checks the digests (`docs/data.md` lists sources, licenses and digests):

```bash
bash scripts/fetch_data.sh alfworld                    # the ALFWorld game files, via alfworld-download
bash scripts/fetch_data.sh scienceworld --synthesize   # the paper's 1308 eval tasks, in sorted order
bash scripts/fetch_data.sh webshop --from <dir>        # the WebShop authors' three catalog files, md5-checked
```

The paper's own ScienceWorld task order is not in the repository: `--synthesize` writes the same 1308 tasks in sorted order, so the recipe's 134-task `scienceworld_unseen` slice (the list's first 134 rows) is a different subset of them, and the ScienceWorld entries you reproduce show the method's effect against the vanilla twin but are not directly comparable with the paper's numbers (those need the campaign's `test.jsonl`, imported with `--from`).

WebShop's `<dir>` holds `items_shuffle_1000.json`, `items_ins_v2_1000.json` and `items_human_ins.json`. Its engine and the frozen goal export come next; the two engine scripts do not read `local.env`, so they take `WEBSHOP_HOME`: `WEBSHOP_HOME=$MILES_AGENT_DATA/webshop bash env_servers/webshop/setup.sh` builds the engine's environment, `WEBSHOP_HOME=$MILES_AGENT_DATA/webshop bash env_servers/webshop/launch_engine.sh` runs it in the foreground (a second terminal), and `bash scripts/fetch_data.sh webshop --regenerate-goals` exports the goals from that running engine. Stop the engine afterwards: a training run starts its own on the same port (`docs/data.md`).

**Base models.** The Hub checkpoints go under `$MILES_AGENT_MODELS/<name>`; `POLICY_MODEL`, `TEACHER_MODEL` and `JUDGE_MODEL` are directory names there:

| model | role |
|---|---|
| `Qwen3.5-2B`, `Qwen3.5-4B` | students of the Qwen3.5-9B teacher |
| `Qwen3.5-9B` | teacher of the Qwen3.5 students; the WebShop judge for every cell |
| `Qwen3-4B`, `Qwen3-8B` | students of the Qwen3-30B-A3B-Instruct-2507 teacher |
| `Qwen3-30B-A3B-Instruct-2507` | teacher of the Qwen3 students |

```bash
uvx --from "huggingface_hub>=1.0" hf download Qwen/Qwen3.5-9B --local-dir "$MILES_AGENT_MODELS/Qwen3.5-9B"   # one line per model
```

## 🤗 Released Checkpoints

The 24 checkpoints of the main table are on the Hugging Face Hub under [TreezzZ](https://huggingface.co/TreezzZ) (bf16 safetensors). The `-teacher-for-<student>` repository of a pair is the repaired teacher that distilled that pair's `-student`, the teacher's base model in its name; the `-student` is the final checkpoint of the 150-step arm. The paper's "teacher alone" row is the unrepaired base teacher of the table above.

| environment | base student | base teacher | repaired teacher | distilled student |
|---|---|---|---|---|
| ALFWorld | Qwen3.5-2B | Qwen3.5-9B | [r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-2b](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-2b) | [r2s-alfworld-qwen3.5-2b-student](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3.5-2b-student) |
| ALFWorld | Qwen3.5-4B | Qwen3.5-9B | [r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b) | [r2s-alfworld-qwen3.5-4b-student](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3.5-4b-student) |
| ALFWorld | Qwen3-4B | Qwen3-30B-A3B-Instruct-2507 | [r2s-alfworld-qwen3-30b-a3b-teacher-for-qwen3-4b](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3-30b-a3b-teacher-for-qwen3-4b) | [r2s-alfworld-qwen3-4b-student](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3-4b-student) |
| ALFWorld | Qwen3-8B | Qwen3-30B-A3B-Instruct-2507 | [r2s-alfworld-qwen3-30b-a3b-teacher-for-qwen3-8b](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3-30b-a3b-teacher-for-qwen3-8b) | [r2s-alfworld-qwen3-8b-student](https://huggingface.co/TreezzZ/r2s-alfworld-qwen3-8b-student) |
| ScienceWorld | Qwen3.5-2B | Qwen3.5-9B | [r2s-scienceworld-qwen3.5-9b-teacher-for-qwen3.5-2b](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3.5-9b-teacher-for-qwen3.5-2b) | [r2s-scienceworld-qwen3.5-2b-student](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3.5-2b-student) |
| ScienceWorld | Qwen3.5-4B | Qwen3.5-9B | [r2s-scienceworld-qwen3.5-9b-teacher-for-qwen3.5-4b](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3.5-9b-teacher-for-qwen3.5-4b) | [r2s-scienceworld-qwen3.5-4b-student](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3.5-4b-student) |
| ScienceWorld | Qwen3-4B | Qwen3-30B-A3B-Instruct-2507 | [r2s-scienceworld-qwen3-30b-a3b-teacher-for-qwen3-4b](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3-30b-a3b-teacher-for-qwen3-4b) | [r2s-scienceworld-qwen3-4b-student](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3-4b-student) |
| ScienceWorld | Qwen3-8B | Qwen3-30B-A3B-Instruct-2507 | [r2s-scienceworld-qwen3-30b-a3b-teacher-for-qwen3-8b](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3-30b-a3b-teacher-for-qwen3-8b) | [r2s-scienceworld-qwen3-8b-student](https://huggingface.co/TreezzZ/r2s-scienceworld-qwen3-8b-student) |
| WebShop | Qwen3.5-2B | Qwen3.5-9B | [r2s-webshop-qwen3.5-9b-teacher-for-qwen3.5-2b](https://huggingface.co/TreezzZ/r2s-webshop-qwen3.5-9b-teacher-for-qwen3.5-2b) | [r2s-webshop-qwen3.5-2b-student](https://huggingface.co/TreezzZ/r2s-webshop-qwen3.5-2b-student) |
| WebShop | Qwen3.5-4B | Qwen3.5-9B | [r2s-webshop-qwen3.5-9b-teacher-for-qwen3.5-4b](https://huggingface.co/TreezzZ/r2s-webshop-qwen3.5-9b-teacher-for-qwen3.5-4b) | [r2s-webshop-qwen3.5-4b-student](https://huggingface.co/TreezzZ/r2s-webshop-qwen3.5-4b-student) |
| WebShop | Qwen3-4B | Qwen3-30B-A3B-Instruct-2507 | [r2s-webshop-qwen3-30b-a3b-teacher-for-qwen3-4b](https://huggingface.co/TreezzZ/r2s-webshop-qwen3-30b-a3b-teacher-for-qwen3-4b) | [r2s-webshop-qwen3-4b-student](https://huggingface.co/TreezzZ/r2s-webshop-qwen3-4b-student) |
| WebShop | Qwen3-8B | Qwen3-30B-A3B-Instruct-2507 | [r2s-webshop-qwen3-30b-a3b-teacher-for-qwen3-8b](https://huggingface.co/TreezzZ/r2s-webshop-qwen3-30b-a3b-teacher-for-qwen3-8b) | [r2s-webshop-qwen3-8b-student](https://huggingface.co/TreezzZ/r2s-webshop-qwen3-8b-student) |

## 🚀 Reproducing the Main Table

The protocol of every cell: 150 steps of on-policy distillation at rollout temperature 0.4 with thinking off (`recipes/<env>/opd.py`, selected with `RUN_SCRIPT=opd.py`, under the paper's one-shot prompt `PROMPT=r2s_oneshot`). A cell is its final checkpoint decoded three times: the in-training eval after step 150 (seed 42) and two eval-only decodes of the same weights (seeds 43 and 44 by default; the paper's own decodes used 42 / 1001 / 1002). The metric is `eval/<slice>/env/won`, averaged over the seen and unseen splits per decode on ALFWorld and ScienceWorld and taken on WebShop's 500 canonical goals (the engine's goal positions 0..499); the table entry is the mean +- std over the three decodes.

The steps below show the ALFWorld cell of Qwen3.5-9B -> Qwen3.5-4B; the other cells change `--recipe`, `POLICY_MODEL`, `TEACHER_MODEL`, step 3's `BASE_MODEL` and the names, while the WebShop judge is `Qwen3.5-9B` in every cell. `<pilot>`, `<tag>`, `<arm>`, `<twin>`, `<anchor>` and `<eval>` are names you choose: a run tag names the run's directory under `$MILES_AGENT_RESULTS` and is never reused; `TAG` names the pipeline's output directory in step 2 and the repaired teacher's model directory in step 3. Each step states its machine count and wall-clock (`docs/r2s.md` §5).

### 1. Pilot Dump

The untrained student plays the arm's protocol at temperature 1.0 with no teacher; nothing trains or saves (one node, ~2 h). Rows land in `$MILES_AGENT_RESULTS/<pilot>/opd_dump/dump_*.jsonl`:

```bash
RUN_SCRIPT=opd.py MODE=debug_rollout_only PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B \
EXTRA_ARGS="--algo grpo --opd-dump --rollout-temperature 1.0" \
  scripts/run_local.sh --recipe alfworld --run-tag <pilot> --topology colocated_1n
```

### 2. Teacher Repair Pipeline

Stages A -> B -> C -> D over the dump on one node (~6 h; stage B dominates). Outputs go to `$MILES_AGENT_DATA/r2s/<tag>/{states,fold,judge,tau,labels}.jsonl`, each with a `.summary.json` that carries the stage's marker; a stage whose marker is already there is skipped on a rerun. WebShop is the same line with `--recipe webshop`: the judge stage J goes in before C by itself (the paper ran it as `STOP_AFTER_FOLD=1`, then `STAGES=J,C,D` under the same `TAG`; `docs/r2s.md` §4.4):

```bash
MODE=r2s_pipeline TAG=<tag> TEACHER_MODEL=Qwen3.5-9B POLICY_MODEL=Qwen3.5-4B \
DUMP_GLOB="$MILES_AGENT_RESULTS/<pilot>/opd_dump/dump_*.jsonl" scripts/run_local.sh --recipe alfworld --run-tag <tag>
```

### 3. Teacher Fine-Tuning

Stage E fine-tunes the base teacher (`BASE_MODEL`) on stage D's labels on one GPU machine (~1-2 h) and promotes it only if the held-out privileged cross-entropy drops: `R2S_FT_PASS` puts the repaired teacher at `$MILES_AGENT_MODELS/<TAG>`, `R2S_FT_FAIL` promotes nothing. This `TAG` is the student arm's `TEACHER_MODEL`; an existing directory of that name is refused, so pick another `TAG` if the released teacher is already there:

```bash
TAG=r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b LABELS=$MILES_AGENT_DATA/r2s/<tag>/labels.jsonl BASE_MODEL=Qwen3.5-9B \
  scripts/r2s_teacher_ft.sh
```

### 4. Student Arm

The R2S arm distills the student from the repaired teacher for 150 steps (~10-12 h) on `async_4n_teacher`: four nodes, the last one serving the teacher. The vanilla OPD twin uses the base teacher and the paper's vanilla seed (43 on ALFWorld and ScienceWorld, 42 on WebShop); the paper's twins ran under `--topology async_2n_teacher --num-nodes 2`, one training node plus the teacher's. `TEACHER_URL=http://<host>:13141`, an already serving teacher, takes one node off either count (`docs/r2s.md` §2.3). On plain machines every node runs the same line with its own `NODE_RANK` (0 = the head, 3 = the teacher) and the head's `HEAD_IP`, with `$MILES_AGENT_RESULTS` on a filesystem all four see; a scheduler wrapper does this per node (`docs/cluster.md`). Keep the head's output, step 5 reads it:

```bash
# the R2S arm: the repaired teacher, seed 42
RUN_SCRIPT=opd.py PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B TEACHER_MODEL=r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b \
  scripts/run_local.sh --recipe alfworld --run-tag <arm> --topology async_4n_teacher --num-nodes 4 2>&1 | tee <arm>.log
# the vanilla OPD twin: the base teacher and the vanilla seed
RUN_SCRIPT=opd.py PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B TEACHER_MODEL=Qwen3.5-9B EXTRA_ARGS="--seed 43" \
  scripts/run_local.sh --recipe alfworld --run-tag <twin> --topology async_4n_teacher --num-nodes 4 2>&1 | tee <twin>.log
```

### 5. Decodes and the Table

Decodes 2 and 3 reload the final checkpoint (`--eval-load-iter 149`: save ids are 0-based) on one node each and must carry the arm's prompt. The paper's anchor rows, teacher alone and student alone, decode the HF weights with `--eval-base` under seeds 42 / 43 / 44. `tools/r2s_decodes.py` reads the last eval record of each run's output (`--logs`, the files kept with `tee`; `--runs` resolves run tags to a scheduler wrapper's job logs) and prints the mean +- std per slice plus the seen/unseen mean per decode; outside a wrapper its `WARN ...: no JOB_PASS` line is expected for every run (the entry's own success marker is `RUN_INNER_PASS`):

```bash
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B scripts/eval_ckpt.sh alfworld --recipe-script opd.py \
  --eval-load-tag <arm> --eval-load-iter 149 --seed 43 --run-tag <arm>_d43 2>&1 | tee <arm>_d43.log
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B scripts/eval_ckpt.sh alfworld --recipe-script opd.py \
  --eval-load-tag <arm> --eval-load-iter 149 --seed 44 --run-tag <arm>_d44 2>&1 | tee <arm>_d44.log
# anchors: the HF weights, rollout-only
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-9B scripts/eval_ckpt.sh alfworld --recipe-script opd.py --eval-base --seed 42 --run-tag <anchor>
# the table entry: the in-training decode (the arm's log) and the two reloads
python3 tools/r2s_decodes.py --logs <arm>.log <arm>_d43.log <arm>_d44.log
```

### 6. Evaluating a Released Checkpoint

A released student is decoded like an anchor: download it under `$MILES_AGENT_MODELS`, keep `POLICY_MODEL` the name of its base model (the trainer derives the model type from that name; `Qwen3-4B` / `Qwen3-8B` for the Qwen3 students) and point `POLICY_HF` at the downloaded directory, then run `--eval-base` with the arm's prompt and `--recipe-script opd.py` (the protocol's temperature, thinking off and per-turn budget), three seeds for a table entry:

```bash
uvx --from "huggingface_hub>=1.0" hf download TreezzZ/r2s-alfworld-qwen3.5-4b-student \
  --local-dir "$MILES_AGENT_MODELS/r2s-alfworld-qwen3.5-4b-student"
PROMPT=r2s_oneshot POLICY_MODEL=Qwen3.5-4B POLICY_HF="$MILES_AGENT_MODELS/r2s-alfworld-qwen3.5-4b-student" \
  scripts/eval_ckpt.sh alfworld --recipe-script opd.py --eval-base --seed 42 --run-tag <eval>
```

A released teacher, downloaded the same way into `$MILES_AGENT_MODELS/r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b`, is the `TEACHER_MODEL` of step 4 as written there: distilling your own student from it skips steps 1-3.

## 📁 Repository Layout

```
miles-agent/
├── miles_agent/
│   ├── launcher.py              # the shared flags: --algo opd, MODEL_FAMILIES, the topologies
│   ├── opd/                     # the student arm: teacher scoring, turn segmentation, the pilot dump
│   └── r2s/                     # the repair pipeline: divergence (A), takeover (B), judge (J), select (C), label (D), teacher_ft (E)
├── recipes/<env>/opd.py         # the paper's protocol over alfworld / scienceworld / webshop (RUN_SCRIPT=opd.py)
├── prompts/<env>/r2s_oneshot.py # the one-shot prompt of every R2S run (PROMPT=r2s_oneshot)
├── scripts/
│   ├── run_local.sh             # the entry: the pilot, the pipeline (MODE=r2s_pipeline) and the student arm
│   ├── r2s_pipeline.sh          # stages A-D (and J) on one machine, sourced by run_local.sh
│   ├── r2s_teacher_ft.sh        # stage E and its promotion gate
│   ├── teacher_node.sh          # the teacher served on the last node of a *_teacher topology
│   └── eval_ckpt.sh             # one decode of a saved checkpoint or of HF weights
├── tools/r2s_decodes.py         # mean +- std over the three decodes of a cell
├── gym_root/, env_servers/      # the NeMo Gym environment servers and the WebShop engine
├── patches/miles/               # the patches applied over the pinned miles submodule
└── docs/r2s.md                  # the reference behind every step above (cited by §)
```

## 📚 Citation

The paper is under review; the BibTeX entry will be added here with the preprint. To cite the software, use `CITATION.cff` (GitHub's "Cite this repository" button).

## ⚖️ License

Apache-2.0 (`LICENSE`). `env_servers/webshop/webshop/` is a vendored fork of WebShop under its own `LICENSE.md`; third-party notices are in `NOTICE`.

`docs/r2s.md` holds the flags, the file contracts, the pitfalls and the reproduction ledger behind every step above.
