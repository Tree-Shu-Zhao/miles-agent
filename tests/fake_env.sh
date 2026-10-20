#!/bin/bash
# tests/fake_env.sh — a complete fake Contract A (docs/cluster.md, section A: every name of tests/contract_vars.txt)
# for a machine that has no job wrapper. Sourced, never run:
#   . tests/fake_env.sh && bash scripts/contract_selftest.sh      # the no-cluster self-test (the pin procedure of a
#                                                                 # wrapper repo: the SHA it last passed against)
#   . tests/fake_env.sh && bash scripts/run_local.sh --dry-run --recipe alfworld --run-tag t1
# Everything hangs under FAKE_ENV_ROOT (default ${TMPDIR:-/tmp}/miles-fake, exported); the addresses are RFC 5737
# documentation addresses (192.0.2.x); the optional names are exported EMPTY: set, so that a listing counts them, and
# empty, so that the entries take their defaults (GYM_VENV_ROOT empty = the `uv run` gym path; no external gym,
# teacher or retriever; no secret). GPUS_PER_NODE and RUN_TAG keep a value already in the environment (defaults 1
# and selftest). The fake TMPDIR is the one thing created here (a TMPDIR has to exist: mktemp and the self-test's
# stub dir write under it); everything else under FAKE_ENV_ROOT is created only by what runs under it (a dry run
# creates nothing more).
FAKE_ENV_ROOT=${FAKE_ENV_ROOT:-${TMPDIR:-/tmp}/miles-fake}
export FAKE_ENV_ROOT
export CONTRACT_VERSION=1
export RUN_TAG=${RUN_TAG:-selftest}
export JOB_ID=1 JOB_RESTART=0 JOB_NAME=miles_run
export NNODES=1 NODE_RANK=0 HEAD_IP=192.0.2.10 NODE_IP=192.0.2.10
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export PROJECT_DIR=$FAKE_ENV_ROOT/project
export MODELS_DIR=$PROJECT_DIR/models DATA_DIR=$PROJECT_DIR/data RESULTS_DIR=$PROJECT_DIR/results RUNS_DIR=$PROJECT_DIR/runs
export TOOLS_DIR=$PROJECT_DIR/tools IMAGES_DIR=$PROJECT_DIR/images STAGE_TARS_DIR=$PROJECT_DIR/stage_tars
export CACHE_DIR=$PROJECT_DIR/cache HF_HOME=$PROJECT_DIR/cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_EGRESS=1
export NODE_LOCAL=$FAKE_ENV_ROOT/node_local/miles-$JOB_ID
export TMPDIR=$FAKE_ENV_ROOT/tmp TRITON_CACHE_DIR=$NODE_LOCAL/jit/triton TORCHINDUCTOR_CACHE_DIR=$NODE_LOCAL/jit/inductor
mkdir -p "$TMPDIR"
export TRITON_PTXAS_PATH=
export STAGE_DIR=$PROJECT_DIR/jobs/fake
export STAGE_SHA=0000000000000000000000000000000000000000 INFRA_SHA=0000000000000000000000000000000000000000
export SYNC_DIR=$FAKE_ENV_ROOT/sync/${JOB_ID}_$JOB_RESTART
export PREEMPT_SENTINEL=$SYNC_DIR/preempt
export WANDB_API_KEY= HF_TOKEN=
export NEMO_GYM_URL= TEACHER_URL= RETRIEVER_URL= GYM_VENV_ROOT=
export ENTRY=scripts/run_local.sh
export SESSION_SERVER_EXTERNAL_HOST= SANDBOX_TRANSPORT=
