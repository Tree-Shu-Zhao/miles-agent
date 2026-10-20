# pyright: reportMissingImports=false
"""Shared Miles launcher for every recipe (search_r1 / alfworld / webshop / scienceworld / tmax).

Two layers:
- The launch platform, shared by EVERY recipe: ``LaunchArgs`` (model, paths,
  model-family plumbing), the checkpoint conversion (``prepare``), the stale
  process ``cleanup``, checkpoints with resume and the preemption sentinel
  (``checkpoint_args``), the session server, router and agent hooks
  (``session_args``), the engines' parser (``sglang_args``), the actors'
  environment (``build_extra_env_vars``) and the submission (``launch``).
  A fix for a miles or image change lands here once, for all recipes.
- The science config the three env campaigns share (optimizer, clip,
  KL, topology arms): ``BaseScriptArgs`` and ``build_train_args``. Each env
  recipe's run.py is a thin module: a ``ScriptArgs(BaseScriptArgs)`` carrying
  its overrides (data paths, batch shape, context budget), a ``RecipeSpec``
  (eval slices, dynamic-sampling filters, env plumbing) and the
  ``cleanup``/``prepare``/``execute``/``main`` entry points that delegate here.
  tmax keeps TMax's own science (``ScriptArgs(LaunchArgs)``) on the same
  platform.

The provenance notes on every non-obvious flag live here once; a recipe spells
out only what genuinely differs.

Topologies (--topology) - the first three are the env campaigns' arms, the two
teacher topologies the R2S paper's OPD arms (algo opd):
- colocated_1n: 1 node x 4 GPUs, synchronous, engines colocated (train.py).
  Matches the nemo-rl baseline's synchronous arm.
- async_3n: 3 nodes = 1 train (4 GPUs) + 2 rollout (8 GPUs), train_async.py.
  Miles/slime drain semantics: in-flight generation syncs BEFORE each weight
  push, every trajectory is version-consistent; staleness dose via
  --update-weights-interval (2 -> age in {1,2}, mean 1.5 - the slime arm that
  matched sync quality).
- fully_async_3n: same node split, train_async.py + --fully-async. A
  persistent worker generates continuously while the trainer drains completed
  groups; weight updates retract in-flight requests and recompute KV on the
  new weights (token-level version mixing - the nemo-rl in-flight analog).
  Staleness bound via --max-weight-staleness.
- async_4n_teacher: async_3n plus a fourth node that serves the OPD teacher
  with sglang outside ray (the job starts it and passes its base URL as
  --teacher-url); the ray-side split is async_3n's. The paper's R2S cohort.
- async_2n_teacher: one ray node split 2 actor + 2 rollout GPUs
  (train_async.py, the same update ordering) plus the teacher node; the
  paper's vanilla and anchor cohort.
  With an already serving teacher (the job's TEACHER_URL) neither has a
  teacher node: the job takes one node fewer (-N 3 / -N 1), every node is
  ray's and the ray-side split is the same; the gate passes --teacher-url
  either way, so the launcher never tells the two apart.
The GPU and node counts behind these figures are --num-gpus-per-node and
--num-nodes (the run's ray nodes; 0 = the topology's own count): the strings
_topology_args builds scale with them, the figures above are their defaults.

Import contract:
- run.py imports this as ``miles_agent.launcher`` and therefore puts the repo
  root on sys.path itself (the training entry runs ``python3 run.py`` from the
  recipe dir with no PYTHONPATH).
- ``execute_train`` (this module, the one call into miles' launch API) is
  resolved at call time through the module attribute:
  recipes/search_r1/eval_full.py wraps it to rewrite the command.

Command line (``python3 -m miles_agent.launcher``, ``main``): a checkpoint dir's
family facts for the serving jobs - its tool-call and reasoning parsers, its
served template (``write_serve_template``) - and the OPD teacher gate's
tokenizer probe. argparse is imported there only.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import miles.utils.external_utils.command_utils as U
from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

# This repo's root (NOT U.repo_base_dir - that is the miles checkout): makes
# the miles_agent package importable by dotted hook paths.
REPO_ROOT = Path(__file__).resolve().parent.parent


def data_path(name: str) -> str:
    """A data file's default location: under MILES_AGENT_DATA, else <repo>/data. The recipes' --prompt-data /
    --eval-data defaults are built from it when the args are instantiated (the entry always passes the prepared
    rows explicitly; a bare `python run.py` sees these)."""
    return os.path.join(os.environ.get("MILES_AGENT_DATA") or str(REPO_ROOT / "data"), name)


def results_dir(recipe: str) -> str:
    """A recipe's default --save-dir: <MILES_AGENT_RESULTS, else <repo>/results>/<recipe>/ (the trailing slash
    the save dirs always had)."""
    return os.path.join(os.environ.get("MILES_AGENT_RESULTS") or str(REPO_ROOT / "results"), recipe, "")


# Dynamic-sampling filters (dotted paths for --dynamic-sampling-filter-path).
# The stock filter drops groups with an ABORTED sample; the wrappers in
# miles_agent/dapo_filter.py add None-reward safety, the DAPO zero-reward-std
# drop, and the transport-failure drop. Rationale for the latter: engine /
# adapter / gym failures are infrastructure, not env feedback - a sample that
# never received a verdict must not train as reward 0, so its group is dropped
# and re-sampled. Env feedback (invalid actions, low scores) is always kept.
STOCK_NO_ABORTED_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"
NO_ABORTED_NONZERO_STD_FILTER = "miles_agent.dapo_filter.check_no_aborted_nonzero_std"
NO_TRANSPORT_FAILURE_FILTER = "miles_agent.dapo_filter.check_no_aborted_no_transport_failure"
NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER = "miles_agent.dapo_filter.check_no_aborted_no_transport_failure_nonzero_std"

# Generate functions (dotted paths for --custom-generate-function-path): miles'
# own agentic episode through NeMo Gym, and the OPD wrapper around it that
# scores each training sample at the teacher (Sample.teacher_log_probs) and
# appends the pilot dump rows (miles_agent/opd/generate.py; the opd arm and
# --opd-dump, build_train_args).
STOCK_AGENTIC_GENERATE_FN = "miles.rollout.generate_hub.agentic_tool_call.generate"
OPD_GENERATE_FN = "miles_agent.opd.generate.generate"


@dataclass(frozen=True)
class ModelFamily:
    """What miles needs to know about a model beyond its checkpoint."""

    # miles' TITO tokenizer (--tito-model): selects the fixed chat template and,
    # through the TITO class, the engines' tool-call parser (tool_call_parser).
    tito_model: str
    # JSON for --apply-chat-template-kwargs; "" = none.
    chat_template_kwargs: str
    # scripts/models/<type>.py of the miles pin; "" = from the model name by
    # Qwen's naming (_megatron_model_type).
    megatron_model_type: str = ""
    # How Megatron reads the HF checkpoint (--megatron-to-hf-mode): "raw" from
    # the torch_dist twin prepare() converts; "bridge" through Megatron-Bridge
    # from the HF dir itself (no twin: the HF dir is --ref-load).
    megatron_to_hf_mode: str = "raw"
    # Every checkpoint of the family is a mixture of experts (is_moe also reads
    # Qwen's "-A3B" marker): the trainer spreads the experts over a node's GPUs
    # (expert_parallel) and replays the engines' routing in its forward
    # (--use-rollout-routing-replay: without it a sigmoid-routed MoE's train and
    # rollout log-probs differ by ~0.28, miles docs/models/nemotron).
    moe: bool = False
    # Megatron's --attention-backend for the trainer.
    attention_backend: str = "flash"
    # The engines' MoE kernels (--sglang-moe-runner-backend); "" = sglang's pick. The routing replay needs kernels that
    # route through sglang's top-k, where it records each token's experts: sglang's pick for Nemotron-H on the engines used,
    # flashinfer_trtllm, routes inside its kernel and returned all-zero routing (seen on a smoke run).
    moe_runner_backend: str = ""
    # Per-family process env for the driver and the actors, the job env winning (_apply_family_env, from
    # prepare() and launch()): such as the 2507 checkpoint's rope theta, which the pin's scripts/models file
    # reads from MODEL_ARGS_ROTARY_BASE.
    env: dict[str, str] = field(default_factory=dict)


# By model name: the longest key prefixing it, else the one key found anywhere in
# it (_listed_family). No entry spells out a tool-call markup: the policy model's
# chat template renders it (the recipes' prompt examples,
# miles_agent/tool_call_markup.py) and the parser miles binds to the TITO
# family reads it back, so a new family is one entry here.
MODEL_FAMILIES = {
    # Thinking off at the template: the per-turn budget holds no reasoning.
    "Qwen3.5-": ModelFamily(tito_model="qwen35", chat_template_kwargs='{"enable_thinking": false}'),
    # Same contract as Qwen3.5 under TITO's qwen36 template (qwen3_coder markup); listed
    # from the pin's registry, not yet run here.
    "Qwen3.6-": ModelFamily(tito_model="qwen36", chat_template_kwargs='{"enable_thinking": false}'),
    # The template's own default: Qwen3's hybrid models think unless told otherwise, the
    # 2507 Instruct models cannot. Under the per-turn budget pass --chat-template-kwargs
    # '{"enable_thinking": false}' for a hybrid one, as the R2S evaluations did.
    "Qwen3-": ModelFamily(tito_model="qwen3", chat_template_kwargs=""),
    # The R2S paper's teacher of the Qwen3-4B/8B students. The pin has no scripts/models file of the
    # checkpoint's name: qwen3-30B-A3B.py is its architecture, with the rope theta defaulting to 1e6 and read
    # from MODEL_ARGS_ROTARY_BASE (its line 11), while the Instruct-2507 checkpoint's config.json has 1e7.
    "Qwen3-30B-A3B-Instruct-2507": ModelFamily(
        tito_model="qwen3",
        chat_template_kwargs="",
        megatron_model_type="qwen3-30B-A3B",
        moe=True,
        env={"MODEL_ARGS_ROTARY_BASE": "10000000"},
    ),
    # Mamba-2 + MoE (128 experts, top 6, sigmoid routing) + attention, with
    # Nemotron-3-Nano-30B-A3B's dimensions and one MTP layer, which the trainer
    # leaves out (MILES_NEMOTRONH_KEEP_MTP unset) and the engines may draft with.
    # Its Mamba layers pick their own kernels (attention backend auto).
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B": ModelFamily(
        tito_model="nemotron3",
        chat_template_kwargs='{"enable_thinking": false}',
        megatron_model_type="nemotron-3-nano-30b-a3b",
        megatron_to_hf_mode="bridge",
        moe=True,
        attention_backend="auto",
        moe_runner_backend="flashinfer_cutlass",
    ),
}


def _listed_family(model_name: str) -> ModelFamily | None:
    """The row of the longest key prefixing the name ("Qwen3-30B-A3B-Instruct-2507" over "Qwen3-"); else, for a
    fine-tune's directory such as "r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b", the row of the unique longest key
    found anywhere in the name, case-insensitively; None when no key matches."""
    prefixes = [key for key in MODEL_FAMILIES if model_name.startswith(key)]
    if prefixes:
        return MODEL_FAMILIES[max(prefixes, key=len)]
    lowered = model_name.lower()
    found = sorted((key for key in MODEL_FAMILIES if key.lower() in lowered), key=len, reverse=True)
    if not found:
        return None
    if len(found) > 1 and len(found[0]) == len(found[1]):
        raise ValueError(f"{model_name!r} names more than one model family: {found[0]!r} and {found[1]!r}")
    return MODEL_FAMILIES[found[0]]


# Qwen names a mixture of experts by its active parameters ("Qwen3-30B-A3B",
# "Qwen3.5-35B-A3B"), and the pin's scripts/models follow that naming, so the
# Qwen rows above need not list each MoE checkpoint.
_ACTIVE_PARAMS = re.compile(r"-A\d+B(?=-|$)")


def is_moe(model_name: str) -> bool:
    """A mixture of experts: the family says so (ModelFamily.moe), or the name does by Qwen's naming. The trainer
    then needs expert parallelism and the engines' routing (parallel_args); an unlisted name without the marker is
    dense."""
    family = _listed_family(model_name)
    return (family is not None and family.moe) or _ACTIVE_PARAMS.search(model_name) is not None


def model_family(model_name: str) -> ModelFamily:
    family = _listed_family(model_name)
    if family is None:
        raise ValueError(
            f"no MODEL_FAMILIES entry for {model_name!r}: add one, or pass --tito-model and --chat-template-kwargs"
        )
    return family


def _megatron_model_type(model_name: str) -> str:
    """scripts/models/<type>.py: the family's, else by miles' naming: "Qwen3.5-4B" -> "qwen3.5-4B"."""
    family = _listed_family(model_name)
    if family is not None and family.megatron_model_type:
        return family.megatron_model_type
    if not model_name.startswith("Qwen"):
        raise ValueError(f"cannot derive the megatron model type of {model_name!r}: pass --megatron-model-type")
    return "qwen" + model_name[len("Qwen") :]


def _check_model_script(model_type: str, model_name: str) -> None:
    """Fail at the flags, not inside miles' conversion after the allocation, when the pin has no
    scripts/models/<type>.py: a fine-tune's directory name ("Qwen3.5-4B-sft") derives one that does not exist."""
    try:
        from miles.utils.external_utils import model_args_utils
    except ImportError:  # the pin's tree is not importable here (the tests stub miles)
        return
    script_dir = getattr(model_args_utils, "_MODEL_SCRIPT_DIR", None)
    if script_dir is None or (Path(script_dir) / f"{model_type}.py").exists():
        return
    raise ValueError(
        f"the miles pin has no scripts/models/{model_type}.py (from --model-name {model_name!r}): pass "
        "--megatron-model-type <the checkpoint's architecture script>, e.g. qwen3.5-4B for a Qwen3.5-4B fine-tune"
    )


@dataclass
class LaunchArgs(U.ExecuteTrainConfig):
    """The fields the shared launch platform reads, for every recipe.

    Recipes subclass this (the env recipes through BaseScriptArgs) and
    re-declare the fields that differ: dataclass inheritance keeps the field
    order and lets a subclass override a default, and U.dataclass_cli reads
    dataclasses.fields()/__init__ of the concrete class, so every field here
    becomes a --flag of each recipe.
    """

    mode: Literal["normal", "debug_rollout_only"] = "normal"
    # Each recipe re-declares this with the topologies it implements.
    topology: str = "colocated_1n"
    # scripts/models/<type>.py in the miles pin; "auto" = from model_name.
    megatron_model_type: str = "auto"
    num_gpus_per_node: int = 4
    # Ray nodes of the run; 0 = the topology's own count: colocated_1n 1, async_3n /
    # fully_async_3n / async_4n_teacher 3 (the teacher node is outside ray),
    # async_2n_teacher 1. Redeclared over the inherited field, whose default is the
    # scheduler's allocation size: a *_teacher job's ray cluster is one node smaller
    # than its allocation. Launcher-internal - it shapes the topology strings
    # (_topology_args) and is never emitted into the train args (train.py has no
    # such flag).
    num_nodes: int = 0
    # The Megatron-LM checkout the actors import (the PYTHONPATH head, the conversion):
    # MEGATRON_PATH in the environment overrides the image path, the flag wins over both.
    megatron_path: str = field(default_factory=lambda: os.environ.get("MEGATRON_PATH", "/root/Megatron-LM"))

    # Paths - the training entry (scripts/run_local.sh)
    # pass --base-dir/--model-name/--hf-checkpoint/--ref-load/--save-dir
    # explicitly for the model they stage.
    skip_prepare: bool = False
    base_dir: str = "/root"
    model_name: str = "Qwen3.5-4B"
    hf_checkpoint: str = "Qwen/Qwen3.5-4B"
    ref_load: str = "/root/Qwen3.5-4B_torch_dist"
    save_dir: str = ""  # recipe-specific, required
    # Model family plumbing: "auto" = model_name's MODEL_FAMILIES entry. The
    # engines' tool-call parser always follows tito_model (tool_call_parser).
    tito_model: str = "auto"
    chat_template_kwargs: str = "auto"
    # raw | bridge (ModelFamily.megatron_to_hf_mode); bridge ignores --ref-load
    # and --base-dir: the reference is the HF checkpoint, nothing is converted.
    megatron_to_hf_mode: str = "auto"
    # The trainer's expert parallelism; 0 = a MoE family's experts over one
    # node's GPUs (num_gpus_per_node), 1 for a dense family.
    expert_parallel: int = 0
    # Megatron's --attention-backend; "" = the family's.
    megatron_attention_backend: str = ""
    # The engines' --sglang-moe-runner-backend; "" = the family's (ModelFamily.moe_runner_backend), "auto" = sglang's.
    sglang_moe_runner_backend: str = ""

    save_interval: int = 50
    # File whose appearance makes miles save the last trained step and stop
    # (--preempt-sentinel); a scheduler wrapper raises it on the preemption and
    # time-limit signals (docs/cluster.md, "C. preemption / time limit"). "" = off.
    preempt_sentinel: str = ""
    # The R2S pilot dump: OPD_GENERATE_FN (installed whatever the algo) appends
    # every training episode, in token space, to <save_dir>/opd_dump, which the
    # actors learn as MILES_OPD_DUMP_DIR - the one carrier, read by the hook and
    # by gym_bridge for the transcript. Under --mode debug_rollout_only the run
    # neither evaluates nor saves (_pilot_dump).
    opd_dump: bool = False

    # Raw miles flags appended verbatim to the train command (after every
    # block, so they win arg-parse precedence). For one-off experiments;
    # promote anything load-bearing into an explicit field.
    extra_train_args: str = ""

    # NeMo Gym settings
    # The agent server(s): NEMO_GYM_URL, else the local agent port scripts/start_gym.sh serves (NEMO_GYM_PORT).
    nemo_gym_url: str = field(default_factory=lambda: os.environ.get("NEMO_GYM_URL", "http://127.0.0.1:8200"))
    # Host the NeMo Gym agents reach the session servers on, when they cannot
    # use the placed node address (--session-server-external-host; it also
    # keeps every session server on the head node). Empty = placed address.
    session_server_external_host: str = ""

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        # Explicit flags win; "auto" follows model_name.
        if self.megatron_model_type == "auto":
            self.megatron_model_type = _megatron_model_type(self.model_name)
        _check_model_script(self.megatron_model_type, self.model_name)
        if "auto" in (self.tito_model, self.chat_template_kwargs):
            family = model_family(self.model_name)
            if self.tito_model == "auto":
                self.tito_model = family.tito_model
            if self.chat_template_kwargs == "auto":
                self.chat_template_kwargs = family.chat_template_kwargs
        # The trainer's side: an unlisted model has ModelFamily's defaults (dense, from its torch_dist twin).
        family = _listed_family(self.model_name) or ModelFamily(tito_model="", chat_template_kwargs="")
        if self.megatron_to_hf_mode == "auto":
            self.megatron_to_hf_mode = family.megatron_to_hf_mode
        if self.megatron_to_hf_mode not in ("raw", "bridge"):
            raise ValueError(f"--megatron-to-hf-mode {self.megatron_to_hf_mode!r}: raw or bridge")
        if not self.expert_parallel:
            self.expert_parallel = self.num_gpus_per_node if is_moe(self.model_name) else 1
        if not self.megatron_attention_backend:
            self.megatron_attention_backend = family.attention_backend
        if not self.sglang_moe_runner_backend:
            self.sglang_moe_runner_backend = family.moe_runner_backend

    @property
    def moe(self) -> bool:
        """A mixture of experts (is_moe: the family's flag or Qwen's "-A3B" marker)."""
        return is_moe(self.model_name)


@dataclass
class BaseScriptArgs(LaunchArgs):
    """The env recipes' fields, with the campaign-wide defaults (see LaunchArgs
    for how a recipe overrides them). Fields with an empty default have no
    campaign-wide value and MUST be overridden.
    """

    topology: Literal[
        "colocated_1n", "async_3n", "fully_async_3n", "async_4n_teacher", "async_2n_teacher"
    ] = "colocated_1n"
    prompt_data: str = ""  # recipe-specific, required
    eval_data: str = ""  # recipe-specific, required

    # Aligned science config - shared verbatim by the nemo-rl and slime
    # baselines; see the search_r1 recipe docstring for provenance.
    num_rollout: int = 500
    rollout_batch_size: int = 64
    n_samples_per_prompt: int = 8
    global_batch_size: int = 512  # = full batch: exactly one update/step (mu=1)
    rollout_temperature: float = 1.0
    # Validation sampling temperature (1 gen/prompt). 1.0 = same policy as
    # training; verl-agent and TCOD validate at 0.4 (the webshop recipe pins it).
    eval_temperature: float = 1.0
    rollout_max_response_len: int = 512  # per-turn generation cap
    # Validation per-turn generation cap; 0 = same as rollout_max_response_len.
    # TCOD evaluates with 4096 max tokens against 512 in training (Table 5).
    eval_max_response_len: int = 0
    # Global RNG seed (--seed: trainer + rollout). miles' default is 1234; the
    # TCOD protocol runs use 42 (the webshop recipe pins it).
    seed: int = 1234
    # The trainer's packing budget by default (--max-tokens-per-gpu, below) and
    # miles' --rollout-max-context-len, which the session-server path the env
    # recipes use does not read: an episode is bounded by the agent's turn cap
    # times the per-turn cap, and a request past the engine's context comes
    # back as a 400 (env/policy_400_truncated). Multi-turn envs train on the
    # FULL transcript (one contiguous sample per episode by prefix extension),
    # so the budget must hold every observation + admissible list + reply of
    # the longest episode.
    rollout_max_context_len: int = 32768
    # Dynamic-batch packing budget for train/logprob passes
    # (--max-tokens-per-gpu). 0 = follow rollout_max_context_len: one
    # multi-turn sample cannot split across GPUs with CP=1, so the budget can
    # never usefully sit below the per-sample context. Recipes pin an explicit
    # value where it must stay put while the context is tuned.
    train_max_tokens_per_gpu: int = 0
    eval_interval: int = 50
    # Adam at a constant lr (_optimizer_args): the campaign's values; the OPD
    # presets pin the paper's (1e-6, no warmup, wd 0.1).
    lr: float = 1e-6
    lr_warmup_iters: int = 142
    weight_decay: float = 0.01

    # async_3n only: weights pushed to the rollout engines every N train steps
    # (trained data is 1..N versions old; N=2 -> mean age 1.5, N=3 -> mean 2.0;
    # both slime arms tracked the sync curve)
    update_weights_interval: int = 2
    # fully_async_3n only: max gap between a group's oldest weight version and
    # the current engine version; older groups are dropped (or retried). 2
    # mirrors the nemo-rl in-flight stress arm's max_trajectory_age_steps=2.
    max_weight_staleness: int = 2

    # Off-policy correction arms (fully-async staleness study). Mutually
    # exclusive - miles asserts this in miles/utils/arguments.py. Default off = the
    # uncorrected baseline the first fa3n campaign ran: old_log_probs are
    # recomputed by the CURRENT actor after the pre-train weight push, so the
    # PPO ratio is identically 1 and clip is inert on stale data.
    # - use_rollout_logprobs: PPO denominator = engine per-token logprobs
    #   captured at decode time (true behavior policy, correct even for
    #   retract-mixed requests).
    # - use_tis: keep the recompute denominator, multiply a clamped
    #   exp(train - rollout) weight into pg_loss (tis_clip default 2.0 - same
    #   cap as the nemo-rl fx3 arm).
    use_rollout_logprobs: bool = False
    use_tis: bool = False

    # Algorithm arm. grpo = campaign baseline. gspo follows the repo's GSPO
    # recipes (sequence-level clip 4e-4, KL off) because token-scale clip
    # values never trigger on a sequence ratio. ppo adds the critic (mirrors
    # actor topology; both init from --ref-load on fresh runs) and, on async
    # drivers, the mandatory behavior-policy denominator via --keep-old-actor.
    # dapo = grpo + clip-higher (already in the baseline) + zero-reward-std
    # group filtering with refill (true DAPO dynamic sampling).
    # opd = the R2S paper's student arm on miles' native sampled on-policy
    # distillation: the train reward zeroed (build_extra_env_vars), the teacher
    # KL the whole objective (_algo_args), no reference KL, no clip-higher.
    algo: Literal["grpo", "gspo", "ppo", "dapo", "opd"] = "grpo"

    # opd only. teacher_url: the base URL of the teacher's sglang server
    # (http://host:13141), passed by the job's teacher gate as --teacher-url;
    # _algo_args appends /generate for miles' --opd-teacher-urls. opd_kl_coef:
    # beta, the per-token weight of the sampled reverse KL (the paper's 1).
    teacher_url: str = ""
    opd_kl_coef: float = 1.0


@dataclass(frozen=True)
class RecipeSpec:
    """What a recipe contributes beyond its ScriptArgs defaults."""

    # Recipe dir name; also the wandb run-name fallback (f"{name}_{topology}").
    name: str
    # The recipe's run.py (__file__): its dir goes on the actors' PYTHONPATH
    # (bare recipe modules such as reward.shape) and its path names the wandb
    # project (miles-<recipe>_run via U.get_default_wandb_args).
    run_file: str
    # --eval-prompt-data entries as (dataset name, "@[start:end]" or "").
    # Slice offsets assume prepare_data preserved the block ordering of the
    # eval file (documented per recipe). Env recipes only.
    eval_slices: tuple[tuple[str, str], ...] = ()
    # REWARD_SHAPING_PATH: the recipe's shaping terms (recipes/<name>/reward.py);
    # None = no shaping module (train == env).
    reward_shaping_path: str | None = "reward.shape"
    # --dynamic-sampling-filter-path for grpo/gspo/ppo and for dapo.
    filter_path: str = STOCK_NO_ABORTED_FILTER
    dapo_filter_path: str = NO_ABORTED_NONZERO_STD_FILTER
    # NEMO_GYM_RUN_TIMEOUT (seconds) for one /run; None keeps gym_bridge's
    # 600 s default, which multi-turn envs exceed under saturation.
    run_timeout_s: int | None = None
    # Env-var prefixes forwarded verbatim from the launcher's environment into
    # the ray runtime env (e.g. ("WEBSHOP_",) for reward knobs the recipe's
    # reward.py reads at import time). The ray runtime env is the only
    # guaranteed path from the job environment to the actors.
    env_passthrough_prefixes: tuple[str, ...] = ()

    @property
    def recipe_dir(self) -> Path:
        return Path(self.run_file).resolve().parent


# pgrep arguments for cleanup(). "sglang" matches process names only: the
# v0.1.2 image installs its Python under /opt/sglang, so a command-line match
# also hits this job's external Ray head (gcs_server, raylet and the dashboard
# that `ray job submit` dials all run from /opt/sglang/...).
CLEANUP_PGREP = ("sglang", "-f train.py", "-f train_async.py", "-f MegatronTrain")


def cleanup():
    """Kill stale sglang servers and miles drivers / train actors to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for match in CLEANUP_PGREP:
        subprocess.run(
            f"pgrep {match} | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)


def _apply_family_env(family: ModelFamily | None, env_vars: dict[str, str] | None = None) -> None:
    """ModelFamily.env into this process, where miles evaluates the family's scripts/models file (the conversion
    of prepare(), the model args of launch()), and into the actors' ``env_vars`` when given; a value the job env
    set wins in both, so the conversion and the training read the same model args."""
    for key, value in (family.env if family is not None else {}).items():
        os.environ.setdefault(key, value)
        if env_vars is not None:
            env_vars.setdefault(key, os.environ[key])


def prepare(args: LaunchArgs):
    """Convert the HF checkpoint to torch_dist format if not already done.

    miles >= 0.1.2 redoes a conversion IN PLACE (under a file lock) when its
    miles_conversion_version.txt stamp is missing or older, i.e. the first
    launch after a bump rewrites a shared <base_dir>/<model>_torch_dist that
    older-image jobs may be loading - point --base-dir elsewhere to avoid it.
    A bridge-mode model has no twin: Megatron-Bridge reads the HF checkpoint.
    """
    # the conversion evaluates scripts/models/<type>.py in this process (base_backend.convert_checkpoint)
    _apply_family_env(_listed_family(args.model_name))
    if args.megatron_to_hf_mode == "bridge":
        return
    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.base_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def train_script(args: LaunchArgs) -> str:
    return "train.py" if args.topology == "colocated_1n" else "train_async.py"


# A pilot's eval and save intervals: never within its steps (the campaign branch's submit script).
_PILOT_INTERVAL = 1000000


def _pilot_dump(args: LaunchArgs) -> bool:
    """The R2S pilot: --opd-dump under --mode debug_rollout_only rolls the base policy out at the arm's caps
    and only appends dump rows, so it skips the eval before training and every periodic eval / save within its
    steps, whatever the preset's intervals; the eval after the last step is miles' own (train.py:201-221 forces
    it at num_rollout - 1) and stays, its save a no-op under --debug-rollout-only."""
    return args.opd_dump and args.mode == "debug_rollout_only"


def checkpoint_args(args: LaunchArgs, *, critic: bool = False, skip_eval_on_resume: bool = False) -> str:
    """Checkpoints, resume and the preemption sentinel. ``critic``: a PPO run, whose
    critic saves to <save_dir>_critic. ``skip_eval_on_resume``: a resumed link skips
    the eval before training (the recipe evaluates on its interval only)."""
    if args.megatron_to_hf_mode == "bridge":
        # the reference (and a fresh run's start) is the HF checkpoint itself; the
        # saved checkpoints are torch_dist all the same, and --load resumes them
        ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint} --megatron-to-hf-mode bridge "
    else:
        ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} "
    # Megatron asserts a --save-interval whenever --save is given, and the
    # sentinel saves: save_interval 0 = no checkpoints at all.
    save_interval = _PILOT_INTERVAL if _pilot_dump(args) else args.save_interval
    if save_interval > 0:
        ckpt_args += f"--save {args.save_dir} --save-interval {save_interval} "
    # Resume after a requeue: fresh-start semantics retrain from zero
    # AND overwrite the prior checkpoints (a run lost 270 steps to a
    # preemption). --load only when a tracker exists (load_checkpoint asserts
    # the dir is non-empty); PPO additionally requires --critic-load, since
    # critic_load defaults to args.load, which would put actor weights into
    # the critic. PPO with an actor tracker but no critic tracker (partial
    # save at the kill) starts fresh rather than resuming a mismatched pair.
    critic_dir = args.save_dir.rstrip("/") + "_critic"
    has_tracker = os.path.exists(os.path.join(args.save_dir, "latest_checkpointed_iteration.txt"))
    has_critic_tracker = os.path.exists(os.path.join(critic_dir, "latest_checkpointed_iteration.txt"))
    if has_tracker and (not critic or has_critic_tracker):
        ckpt_args += f"--load {args.save_dir} "
        if critic:
            ckpt_args += f"--critic-load {critic_dir} "
        if skip_eval_on_resume:
            ckpt_args += "--skip-eval-before-train "
    if args.preempt_sentinel and save_interval > 0:
        ckpt_args += f"--preempt-sentinel {args.preempt_sentinel} "
    elif args.preempt_sentinel:
        # the wrapper still raises it on a preemption or a lost env server and waits for the save
        print(
            "WARN: --preempt-sentinel dropped: save_interval 0 saves nothing, so a preemption or env-loss"
            " signal cannot stop this run at a checkpoint (it ends with the allocation)",
            file=sys.stderr,
        )
    return ckpt_args


def _rollout_args(args: BaseScriptArgs) -> str:
    return (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-temperature {args.rollout_temperature} "
        "--rollout-top-p 1.0 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--seed {args.seed} "
    )


def _eval_args(args: BaseScriptArgs, spec: RecipeSpec) -> str:
    # 1 gen/prompt at args.eval_temperature on the recipe's fixed validation file.
    slices = "".join(f"{name} {args.eval_data}{sl} " for name, sl in spec.eval_slices)
    cap = f"--eval-max-response-len {args.eval_max_response_len} " if args.eval_max_response_len > 0 else ""
    eval_interval = _PILOT_INTERVAL if _pilot_dump(args) else args.eval_interval
    # the pilot skips the step-0 eval as well (train.py:134; the last step's eval is forced regardless)
    skip_first = "--skip-eval-before-train " if _pilot_dump(args) else ""
    return (
        f"--eval-interval {eval_interval} "
        f"{skip_first}"
        "--eval-prompt-data "
        f"{slices}"
        "--n-samples-per-eval-prompt 1 "
        f"--eval-temperature {args.eval_temperature} "
        "--eval-top-p 1.0 "
        f"{cap}"
    )


def _algo_args(args: BaseScriptArgs) -> str:
    if args.algo == "opd":
        # The paper's student arm on miles' native sampled OPD. With the train
        # reward zeroed (build_extra_env_vars) the grpo estimator gives
        # advantage 0 and apply_opd_kl_to_advantages (miles/backends/
        # training_utils/loss/hub/opd.py) leaves -beta * (old student log-prob
        # - teacher log-prob) per token: the sampled reverse KL inside the PPO
        # surrogate, clip 0.2 inert at ratio 1; train/opd_reverse_kl is miles'
        # own metric. No reference KL, no entropy term, no clip-higher.
        # --opd-type sglang names where the teacher log-probs come from; the
        # scoring is OPD_GENERATE_FN's (one prefill per training sample into
        # Sample.teacher_log_probs), not miles' RM-hook path, so the reward slot
        # stays reward_routing's and there is no --rm-url.
        teacher = (
            f"--opd-teacher-urls default={args.teacher_url.rstrip('/')}/generate " if args.teacher_url else ""
        )
        return (
            "--advantage-estimator grpo "
            "--entropy-coef 0.0 "
            "--eps-clip 0.2 "
            "--use-opd "
            "--opd-type sglang "
            f"--opd-kl-coef {args.opd_kl_coef} "
            "--opd-log-prob-top-k 0 "
            f"{teacher}"
        )
    if args.algo == "gspo":
        # Canonical GSPO per the repo's Qwen3 recipes: the sequence-level
        # geometric-mean ratio needs a sequence-scale clip (4e-4; 0.2-style
        # token values disable clipping entirely, and nothing asserts) and
        # every in-repo GSPO recipe keeps KL off. eps-clip-high defaults to
        # eps-clip, i.e. symmetric.
        return "--advantage-estimator gspo --eps-clip 4e-4 --entropy-coef 0.0 "
    if args.algo == "ppo":
        # Critic mirrors the actor's placement/parallelism (miles overwrites
        # critic_num_*); fresh runs init actor AND critic from --ref-load, the
        # value head starts random (hence 1 critic-only warmup step) and the
        # critic lr follows the repo PPO recipes (10x actor). Async drivers
        # hard-require a behavior-policy denominator for PPO; --keep-old-actor
        # recomputes old_log_probs under rollout-time weights (TIS and engine
        # logprobs were rejected by the day-2 staleness study). On resume,
        # additionally pass --critic-load <save>_critic: miles defaults
        # critic_load to the ACTOR checkpoint dir.
        algo_args = (
            "--advantage-estimator ppo "
            "--use-kl-loss "
            "--kl-loss-coef 0.001 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.0 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
            "--value-clip 0.2 "
            "--critic-lr 1e-5 "
            "--num-critic-only-steps 1 "
        )
        # Async drivers hard-require a behavior-policy correction with a
        # critic (validate_async_off_policy_correction) - the fa3n_ppo_r1
        # post-mortem confirmed why: uncorrected ratio==1 collapses. PPO
        # arms must also pass --extra-train-args=--no-offload-train
        # (offload x distributed weight push = CUDA illegal memory access; patches/miles/README.md).
        if args.topology != "colocated_1n":
            algo_args += "--keep-old-actor "
        return algo_args
    # grpo baseline; dapo = same loss, filter swapped in _agent_args
    return (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.001 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        # Deliberate divergence from the 0.2-symmetric baselines: DAPO-style
        # clip-higher.
        "--eps-clip-high 0.28 "
    )


def _num(value: float) -> str:
    """The shortest spelling of a float without a zero-padded exponent ("1e-6", not Python's "1e-06"): the
    optimizer flags read as the constants they were."""
    return format(value, "g").replace("e-0", "e-").replace("e+0", "e+")


def _optimizer_args(args: BaseScriptArgs) -> str:
    """Adam at a constant lr. The OPD arm adds Megatron's --use-checkpoint-opt-param-scheduler (the campaign
    branch's submit script): on a requeued link the checkpoint's schedule state wins over the flags."""
    scheduler = "--use-checkpoint-opt-param-scheduler " if args.algo == "opd" else ""
    return (
        "--optimizer adam "
        f"--lr {_num(args.lr)} "
        "--lr-decay-style constant "
        # constant style never reads the value; it only satisfies megatron's
        # lr_warmup_steps < lr_decay_steps assert (fails when steps < 142)
        "--lr-decay-iters 10000 "
        f"--lr-warmup-iters {args.lr_warmup_iters} "
        "--lr-warmup-init 0.0 "
        f"--weight-decay {_num(args.weight_decay)} "
        "--adam-beta1 0.9 "
        # slime baseline value; the nemo-rl baseline ran 0.999 - the one knob
        # the two proven baselines themselves disagree on
        "--adam-beta2 0.98 "
        f"{scheduler}"
    )


def parallel_args(args: LaunchArgs) -> str:
    """The trainer's parallelism: TP, PP and CP 1 (each family's activations fit a GPU at the recipes' contexts);
    a MoE family's experts over expert_parallel GPUs, exchanged all-to-all, and its routing replayed from the
    engines (is_moe)."""
    out = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.expert_parallel} "
        "--expert-tensor-parallel-size 1 "
    )
    if args.expert_parallel > 1:
        out += "--moe-token-dispatcher-type alltoall "
    if args.moe:
        # Requests stay in place across a weight update, so every token keeps the routing of the forward that gave
        # its log-prob, and a session takes only each turn's new routing rows. Retracted requests are re-prefilled
        # with the new weights, which records new routing for the tokens they already generated.
        out += "--use-rollout-routing-replay --pause-generation-mode in_place "
    return out


def _perf_args(args: BaseScriptArgs) -> str:
    max_tokens_per_gpu = args.train_max_tokens_per_gpu or args.rollout_max_context_len
    return (
        f"{parallel_args(args)}"
        "--sequence-parallel "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
        # Real entropy in train/entropy_loss; detached (no gradient effect)
        # while entropy-coef stays 0.
        "--observe-training-entropy "
    )


def tool_call_parser(tito_model: str) -> str:
    """The parser for the family's markup: the one miles binds on its TITO class."""
    _, parser = resolve_reasoning_and_tool_call_parser(tito_model)
    if parser is None:
        raise ValueError(f"--tito-model {tito_model} binds no tool-call parser")
    return parser


def reasoning_parser(tito_model: str) -> str:
    """The parser for the family's think block, bound on the same TITO class (--reasoning-parser of a server
    that must hand the answer back without it)."""
    parser, _ = resolve_reasoning_and_tool_call_parser(tito_model)
    if parser is None:
        raise ValueError(f"--tito-model {tito_model} binds no reasoning parser")
    return parser


def write_serve_template(tito_model: str, out_path: str | Path, *, thinking: str = "off") -> str:
    """Write the TITO family's fixed chat template for a plain sglang server that renders as the training engines
    do (scripts/serve_model.sh) and return its tool-call parser. ``thinking`` "off" renders thinking off unless a
    request turns it on, as MODEL_FAMILIES' enable_thinking false does in training; "template" keeps the
    template's own default."""
    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, TITOTokenizerType

    cls = TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(tito_model))
    fixed = cls.FIXED_TEMPLATE
    if fixed.template is None or not cls.tool_call_parser:
        raise ValueError(f"FATAL: TITO family {tito_model} has no fixed template or tool-call parser")
    text = (TEMPLATE_DIR / fixed.template).read_text()
    if thinking == "off":
        # the request's enable_thinking still wins; unset now means false, as the launch default does in training
        old = "{%- if enable_thinking is defined and enable_thinking is false %}"
        new = "{%- if enable_thinking is not defined or enable_thinking is false %}"
        if text.count(old) != 1:
            raise ValueError(
                f"FATAL: {fixed.template} has {text.count(old)} copies of {old!r}, not 1: cannot default thinking off"
            )
        text = text.replace(old, new)
    elif thinking != "template":
        raise ValueError(f"FATAL: SERVE_THINKING={thinking}: off or template")
    Path(out_path).write_text(text)
    # rendered per request by the session server in training; a plain server renders without them
    print(
        f"SERVE_TEMPLATE {fixed.template} thinking={thinking} (TITO extra_kwargs {fixed.extra_kwargs} not applied)",
        file=sys.stderr,
    )
    return cls.tool_call_parser


# tp=1, dp maxed: the model fits a single GPU, so one engine per GPU.
# --sglang-tool-call-parser must be passed explicitly: miles binds one per TITO
# family, but only its test runner consumes that binding - the production
# engine launch path does not. Without it the engines return the call markup
# as text and the gym agent sees no structured tool_calls: env_loop_agent
# counts every step as an action_format_error, tool_call_agent ends every
# episode after one turn with no search and no <answer>.
def sglang_args(args: LaunchArgs, *, context_length: int = 0) -> str:
    """The rollout engines; ``context_length`` > 0 sets their context window."""
    context = f"--sglang-context-length {context_length} " if context_length > 0 else ""
    moe = f"--sglang-moe-runner-backend {args.sglang_moe_runner_backend} " if args.sglang_moe_runner_backend else ""
    return (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        f"{context}"
        f"{moe}"
        f"--sglang-tool-call-parser {tool_call_parser(args.tito_model)} "
    )


def session_args(args: LaunchArgs, *, filter_path: str, generate_fn: str = STOCK_AGENTIC_GENERATE_FN) -> str:
    """The agentic rollout through NeMo Gym: generate/agent/reward hooks, the
    group filter (--dynamic-sampling-filter-path), session servers, router, TITO.
    ``generate_fn``: the custom generate function (OPD_GENERATE_FN for the opd
    arm and the pilot dump, build_train_args)."""
    external_host = (
        f"--session-server-external-host {args.session_server_external_host} "
        if args.session_server_external_host
        else ""
    )
    template_kwargs = (
        f"--apply-chat-template-kwargs {shlex.quote(args.chat_template_kwargs)} " if args.chat_template_kwargs else ""
    )
    return (
        f"--custom-generate-function-path {generate_fn} "
        "--custom-agent-function-path miles_agent.gym_bridge.run "
        "--custom-rm-path miles_agent.reward_routing.reward_func "
        f"--dynamic-sampling-filter-path {filter_path} "
        "--use-session-server "
        # Bind address only: agents get the placed node's address (or
        # --session-server-external-host), so bind every interface.
        "--session-server-ip 0.0.0.0 "
        # sub-9000 quiet band: nodes with ip_local_port_range=9000-65000 (common on big boxes)
        # so any pinned listener >=9000 eventually loses a race with an
        # ephemeral socket (ray opens thousands of connections) and the
        # session-server port check dies. 7100-7131 sits below the ephemeral
        # floor, clear of the router (7200, below), the webshop engine (6200),
        # ray GCS (6379), and the dashboard (8265).
        "--session-server-port 7100 "
        "--session-server-workers 32 "
        f"{external_host}"
        # Same band, for the router: miles >= 0.1.2 allocates it dynamically
        # from 20000 (the old pin picked 3000-4000), i.e. inside the ephemeral
        # range. The prometheus port and the engines stay dynamic.
        "--sglang-router-port 7200 "
        # miles >= 0.1.2 defaults to subproc (one child process per episode);
        # inline keeps gym_bridge.run on the rollout event loop as before.
        "--custom-agent-function-mode inline "
        f"--tito-model {args.tito_model} "
        f"{template_kwargs}"
    )


# Reward routing (train/eval decoupling): miles_agent/reward_routing.py
# returns {"env": verify score, "train": shaped}. GRPO optimizes "train";
# every eval/* metric reads "env" - shaping can never leak into eval curves.
# The score's semantics (EM for wiki_search, terminal won for alfworld, raw
# task_score for webshop) are owned by the env alone; a recipe's shaping
# terms live in its reward.py (REWARD_SHAPING_PATH below).
REWARD_ARGS = "--reward-key train --eval-reward-key env "

def _disagg_args(gpus_per_node: int, nodes: int) -> str:
    """The disaggregated split over ``nodes`` ray nodes of ``gpus_per_node`` GPUs each: one train node, the rest
    rollout. Pin the RolloutManager (and its session servers) to the ray head node: NeMo Gym runs there, so every
    session URL its agents dial stays on that box."""
    return (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {gpus_per_node} "
        f"--rollout-num-gpus {(nodes - 1) * gpus_per_node} --pin-rollout-manager-to-head "
    )


# The 3-node arms at the campaign's defaults: 1 train (4 GPUs) + 2 rollout (8 GPUs) -
# the fastest split in the slime topology runs.
_DISAGG_ARGS = _disagg_args(4, 3)

# The ray node count a topology assumes when --num-nodes is 0.
_TOPOLOGY_NODES = {
    "colocated_1n": 1,
    "async_3n": 3,
    "fully_async_3n": 3,
    "async_4n_teacher": 3,
    "async_2n_teacher": 1,
}


def _topology_args(args: BaseScriptArgs) -> str:
    """The ray-side split from --num-gpus-per-node and --num-nodes (0 = the topology's own count); at the
    defaults every string is the campaign's byte for byte."""
    gpus = args.num_gpus_per_node
    nodes = args.num_nodes or _TOPOLOGY_NODES.get(args.topology, 3)
    if args.topology == "colocated_1n":
        return f"--colocate --actor-num-nodes 1 --actor-num-gpus-per-node {gpus} "
    if args.topology == "async_2n_teacher":
        # One ray node, distinct GPUs for the actor (DP2) and the engines (DP2);
        # the second node serves the teacher outside ray (none under TEACHER_URL,
        # -N 1). train_async.py with the same update ordering: the paper's
        # vanilla and anchor cohort (the paper branch's launcher.py:573, verbatim).
        if gpus < 2:
            raise ValueError(f"--num-gpus-per-node must be >= 2 for {args.topology} (got {gpus})")
        return (
            f"--actor-num-nodes 1 --actor-num-gpus-per-node {gpus // 2} --rollout-num-gpus {gpus - gpus // 2} "
            "--pin-rollout-manager-to-head "
            f"--update-weights-interval {args.update_weights_interval} "
        )
    if nodes < 2:
        raise ValueError(f"--num-nodes must be >= 2 for {args.topology} (got {nodes})")
    if args.topology in ("async_3n", "async_4n_teacher"):
        # async_4n_teacher: the teacher node never joins ray, so the ray-side
        # split is async_3n's: the job hands ray NNODES - 1 nodes, or every
        # node when TEACHER_URL names an already serving teacher (-N 3).
        return _disagg_args(gpus, nodes) + f"--update-weights-interval {args.update_weights_interval} "
    topology_args = _disagg_args(gpus, nodes) + f"--fully-async --max-weight-staleness {args.max_weight_staleness} "
    if args.algo == "dapo":
        # Filter drops shrink the kept-group rate to ~1/5; raise the
        # in-flight budget (default = rollout_batch_size) so the producer
        # keeps the engines saturated while refilling.
        topology_args += "--async-max-concurrent-samples 128 "
    return topology_args


def misc_args(args: LaunchArgs) -> str:
    return (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        f"--attention-backend {args.megatron_attention_backend} "
        # miles' rollout-side default is 8; on 4-GPU nodes the engine
        # addr allocator then treats engines on two nodes as one node group
        # and hands every engine the first node's IP (async_3n: engines on
        # the second rollout node timed out dialing a TCPStore on the first).
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )


def _correction_args(args: BaseScriptArgs) -> str:
    correction_args = ""
    if args.use_rollout_logprobs:
        correction_args += "--use-rollout-logprobs "
    if args.use_tis:
        correction_args += "--use-tis "
    return correction_args


def wandb_args(args: LaunchArgs, spec: RecipeSpec, *, run_id: str | None = None) -> str:
    # No-op without WANDB_API_KEY in the environment. Prefix with the arm
    # name (``run_id``, else the save-dir basename): a topology-only prefix
    # makes concurrent arms indistinguishable in wandb.
    arm_name = args.save_dir.rstrip("/").rsplit("/", 1)[-1]
    run_name_prefix = run_id or arm_name or f"{spec.name}_{args.topology}"
    # Stable group (= RUN_TAG, no per-boot timestamp; commit 1dc9ddc, extended
    # to every recipe 2026-09-03): each chain link and crash-restart of one
    # training run lands in the same wandb group, so the UI's group view shows
    # one continuous curve instead of per-boot fragments. Relies on the
    # discipline that a RUN_TAG is never reused.
    return U.get_default_wandb_args(spec.run_file, run_id=run_name_prefix)


def build_train_args(args: BaseScriptArgs, spec: RecipeSpec) -> str:
    # DAPO swaps in the wrapper that also drops zero-reward-std groups
    # (dict/None-reward safe; see miles_agent/dapo_filter.py). Campaign logs
    # show all_zero + all_one ~ 0.75-0.85 of groups late in training, so the
    # refill costs ~4-7x generation per kept batch on these tasks.
    filter_path = spec.dapo_filter_path if args.algo == "dapo" else spec.filter_path
    # The opd arm and the pilot dump wrap the stock agentic episode: the same
    # /run, then the teacher scoring / the dump row on the training samples.
    generate_fn = OPD_GENERATE_FN if args.algo == "opd" or args.opd_dump else STOCK_AGENTIC_GENERATE_FN
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    return (
        f"{checkpoint_args(args, critic=args.algo == 'ppo')}"
        f"{_rollout_args(args)}"
        f"{_eval_args(args, spec)}"
        f"{_algo_args(args)}"
        f"{_optimizer_args(args)}"
        f"{_perf_args(args)}"
        f"{sglang_args(args)}"
        f"{session_args(args, filter_path=filter_path, generate_fn=generate_fn)}"
        f"{REWARD_ARGS}"
        f"{_topology_args(args)}"
        f"{misc_args(args)}"
        f"{_correction_args(args)}"
        f"{wandb_args(args, spec)}"
        f"{debug_args}"
        f"{args.extra_train_args} "
    )


def build_extra_env_vars(args: LaunchArgs, spec: RecipeSpec) -> dict[str, str]:
    extra_env_vars = {
        # recipe dir: bare recipe modules (reward.shape); REPO_ROOT: the
        # miles_agent package (dotted paths, no shadowing possible)
        "PYTHONPATH": f"{args.megatron_path}:{spec.recipe_dir}:{REPO_ROOT}:{U.repo_base_dir}",
        "NEMO_GYM_URL": args.nemo_gym_url,
    }
    if spec.reward_shaping_path:
        # The recipe's shaping terms (recipes/<name>/reward.py); coefficients
        # arrive via job env (default 0 / the env's standard reward).
        extra_env_vars["REWARD_SHAPING_PATH"] = spec.reward_shaping_path
    if isinstance(args, BaseScriptArgs) and args.algo == "opd":
        # The paper's arm trains on the teacher KL alone: the task reward is
        # zeroed for training (the recipes' shaping terms belong to the GRPO
        # arms) while eval/* and rollout/episode_raw_reward keep the env score.
        extra_env_vars["REWARD_SHAPING_PATH"] = "miles_agent.opd.zero_shape"
    if args.opd_dump:
        extra_env_vars["MILES_OPD_DUMP_DIR"] = os.path.join(args.save_dir.rstrip("/"), "opd_dump")
    if spec.run_timeout_s is not None:
        extra_env_vars["NEMO_GYM_RUN_TIMEOUT"] = str(spec.run_timeout_s)
    # The wandb run lives in the ray driver, which receives every actor's
    # forwarded stdout (sglang decode chatter): default console capture
    # uploads ~300MB/run of output.log. Metrics are unaffected; the full
    # console already lands in the job log.
    extra_env_vars["WANDB_CONSOLE"] = "off"
    if spec.env_passthrough_prefixes:
        for key, value in os.environ.items():
            if key.startswith(spec.env_passthrough_prefixes):
                extra_env_vars.setdefault(key, value)
    return extra_env_vars


def execute_train(args: LaunchArgs, **kwargs) -> None:
    """Submit through miles' launch API: the backend args (an ExecuteTrainConfig) select."""
    args.create_backend().execute_train(**kwargs)


def launch(args: LaunchArgs, spec: RecipeSpec, train_args: str, *, extra_env_vars: dict[str, str] | None = None):
    """Submit ``train_args`` with the shared actor environment; ``extra_env_vars`` win, and the family's process
    env (ModelFamily.env) fills in what the job env left unset, here and in the actors."""
    env_vars = build_extra_env_vars(args, spec) | (extra_env_vars or {})
    _apply_family_env(_listed_family(args.model_name), env_vars)
    execute_train(
        args,
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script=train_script(args),
        extra_env_vars=env_vars,
    )


def _validate_opd(args: BaseScriptArgs) -> None:
    """The OPD arm's protocol, refused at the flags rather than after the allocation: a teacher (its base URL)
    unless the run only evaluates, the reward slot and the scoring as wired here, no off-policy correction,
    thinking off (the template kwargs must parse)."""
    if args.topology.endswith("_teacher") and args.algo != "opd":
        raise ValueError(f"--topology {args.topology} serves an OPD teacher on its last node: it needs --algo opd")
    if args.teacher_url and args.algo != "opd":
        raise ValueError("--teacher-url is an --algo opd knob (the pilot dump scores nothing)")
    if args.algo != "opd":
        return
    if not args.teacher_url and args.num_rollout != 0:
        raise ValueError(
            "--algo opd needs --teacher-url (the job's teacher gate passes it); only an eval-only run "
            "(--num-rollout 0) trains nothing and scores nothing"
        )
    if args.teacher_url:
        parts = urlsplit(args.teacher_url)
        if parts.path.strip("/") or parts.query or parts.fragment:  # ?x=1 would become ...?x=1/generate
            raise ValueError(
                f"--teacher-url {args.teacher_url}: pass the server's base URL (http://host:13141, no path, query "
                "or fragment); _algo_args appends /generate for miles' --opd-teacher-urls"
            )
    tokens = shlex.split(args.extra_train_args)
    for flag in ("--rm-url", "--custom-rm-path"):
        if any(token == flag or token.startswith(f"{flag}=") for token in tokens):
            raise ValueError(
                f"{flag} in --extra-train-args: the teacher scoring is the generate hook's (OPD_GENERATE_FN) and "
                "the reward slot stays miles_agent.reward_routing.reward_func"
            )
    if args.use_tis or args.use_rollout_logprobs:
        raise ValueError("--use-tis / --use-rollout-logprobs are not in the OPD protocol (ratio 1, clip inert)")
    try:
        kwargs = json.loads(args.chat_template_kwargs or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(
            f"--chat-template-kwargs {args.chat_template_kwargs!r} is not JSON ({error.msg} at char {error.pos}): "
            "pass a JSON object such as '{\"enable_thinking\": false}'"
        ) from None
    if not isinstance(kwargs, dict):
        raise ValueError(f"--chat-template-kwargs {args.chat_template_kwargs!r} is not a JSON object")
    if kwargs.get("enable_thinking") is not False:
        print(
            "WARN: --algo opd with thinking not disabled: the paper trained every family, Qwen3 hybrids included, "
            "with --chat-template-kwargs '{\"enable_thinking\": false}'",
            file=sys.stderr,
        )


def execute(args: BaseScriptArgs, spec: RecipeSpec):
    for name in ("save_dir", "prompt_data", "eval_data"):
        if not getattr(args, name):
            raise ValueError(f"{type(args).__name__}.{name} is empty: the recipe must default it or pass --{name}")
    if not spec.eval_slices:
        # --eval-prompt-data takes one or more values (miles argparse nargs="+"): an empty list
        # would fail at train time, after the allocation, with argparse's message
        raise ValueError(f"RecipeSpec {spec.name!r} has no eval_slices: an env recipe evaluates on its fixed file")
    _validate_opd(args)
    launch(args, spec, build_train_args(args, spec))


# The OPD teacher gate's tokenizer probe: the teacher scores the student's token ids, so the two tokenizers must
# agree in size and on the ids of plain ASCII, CJK and the student's own tool-call markup.
_PROBE_TEXT = (
    "The quick brown fox jumps over the lazy dog 0123456789 <>[]{}\n"
    "你好，世界。日本語のテキスト 한국어 "
)
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "One command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


def tokenizer_probe(policy_hf: str, teacher_hf: str) -> str | None:
    """Why the teacher at ``teacher_hf`` cannot score the tokens of the student at ``policy_hf``, or None when
    their tokenizers agree in size and on the probe's ids (_PROBE_TEXT + the student's rendered tool call)."""
    from transformers import AutoTokenizer  # heavy; the gate runs in the container

    from miles_agent.tool_call_markup import hf_render_chat, render_tool_call

    probe = _PROBE_TEXT + render_tool_call(hf_render_chat(policy_hf), _PROBE_TOOL, {"command": "go to desk 1"})
    student, teacher = (AutoTokenizer.from_pretrained(path) for path in (policy_hf, teacher_hf))
    if len(student) != len(teacher):
        return f"vocabulary sizes differ: student {len(student)}, teacher {len(teacher)}"
    if student.encode(probe) != teacher.encode(probe):
        return "the probe string tokenizes differently"
    return None


def _path_family(model: str) -> ModelFamily:
    """The family of a HF checkpoint dir or a model name: its basename's MODEL_FAMILIES row."""
    return model_family(Path(model).name)


def main(argv: list[str] | None = None) -> None:
    """``python3 -m miles_agent.launcher``: a checkpoint dir's family facts for the serving jobs and the OPD
    teacher gate's tokenizer probe; a failure is one line on stderr and exit 1."""
    import argparse

    cli = argparse.ArgumentParser(prog="python3 -m miles_agent.launcher", description=main.__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    model_help = "HF checkpoint dir or model name: its basename picks the MODEL_FAMILIES row"
    commands.add_parser("parser", help="print the family's tool-call parser").add_argument("model", help=model_help)
    commands.add_parser("reasoning-parser", help="print the family's reasoning parser").add_argument(
        "model", help=model_help
    )
    serve = commands.add_parser("serve-template", help="write the family's fixed template; print its tool-call parser")
    serve.add_argument("model", help=model_help)
    serve.add_argument("out", help="the template file to write")
    serve.add_argument("--thinking", default="off", help="off (default): thinking off unless a request turns it on")
    serve.add_argument("--tito-model", default="", help="this TITO family instead of the model's row")
    probe = commands.add_parser("tokenizer-probe", help="TOKENIZER_PROBE_OK when the teacher can score the student")
    probe.add_argument("policy_hf", help="the student's HF checkpoint dir")
    probe.add_argument("teacher_hf", help="the teacher's HF checkpoint dir")
    ns = cli.parse_args(argv)
    try:
        if ns.command == "tokenizer-probe":
            drift = tokenizer_probe(ns.policy_hf, ns.teacher_hf)
            if drift:
                sys.exit(f"TOKENIZER_PROBE_FAIL {drift}")
            print("TOKENIZER_PROBE_OK")
        elif ns.command == "serve-template":
            tito = ns.tito_model or _path_family(ns.model).tito_model
            print(write_serve_template(tito, ns.out, thinking=ns.thinking))
        elif ns.command == "parser":
            print(tool_call_parser(_path_family(ns.model).tito_model))
        else:
            print(reasoning_parser(_path_family(ns.model).tito_model))
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
