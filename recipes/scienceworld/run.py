# pyright: reportMissingImports=false
"""scienceworld recipe launcher: Miles <-> env_loop_agent GRPO
on the scienceworld_env environment.

ScienceWorld campaign (env #4, the R2S paper's fourth environment): elementary-science
text tasks over the Gymnasium reset/step protocol - the model calls the `act` tool
once per turn with a command composed from the action commands and objects the env
lists, the env replies with the simulator feedback plus the updated lists, and the
episode ends on completion or at the 30-step TCOD cap. The science config is carried
over verbatim from the alfworld campaign (itself the search_r1 config: eps-clip-high
0.28, adam-beta2 0.98) so curves are comparable arm-for-arm; the episode-shape knobs are
alfworld's too (per-turn response 512 / context 32768 - see the inline notes at their
definitions).
64 prompts x 8 samples = 512 rollouts/step, global batch 512 (mu=1), lr 1e-6
constant + warmup 142, KL 0.001 low_var_kl, clip 0.2/0.28, temp/top_p 1.0,
500 steps, eval every 50 on the 274-task validation set (per-split slices).

Splits (TCOD-aligned, arXiv 2604.24005; recipes/scienceworld/prepare_data.py): train = the
17 train task types x their first 50% variations (2294 tasks); scienceworld_seen = 140
held-out variations of the train task types (our addition, TCOD has no seen-task eval);
scienceworld_unseen = the first 134 rows of the frozen 1308-row TCOD test taskset (the 13
unseen task types).

Metrics: the eval/* top-level numbers are the env's dense task score - the episode's
reward sum = the simulator's best score / 100, TCOD's reward in [0, 1]. Success is the
simulator's completion flag, reported as env/won (the R2S paper's metric;
docs/agentic_rl_metrics.md). Unlike alfworld's binary score the two differ here (partial
credit is common), so success rates are read from won, never from the score.

Topologies (--topology): colocated_1n | async_3n | fully_async_3n - see
miles_agent/launcher.py for the drain/staleness semantics of each arm.

Prerequisites:
- A NeMo Gym deployment serving scienceworld_env + env_loop_agent
  (scripts/start_gym.sh); point NEMO_GYM_URL at the env_loop_agent server.
  The scienceworld_env server hosts one warm JVM per worker process on the
  pinned scienceworld.jar + JDK 17; the training entry hands it jar_path /
  java_home / env_workers as gym overrides (docs/scienceworld_integration.md).
- The policy model's family plumbing (TITO tokenizer, chat template kwargs, the
  engines' tool-call parser) follows --model-name (miles_agent/launcher.py
  MODEL_FAMILIES; default Qwen3.5-4B).
- Prompt/eval data from recipes/scienceworld/prepare_data.py --model <the
  policy model's HF dir> (a prompt's act calls are the policy model's own markup).
  The eval file must be scienceworld_eval274.jsonl: 274 rows in split blocks (140
  valid_seen then 134 valid_unseen, in that order) - the per-split slices
  below index into it.

Usage:
    NEMO_GYM_URL=http://<gym-host>:<port> python run.py --topology colocated_1n
    python run.py --topology async_3n         # expects external ray, 3 nodes
    python run.py --topology fully_async_3n   # expects external ray, 3 nodes
    python run.py --mode debug_rollout_only
    # the training entry: RECIPE=scienceworld selects this recipe dir
    #   RECIPE=scienceworld scripts/run_local.sh --run-tag <tag> [--topology ...]
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# The shared launcher lives in this repo's miles_agent package. The training entry
# runs `python3 run.py` from this dir with no PYTHONPATH, so make the repo root
# importable here (the ray actors get it through the runtime env instead).
_REPO_ROOT = str(SCRIPT_DIR.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import miles.utils.external_utils.command_utils as U  # noqa: E402
import typer  # noqa: E402

from miles_agent import launcher  # noqa: E402
from miles_agent.launcher import (  # noqa: E402
    NO_TRANSPORT_FAILURE_FILTER,
    NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER,
    BaseScriptArgs,
    RecipeSpec,
)


@dataclass
class ScriptArgs(BaseScriptArgs):
    # Under MILES_AGENT_RESULTS / MILES_AGENT_DATA (launcher.results_dir / data_path); the entry passes all three.
    save_dir: str = field(default_factory=lambda: launcher.results_dir("scienceworld"))
    prompt_data: str = field(default_factory=lambda: launcher.data_path("scienceworld_train.jsonl"))
    eval_data: str = field(default_factory=lambda: launcher.data_path("scienceworld_eval274.jsonl"))

    rollout_max_response_len: int = 512  # per-turn generation cap: brief deliberation + one action
    # ScienceWorld turns are longer than ALFWorld's (room descriptions, the
    # re-listed action commands and objects; obs capped at 4096 chars), but the
    # paper's episodes stayed far below this (~1.2-1.5k response tokens per
    # episode, no truncation at 32768); flows into the per-sample context limit only.
    rollout_max_context_len: int = 32768
    # Dynamic-batch packing budget for train/logprob passes, pinned at the
    # per-sample context limit like alfworld's (65536 OOMed there: the fp32
    # logits tensor alone is tokens x vocab x 4 B per microbatch).
    train_max_tokens_per_gpu: int = 32768


SPEC = RecipeSpec(
    name="scienceworld",
    # Infrastructure failures (worker respawn -> 500, all JVM slots busy -> 503,
    # gym /run failure) are not env feedback: the group is dropped and
    # re-sampled instead of training the episode as reward 0 (the alfworld
    # decision, 2026-09-03). Env feedback is never filtered.
    filter_path=NO_TRANSPORT_FAILURE_FILTER,
    dapo_filter_path=NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER,
    run_file=__file__,
    # 274-task validation, 1 gen/prompt, temp 1.0. Slice offsets assume
    # prepare_data preserved the split-block ordering (valid_seen first).
    eval_slices=(
        ("scienceworld_seen", "@[0:140]"),
        ("scienceworld_unseen", "@[140:274]"),
    ),
    # 30 model calls per episode under saturation (plus the wait for a free
    # worker JVM) exceeds the 600 s default /run timeout.
    run_timeout_s=1200,
)

cleanup = launcher.cleanup
prepare = launcher.prepare


def execute(args: ScriptArgs):
    launcher.execute(args, SPEC)


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
