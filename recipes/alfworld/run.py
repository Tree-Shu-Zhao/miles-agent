# pyright: reportMissingImports=false
"""alfworld recipe launcher: Miles <-> env_loop_agent GRPO
on the alfworld_env environment.

ALFWorld campaign (env #2): free-text household episodes over the Gymnasium
reset/step protocol - the model calls the `act` tool once per turn, the env
replies with the observation + admissible actions, reward 1.0 only on a won
terminal state. The science config is carried over verbatim from the
search_r1 campaign (same deliberate deviations: eps-clip-high 0.28,
adam-beta2 0.98) so alfworld curves are comparable arm-for-arm; only the
episode-shape knobs differ (per-turn response 512 / context 32768 - see the
inline notes at their definitions).
64 prompts x 8 samples = 512 rollouts/step, global batch 512 (mu=1), lr 1e-6
constant + warmup 142, KL 0.001 low_var_kl, clip 0.2/0.28, temp/top_p 1.0,
500 steps, eval every 50 on the 274-game validation set (per-split slices).

Topologies (--topology): colocated_1n | async_3n | fully_async_3n - see
miles_agent/launcher.py for the drain/staleness semantics of each arm.

Prerequisites:
- A NeMo Gym deployment serving alfworld_env + env_loop_agent
  (scripts/start_gym.sh); point NEMO_GYM_URL at the env_loop_agent server.
  The alfworld_env server resolves metadata.game_file under its own
  $ALFWORLD_DATA - no retriever or other external service is needed.
- The policy model's family plumbing (TITO tokenizer, chat template kwargs, the
  engines' tool-call parser) follows --model-name (miles_agent/launcher.py
  MODEL_FAMILIES; default Qwen3.5-4B).
- Prompt/eval data from recipes/alfworld/prepare_data.py --model <the
  policy model's HF dir> (its example act call is the policy model's own markup). The
  eval file must be alfworld_eval274.jsonl: 274 rows in split blocks (140
  valid_seen then 134 valid_unseen, in that order) - the per-split slices
  below index into it.

Usage:
    NEMO_GYM_URL=http://<gym-host>:<port> python run.py --topology colocated_1n
    python run.py --topology async_3n         # expects external ray, 3 nodes
    python run.py --topology fully_async_3n   # expects external ray, 3 nodes
    python run.py --mode debug_rollout_only
    # the training entry: RECIPE=alfworld selects this recipe dir
    #   RECIPE=alfworld scripts/run_local.sh --run-tag <tag> [--topology ...]
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
    save_dir: str = field(default_factory=lambda: launcher.results_dir("alfworld"))
    prompt_data: str = field(default_factory=lambda: launcher.data_path("alfworld_train.jsonl"))
    eval_data: str = field(default_factory=lambda: launcher.data_path("alfworld_eval274.jsonl"))

    rollout_max_response_len: int = 512  # per-turn generation cap: think + one action
    # 30 turns x (obs + admissible list) + per-turn generations ~ 26k worst
    # case; flows into the per-sample context limit only.
    rollout_max_context_len: int = 32768
    # Dynamic-batch packing budget for train/logprob passes. 65536 OOMs:
    # the fp32 logits tensor alone is tokens x 152064 vocab x 4 B = 39.9 GB
    # per microbatch, which tipped GPU-alloc retries and then host oom_kill
    # after ~90 steps; packing gain was nil (actor_train time
    # unchanged), so this stays at the per-sample context limit.
    train_max_tokens_per_gpu: int = 32768


SPEC = RecipeSpec(
    name="alfworld",
    # Infrastructure failures (worker respawn -> 500, gym /run failure) are
    # not env feedback: the group is dropped and re-sampled instead of
    # training the episode as reward 0 (user decision 2026-09-03, after the
    # alfworld reproduction finished). Env feedback is never filtered.
    filter_path=NO_TRANSPORT_FAILURE_FILTER,
    dapo_filter_path=NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER,
    run_file=__file__,
    # 274-game validation, 1 gen/prompt, temp 1.0. Slice offsets assume
    # prepare_data preserved the split-block ordering (valid_seen first).
    eval_slices=(
        ("alfworld_seen", "@[0:140]"),
        ("alfworld_unseen", "@[140:274]"),
    ),
    # 30 model calls per episode under saturation exceeds the 600 s default
    # /run timeout. The recipe's shaping coefficients arrive via job env
    # (ALFWORLD_STEP_PENALTY/...), default 0.
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
