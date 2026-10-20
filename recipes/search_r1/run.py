# pyright: reportMissingImports=false
"""search_r1 recipe launcher: Miles <-> tool_call_agent GRPO
on the wiki_search environment.

Cross-framework alignment run: the science config below follows the protocol
both proven baselines share, with two documented deviations
(eps-clip-high 0.28 and adam-beta2 0.98; see the inline notes at their
definitions in miles_agent/launcher.py) -
- nemo-rl: its GRPO config for this task (sync colocated, 3 nodes
  optimum) + grpo_async_2n/3n (disaggregated fully-async),
- slime:   the r2s nrl-align job's ALIGN_ARGS
  (drain-async, 3 nodes = 2 train + 1 rollout).
64 prompts x 8 samples = 512 rollouts/step, global batch 512 (mu=1), lr 1e-6
constant + warmup 142, KL 0.001 low_var_kl, clip 0.2/0.28, temp/top_p 1.0,
per-turn response 1024 / context 16384, 500 steps, eval every 50 on the exact
893-row nemo-rl validation set (per-source slices). Do not change any of these
without re-running the baselines.

Topologies (--topology): colocated_1n | async_3n | fully_async_3n - see
miles_agent/launcher.py for the drain/staleness semantics of each arm.

Prerequisites:
- A NeMo Gym deployment serving wiki_search + tool_call_agent
  (scripts/start_gym.sh); point NEMO_GYM_URL at the tool_call_agent server.
- The wiki_search resources server needs a reachable FlashRAG-style retriever
  (its configs/wiki_search.yaml retriever_base_url).
- The policy model's family plumbing (TITO tokenizer, chat template kwargs, the
  engines' tool-call parser) follows --model-name (miles_agent/launcher.py
  MODEL_FAMILIES; default Qwen3.5-4B).
- Prompt/eval data from recipes/search_r1/prepare_data.py. The eval file must
  be the converted nemo-rl validation.jsonl: 893 rows in source blocks
  (128 each of nq/triviaqa/popqa/hotpotqa/2wiki/musique + 125 bamboogle, in
  that order) - the per-source slices below index into it.

Usage:
    NEMO_GYM_URL=http://<gym-host>:<port> python run.py --topology colocated_1n
    python run.py --topology async_3n         # expects external ray, 3 nodes
    python run.py --topology fully_async_3n   # expects external ray, 3 nodes
    python run.py --mode debug_rollout_only
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
from miles_agent.launcher import BaseScriptArgs, RecipeSpec  # noqa: E402


@dataclass
class ScriptArgs(BaseScriptArgs):
    # Under MILES_AGENT_RESULTS / MILES_AGENT_DATA (launcher.results_dir / data_path); the entry passes all three.
    save_dir: str = field(default_factory=lambda: launcher.results_dir("search_r1"))
    prompt_data: str = field(default_factory=lambda: launcher.data_path("search_r1_train.jsonl"))
    eval_data: str = field(default_factory=lambda: launcher.data_path("search_r1_eval893.jsonl"))

    rollout_max_response_len: int = 1024  # per-turn generation cap
    rollout_max_context_len: int = 16384  # whole-episode context budget
    # --max-tokens-per-gpu follows rollout_max_context_len (launcher default 0).


SPEC = RecipeSpec(
    name="search_r1",
    run_file=__file__,
    # Exact nemo-rl training-time validation: 893 rows, 1 gen/prompt, temp 1.0.
    # Slice offsets assume prepare_data preserved the source-block ordering.
    eval_slices=(
        ("nq", "@[0:128]"),
        ("triviaqa", "@[128:256]"),
        ("popqa", "@[256:384]"),
        ("hotpotqa", "@[384:512]"),
        ("2wikimultihopqa", "@[512:640]"),
        ("musique", "@[640:768]"),
        ("bamboogle", "@[768:893]"),
    ),
    # Shaping coefficients arrive via job env (REWARD_FORMAT_COEF/...),
    # default 0. Wiki episodes are a few short turns: gym_bridge's 600 s /run
    # default holds.
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
