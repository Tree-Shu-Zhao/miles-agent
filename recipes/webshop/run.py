# pyright: reportMissingImports=false
"""webshop recipe launcher: Miles <-> env_loop_agent GRPO
on the webshop_env environment.

WebShop campaign (env #3): text shopping episodes over the Gymnasium reset/step
protocol - the model calls the `act` tool once per turn (command = a WebShop
action), the env replies the page + admissible actions, reward is the raw
WebShop score at click[buy now]. The optimizer / loss / topology config is
carried over verbatim from the search_r1 / alfworld campaigns
(eps-clip-high 0.28, adam-beta2 0.98) so arms stay comparable; what differs is
the batch shape (16 x 8 = 128 rollouts/step, global batch 128, vs 64 x 8 = 512
there) and the episode-shape knobs (per-turn response 512 / context 32768 -
see the ScriptArgs notes).

16 prompts x 8 samples = 128 rollouts/step, global batch 128 (mu=1), lr 1e-6
constant + warmup 142, KL 0.001 low_var_kl, clip 0.2/0.28, temp/top_p 1.0,
500 steps by default (the campaign runs pass --num-rollout=250).

Protocol (docs/webshop_integration.md "TCOD protocol"): the split layout comes
from WEBSHOP_PROTOCOL (env_servers/webshop/defaults.sh, default canonical = the
WebShop paper's fixed held-out test idx 0..499; tcod = TCOD's layout) and must
match what recipes/webshop/prepare_data.py emitted - the training entry
regenerates the data with the same variable. Validation (TCOD-aligned):
1 sample/goal at temperature 0.4 (TCOD Table 4/5), per-turn cap 4096 tokens
(TCOD Table 5; training keeps 512), every 50 steps, seed 42. eval/<slice> =
mean raw task_score; the success metric is env/won (score == 1). WebShop's
done flag (buy now reached at any score) is NOT a success measure and is not
exported.

Failure semantics: env feedback (invalid actions, no-op clicks, bad purchases,
low scores) is always trained on. Infrastructure failures (engine worker
crash/respawn/timeout, adapter 5xx, any gym /run failure) surface as
env_metrics.run_transport_failed and the whole GRPO group is dropped and
re-sampled by miles_agent.dapo_filter.check_no_aborted_no_transport_failure
(never trained as reward 0).

Topologies (--topology): colocated_1n | async_3n | fully_async_3n - see
miles_agent/launcher.py for the drain/staleness semantics of each arm.

Prerequisites:
- A NeMo Gym deployment serving webshop_env + env_loop_agent
  (the training entry under RECIPE=webshop starts it and the py3.10
  engine server); point NEMO_GYM_URL at the env_loop_agent server.
- The policy model's family plumbing (TITO tokenizer, chat template kwargs, the
  engines' tool-call parser) follows --model-name (miles_agent/launcher.py
  MODEL_FAMILIES; default Qwen3.5-4B).
- Prompt/eval data from recipes/webshop/prepare_data.py --protocol
  $WEBSHOP_PROTOCOL --model <the policy model's HF dir> (webshop_train.jsonl,
  webshop_eval.jsonl; row layout per protocol in EVAL_SLICES below). Rows
  carry metadata.goal_id, the CONTENT id of a goal from the frozen goals
  export (WEBSHOP_GOALS_FILE, rebuilt by scripts/fetch_data.sh webshop
  --regenerate-goals; not a position in the engine's goal list); the engine
  resolves it on /reset and
  the training entry refuses to start unless the engine's goal-set digest
  equals the dataset's, so a run can never silently score other questions.

Usage:
    NEMO_GYM_URL=http://<gym-host>:<port> python run.py --topology colocated_1n
    # the training entry: RECIPE=webshop selects this recipe dir
    #   RECIPE=webshop scripts/run_local.sh --run-tag <tag> [--topology ...]
"""

import os
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

# Row layout of webshop_eval.jsonl per protocol (recipes/webshop/prepare_data.py
# eval_slices(); the row order is contractual):
#   canonical webshop         rows 0..499  = WebShop paper test set (goals-file
#   (default)                 pos < 500, never trained) - the mainstream held-out number;
#             webshop_pool    rows 500..755 = 256 goals sampled from the WHOLE
#                             pool, trained goals included - the distribution
#                             verl-agent/GiGPO validate on.
#   tcod      webshop         rows 0..99   = TCOD's test set (goals-file pos 4096..4195;
#                             TCOD_examples/webshop/get_webshop_data.py);
#             webshop_heldout rows 100..499 = pos 4196..4595, never trained,
#                             a lower-noise companion (100 goals = 1% granularity).
EVAL_SLICES = {
    "tcod": (("webshop", "@[0:100]"), ("webshop_heldout", "@[100:500]")),
    "canonical": (("webshop", "@[0:500]"), ("webshop_pool", "@[500:756]")),
}
WEBSHOP_PROTOCOL = os.environ.get("WEBSHOP_PROTOCOL", "canonical")
if WEBSHOP_PROTOCOL not in EVAL_SLICES:
    raise SystemExit(f"WEBSHOP_PROTOCOL={WEBSHOP_PROTOCOL!r} is not one of {sorted(EVAL_SLICES)}")


@dataclass
class ScriptArgs(BaseScriptArgs):
    # Under MILES_AGENT_RESULTS / MILES_AGENT_DATA (launcher.results_dir / data_path); the entry passes all three.
    save_dir: str = field(default_factory=lambda: launcher.results_dir("webshop"))
    prompt_data: str = field(default_factory=lambda: launcher.data_path("webshop_train.jsonl"))
    eval_data: str = field(default_factory=lambda: launcher.data_path("webshop_eval.jsonl"))

    # TCOD validation protocol (Table 4/5): temperature 0.4, top-p 1.0, max
    # 4096 tokens per turn (training keeps 512), seed 42. TCOD evaluates every
    # 5 steps on 10 sampled tasks; we evaluate the whole slice set (756 episodes
    # on canonical, 500 on tcod) every 50 steps, the launcher default: at every
    # 10 steps a 250-step run spent 26 evals (about an hour) on evaluation.
    eval_temperature: float = 0.4
    eval_max_response_len: int = 4096
    eval_interval: int = 50
    seed: int = 42

    # Batch shape: 16 x 8 = 128 concurrent episodes per step (the engine tier
    # is sized for this), one update/step (mu=1).
    rollout_batch_size: int = 16
    global_batch_size: int = 128
    rollout_max_response_len: int = 512  # per-turn generation cap: think + one action
    # Context budget. The transcript is the FULL multi-turn history: miles
    # assembles ONE contiguous sample per episode by prefix extension, so a
    # verl-agent-style 2-step sliding window (a single-turn re-render) is not
    # portable here. Each WebShop step costs ~1.0-1.4k tokens of page +
    # admissible list plus the reply, so 15 steps need ~17-22k tokens; at
    # 16384 long episodes hit the policy 400 around step 11-14 and trained as
    # reward-0 truncations. 32768 fits the GPUs used (same as alfworld); 65536 OOMs
    # (see the alfworld recipe).
    rollout_max_context_len: int = 32768
    # Dynamic-batch packing budget; kept at the per-sample context limit (one
    # multi-turn sample cannot split across GPUs with CP=1).
    train_max_tokens_per_gpu: int = 32768


SPEC = RecipeSpec(
    name="webshop",
    run_file=__file__,
    # Two validation views of one file; see EVAL_SLICES.
    eval_slices=EVAL_SLICES[WEBSHOP_PROTOCOL],
    # Infra failures (engine/adapter/gym) drop the group instead of training
    # as reward 0; env feedback is never filtered. DAPO adds the
    # zero-reward-std drop on top.
    filter_path=NO_TRANSPORT_FAILURE_FILTER,
    dapo_filter_path=NO_TRANSPORT_FAILURE_NONZERO_STD_FILTER,
    # 15 model calls per episode under saturation exceeds the 600 s default.
    run_timeout_s=1200,
    # Reward knobs (recipes/webshop/reward.py reads WEBSHOP_DENSE_REWARD /
    # WEBSHOP_REWARD_SCALE / WEBSHOP_*_PENALTY at import time inside the ray
    # actors): forward every WEBSHOP_* var present in the job environment -
    # the ray runtime env is the only guaranteed path to the actors. Defaults
    # give the standard 10-iff-perfect WebShop training reward.
    env_passthrough_prefixes=("WEBSHOP_",),
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
