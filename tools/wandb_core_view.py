#!/usr/bin/env python3
"""Build the pinned "agentic-rl-core" wandb saved view of a recipe's project.

One view per recipe (search_r1, alfworld, webshop, scienceworld, tmax),
same four pinned sections as docs/agentic_rl_metrics.md (Tier 0-3): only
the keys that recipe's runs actually log, so no empty panels. Tier 0 is
recipe-specific (its success metrics and eval slices); Tier 1-3 are shared, with
the async-only panels (staleness, TIS, DAPO drops, PPO critic) kept for
search_r1 alone. tmax lays out all four itself (tmax_sections) and filters/groups
its runset. The env recipes with an OPD
preset (recipes/<env>/opd.py, the R2S student arm; docs/r2s.md) add a fifth
section for those arms, one metric a panel (opd_section).

    python tools/wandb_core_view.py --recipe webshop            # create a new view
    python tools/wandb_core_view.py --recipe webshop --update   # rewrite the view recorded in tools/wandb_views.local.json
    python tools/wandb_core_view.py --recipe alfworld webshop --dry-run   # list panels and keys

wandb_workspaces.Workspace.save() creates a NEW saved view every time it is
called on a fresh object (the display name is not a key); pass --update to
load the recorded view by URL and rewrite it in place instead.

Requires: pip install wandb wandb-workspaces; WANDB_API_KEY and WANDB_ENTITY (or --entity) in the env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = os.environ.get("WANDB_ENTITY", "")
VIEW_NAME = "agentic-rl-core"

# Saved-view URLs, recorded by the first --recipe run in the untracked tools/wandb_views.local.json
# ({"<recipe>": "<url>"}); --update rewrites the view at that URL.
VIEWS_FILE = Path(__file__).with_name("wandb_views.local.json")


def _load_views() -> dict[str, str]:
    try:
        return json.loads(VIEWS_FILE.read_text())
    except FileNotFoundError:
        return {}


VIEW_URLS: dict[str, str] = _load_views()

L = wr.LinePlot
SEARCH_EVAL_DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
# The env recipes' eval slices (recipes/<env>/run.py eval_slices): the seen / unseen splits, WebShop's
# canonical 500 goals and its 256-goal pool.
ALFWORLD_SLICES = ["alfworld_seen", "alfworld_unseen"]
WEBSHOP_SLICES = ["webshop", "webshop_pool"]
SCIENCEWORLD_SLICES = ["scienceworld_seen", "scienceworld_unseen"]


def sec(name: str, panels: list) -> ws.Section:
    return ws.Section(name=name, panels=panels, is_open=True, pinned=True)


# --------------------------------------------------------------------- Tier 0
def tier0_search() -> list:
    return [
        L(title="Train reward (raw)", x="rollout/step", y=["rollout/episode_raw_reward"]),
        # Macro-EM matches the Search-R1 paper's Avg. column (verified: the
        # printed Avg equals the unweighted mean of the 7 EMs).
        L(
            title="Held-out EM (macro avg, Search-R1 style)",
            x="eval/step",
            y=[f"eval/{d}" for d in SEARCH_EVAL_DATASETS],
            custom_expressions=["(" + " + ".join(f"${{eval/{d}}}" for d in SEARCH_EVAL_DATASETS) + ") / 7"],
        ),
        *[L(title=f"EM {d}", x="eval/step", y=[f"eval/{d}"]) for d in SEARCH_EVAL_DATASETS],
        L(title="Response length", x="rollout/step", y=["rollout/response_len/mean"]),
        L(title="Truncation ratio (alarm >=0.15)", x="rollout/step", y=["rollout/truncated_ratio"]),
    ]


def env_tier0_common(slices: list[str]) -> list:
    """Panels every env recipe shares: lengths, step budget, action faults."""
    return [
        L(title="Steps per episode (train)", x="rollout/step", y=["rollout/env/step_count"]),
        L(
            title="Steps per episode (held-out)",
            x="eval/step",
            y=[f"eval/{s}/env/step_count" for s in slices],
        ),
        L(
            title="Episodes hitting the step cap (env_truncated)",
            x="rollout/step",
            y=["rollout/env/env_truncated"],
        ),
        L(
            title="Action faults per episode (train)",
            x="rollout/step",
            y=["rollout/env/action_format_errors", "rollout/env/invalid_actions", "rollout/env/extra_tool_calls"],
        ),
        L(
            title="Action faults per episode (held-out)",
            x="eval/step",
            y=[f"eval/{s}/env/{k}" for s in slices for k in ("action_format_errors", "invalid_actions")],
        ),
        L(
            title="Response length: per turn / per episode",
            x="rollout/step",
            y=["rollout/response_len/mean", "rollout/episode_total_response_length/mean"],
        ),
        L(title="Truncation ratio (alarm >=0.15)", x="rollout/step", y=["rollout/truncated_ratio"]),
    ]


def tier0_alfworld() -> list:
    slices = ALFWORLD_SLICES
    return [
        # ALFWorld's score is binary, so raw reward == won; one panel.
        L(title="Train success rate (env/won)", x="rollout/step", y=["rollout/env/won"]),
        L(
            title="Held-out success rate (env/won): seen vs unseen",
            x="eval/step",
            y=[f"eval/{s}/env/won" for s in slices],
        ),
        L(
            title="Held-out failure mix: lost vs step cap",
            x="eval/step",
            y=[f"eval/{s}/env/{k}" for s in slices for k in ("lost", "env_truncated")],
        ),
        L(title="Train failure mix: lost", x="rollout/step", y=["rollout/env/lost"]),
        *env_tier0_common(slices),
    ]


def tier0_webshop() -> list:
    slices = WEBSHOP_SLICES
    return [
        # task_score is the raw WebShop score in [0,1]; won = score == 1 (the
        # success metric; WebShop's done flag / "purchased" is not success).
        L(
            title="Train: success (env/won) and raw score (env/task_score)",
            x="rollout/step",
            y=["rollout/env/won", "rollout/env/task_score"],
        ),
        L(
            title="Held-out success rate (env/won): canonical 500 vs pool 256",
            x="eval/step",
            y=[f"eval/{s}/env/won" for s in slices],
        ),
        L(
            title="Held-out raw score (env/task_score): canonical 500 vs pool 256",
            x="eval/step",
            y=[f"eval/{s}/env/task_score" for s in slices],
        ),
        *env_tier0_common(slices),
    ]


def tier0_scienceworld() -> list:
    slices = SCIENCEWORLD_SLICES
    return [
        # The score is TCOD's dense task score (the simulator's best score / 100) and the
        # top-level eval/<slice> key is that score; success is the simulator's completion
        # flag, env/won (the paper's metric), which differs from it under partial credit.
        L(title="Train success rate (env/won)", x="rollout/step", y=["rollout/env/won"]),
        L(title="Train raw score (env/task_score)", x="rollout/step", y=["rollout/env/task_score"]),
        L(
            title="Held-out success rate (env/won): seen vs unseen",
            x="eval/step",
            y=[f"eval/{s}/env/won" for s in slices],
        ),
        L(
            title="Held-out raw score (env/task_score): seen vs unseen",
            x="eval/step",
            y=[f"eval/{s}/env/task_score" for s in slices],
        ),
        *env_tier0_common(slices),
    ]


# ------------------------------------------------------------------ Tier 1-3
def tier1(async_topology: bool) -> list:
    panels = [
        L(title="Policy divergence / clip", x="train/step", y=["train/ppo_kl", "train/pg_clipfrac"]),
        L(
            title="Engine-trainer mismatch (abs_diff <=0.03)",
            x="train/step",
            y=["train/train_rollout_kl", "train/train_rollout_logprob_abs_diff"],
        ),
        L(
            title="Mixed-version dose (flat 0 = blind spot)",
            x="rollout/step",
            y=["rollout/weight_version/mixed_version_ratio"],
        ),
    ]
    if async_topology:
        panels += [
            L(title="TIS correction pressure", x="train/step", y=["train/tis", "train/tis_clipfrac"]),
            L(
                title="Staleness age",
                x="rollout/step",
                y=["rollout/fully_async/avg_staleness", "rollout/fully_async/max_staleness"],
            ),
        ]
    else:
        panels.append(L(title="Importance-weight ESS (train/ess_ratio)", x="train/step", y=["train/ess_ratio"]))
    return panels


def tier2(async_topology: bool, env_health: list[str]) -> list:
    panels = [
        L(
            title="Env failure rates",
            x="rollout/step",
            y=["rollout/env/run_transport_failed", "rollout/env/policy_400_truncated"],
        ),
    ]
    if env_health:
        panels.append(L(title="Env worker health", x="rollout/step", y=env_health))
    panels += [
        L(title="Step time / rollout time", x="rollout/step", y=["perf/step_time", "perf/rollout_time"]),
        L(title="Trainer wait fraction", x="rollout/step", y=["perf/wait_time_ratio"]),
    ]
    if async_topology:
        panels.append(L(title="Async buffer depth", x="rollout/step", y=["rollout/fully_async/queue_size"]))
    return panels


def tier3(async_topology: bool) -> list:
    panels = [
        L(
            title="Zero-std group fraction (under DAPO see drops)",
            x="rollout/step",
            y=["rollout/zero_std/all_zero_percentage", "rollout/zero_std/all_one_percentage"],
        ),
    ]
    if async_topology:
        panels.append(
            L(
                title="DAPO drop counts",
                x="rollout/step",
                y=["rollout/dynamic_filter/drop_zero_std_0.0", "rollout/dynamic_filter/drop_zero_std_1.0"],
            )
        )
    panels.append(L(title="Numerical health", x="train/step", y=["train/grad_norm", "train/loss"]))
    if async_topology:
        # train/entropy_loss was a placeholder 0 on the search arms (entropy
        # coef 0, observe-training-entropy off); -mean sampled logprob is the
        # proxy that exists on every run.
        panels.append(
            L(
                title="Entropy proxy (-mean sampled logprob)",
                x="rollout/step",
                y=["rollout/log_probs"],
                custom_expressions=["0 - ${rollout/log_probs}"],
            )
        )
        panels.append(L(title="PPO critic", x="train/step", y=["train/critic-value_loss", "train/critic-grad_norm"]))
    else:
        # run.py observes training entropy on these recipes, so the key is real.
        panels.append(L(title="Entropy (train/entropy_loss)", x="train/step", y=["train/entropy_loss"]))
        panels.append(L(title="KL to reference (train/kl_loss)", x="train/step", y=["train/kl_loss"]))
    return panels


# ---------------------------------------------------------------------- tmax
def tmax_sections() -> list:
    """Terminal-agent RL (recipes/tmax): one sample is a whole Harbor trial, scored offline (TB-Lite / TB 2.x), so
    there is no in-training eval; the learning curve is the train reward and the exit mix. Reply length and
    repetition sit next to the reward because a run that drifts into repeated thinking shows it there first.

    One metric a panel: the runset is grouped by RUN_TAG, which gives every line of a group one color, so several
    metrics in a panel differ only by dash pattern and blur together."""

    def rollout(title: str, key: str) -> wr.LinePlot:
        return L(title=title, x="rollout/step", y=[key])

    def train(title: str, key: str) -> wr.LinePlot:
        return L(title=title, x="train/step", y=[key])

    learning = [
        rollout("Train reward, every finished group", "rollout/raw_reward_unfiltered"),
        rollout("Train reward, the trained batch", "rollout/raw_reward"),
        rollout("Exit: submitted", "rollout/env/exit_submitted"),
        rollout("Exit: turn limit", "rollout/env/exit_limit"),
        rollout("Exit: context overflow", "rollout/env/exit_context"),
        rollout("Exit: no tool call", "rollout/env/exit_no_tool_call"),
        rollout("Exit: timeout", "rollout/env/exit_timeout"),
        rollout("Turns per trial", "rollout/env/agent_steps"),
        rollout("Output tokens per turn (thinking + action)", "rollout/env/agent_output_tokens_per_step"),
        rollout("Output tokens per trial", "rollout/env/agent_output_tokens"),
        rollout("Trials with a repeating (highly compressible) window", "rollout/repetition_frac"),
        rollout("Trials ending on a length cap", "rollout/truncated_ratio"),
    ]
    stability = [
        rollout("Mean sampled log-prob (entropy proxy)", "rollout/rollout_log_probs"),
        train("Training entropy (--fp32-lm-head runs)", "train/entropy_loss"),
        train("Engine-trainer log-prob gap (abs diff)", "train/train_rollout_logprob_abs_diff"),
        train("Engine-trainer KL", "train/train_rollout_kl"),
        train("DPPO masked token fraction", "train/dppo_masked_frac"),
        train("Policy KL to the sampling policy (ppo_kl)", "train/ppo_kl"),
        train("Grad norm", "train/grad_norm"),
    ]
    infra = [
        rollout("Step time (s)", "perf/step_time"),
        rollout("Train time a step (s)", "perf/train_time"),
        rollout("Trainer wait fraction", "perf/wait_time_ratio"),
        rollout("Seconds per trial", "rollout/env/trial_s"),
        rollout("Agent seconds per trial", "rollout/env/agent_s"),
        rollout("Exit: error", "rollout/env/exit_error"),
        rollout("Transport failures", "rollout/env/run_transport_failed"),
        rollout("Verifier failures", "rollout/env/verifier_failed"),
        rollout("MTP accept length", "rollout/spec_accept_length"),
        rollout("Token-in-token-out session mismatch", "rollout/tito_session_mismatch_rate/v1"),
    ]
    off_policy = [
        rollout("Mean staleness (weight versions)", "rollout/fully_async/avg_staleness"),
        rollout("Max staleness (weight versions)", "rollout/fully_async/max_staleness"),
        rollout("Stale groups dropped", "rollout/fully_async/stale_groups_filtered"),
        rollout("All-fail groups", "rollout/zero_std/all_zero_percentage"),
        rollout("All-pass groups", "rollout/zero_std/all_one_percentage"),
        rollout("Samples spanning several weight versions", "rollout/weight_version/mixed_version_ratio"),
    ]
    return [
        sec("Tier 0 · Learning curves", learning),
        sec("Tier 1 · Training stability", stability),
        sec("Tier 2 · Env / Infra", infra),
        sec("Tier 3 · Off-policy / Data efficacy", off_policy),
    ]


# ---------------------------------------------------------------- OPD arms
OPD_SECTION = "Tier 4 · OPD arm (RUN_SCRIPT=opd.py)"


def opd_section(slices: list[str]) -> ws.Section:
    """The R2S student arm in the recipe's project (recipes/<env>/opd.py, --algo opd; docs/r2s.md): the
    sampled reverse KL that is its whole loss, the env score it logs but does not train on, the teacher
    scoring health and the deliberation shape of its turns (miles_agent/opd), and the paper's success
    metric per slice. One metric a panel: the arms are read grouped by RUN_TAG, and several metrics in
    one panel of a group differ only by dash pattern."""
    panels = [
        L(title="Reverse KL to the teacher (the loss)", x="train/step", y=["train/opd_reverse_kl"]),
        L(title="Train env reward (logged, zeroed for the loss)", x="rollout/step", y=["rollout/episode_raw_reward"]),
        L(
            title="Teacher scoring failures per episode (alarm > 0: aborted samples, dropped groups)",
            x="rollout/step",
            y=["rollout/env/opd_teacher_scoring_failed"],
        ),
        L(title="Turns per episode", x="rollout/step", y=["rollout/env/opd_turns"]),
        L(title="Deliberation tokens per turn", x="rollout/step", y=["rollout/env/opd_delib_tokens_mean"]),
        L(
            title="Turns ending in an action (fraction)",
            x="rollout/step",
            y=["rollout/env/opd_turns_with_action_frac"],
        ),
        L(title="Teacher scoring latency per episode (s)", x="rollout/step", y=["rollout/env/opd_teacher_latency_s"]),
    ]
    panels += [
        L(title=f"Held-out success rate, {s} (env/won)", x="eval/step", y=[f"eval/{s}/env/won"]) for s in slices
    ]
    return sec(OPD_SECTION, panels)


# ------------------------------------------------------------------- recipes
RECIPES = {
    "search_r1": dict(project="miles-search_r1_run", tier0=tier0_search, async_topology=True, env_health=[]),
    "alfworld": dict(
        project="miles-alfworld_run",
        tier0=tier0_alfworld,
        async_topology=False,
        env_health=["rollout/env/env_worker_respawns", "rollout/env/env_load_retries", "rollout/env/env_pool_hit"],
        opd_slices=ALFWORLD_SLICES,
    ),
    "webshop": dict(
        project="miles-webshop_run",
        tier0=tier0_webshop,
        async_topology=False,
        env_health=[],
        opd_slices=WEBSHOP_SLICES,
    ),
    "scienceworld": dict(
        project="miles-scienceworld_run",
        tier0=tier0_scienceworld,
        async_topology=False,
        env_health=["rollout/env/env_worker_respawns", "rollout/env/env_free_workers"],
        opd_slices=SCIENCEWORLD_SLICES,
    ),
    "tmax": dict(
        project="miles-tmax_run",
        sections=tmax_sections,
        # One curve per RUN_TAG (chain links and restarts are separate wandb runs of one group); smoke runs hidden.
        runset=ws.RunsetSettings(filters=[ws.Config("num_rollout") >= 20], groupby=[ws.Metric("Group")]),
    ),
}


def sections(recipe: str) -> list:
    r = RECIPES[recipe]
    if "sections" in r:
        return r["sections"]()
    a = r["async_topology"]
    tiers = [
        sec("Tier 0 · Learning curves", r["tier0"]()),
        sec("Tier 1 · Off-policy health", tier1(a)),
        sec("Tier 2 · Env / Infra", tier2(a, r["env_health"])),
        sec("Tier 3 · Data efficacy / Algorithm", tier3(a)),
    ]
    if "opd_slices" in r:
        tiers.append(opd_section(r["opd_slices"]))
    return tiers


def build(recipe: str, entity: str) -> ws.Workspace:
    workspace = ws.Workspace(
        entity=entity,
        project=RECIPES[recipe]["project"],
        name=VIEW_NAME,
        sections=sections(recipe),
        settings=ws.WorkspaceSettings(ignore_outliers=True),
    )
    if "runset" in RECIPES[recipe]:
        workspace.runset_settings = RECIPES[recipe]["runset"]
    return workspace


def panel_keys(recipe: str) -> list[str]:
    keys: list[str] = []
    for s in sections(recipe):
        for p in s.panels:
            keys.extend(p.y)
    return sorted(set(keys))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", nargs="+", choices=sorted(RECIPES), required=True)
    ap.add_argument(
        "--update", action="store_true", help="rewrite the view recorded in tools/wandb_views.local.json in place"
    )
    ap.add_argument("--dry-run", action="store_true", help="print sections, panels and metric keys; do not save")
    ap.add_argument("--entity", default=ENTITY, help="the wandb entity owning the projects (default: WANDB_ENTITY)")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.entity:
        print("FATAL: --entity or WANDB_ENTITY is required", file=sys.stderr)
        return 2
    for recipe in args.recipe:
        if args.dry_run:
            print(f"== {recipe} -> {args.entity or '<WANDB_ENTITY>'}/{RECIPES[recipe]['project']} view {VIEW_NAME!r}")
            for s in sections(recipe):
                print(f"  [{s.name}]")
                for p in s.panels:
                    print(f"    {p.title}: x={p.x} y={p.y}")
            print("  keys:", " ".join(panel_keys(recipe)))
            continue
        if args.update:
            url = VIEW_URLS.get(recipe)
            if not url:
                print(f"{recipe}: no URL recorded in {VIEWS_FILE.name}; run once without --update", file=sys.stderr)
                return 2
            workspace = ws.Workspace.from_url(url)
            workspace.name = VIEW_NAME
            workspace.sections = sections(recipe)
            workspace.settings = ws.WorkspaceSettings(ignore_outliers=True)
            if "runset" in RECIPES[recipe]:
                workspace.runset_settings = RECIPES[recipe]["runset"]
        else:
            workspace = build(recipe, args.entity)
        workspace.save()
        if not args.update:  # record the new view's URL for --update
            views = _load_views()
            views[recipe] = workspace.url
            VIEWS_FILE.write_text(json.dumps(views, indent=2, sort_keys=True) + "\n")
        print(f"{recipe}: {workspace.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
