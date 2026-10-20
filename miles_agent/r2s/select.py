"""Stage C of the R2S pipeline: the per-env table and the one selection rule.

The paper's rule (4_method.tex:16; x_appendix_analysis_method.tex:62 for tab:main): a state's
takeovers pass a check that tells a finishing takeover from a failing one, the passing ones are
grouped by how they end, the state is kept when the largest group holds at least m of them, and
the shortest takeover of that group (then the lowest sample) is tau. Two checks, one function:
``vote`` admits the takeovers that ended inside the turn budget (ScienceWorld: on the simulator's
completion message, ``done_regex``) and groups them by the terminal observation; ``judge`` admits
the ones stage J approved (WebShop: each purchase read against the instruction) and groups them
by the item bought (``purchased_asin``). ENV_DEFAULTS are the campaign's values
(the campaign branch's submit script): ALFWorld m=2, ScienceWorld
m=1 with "Task completed.", WebShop the judge with m=1 and no label gate; takeovers with a
format-faulted step admitted everywhere (ALLOW_INVALID 1). An override is --env-config (a json
table laid over ENV_DEFAULTS) or a flag, never a new default, so a selection run alone is a
config change. No gate here: the paper gated the labeled fraction at stage D.

    python -m miles_agent.r2s.select --env alfworld --fold fold.jsonl --out tau.jsonl
    python -m miles_agent.r2s.select --env webshop --fold fold.jsonl --judge-file judge.jsonl --out tau.jsonl

Rows {state_id, sample, group_size, n_eligible}; summary {n_states, n_selected, coverage, n_takeovers,
eligible_total, judge_missing, config, SELECT_DONE}, the counts present on an empty fold too. label.py reads
--tau or calls select_tau inline with the same config.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from miles_agent.r2s import common

MARKER = "SELECT_DONE"
CHECKS = ("vote", "judge")
VOTE_KEYS = ("terminal", "purchased_asin")

# Stage knobs the campaign never changed (REF's stage CLIs).
TOP_N = 2400  # A: states kept after the per-goal cap
MAX_PER_GOAL = 3
MIN_DELIB_TOKENS = 3
MIN_DIV_TURN = 0
K = 8  # B: takeovers per state
TAKEOVER_T = 1.0
TOP_P = 1.0
MAX_TEACHER_TOKENS = 512
LABEL_K = 16  # D: top-k targets per position
MAX_CTX = 32768
MIN_STATES = 100  # the job's gate after A
JUDGE_THINKING = True  # J: Qwen3.5-9B with reasoning mode on (appendix)
JUDGE_MAX_TOKENS = 2048
JUDGE_CONCURRENCY = 64
JUDGE_T = 0.0
FT_EPOCHS, FT_LR, FT_GRAD_ACCUM, FT_WARMUP_STEPS = 1, 2e-6, 8, 10  # E: REF teacher_ft.py:206-213
FT_MIN_LR_RATIO, FT_CLIP, FT_VAL_PCT, SEED = 0.1, 1.0, 10, 42


@dataclass(frozen=True)
class EnvConfig:
    """The per-env instantiation of the rule, with the env facts the other stages read."""

    goal_keys: tuple[str, ...]  # the per-goal cap key (stage A) and the FT split key (stage E), D17
    action_tool: str = "act"  # the env's action tool (stage B's command rule)
    vote_key: str = "terminal"  # how takeovers are grouped: terminal-observation fingerprint | purchased ASIN
    check: str = "vote"  # vote (the eligibility below) | judge (stage J's match_ok)
    m: int = 2  # the largest group must hold m takeovers
    done_regex: str = ""  # vote: the terminal observation must match (ScienceWorld's completion message)
    label_frac_gate: float = 0.25  # stage D: LABEL_FAIL below this labeled fraction
    allow_invalid: bool = True  # a takeover with a format-faulted step may still be tau (ALLOW_INVALID 1)
    require_action: bool = True  # stage A: the divergence turn must end in a tool call
    max_steps: int = 30  # stage B's turn budget for a row without metadata["max_episode_steps"] (D16)

    def __post_init__(self) -> None:
        if self.check not in CHECKS or self.vote_key not in VOTE_KEYS or self.m < 1:
            raise ValueError(f"check in {CHECKS}, vote_key in {VOTE_KEYS} and m >= 1; got {self}")


ENV_DEFAULTS = {
    "alfworld": EnvConfig(
        ("game_file",), vote_key="terminal", check="vote", m=2, done_regex="", label_frac_gate=0.25, max_steps=30
    ),
    "scienceworld": EnvConfig(
        ("task_name", "var_num"),
        vote_key="terminal",
        check="vote",
        m=1,
        done_regex="Task completed.",  # the campaign's exact export (dot unescaped); "Task finished." is a failure
        label_frac_gate=0.25,
        max_steps=30,
    ),
    "webshop": EnvConfig(
        ("goal_id",), vote_key="purchased_asin", check="judge", m=1, done_regex="", label_frac_gate=0.0, max_steps=15
    ),
}


def load_env_config(env: str, env_config_path: str | None = None, **overrides: Any) -> EnvConfig:
    """ENV_DEFAULTS[env], then the --env-config table's entry for ``env`` (json {env: {field: value}};
    an env the table alone defines needs its goal_keys), then the flags that were given (None is
    "not given"). A field no EnvConfig has is an error, not a silent no-op."""
    values = dataclasses.asdict(ENV_DEFAULTS[env]) if env in ENV_DEFAULTS else {}
    if env_config_path:
        with open(env_config_path, encoding="utf-8") as f:
            values.update(json.load(f).get(env) or {})
    values.update({k: v for k, v in overrides.items() if v is not None})
    if not values:
        raise ValueError(f"unknown env {env!r}: ENV_DEFAULTS has {sorted(ENV_DEFAULTS)}; --env-config can add one")
    unknown = sorted(set(values) - {f.name for f in dataclasses.fields(EnvConfig)})
    if unknown:
        raise ValueError(f"unknown EnvConfig field(s) {unknown} for env {env!r}")
    keys = values.get("goal_keys")
    if not keys:  # the per-goal cap and the FT split have no default key
        raise ValueError(f"env {env!r} needs goal_keys (--env-config)")
    values["goal_keys"] = (keys,) if isinstance(keys, str) else tuple(keys)  # json lists; a lone key as a string
    return EnvConfig(**values)


def goal_of(metadata: Mapping[str, Any], cfg: EnvConfig) -> str:
    """The per-goal key of stage A's cap and stage E's split: the env's goal_keys joined with "/"
    (ScienceWorld rows carry task_name and var_num and no task_id, D17; REF common.py:43-53)."""
    missing = [k for k in cfg.goal_keys if metadata.get(k) is None]
    if missing:
        raise KeyError(f"metadata lacks {missing} (goal_keys {cfg.goal_keys})")
    return "/".join(str(metadata[k]) for k in cfg.goal_keys)


def eligible(tau: Mapping[str, Any], cfg: EnvConfig) -> bool:
    """A takeover the rule considers before the env's check (REF label.py:78-92): replayed, done, a
    terminal key, at least one step, every step format_ok unless allow_invalid, and the done_regex on
    the terminal observation. Stage J judges exactly this set, so a takeover without a verdict is a
    lost call, not a rule gap."""
    if not (tau.get("replay_ok") and tau.get("done") and tau.get("terminal_key")):
        return False
    steps = tau.get("takeover_steps") or []
    if not steps:
        return False
    if not cfg.allow_invalid and any(not st.get("format_ok") for st in steps):
        return False
    return not cfg.done_regex or bool(re.search(cfg.done_regex, tau.get("last_observation") or ""))


def select_tau(
    group: list[dict[str, Any]], cfg: EnvConfig, judge_rows: Mapping[tuple[Any, Any], Mapping[str, Any]] | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """tau of one state's takeovers (stage B rows), or None, with the REF stats. The ordering is
    REF label.py:66-109 pick_tau_vote_terminal's byte for byte: ``eligible``, then the judge's
    approval when ``check`` is judge (``judge_rows`` keyed (state_id, sample); a missing row is not
    an approval and is counted, F9), then the vote by terminal_key: the modal key needs m votes,
    ties go to the greater key, and the shortest takeover of the modal group (then the lowest
    sample) is tau."""
    ok = [t for t in group if t.get("replay_ok")]
    stats: dict[str, Any] = {
        "n_group": len(ok),
        "p_won": (sum(bool(t.get("won")) for t in ok) / len(ok)) if ok else 0.0,
    }
    elig = [t for t in ok if eligible(t, cfg)]
    stats["vote_n_eligible"] = len(elig)
    if cfg.check == "judge":
        if judge_rows is None:
            raise ValueError("check 'judge' needs the judge rows (stage J, --judge-file)")
        stats["judge_missing"] = sum((t["state_id"], t["sample"]) not in judge_rows for t in elig)
        elig = [t for t in elig if judge_rows.get((t["state_id"], t["sample"]), {}).get("match_ok", False)]
        stats["judge_n_ok"] = len(elig)
    votes = collections.Counter(t["terminal_key"] for t in elig)
    if not votes or max(votes.values()) < cfg.m:
        return None, stats
    modal, cnt = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
    stats["vote_modal_count"] = cnt
    cands = [t for t in elig if t["terminal_key"] == modal]
    return min(cands, key=lambda t: (len(t["takeover_steps"]), t.get("sample", 0))), stats


def main(argv: list[str] | None = None) -> int:
    ap = common.stage_parser(__doc__)
    ap.add_argument("--fold", required=True, help="stage B rows")
    ap.add_argument("--check", choices=CHECKS, help="override the env's check")
    ap.add_argument("--m", type=int, help="override the env's m")
    ap.add_argument("--done-regex", help="override the env's done_regex (vote)")
    ap.add_argument("--judge-file", help="stage J rows (check judge)")
    args = common.parse_stage_args(ap, argv)
    cfg = load_env_config(args.env, args.env_config, check=args.check, m=args.m, done_regex=args.done_regex)
    if common.skip_if_done(args, MARKER):
        return 0
    if cfg.check == "judge" and not args.judge_file:
        raise SystemExit(f"--env {args.env} selects with check 'judge': --judge-file is required")
    if common.stop_requested(args.stop_file):
        print(f"stop file {args.stop_file} present before selection; exiting without {MARKER}")
        return 0
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in common.read_jsonl(args.fold):
        groups.setdefault(row["state_id"], []).append(row)
    if args.limit is not None:
        groups = dict(list(groups.items())[: args.limit])
    judge_rows = (
        {(v["state_id"], v["sample"]): v for v in common.read_jsonl(args.judge_file)} if args.judge_file else None
    )
    rows = []
    totals = collections.Counter(n_takeovers=0, eligible_total=0, judge_missing=0)  # written on an empty fold too
    for state_id, group in groups.items():
        tau, stats = select_tau(group, cfg, judge_rows)
        totals["n_takeovers"] += len(group)
        totals["eligible_total"] += stats["vote_n_eligible"]
        totals["judge_missing"] += stats.get("judge_missing", 0)
        if tau is None:
            continue
        rows.append(
            {
                "state_id": state_id,
                "sample": tau["sample"],
                "group_size": len(group),
                "n_eligible": stats.get("judge_n_ok", stats["vote_n_eligible"]),
            }
        )
    common.write_jsonl(args.out, rows)
    summary = {
        "n_states": len(groups),
        "n_selected": len(rows),
        "coverage": len(rows) / len(groups) if groups else None,
        **totals,
        "config": dataclasses.asdict(cfg),
    }
    common.write_summary(args.summary, summary, MARKER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
