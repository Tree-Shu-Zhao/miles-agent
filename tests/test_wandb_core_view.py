"""tools/wandb_core_view.py: the OPD arm's pinned section of the env recipes' views (docs/r2s.md), one metric a
panel, and the GRPO-only views left as they were.

wandb_workspaces is not installed on dev boxes: the surface the tool touches (LinePlot / Section / Workspace and
its settings, the runset filter expressions) is stood in by plain attribute records for the module load, and a
dry run saves nothing. Under the real package the same test runs unstubbed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# design §2.8: the arm's keys; plus eval/<slice>/env/won per slice of the recipe
OPD_KEYS = {
    "train/opd_reverse_kl",
    "rollout/episode_raw_reward",
    "rollout/env/opd_teacher_scoring_failed",
    "rollout/env/opd_turns",
    "rollout/env/opd_delib_tokens_mean",
    "rollout/env/opd_turns_with_action_frac",
    "rollout/env/opd_teacher_latency_s",
}
# recipes/<env>/run.py eval_slices (webshop: the default canonical protocol)
SLICES = {
    "alfworld": ["alfworld_seen", "alfworld_unseen"],
    "webshop": ["webshop", "webshop_pool"],
    "scienceworld": ["scienceworld_seen", "scienceworld_unseen"],
}
_FAKE_KEYS = (
    "wandb_workspaces",
    "wandb_workspaces.reports",
    "wandb_workspaces.reports.v2",
    "wandb_workspaces.workspaces",
)


def _record(name: str) -> type:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)

    return type(name, (), {"__init__": __init__})


class _Config:
    """ws.Config("num_rollout") >= 20, the runset filter the tmax entry builds at import."""

    def __init__(self, key: str):
        self.key = key

    def __ge__(self, other):
        return (self.key, ">=", other)


def _fake_wandb_workspaces() -> dict[str, types.ModuleType]:
    pkg, reports, v2, workspaces = (types.ModuleType(k) for k in _FAKE_KEYS)
    v2.LinePlot = _record("LinePlot")
    for name in ("Section", "Workspace", "WorkspaceSettings", "RunsetSettings", "Metric"):
        setattr(workspaces, name, _record(name))
    workspaces.Config = _Config
    pkg.reports, pkg.workspaces, reports.v2 = reports, workspaces, v2
    return dict(zip(_FAKE_KEYS, (pkg, reports, v2, workspaces), strict=True))


@pytest.fixture(scope="module")
def view():
    fakes = {} if importlib.util.find_spec("wandb_workspaces") else _fake_wandb_workspaces()
    saved = {key: sys.modules.get(key) for key in fakes}
    sys.modules.update(fakes)
    try:
        spec = importlib.util.spec_from_file_location("wandb_core_view", REPO / "tools" / "wandb_core_view.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def test_wandb_core_view_opd_keys(view, capsys):
    for recipe, slices in SLICES.items():
        sections = view.sections(recipe)
        assert [s.name for s in sections][:4] == [s.name for s in view.sections("search_r1")][:4], recipe
        opd = [s for s in sections if s.name == view.OPD_SECTION]
        assert len(opd) == 1 and sections[-1] is opd[0], recipe
        panels = opd[0].panels
        assert all(len(p.y) == 1 for p in panels), [p.title for p in panels if len(p.y) != 1]
        assert {p.y[0] for p in panels} == OPD_KEYS | {f"eval/{s}/env/won" for s in slices}, recipe
        assert len(panels) == len(OPD_KEYS) + len(slices), recipe  # no key in two panels
        assert OPD_KEYS <= set(view.panel_keys(recipe)), recipe
    # the views without an OPD preset are unchanged: no section, none of the arm's own keys (the raw train
    # reward is a Tier 0 key of search_r1 already)
    for recipe in ("search_r1", "tmax"):
        assert all(s.name != view.OPD_SECTION for s in view.sections(recipe)), recipe
        assert not (OPD_KEYS - {"rollout/episode_raw_reward"}) & set(view.panel_keys(recipe)), recipe
    # the dry run lists every OPD panel with its one key and saves nothing
    assert view.main(["--recipe", "alfworld", "scienceworld", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count(f"[{view.OPD_SECTION}]") == 2
    assert "y=['train/opd_reverse_kl']" in out
    assert "y=['eval/alfworld_unseen/env/won']" in out and "y=['eval/scienceworld_unseen/env/won']" in out
    assert "y=['rollout/env/opd_turns', " not in out
