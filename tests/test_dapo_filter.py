"""Behavior tests for miles_agent.dapo_filter (the dynamic-sampling filter wrappers).

miles (torch) is not importable on dev boxes: the two miles filter_hub modules
the wrappers import are stubbed with torch-free equivalents for this module.
"""

import dataclasses
import importlib
import statistics
import sys
import types
from types import SimpleNamespace

import pytest

_STUB_KEYS = (
    "miles",
    "miles.rollout",
    "miles.rollout.filter_hub",
    "miles.rollout.filter_hub.base_types",
    "miles.rollout.filter_hub.dynamic_sampling_filters",
)
ARGS = SimpleNamespace(reward_key="train")
ABORTED = "ABORTED"


class _Sample:
    def __init__(self, reward, *, status="COMPLETED", transport_failed=None, metadata=None):
        self.status = status
        self.reward = reward
        self.metadata = {} if metadata is None else metadata
        if transport_failed is not None:
            self.metadata.setdefault("env_metrics", {})["run_transport_failed"] = transport_failed

    def get_reward_value(self, args):
        if self.reward is None:
            return None
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def _r(v):
    return {"env": v, "train": v}


@pytest.fixture(scope="module")
def df():
    saved = {k: sys.modules.get(k) for k in (*_STUB_KEYS, "miles_agent.dapo_filter")}
    pkgs = {}
    for name in ("miles", "miles.rollout", "miles.rollout.filter_hub"):
        m = types.ModuleType(name)
        m.__path__ = []
        pkgs[name] = m
    base_types = types.ModuleType("miles.rollout.filter_hub.base_types")

    @dataclasses.dataclass
    class DynamicFilterOutput:
        keep: bool
        reason: str | None = None

    base_types.DynamicFilterOutput = DynamicFilterOutput
    dsf = types.ModuleType("miles.rollout.filter_hub.dynamic_sampling_filters")

    def _flatten(samples):
        for s in samples:
            if isinstance(s, list):
                yield from s
            else:
                yield s

    def check_no_aborted(args, samples, **kwargs):
        if any(s.status == ABORTED for s in _flatten(samples)):
            return DynamicFilterOutput(keep=False, reason="group_has_aborted")
        return DynamicFilterOutput(keep=True)

    def check_reward_nonzero_std(args, samples, **kwargs):
        rewards = [s.get_reward_value(args) for s in _flatten(samples)]
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        keep = std > 1e-8
        return DynamicFilterOutput(keep=keep, reason=None if keep else f"zero_std_{round(rewards[0], 1)}")

    dsf.check_no_aborted = check_no_aborted
    dsf.check_reward_nonzero_std = check_reward_nonzero_std
    sys.modules.update(pkgs)
    sys.modules["miles.rollout.filter_hub.base_types"] = base_types
    sys.modules["miles.rollout.filter_hub.dynamic_sampling_filters"] = dsf
    sys.modules.pop("miles_agent.dapo_filter", None)
    mod = importlib.import_module("miles_agent.dapo_filter")
    yield mod
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ---------------------------------------------------------------- transport-failure drop


def test_clean_group_is_kept(df):
    group = [_Sample(_r(0.0), transport_failed=0.0), _Sample(_r(1.0), transport_failed=0.0)]
    out = df.check_no_aborted_no_transport_failure(ARGS, group)
    assert out.keep and out.reason is None


def test_env_feedback_is_never_filtered(df):
    """All-zero rewards, many invalid actions, low task_score: legitimate signal, kept."""
    em = {"run_transport_failed": 0.0, "invalid_actions": 9, "task_score": 0.1, "won": 0.0}
    group = [_Sample(_r(0.0), metadata={"env_metrics": dict(em)}) for _ in range(4)]
    assert df.check_no_aborted_no_transport_failure(ARGS, group).keep


def test_one_transport_failure_drops_the_group(df):
    group = [_Sample(_r(1.0), transport_failed=0.0), _Sample(_r(0.0), transport_failed=1.0)]
    out = df.check_no_aborted_no_transport_failure(ARGS, group)
    assert not out.keep and out.reason == "group_has_transport_failure"


def test_transport_failure_detected_in_nested_groups(df):
    group = [[_Sample(_r(1.0), transport_failed=0.0)], [_Sample(_r(0.0), transport_failed=1.0)]]
    assert df.check_no_aborted_no_transport_failure(ARGS, group).reason == "group_has_transport_failure"


def test_missing_env_metrics_means_no_failure(df):
    group = [_Sample(_r(1.0)), _Sample(_r(0.0), metadata={"env_metrics": {}}), _Sample(_r(0.5), metadata={"x": 1})]
    assert df.check_no_aborted_no_transport_failure(ARGS, group).keep


def test_aborted_checked_before_transport(df):
    group = [_Sample(_r(1.0), status=ABORTED, transport_failed=1.0)]
    assert df.check_no_aborted_no_transport_failure(ARGS, group).reason == "group_has_aborted"


# ---------------------------------------------------------------- DAPO variants


def test_no_transport_nonzero_std_chain(df):
    f = df.check_no_aborted_no_transport_failure_nonzero_std
    assert f(ARGS, [_Sample(_r(0.0), transport_failed=1.0), _Sample(_r(1.0))]).reason == "group_has_transport_failure"
    assert f(ARGS, [_Sample(None), _Sample(_r(1.0))]).reason == "group_has_none_reward"
    assert f(ARGS, [_Sample(_r(0.0)), _Sample(_r(0.0))]).reason == "zero_std_0.0"
    assert f(ARGS, [_Sample(_r(10.0)), _Sample(_r(10.0))]).reason == "zero_std_10.0"
    assert f(ARGS, [_Sample(_r(0.0)), _Sample(_r(10.0))]).keep
    assert f(ARGS, [_Sample(_r(1.0), status=ABORTED), _Sample(_r(0.0))]).reason == "group_has_aborted"


def test_check_no_aborted_nonzero_std_unchanged(df):
    f = df.check_no_aborted_nonzero_std
    assert f(ARGS, [_Sample(_r(1.0), status=ABORTED), _Sample(_r(0.0))]).reason == "group_has_aborted"
    assert f(ARGS, [_Sample(None), _Sample(_r(1.0))]).reason == "group_has_none_reward"
    assert f(ARGS, [_Sample(_r(1.0)), _Sample(_r(1.0))]).reason == "zero_std_1.0"
    assert f(ARGS, [_Sample(_r(0.0)), _Sample(_r(1.0))]).keep
    # a transport failure is NOT a criterion for this (alfworld/search_r1 dapo) filter
    assert f(ARGS, [_Sample(_r(0.0), transport_failed=1.0), _Sample(_r(1.0))]).keep


def test_reward_key_resolution(df):
    scalar_args = SimpleNamespace(reward_key=None)
    group = [_Sample(0.0), _Sample(1.0)]
    assert df.check_no_aborted_nonzero_std(scalar_args, group).keep
    assert df.check_no_aborted_nonzero_std(scalar_args, [_Sample(0.5), _Sample(0.5)]).reason == "zero_std_0.5"
