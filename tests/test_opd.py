"""miles_agent/opd: the generate hook, the teacher scoring, turns, the pilot dump row and the zero-shaping plugin.

miles (torch) is not importable on dev boxes, so the miles surface the hook builds on - Sample (utils/types.py),
GenerateFnInput/GenerateFnOutput (rollout/base_types.py), the agentic generate it wraps and on_policy_distillation's
scoring helpers - is stubbed in sys.modules for this module only when the real one is missing, field for field;
tests/test_contracts.py pins that surface against the pin wherever miles imports. The hook runs with a fake inner
generate, the test vocabulary's marker and a fake post_json.
"""

import argparse
import asyncio
import dataclasses
import importlib
import importlib.util
import inspect
import json
import logging
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from test_tool_call_markup import ACT_FLAT, FakeTokenizer, tito_stub  # noqa: F401  (the fixture joins this module)

from miles_agent import gym_bridge
from miles_agent.opd import dump, turns, zero_shape

REPO = Path(__file__).resolve().parent.parent
TEACHER = "http://t:13141/generate"
MARKER = 7  # the call-opening token of the test vocabulary (a launch derives its own, test_marker_id_*)
PROMPT = [{"role": "system", "content": "sys"}]
TRANSCRIPT = [
    *PROMPT,
    {"role": "user", "content": "obs0"},
    {
        "role": "assistant",
        "content": "Going north.",
        "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "act", "arguments": '{"command": "go north"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "tc1", "content": "obs1"},
]
# prompt 3 tokens; response = turn 1 (3 deliberation tokens, the marker, 2 more) | observation | turn 2 (marker first)
PROMPT_IDS = [1, 2, 3]
RESPONSE = [10, 11, 12, MARKER, 20, 21, 90, 91, 92, MARKER, 30, 31]
LOSS_MASK = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
TURNS = [[0, 6, 3], [9, 12, 9]]


def _miles_importable() -> bool:
    try:
        import miles.utils.types  # noqa: F401
    except ImportError:
        return False
    return True


MILES_IMPORTABLE = _miles_importable()
_STUB_KEYS = (
    "miles",
    "miles.utils",
    "miles.utils.types",
    "miles.utils.misc",
    "miles.rollout",
    "miles.rollout.base_types",
    "miles.rollout.generate_hub",
    "miles.rollout.generate_hub.agentic_tool_call",
    "miles.rollout.on_policy_distillation",
)
_HOOK_MODULES = ("miles_agent.opd.teacher_scoring", "miles_agent.opd.generate")


def _make_stubs() -> dict[str, types.ModuleType]:
    """The miles surface the hook touches, mirrored from the pin: utils/types.py Sample (the fields the hook and the
    dump read, Status, validate's length checks), utils/misc.py load_function (minus the registry),
    rollout/base_types.py GenerateFnInput/Output, generate_hub/agentic_tool_call.py generate (+ add_arguments),
    rollout/on_policy_distillation.py parse_teacher_urls / _score_payload / _post_json / _trim_input_field."""
    pkgs = {}
    for name in ("miles", "miles.utils", "miles.rollout", "miles.rollout.generate_hub"):
        m = types.ModuleType(name)
        m.__path__ = []
        pkgs[name] = m

    @dataclasses.dataclass
    class Sample:
        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"
            FAILED = "failed"

        group_index: int | None = None
        index: int | None = None
        rollout_id: int | None = None
        prompt: str | list = ""
        tokens: list = dataclasses.field(default_factory=list)
        response: str = ""
        response_length: int = 0
        reward: float | dict | None = None
        loss_mask: list | None = None
        rollout_log_probs: list | None = None
        teacher_log_probs: list | None = None
        opd_reverse_kl: list | None = None
        status: Status = Status.PENDING
        metadata: dict = dataclasses.field(default_factory=dict)

        def validate(self):
            if self.loss_mask is not None:
                assert len(self.loss_mask) == self.response_length
            if self.teacher_log_probs is not None:
                assert len(self.teacher_log_probs) == self.response_length

    types_mod = types.ModuleType("miles.utils.types")
    types_mod.Sample = Sample

    misc = types.ModuleType("miles.utils.misc")

    def load_function(path, *, sync_required=False):
        if not path:
            return None
        module_path, _, attr = path.rpartition(".")
        return getattr(importlib.import_module(module_path), attr)

    misc.load_function = load_function

    @dataclasses.dataclass(frozen=True)
    class GenerateFnInput:
        state: object
        sample: Sample
        sampling_params: dict
        evaluation: bool

        @property
        def args(self):
            return self.state.args

    @dataclasses.dataclass(frozen=True)
    class GenerateFnOutput:
        samples: object

    base_types = types.ModuleType("miles.rollout.base_types")
    base_types.GenerateFnInput, base_types.GenerateFnOutput = GenerateFnInput, GenerateFnOutput

    agentic = types.ModuleType("miles.rollout.generate_hub.agentic_tool_call")

    async def generate(input):
        raise AssertionError("the tests install their own inner generate")

    def _add_arguments(parser):
        parser.add_argument("--custom-agent-function-path", type=str)
        parser.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len")

    generate.add_arguments = _add_arguments
    agentic.generate = generate

    opd = types.ModuleType("miles.rollout.on_policy_distillation")

    def parse_teacher_urls(values):
        url_map = {}
        for value in values or []:
            name, sep, url = value.partition("=")
            name, url = name.strip(), url.strip()
            if not sep or not name or not url:
                raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL.")
            if name in url_map:
                raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
            url_map[name] = url
        return url_map

    def _score_payload(input_ids, top_k=0, token_ids=None, token_ids_positions=None):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        if top_k > 0:
            payload["top_logprobs_num"] = top_k
        return payload

    async def _post_json(url, payload, timeout_secs=None):
        raise AssertionError("the tests fake teacher_scoring.post_json or this transport")

    def _trim_input_field(meta_info, field, response_length):
        values = meta_info.get(field)
        if values is None:
            raise ValueError(f"Teacher response is missing meta_info.{field}.")
        return values[1:][-response_length:] if response_length > 0 else []

    opd.parse_teacher_urls, opd._score_payload = parse_teacher_urls, _score_payload
    opd._post_json, opd._trim_input_field = _post_json, _trim_input_field

    pkgs["miles"].utils, pkgs["miles"].rollout = pkgs["miles.utils"], pkgs["miles.rollout"]
    pkgs["miles.utils"].types, pkgs["miles.utils"].misc = types_mod, misc
    pkgs["miles.rollout"].base_types, pkgs["miles.rollout"].on_policy_distillation = base_types, opd
    pkgs["miles.rollout"].generate_hub = pkgs["miles.rollout.generate_hub"]
    pkgs["miles.rollout.generate_hub"].agentic_tool_call = agentic
    return {
        **pkgs,
        "miles.utils.types": types_mod,
        "miles.utils.misc": misc,
        "miles.rollout.base_types": base_types,
        "miles.rollout.generate_hub.agentic_tool_call": agentic,
        "miles.rollout.on_policy_distillation": opd,
    }


@pytest.fixture(scope="module")
def miles_surface():
    """The real miles where it imports, else the stubs above, installed for this module; the hook modules are
    re-imported under it (and dropped after) so every test sees one Sample class."""
    saved = {k: sys.modules.get(k) for k in (*_STUB_KEYS, *_HOOK_MODULES)}
    if not MILES_IMPORTABLE:
        sys.modules.update(_make_stubs())
    for name in _HOOK_MODULES:
        sys.modules.pop(name, None)
    yield
    restore = saved if not MILES_IMPORTABLE else {k: saved[k] for k in _HOOK_MODULES}
    for k, v in restore.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture(scope="module")
def hook(miles_surface):
    # teacher_scoring first: generate's `from miles_agent.opd import teacher_scoring` takes the package attribute,
    # which the pop above leaves stale when another module imported it at collection (test_r2s via r2s.divergence)
    teacher_scoring = importlib.import_module("miles_agent.opd.teacher_scoring")
    generate = importlib.import_module("miles_agent.opd.generate")
    from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
    from miles.utils.types import Sample

    return SimpleNamespace(
        generate=generate,
        teacher_scoring=teacher_scoring,
        agentic=generate.agentic_tool_call,
        Sample=Sample,
        GenerateFnInput=GenerateFnInput,
        GenerateFnOutput=GenerateFnOutput,
    )


def _reply(tokens, *, k=0, drop_first_none=False, wrong_id_at=None):
    """A /generate reply for ``tokens`` (the sglang contract: position 0 None, [logprob, id, text] after; the
    log-prob at input position p is -0.5 - p)."""
    token_lps, top_lps = [None], [None]
    for pos in range(1, len(tokens)):
        tid = tokens[pos] if wrong_id_at != pos else tokens[pos] + 1
        token_lps.append([-0.5 - pos, tid, "x"])
        top_lps.append([[-0.1 * j - pos, 1000 + j + pos, "t"] for j in range(k)])
    if drop_first_none:
        token_lps, top_lps = token_lps[1:], top_lps[1:]
    meta = {"input_token_logprobs": token_lps}
    if k:
        meta["input_top_logprobs"] = top_lps
    return {"meta_info": meta}


class FakeTeacher:
    """teacher_scoring.post_json stand-in: position-aligned log-probs for the payload's input_ids, or the error it
    is told to raise."""

    def __init__(self):
        self.calls, self.error, self.wrong_id_at = [], None, None

    async def __call__(self, url, payload, **kwargs):
        self.calls.append((url, payload, kwargs))
        if self.error is not None:
            raise self.error
        return _reply(payload["input_ids"], k=payload.get("top_logprobs_num", 0), wrong_id_at=self.wrong_id_at)


@pytest.fixture
def teacher(hook, monkeypatch):
    fake = FakeTeacher()
    monkeypatch.setattr(hook.teacher_scoring, "post_json", fake)
    return fake


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_OPD_DUMP_DIR", str(tmp_path))
    yield tmp_path
    for handle in dump._handles.values():
        handle.close()
    dump._handles.clear()


def _rows(directory: Path) -> list[dict]:
    files = sorted(directory.glob("dump_*.jsonl"))
    return [json.loads(line) for path in files for line in path.read_text().splitlines()]


def _args(*, use_opd=True, teacher=TEACHER):
    """input.args as miles parses them for an opd launch: --use-opd, --opd-teacher-urls default=<url>, the hook's
    own flags at their defaults, --tito-model / --hf-checkpoint / --apply-chat-template-kwargs (folded dict)."""
    return SimpleNamespace(
        use_opd=use_opd,
        opd_teacher_urls=[f"default={teacher}"] if teacher else None,
        opd_score_concurrency=32,
        opd_score_timeout_s=900.0,
        opd_score_attempts=4,
        tito_model="qwen35",
        hf_checkpoint="/models/Qwen3.5-4B",
        apply_chat_template_kwargs={"enable_thinking": False},
    )


def _sample(hook, *, index=1, status="completed", tokens=None, response_length=None, loss_mask=None, transcript=None):
    metadata = {
        "game_file": "g",
        "split": "train",
        "max_episode_steps": 30,
        "tools": [ACT_FLAT],
        "reward": 1.0,
        "env_metrics": {"won": 1.0},
        "session_server_id": "1.2.3.4:7100",
    }
    if transcript is not None:
        metadata[gym_bridge.TRANSCRIPT_KEY] = transcript
    full = tokens is None
    return hook.Sample(
        index=index,
        rollout_id=index,
        prompt=PROMPT,
        tokens=PROMPT_IDS + RESPONSE if full else tokens,
        response_length=len(RESPONSE) if full and response_length is None else (response_length or 0),
        loss_mask=list(LOSS_MASK) if full else loss_mask,
        status=hook.Sample.Status(status),
        metadata=metadata,
    )


def _input(hook, sample, *, evaluation=False, **args):
    return hook.GenerateFnInput(
        state=SimpleNamespace(args=_args(**args)),
        sample=sample,
        sampling_params={"temperature": 0.4, "max_new_tokens": 512},
        evaluation=evaluation,
    )


def _drive(hook, monkeypatch, produced):
    """The fake inner generate (returns ``produced``, one Sample or a list as miles' v1/v2 paths do) and the test
    vocabulary's marker; returns the inputs the inner generate saw."""
    seen = []

    async def inner(input):
        seen.append(input)
        return hook.GenerateFnOutput(samples=produced)

    monkeypatch.setattr(hook.agentic, "generate", inner)
    monkeypatch.setattr(hook.generate, "marker_id", lambda args, tools: MARKER)
    return seen


# ---------------------------------------------------------------- turns


def test_segment_turns_marks_delib_end_at_marker():
    got = turns.segment_turns(RESPONSE, LOSS_MASK, MARKER)
    assert [[t.start, t.end, t.delib_end] for t in got] == TURNS
    assert [t.delib_tokens for t in got] == [3, 0] and [t.length for t in got] == [6, 3]
    assert all(t.has_action for t in got)
    assert turns.turn_metrics(got) == {
        "opd_turns": 2.0,
        "opd_delib_tokens_mean": 1.5,
        "opd_turns_with_delib_frac": 0.5,
        "opd_turns_with_action_frac": 1.0,
    }
    # a turn without the marker is all deliberation (a format fault); any other id is just a token
    got = turns.segment_turns([5, 6, 8, 40, 9, 9], [1, 1, 1, 0, 1, 1], MARKER)
    assert [[t.start, t.end, t.delib_end] for t in got] == [[0, 3, 3], [4, 6, 6]]
    assert not any(t.has_action for t in got) and turns.turn_metrics(got)["opd_turns_with_action_frac"] == 0.0
    # no loss_mask: one turn over the whole response; no turns: zeros
    assert turns.segment_turns([1, MARKER, 2], None, MARKER) == [turns.Turn(0, 3, 1)]
    assert turns.turn_metrics([]) == {
        "opd_turns": 0.0,
        "opd_delib_tokens_mean": 0.0,
        "opd_turns_with_delib_frac": 0.0,
        "opd_turns_with_action_frac": 0.0,
    }
    with pytest.raises(AssertionError, match="loss_mask length"):
        turns.segment_turns([1, 2], [1], MARKER)


class _Status(Enum):
    COMPLETED = "completed"


# A campaign-style dump row (REF miles_agent/opd/dump.py keys) in the test vocabulary: three turns, the second a
# format fault (no call), the third a call without deliberation.
CAMPAIGN_ROW = {
    "rollout_id": 4,
    "index": 4,
    "status": "completed",
    "reward": 1.0,
    "metadata": {"game_file": "g", "split": "train", "max_episode_steps": 30, "tools": [ACT_FLAT]},
    "env_metrics": {"won": 1.0, "step_count": 3.0},
    "tokens": [1, 2, 3, 10, 11, 12, MARKER, 20, 21, 90, 91, 92, 30, 31, 32, 93, MARKER, 40, 41],
    "response_length": 16,
    "loss_mask": [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    "turns": [[0, 6, 3], [9, 12, 12], [13, 16, 13]],
}


def test_segment_turns_matches_campaign_row(dump_dir):
    """The stored turns of a campaign-style row are what segment_turns yields from its tokens/loss_mask with the
    marker, and dump_row writes the triple back as stored (byte-equal rows across the port)."""
    row = CAMPAIGN_ROW
    got = turns.segment_turns(row["tokens"][-row["response_length"] :], row["loss_mask"], MARKER)
    assert [[t.start, t.end, t.delib_end] for t in got] == row["turns"]
    assert [t.has_action for t in got] == [True, False, True] and [t.delib_tokens for t in got] == [3, 3, 0]
    sample = SimpleNamespace(
        rollout_id=4,
        index=4,
        status=_Status.COMPLETED,
        prompt=PROMPT,
        tokens=row["tokens"],
        response_length=row["response_length"],
        loss_mask=row["loss_mask"],
        metadata={**row["metadata"], "reward": row["reward"], "env_metrics": row["env_metrics"]},
    )
    dump.dump_row(sample, got, None, _args())
    (written,) = _rows(dump_dir)
    assert {k: written[k] for k in row} == row
    assert json.dumps(written["turns"]) == json.dumps(row["turns"])


# ---------------------------------------------------------------- teacher scoring


def test_teacher_log_probs_alignment_and_zero_fill(hook):
    ts = hook.teacher_scoring
    tokens = [5, 6, 7, 8, 9, 10]  # prompt 2, response 4: response position r is input position 2 + r
    got = ts.sampled_logprobs(_reply(tokens), tokens, 4, [1, 0, 1, 1])
    assert got == [-2.5, 0.0, -4.5, -5.5]
    assert ts.sampled_logprobs(_reply(tokens, drop_first_none=True), tokens, 4, [1, 0, 1, 1]) == got
    assert ts.sampled_logprobs(_reply(tokens), tokens, 4, None) == [-2.5, -3.5, -4.5, -5.5]
    # the whole stream as the response: input position 0 has no prediction, so 0.0 rather than an error
    assert ts.sampled_logprobs(_reply(tokens), tokens, 6, None) == [0.0, -1.5, -2.5, -3.5, -4.5, -5.5]
    assert ts.sampled_logprobs(_reply(tokens), tokens, 0, []) == []
    for bad in ({}, {"meta_info": {}}, {"meta_info": {"input_token_logprobs": [None] * 4}}):
        with pytest.raises(ts.TeacherScoringError):
            ts.sampled_logprobs(bad, tokens, 4, None)
    with pytest.raises(ts.TeacherScoringError, match="teacher token id None"):
        ts.sampled_logprobs({"meta_info": {"input_token_logprobs": [None] * 6}}, tokens, 4, None)


async def test_teacher_log_probs_rejects_token_mismatch(hook, teacher, monkeypatch):
    """The teacher's token id must be the student's at every scored position (a tokenizer or template drift is
    never trained on); prefill, the pipeline's prefill, checks every position >= 1 of its own ids."""
    ts = hook.teacher_scoring
    tokens = [5, 6, 7, 8]
    with pytest.raises(ts.TeacherScoringError, match="teacher token id 9 != student token 8 at position 3"):
        ts.sampled_logprobs(_reply(tokens, wrong_id_at=3), tokens, 2, None)
    teacher.wrong_id_at = 1
    with pytest.raises(ts.TeacherScoringError, match="server token 7 != ours 6 at position 1"):
        await ts.prefill("http://s:13142/generate", tokens)
    teacher.wrong_id_at = None
    plain = await ts.prefill("http://s:13142/generate", tokens)
    assert plain == {"lps": [None, -1.5, -2.5, -3.5], "top": [None] * 4}
    teacher.calls.clear()
    top = await ts.prefill("http://s:13142/generate", tokens, top_k=2, timeout_s=30.0)
    assert top["lps"] == plain["lps"] and top["top"][0] is None and [len(t) for t in top["top"][1:]] == [2, 2, 2]
    ((url, payload, kwargs),) = teacher.calls
    assert url == "http://s:13142/generate" and payload["top_logprobs_num"] == 2 and kwargs == {"timeout_s": 30.0}
    partial_reply = _reply(tokens)
    partial_reply["meta_info"]["input_token_logprobs"][2] = None

    async def partial(url, payload, **kwargs):
        return partial_reply

    monkeypatch.setattr(hook.teacher_scoring, "post_json", partial)
    with pytest.raises(ts.TeacherScoringError, match="no logprob at position 2"):
        await ts.prefill("http://s:13142/generate", tokens)


def test_sampled_logprobs_trim_matches_miles(hook):
    """On the scored positions sampled_logprobs reads the numbers miles' own trim of the reply yields (the stock OPD
    reward's _trim_input_field(...)[r][0]); the masked positions are 0.0."""
    if not MILES_IMPORTABLE:
        pytest.skip("miles (and torch) not importable here")
    from miles.rollout.on_policy_distillation import _trim_input_field

    tokens = list(range(100, 112))
    response_length, loss_mask = 7, [1, 1, 0, 1, 0, 1, 1]
    reply = _reply(tokens)
    ours = hook.teacher_scoring.sampled_logprobs(reply, tokens, response_length, loss_mask)
    theirs = [e[0] for e in _trim_input_field(reply["meta_info"], "input_token_logprobs", response_length)]
    assert len(theirs) == response_length
    scored = [o for o, m in zip(ours, loss_mask, strict=True) if m]
    assert scored == [t for t, m in zip(theirs, loss_mask, strict=True) if m]
    assert [o for o, m in zip(ours, loss_mask, strict=True) if not m] == [0.0, 0.0]


async def test_post_json_retries_transport_and_fails_fast_on_4xx(hook, monkeypatch):
    """miles' _post_json under our retry policy: transport failures, timeouts and 5xx are retried, a 4xx is final and
    a 400 is the teacher's context-window refusal (F13), told apart on the error."""
    ts = hook.teacher_scoring
    script, seen = [], []

    async def transport(url, payload, timeout_secs=None):
        seen.append((url, timeout_secs))
        outcome = script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def http(status):
        return aiohttp.ClientResponseError(SimpleNamespace(real_url="u"), (), status=status, message="m")

    monkeypatch.setattr(sys.modules["miles.rollout.on_policy_distillation"], "_post_json", transport)
    monkeypatch.setattr(ts, "_BACKOFF_S", (0.0, 0.0))
    script[:] = [aiohttp.ClientConnectionError("down"), http(503), TimeoutError(), {"ok": 1}]
    assert await ts.post_json("http://t/generate", {"a": 1}, timeout_s=5.0, attempts=4) == {"ok": 1}
    assert seen == [("http://t/generate", 5.0)] * 4
    script[:] = [http(400), http(404)]
    with pytest.raises(ts.TeacherScoringError, match="answered 400") as refused:
        await ts.post_json("http://t/generate", {}, attempts=4)
    assert refused.value.ctx_overflow and len(script) == 1
    with pytest.raises(ts.TeacherScoringError, match="answered 404") as refused:
        await ts.post_json("http://t/generate", {}, attempts=4)
    assert not refused.value.ctx_overflow and script == []
    script[:] = [aiohttp.ClientConnectionError("down"), ValueError("not json")]
    with pytest.raises(ts.TeacherScoringError, match="unreachable after 2 attempts") as failed:
        await ts.post_json("http://t/generate", {}, attempts=2)
    assert not failed.value.ctx_overflow and script == []


# ---------------------------------------------------------------- the hook


async def test_generate_skips_eval_and_pops_transcript(hook, monkeypatch, teacher, dump_dir):
    sample = _sample(hook, transcript=TRANSCRIPT)
    seen = _drive(hook, monkeypatch, sample)
    input = _input(hook, sample, evaluation=True)
    out = await hook.generate.generate(input)
    assert seen == [input] and out.samples is sample
    assert gym_bridge.TRANSCRIPT_KEY not in sample.metadata
    assert sample.teacher_log_probs is None and teacher.calls == [] and _rows(dump_dir) == []
    assert sample.metadata["env_metrics"] == {"won": 1.0} and sample.status is hook.Sample.Status.COMPLETED
    # a training call pops it from every sample of a list output too, before the dump rows are built
    a, b = _sample(hook, index=1, transcript=TRANSCRIPT), _sample(hook, index=2, transcript=TRANSCRIPT)
    _drive(hook, monkeypatch, [a, b])
    assert (await hook.generate.generate(_input(hook, a))).samples == [a, b]
    assert all(gym_bridge.TRANSCRIPT_KEY not in s.metadata for s in (a, b))
    rows = _rows(dump_dir)
    assert [r["index"] for r in rows] == [1, 2] and [r["transcript"] for r in rows] == [TRANSCRIPT, TRANSCRIPT]
    assert all(gym_bridge.TRANSCRIPT_KEY not in r["metadata"] for r in rows)


async def test_generate_scores_truncated_but_not_aborted_or_empty(hook, monkeypatch, teacher):
    completed, truncated = _sample(hook, index=1), _sample(hook, index=2, status="truncated")
    aborted = _sample(hook, index=3, status="aborted", tokens=[], response_length=0, loss_mask=None)
    _drive(hook, monkeypatch, [completed, truncated, aborted])
    await hook.generate.generate(_input(hook, completed))
    for s in (completed, truncated):
        # response position r is input position 3 + r: -0.5 - pos, 0.0 on the observation tokens
        assert s.teacher_log_probs == [-3.5, -4.5, -5.5, -6.5, -7.5, -8.5, 0.0, 0.0, 0.0, -12.5, -13.5, -14.5]
        s.validate()
        m = s.metadata["env_metrics"]
        assert m["won"] == 1.0 and m["opd_turns"] == 2.0 and m["opd_delib_tokens_mean"] == 1.5
        assert m["opd_turns_with_action_frac"] == 1.0 and m["opd_turns_with_delib_frac"] == 0.5
        assert m["opd_teacher_scoring_failed"] == 0.0 and m["opd_teacher_latency_s"] >= 0.0
    assert completed.status is hook.Sample.Status.COMPLETED and truncated.status is hook.Sample.Status.TRUNCATED
    assert aborted.status is hook.Sample.Status.ABORTED and aborted.teacher_log_probs is None
    assert aborted.metadata["env_metrics"] == {"won": 1.0}
    assert [(url, payload["input_ids"]) for url, payload, _ in teacher.calls] == [
        (TEACHER, completed.tokens),
        (TEACHER, truncated.tokens),
    ]
    _, payload, kwargs = teacher.calls[0]
    assert kwargs == {"timeout_s": 900.0, "attempts": 4, "concurrency": 32}
    assert payload["sampling_params"]["max_new_tokens"] == 0 and payload["return_logprob"] is True
    assert "top_logprobs_num" not in payload  # sampled OPD: the sampled token's log-prob only


async def test_generate_handles_empty_non_aborted_sample(hook, monkeypatch, teacher, dump_dir):
    """status != ABORTED with response_length 0 (F3): no turns to segment, nothing to score, not aborted; under opd an
    empty teacher list satisfies miles' length checks; the pilot leaves the field None like the rest of its batch."""
    empty = _sample(hook, tokens=PROMPT_IDS, response_length=0, loss_mask=[])
    _drive(hook, monkeypatch, empty)
    await hook.generate.generate(_input(hook, empty))
    assert empty.teacher_log_probs == [] and empty.status is hook.Sample.Status.COMPLETED
    empty.validate()
    assert empty.metadata["env_metrics"] == {"won": 1.0, "opd_empty_response": 1.0}
    assert teacher.calls == [] and _rows(dump_dir) == []
    other = _sample(hook, index=2, tokens=PROMPT_IDS, response_length=0, loss_mask=None)
    _drive(hook, monkeypatch, other)
    await hook.generate.generate(_input(hook, other, use_opd=False, teacher=None))
    assert other.teacher_log_probs is None and other.metadata["env_metrics"]["opd_empty_response"] == 1.0


async def test_generate_aborts_on_scoring_failure(hook, monkeypatch, teacher, dump_dir, caplog):
    ts = hook.teacher_scoring
    failed = _sample(hook, index=1)
    teacher.error = ts.TeacherScoringError("teacher http://t:13141/generate unreachable after 4 attempts")
    _drive(hook, monkeypatch, failed)
    with caplog.at_level(logging.ERROR):
        await hook.generate.generate(_input(hook, failed))
    assert failed.status is hook.Sample.Status.ABORTED and failed.teacher_log_probs is None
    m = failed.metadata["env_metrics"]
    assert m["opd_teacher_scoring_failed"] == 1.0 and m["opd_teacher_ctx_overflow"] == 0.0
    assert m["opd_turns"] == 2.0 and m["opd_teacher_latency_s"] >= 0.0 and m["won"] == 1.0
    assert any("OPD_TEACHER_SCORING_FAILED sample index=1" in r.getMessage() for r in caplog.records)
    # the dump row was written before the scoring: stage A re-scores, the rows carry no teacher log-probs
    (row,) = _rows(dump_dir)
    assert row["index"] == 1 and row["status"] == "completed" and "teacher_log_probs" not in row
    # a context-window refusal is told apart in the metrics (F13); a token drift aborts like any other failure
    overflow = _sample(hook, index=2)
    teacher.error = ts.TeacherScoringError("teacher answered 400: Bad Request", ctx_overflow=True)
    _drive(hook, monkeypatch, overflow)
    await hook.generate.generate(_input(hook, overflow))
    assert overflow.status is hook.Sample.Status.ABORTED
    assert overflow.metadata["env_metrics"]["opd_teacher_ctx_overflow"] == 1.0
    drift = _sample(hook, index=3)
    teacher.error, teacher.wrong_id_at = None, 5
    _drive(hook, monkeypatch, drift)
    await hook.generate.generate(_input(hook, drift))
    assert drift.status is hook.Sample.Status.ABORTED and drift.teacher_log_probs is None
    assert drift.metadata["env_metrics"]["opd_teacher_scoring_failed"] == 1.0
    assert len(teacher.calls) == 3 and len(_rows(dump_dir)) == 3


async def test_generate_dumps_without_scoring_when_opd_is_off(hook, monkeypatch, teacher, dump_dir):
    """The pilot (--algo grpo --opd-dump): the same hook, the dump row and the turn metrics, no teacher traffic; the
    env var MILES_OPD_DUMP_DIR is the dump's only switch."""
    sample = _sample(hook, transcript=TRANSCRIPT)
    _drive(hook, monkeypatch, sample)
    await hook.generate.generate(_input(hook, sample, use_opd=False, teacher=None))
    assert teacher.calls == [] and sample.teacher_log_probs is None and sample.status is hook.Sample.Status.COMPLETED
    m = sample.metadata["env_metrics"]
    assert m["opd_turns"] == 2.0 and "opd_teacher_scoring_failed" not in m
    (row,) = _rows(dump_dir)
    assert row["tokens"] == PROMPT_IDS + RESPONSE and row["loss_mask"] == LOSS_MASK and row["turns"] == TURNS
    assert row["transcript"] == TRANSCRIPT and row["v"] == 2 and row["env_metrics"]["opd_turns"] == 2.0
    assert row["reward"] == 1.0 and row["metadata"]["tools"] == [ACT_FLAT]
    assert "session_server_id" not in row["metadata"]
    monkeypatch.delenv("MILES_OPD_DUMP_DIR")
    other = _sample(hook, index=9)
    _drive(hook, monkeypatch, other)
    await hook.generate.generate(_input(hook, other, use_opd=False, teacher=None))
    assert len(_rows(dump_dir)) == 1 and other.metadata["env_metrics"]["opd_turns"] == 2.0


async def test_generate_requires_the_rows_action_tool(hook, monkeypatch, teacher):
    sample = _sample(hook)
    del sample.metadata["tools"]
    _drive(hook, monkeypatch, sample)
    with pytest.raises(ValueError, match="declares no tools"):
        await hook.generate.generate(_input(hook, sample))


def test_generate_add_arguments_composes_the_inner_flags(hook):
    """miles registers only the custom generate function's add_arguments, so the hook's must carry the wrapped
    function's flags (--custom-agent-function-path is one of them) plus the scoring knobs at the campaign values."""
    parser = argparse.ArgumentParser()
    hook.generate.generate.add_arguments(parser)
    args = parser.parse_args(["--custom-agent-function-path", "miles_agent.gym_bridge.run"])
    assert args.custom_agent_function_path == "miles_agent.gym_bridge.run" and args.max_seq_len is None
    assert (args.opd_score_concurrency, args.opd_score_timeout_s, args.opd_score_attempts) == (32, 900.0, 4)
    assert parser.parse_args(["--opd-score-timeout-s", "30", "--opd-score-attempts", "1"]).opd_score_timeout_s == 30.0


@pytest.mark.usefixtures("tito_stub")
def test_marker_id_derives_from_the_launch_template_once(hook, monkeypatch, template_render):
    """The marker is tool_call_markup.marker_token_id over the launch's own render (--tito-model and the folded
    --apply-chat-template-kwargs through the --hf-checkpoint tokenizer, around the rows' tool), loaded once per
    launch; a tokenizer that spells the opening in pieces is refused naming the family."""
    from miles_agent.tool_call_markup import marker_token_id, tito_render_chat

    whole = FakeTokenizer(template_render)
    expected = marker_token_id(whole, tito_render_chat("qwen35", whole, ""), ACT_FLAT)
    split = FakeTokenizer(template_render, added={k: v for k, v in whole.added.items() if v != expected})
    loads = []
    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, trust_remote_code=False):
            loads.append(path)
            return {"/models/whole": whole, "/models/split": split}[path]

    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    hook.generate._marker_id.cache_clear()
    args = _args()
    args.hf_checkpoint = "/models/whole"
    assert hook.generate.marker_id(args, [ACT_FLAT]) == expected
    assert hook.generate.marker_id(args, [ACT_FLAT]) == expected and loads == ["/models/whole"]
    args.hf_checkpoint = "/models/split"
    with pytest.raises(ValueError, match="--tito-model qwen35") as refused:
        hook.generate.marker_id(args, [ACT_FLAT])
    assert "whole added token" in str(refused.value) and loads == ["/models/whole", "/models/split"]
    hook.generate._marker_id.cache_clear()


# ---------------------------------------------------------------- dump row, zero_shape, gym_bridge


def test_dump_row_v2_keys(dump_dir):
    """REF's keys first (reward = metadata["reward"], metadata minus the bulky and per-run fields with tools kept),
    the v2 fields after; one file per process, appended and flushed per row."""
    sample = SimpleNamespace(
        rollout_id=3,
        index=7,
        status=_Status.COMPLETED,
        prompt=PROMPT,
        tokens=[1, 2, 3, 4],
        response_length=2,
        loss_mask=[1, 1],
        metadata={
            "reward": 1.0,
            "raw_reward": 1.0,
            "game_file": "g",
            "split": "train",
            "tools": [ACT_FLAT],
            "env_metrics": {"won": 1.0},
            "agent_metrics": {"total_tool_time": 0.1},
            "session_server_id": "x",
            "session_server_instance_id": 2,
            gym_bridge.TRANSCRIPT_KEY: TRANSCRIPT,
            "opaque": object(),
        },
    )
    dump.dump_row(sample, turns.segment_turns([3, 4], [1, 1], MARKER), TRANSCRIPT, _args())
    (row,) = _rows(dump_dir)
    assert list(row) == [
        "rollout_id",
        "index",
        "status",
        "reward",
        "metadata",
        "env_metrics",
        "tokens",
        "response_length",
        "loss_mask",
        "turns",
        "v",
        "prompt",
        "transcript",
        "tito_model",
        "chat_template_kwargs",
    ]
    assert (row["rollout_id"], row["index"], row["status"], row["reward"]) == (3, 7, "completed", 1.0)
    assert row["metadata"] == {"game_file": "g", "split": "train", "tools": [ACT_FLAT]}
    assert row["env_metrics"] == {"won": 1.0} and row["tokens"] == [1, 2, 3, 4] and row["response_length"] == 2
    assert row["loss_mask"] == [1, 1] and row["turns"] == [[0, 2, 2]]
    assert row["v"] == 2 and row["prompt"] == PROMPT and row["transcript"] == TRANSCRIPT
    assert row["tito_model"] == "qwen35" and row["chat_template_kwargs"] == {"enable_thinking": False}
    sample.loss_mask = None
    dump.dump_row(sample, [], None, _args())
    assert len(list(dump_dir.glob("dump_*.jsonl"))) == 1
    second = _rows(dump_dir)[1]
    assert second["transcript"] is None and second["loss_mask"] is None and second["turns"] == []


def test_zero_shape_zeroes_train_but_not_env(miles_surface, monkeypatch):
    """REWARD_SHAPING_PATH=miles_agent.opd.zero_shape (what the launcher exports for the opd arm) through
    reward_routing: the training reward is 0, the env score still lands in reward["env"] and metadata["raw_reward"]."""
    assert zero_shape(0.7, SimpleNamespace(metadata={"env_metrics": {"won": 1.0}})) == 0.0
    # reward_routing calls _shape(score, sample)
    assert list(inspect.signature(zero_shape).parameters) == ["env_score", "sample"]
    monkeypatch.setenv("REWARD_SHAPING_PATH", "miles_agent.opd.zero_shape")
    routing_path = REPO / "miles_agent" / "reward_routing.py"
    spec = importlib.util.spec_from_file_location("reward_routing_zero_shape", routing_path)
    routing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routing)
    sample = SimpleNamespace(metadata={"reward": 1.0, "env_metrics": {"won": 1.0}}, response="")
    assert asyncio.run(routing.reward_func(None, sample)) == {"env": 1.0, "train": 0.0}
    assert sample.metadata["raw_reward"] == 1.0
    failed = SimpleNamespace(metadata={}, response=None)  # /run transport failure: no env score
    assert asyncio.run(routing.reward_func(None, [sample, failed])) == [
        {"env": 1.0, "train": 0.0},
        {"env": 0.0, "train": 0.0},
    ]


async def test_gym_bridge_forwards_transcript_only_under_dump_env(monkeypatch, tmp_path):
    response = {
        "reward": 1.0,
        "won": 1.0,
        "turns": 2.0,
        "last_observation": "obs2",
        "messages": TRANSCRIPT,
        "agent_metrics": {"total_tool_time": 0.5},
    }

    async def fake_post(url, payload):
        return dict(response)

    monkeypatch.setattr(gym_bridge, "post_json", fake_post)
    monkeypatch.delenv("MILES_OPD_DUMP_DIR", raising=False)
    metadata = {"tools": [ACT_FLAT], "game_file": "g"}
    plain = await gym_bridge.run("http://session:1", PROMPT, {}, metadata)
    assert gym_bridge.TRANSCRIPT_KEY not in plain and plain["reward"] == 1.0
    # strings and lists are not counters
    assert plain["env_metrics"] == {"won": 1.0, "turns": 2.0, "run_transport_failed": 0.0}
    monkeypatch.setenv("MILES_OPD_DUMP_DIR", str(tmp_path))
    dumped = await gym_bridge.run("http://session:1", PROMPT, {}, metadata)
    assert dumped[gym_bridge.TRANSCRIPT_KEY] == TRANSCRIPT and dumped["env_metrics"] == plain["env_metrics"]
    assert dumped["agent_metrics"] == {"total_tool_time": 0.5}
    del response["messages"]  # an agent without a transcript (tool_call_agent) forwards none
    assert gym_bridge.TRANSCRIPT_KEY not in await gym_bridge.run("http://session:1", PROMPT, {}, metadata)
