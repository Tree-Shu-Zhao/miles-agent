"""EngineSession retry / decode behavior against a scripted httpx.MockTransport.

No engine process: the transport replies from a per-test outcome queue (an
httpx.Response or an exception to raise), records every request, and
asyncio.sleep is stubbed so the back-off schedule is asserted, not waited on.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from webshop_env._engine_client import EngineError, EngineSession

BASE = "http://engine.test"
RESET_OK = {"obs": "Amazon Shopping Game [SEP] Instruction: [SEP] buy a mug [SEP] Search", "info": {"goal_idx": 7}}
STEP_OK = {"obs": "page", "reward": 10.0, "done": True, "info": {"task_score": 1.0}}


def _bodies(requests: list[httpx.Request]) -> list[dict]:
    return [json.loads(r.content) for r in requests]


@pytest.fixture
async def rig(monkeypatch):
    outcomes: list = []
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request):
        requests.append(request)
        outcome = outcomes.pop(0) if outcomes else httpx.Response(200, json={})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE)
    monkeypatch.setattr(EngineSession, "_clients", {BASE: client})
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    yield SimpleNamespace(outcomes=outcomes, requests=requests, sleeps=sleeps, client=client)
    await client.aclose()


def _session(retries: int = 2) -> EngineSession:
    return EngineSession(BASE, step_timeout=30.0, retries=retries)


# ───────────────────────────────────────────────────────────── constructor
def test_empty_base_url_is_engine_error():
    with pytest.raises(EngineError):
        EngineSession("", 30.0)


def test_engine_error_status_defaults_to_none():
    assert EngineError("x").status is None
    assert EngineError("x", status=410).status == 410


# ─────────────────────────────────────────────────────────────── /step
async def test_step_wire_body_and_decode(rig):
    rig.outcomes.append(httpx.Response(200, json=STEP_OK))
    s = _session()
    obs, reward, done, info = await s.step("search[mug]")
    assert (obs, reward, done, info) == ("page", 10.0, True, {"task_score": 1.0})
    assert isinstance(reward, float)
    [req] = rig.requests
    assert req.method == "POST" and req.url.path == "/step"
    assert json.loads(req.content) == {"session_id": s.session_id, "action": "search[mug]"}


async def test_step_5xx_single_attempt_status_500(rig):
    rig.outcomes.append(httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(EngineError) as ei:
        await _session().step("look")
    assert ei.value.status == 500
    assert len(rig.requests) == 1 and rig.sleeps == []  # a /step that reached the engine is never retried


async def test_step_410_single_attempt_status_410(rig):
    rig.outcomes.append(httpx.Response(410, json={"detail": "unknown session"}))
    with pytest.raises(EngineError) as ei:
        await _session().step("look")
    assert ei.value.status == 410
    assert len(rig.requests) == 1 and rig.sleeps == []


async def test_step_connect_error_retries_with_backoff(rig):
    rig.outcomes.extend(httpx.ConnectError("refused") for _ in range(3))
    s = _session(retries=2)
    with pytest.raises(EngineError) as ei:
        await s.step("look")
    assert ei.value.status is None  # transport failure: no HTTP verdict
    assert len(rig.requests) == 3
    assert rig.sleeps == [2.0, 4.0]
    assert {b["session_id"] for b in _bodies(rig.requests)} == {s.session_id}  # /step keeps its id


async def test_step_connect_error_then_success(rig):
    rig.outcomes.extend([httpx.ConnectTimeout("slow"), httpx.Response(200, json=STEP_OK)])
    obs, *_ = await _session().step("look")
    assert obs == "page" and len(rig.requests) == 2 and rig.sleeps == [2.0]


async def test_step_read_timeout_not_retried(rig):
    # The request may have reached the engine and mutated the episode: no blind retry.
    rig.outcomes.append(httpx.ReadTimeout("engine wedged"))
    with pytest.raises(EngineError) as ei:
        await _session().step("look")
    assert ei.value.status is None and len(rig.requests) == 1 and rig.sleeps == []


# ─────────────────────────────────────────────────────────────── /reset
async def test_reset_wire_body_and_decode(rig):
    rig.outcomes.append(httpx.Response(200, json=RESET_OK))
    s = _session()
    obs, info = await s.reset(7, "eval")
    assert obs == RESET_OK["obs"] and info == {"goal_idx": 7}
    [req] = rig.requests
    assert req.url.path == "/reset"
    assert json.loads(req.content) == {"session_id": s.session_id, "task_id": 7, "split": "eval"}


async def test_reset_5xx_retries_with_distinct_session_ids(rig):
    rig.outcomes.extend(httpx.Response(503, json={"detail": "worker respawning"}) for _ in range(3))
    s = _session(retries=2)
    with pytest.raises(EngineError) as ei:
        await s.reset(7)
    assert ei.value.status == 503
    assert len(rig.requests) == 3 and rig.sleeps == [2.0, 4.0]
    ids = [b["session_id"] for b in _bodies(rig.requests)]
    assert len(set(ids)) == 3  # F02: a fresh id per attempt
    assert s.session_id == ids[-1]


async def test_reset_success_on_second_attempt_keeps_that_id(rig):
    rig.outcomes.extend([httpx.Response(500), httpx.Response(200, json=RESET_OK)])
    s = _session()
    first_id = s.session_id
    await s.reset(7)
    ids = [b["session_id"] for b in _bodies(rig.requests)]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert s.session_id == ids[1]  # the id the engine actually registered
    assert s.session_id != first_id  # the constructor's id is never sent as-is
    assert rig.sleeps == [2.0]


async def test_reset_transport_error_retries_with_distinct_ids(rig):
    rig.outcomes.extend([httpx.ReadTimeout("slow"), httpx.Response(200, json=RESET_OK)])
    s = _session()
    await s.reset(7)
    ids = [b["session_id"] for b in _bodies(rig.requests)]
    assert len(ids) == 2 and ids[0] != ids[1] and s.session_id == ids[1]


async def test_reset_4xx_single_attempt(rig):
    rig.outcomes.append(httpx.Response(400, json={"detail": "bad goal index"}))
    with pytest.raises(EngineError) as ei:
        await _session().reset(-1)
    assert ei.value.status == 400
    assert len(rig.requests) == 1 and rig.sleeps == []


async def test_status_reflects_the_last_failure(rig):
    rig.outcomes.extend([httpx.Response(500), httpx.ConnectError("refused"), httpx.Response(502)])
    with pytest.raises(EngineError) as ei:
        await _session().reset(7)
    assert ei.value.status == 502
    assert ei.value.__cause__ is not None and isinstance(ei.value.__cause__, httpx.HTTPStatusError)


# ────────────────────────────────────────────────────── malformed 2xx bodies
async def test_invalid_json_body_is_engine_error(rig):
    rig.outcomes.append(httpx.Response(200, content=b"<html>oops</html>"))
    with pytest.raises(EngineError, match="non-JSON") as ei:
        await _session().reset(7)
    assert ei.value.status is None
    assert len(rig.requests) == 1  # a decode fault is not a retry trigger


async def test_non_object_json_is_engine_error(rig):
    rig.outcomes.append(httpx.Response(200, json=["obs"]))
    with pytest.raises(EngineError, match="expected an object"):
        await _session().reset(7)


async def test_missing_obs_is_engine_error(rig):
    rig.outcomes.append(httpx.Response(200, json={"info": {}}))
    with pytest.raises(EngineError, match="missing 'obs'"):
        await _session().reset(7)


@pytest.mark.parametrize(
    "body",
    [
        {"obs": "page", "done": False, "info": {}},  # missing reward
        {"obs": "page", "reward": "high", "done": False, "info": {}},  # mistyped reward
        {"obs": "page", "reward": 0.0, "info": {}},  # missing done
        {"obs": None, "reward": 0.0, "done": False, "info": {}},  # mistyped obs
        {"obs": "page", "reward": 0.0, "done": False, "info": None},  # mistyped info
    ],
)
async def test_malformed_step_payload_is_engine_error(rig, body):
    rig.outcomes.append(httpx.Response(200, json=body))
    with pytest.raises(EngineError) as ei:
        await _session().step("look")
    assert ei.value.status is None


async def test_missing_info_defaults_to_empty_dict(rig):
    rig.outcomes.append(httpx.Response(200, json={"obs": "page", "reward": 0, "done": False}))
    obs, reward, done, info = await _session().step("look")
    assert (obs, reward, done, info) == ("page", 0.0, False, {})


# ─────────────────────────────────────────────────────────── close / forget
async def test_close_posts_once_and_is_idempotent(rig):
    rig.outcomes.append(httpx.Response(200, json={}))
    s = _session()
    assert s.closed is False
    await s.close()
    await s.close()
    assert s.closed is True
    assert [r.url.path for r in rig.requests] == ["/close"]
    assert json.loads(rig.requests[0].content) == {"session_id": s.session_id}


async def test_close_after_forget_makes_no_request(rig):
    s = _session()
    s.forget()
    assert s.closed is True
    await s.close()
    assert rig.requests == []


async def test_close_swallows_engine_error_and_logs_status(rig, caplog):
    rig.outcomes.append(httpx.Response(500, json={"detail": "boom"}))
    s = _session()
    with caplog.at_level(logging.INFO, logger="webshop_env.engine_client"):
        await s.close()  # must not raise
    assert s.closed is True
    assert len(rig.requests) == 1 and rig.sleeps == []  # 5xx on /close is not retried
    assert any("status=500" in rec.getMessage() and rec.levelno == logging.INFO for rec in caplog.records)


async def test_close_retries_connect_error_once(rig):
    rig.outcomes.extend([httpx.ConnectError("refused"), httpx.Response(200, json={})])
    await _session().close()
    assert len(rig.requests) == 2 and rig.sleeps == [2.0]


async def test_close_gives_up_after_one_connect_retry(rig):
    rig.outcomes.extend(httpx.ConnectError("refused") for _ in range(3))
    await _session().close()  # swallowed
    assert len(rig.requests) == 2 and rig.sleeps == [2.0]


# ──────────────────────────────────────────────────────────── client cache
async def test_client_is_shared_per_base_url_and_recreated_when_closed(monkeypatch):
    monkeypatch.setattr(EngineSession, "_clients", {})
    a = EngineSession._client(BASE, 30.0)
    assert EngineSession._client(BASE, 30.0) is a
    assert EngineSession._client("http://other", 30.0) is not a
    await a.aclose()
    b = EngineSession._client(BASE, 30.0)
    assert b is not a and not b.is_closed
    await b.aclose()
    await EngineSession._client("http://other", 30.0).aclose()


async def test_trailing_slash_base_url_hits_same_client(rig):
    rig.outcomes.append(httpx.Response(200, json=RESET_OK))
    await EngineSession(BASE + "/", 30.0).reset(7)
    assert rig.requests[0].url.path == "/reset"
