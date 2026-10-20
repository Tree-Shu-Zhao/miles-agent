"""miles_agent.gym_bridge's transport: /abort fan-out, the spread of episodes over every NEMO_GYM_URL server (four tests moved here from the deleted harness agent's test file), the /run body, metric forwarding and the retry."""

import asyncio
from unittest.mock import MagicMock

import pytest
import httpx

from miles_agent import gym_bridge

_REAL_CLIENT = httpx.AsyncClient  # captured once: a test that swaps the transport twice must not wrap the previous wrapper


def _bridge_with_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(gym_bridge.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(transport=transport, **kw))
    monkeypatch.setenv("NEMO_GYM_URL", "http://gym:8201")
    monkeypatch.setattr(gym_bridge, "_in_flight", {})
    return gym_bridge


# ---- the four tests moved here from the harness agent's test file (bodies unchanged; the helper above is theirs, with
# the real client captured at module level and the _in_flight reset added) ----
async def test_bridge_abort_asks_the_agent_server_to_stop(monkeypatch):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"aborted": 3})

    gym_bridge = _bridge_with_transport(monkeypatch, handler)
    await gym_bridge.abort(MagicMock())
    assert seen == ["http://gym:8201/abort"]


async def test_bridge_spreads_episodes_over_every_agent_server(monkeypatch):
    hosts, release = [], asyncio.Event()

    async def handler(request):
        hosts.append(request.url.host)
        await release.wait()
        return httpx.Response(200, json={"reward": 1.0})

    gym_bridge = _bridge_with_transport(monkeypatch, handler)
    monkeypatch.setenv("NEMO_GYM_URL", "http://gym-a:8201, http://gym-b:8201/")
    runs = [asyncio.create_task(gym_bridge.run("http://session:1", "fix it", {}, {})) for _ in range(5)]
    while len(hosts) < 5:
        await asyncio.sleep(0.01)
    assert sorted(hosts) == ["gym-a"] * 3 + ["gym-b"] * 2  # the fewest in flight, ties to the first
    release.set()
    assert [result["reward"] for result in await asyncio.gather(*runs)] == [1.0] * 5
    assert [gym_bridge._in_flight[url] for url in ("http://gym-a:8201", "http://gym-b:8201")] == [0, 0]


async def test_bridge_abort_asks_every_agent_server(monkeypatch):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404 if request.url.host == "gym-a" else 200, json={"aborted": 1})

    gym_bridge = _bridge_with_transport(monkeypatch, handler)
    monkeypatch.setenv("NEMO_GYM_URL", "http://gym-a:8201,http://gym-b:8201")
    await gym_bridge.abort(MagicMock())
    assert seen == ["http://gym-a:8201/abort", "http://gym-b:8201/abort"]


@pytest.mark.parametrize("status", [404, 500])
async def test_bridge_abort_never_raises(monkeypatch, status):
    gym_bridge = _bridge_with_transport(monkeypatch, lambda request: httpx.Response(status, json={}))
    await gym_bridge.abort(MagicMock())  # 404 = agent without /abort (env_loop_agent): drain as before


# ---- new ----
def test_agent_urls_default_is_the_local_agent_port_and_splits_a_list(monkeypatch):
    monkeypatch.delenv("NEMO_GYM_URL", raising=False)
    assert gym_bridge.agent_urls() == ["http://127.0.0.1:8200"]  # C2's default (DESIGN D-knobs row: 8200)
    monkeypatch.setenv("NEMO_GYM_URL", " http://a:1/ ,, http://b:2")
    assert gym_bridge.agent_urls() == ["http://a:1", "http://b:2"]


def test_build_run_body_translates_sampling_to_the_responses_dialect():
    body = gym_bridge.build_run_body("http://s:1", "q", {"temperature": 0.7, "top_p": None, "max_tokens": 32}, {"tools": [{"n": 1}], "k": "v"})
    assert body["policy_base_url"] == "http://s:1/v1"
    assert body["responses_create_params"] == {"input": [{"role": "user", "content": "q"}], "tools": [{"n": 1}], "temperature": 0.7, "max_output_tokens": 32}
    assert body["k"] == "v" and "tools" not in body
    assert gym_bridge.build_run_body("http://s:1", [{"role": "user", "content": "x"}], {}, {})["responses_create_params"]["input"] == [{"role": "user", "content": "x"}]


async def test_run_forwards_numeric_counters_and_agent_metrics(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"reward": 1.0, "tool_errors_env_timeout": 2, "ok": True, "name": "x",
                                         "agent_metrics": {"total_tool_time": 1.5, "label": "y"}})
    b = _bridge_with_transport(monkeypatch, handler)
    out = await b.run("http://s:1", "q", {}, {})
    assert out["reward"] == 1.0
    assert out["env_metrics"] == {"tool_errors_env_timeout": 2.0, "run_transport_failed": 0.0}  # bools and strings are not counters
    assert out["agent_metrics"] == {"total_tool_time": 1.5}
    assert b._in_flight["http://gym:8201"] == 0


async def test_run_marks_a_transport_failure_and_releases_the_slot(monkeypatch):
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json={"reward": 1.0})
    b = _bridge_with_transport(monkeypatch, slow)
    monkeypatch.setenv("NEMO_GYM_RUN_TIMEOUT", "0.05")
    assert await b.run("http://s:1", "q", {}, {}) == {"env_metrics": {"run_transport_failed": 1.0}}
    assert b._in_flight["http://gym:8201"] == 0
    # a 5xx is an HTTPStatusError, not a TransportError: no retry, the episode is marked and the slot released
    b = _bridge_with_transport(monkeypatch, lambda request: httpx.Response(500, json={}))
    assert (await b.run("http://s:1", "q", {}, {}))["env_metrics"] == {"run_transport_failed": 1.0}
    assert b._in_flight["http://gym:8201"] == 0


async def test_post_json_retries_transport_errors_then_raises(monkeypatch):
    calls = []
    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"ok": 1})
    b = _bridge_with_transport(monkeypatch, handler)
    monkeypatch.setattr(b, "_POST_BACKOFF_S", (0.0, 0.0))  # the 1-5 s back-off becomes asyncio.sleep(0)
    assert await b.post_json("http://gym:8201/run", {}) == {"ok": 1} and len(calls) == 3
    def always_down(request):
        raise httpx.ConnectError("down")
    b = _bridge_with_transport(monkeypatch, always_down)
    with pytest.raises(httpx.ConnectError):
        await b.post_json("http://gym:8201/run", {})
