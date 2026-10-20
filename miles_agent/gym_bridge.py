"""Miles <-> NeMo Gym adapter (agent function). Environment-agnostic.

Miles calls ``run`` once per sample via
``--custom-agent-function-path miles_agent.gym_bridge.run`` (with
``--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate``).
Each call POSTs the task to the tool_call_agent server's ``/run``, handing over
the session's OpenAI-compatible URL as ``policy_base_url``; every model call
the gym agent makes lands on Miles' session server and is recorded losslessly.
The returned dict is merged into ``sample.metadata`` for the reward hook
(``--custom-rm-path miles_agent.reward_routing.reward_func``).

Deliberately no miles imports (torch): this module must load on CPU-only
machines for the offline tests and API-policy scans.

Env vars:
  NEMO_GYM_URL   base URL of the tool_call_agent server
                 (default: http://127.0.0.1:8200), or several comma-separated
                 (one per agent server): each episode goes to the one with
                 the fewest episodes in flight, an abort to all.
  NEMO_GYM_RUN_TIMEOUT  hard wall-clock cap in seconds for one /run call
                 (default: 600; wiki episodes are a few short turns).
  MILES_OPD_DUMP_DIR  the pilot dump (launcher --opd-dump): the agent's structured
                 transcript (``messages``) is forwarded as metadata[TRANSCRIPT_KEY]
                 for the OPD generate hook's dump rows, which pops it.

Agents that cannot reach the session server's placed address get a reachable
``base_url`` from miles itself (``--session-server-external-host``).
"""

import asyncio
import logging
import os
import random
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_POST_ATTEMPTS = 3
_POST_BACKOFF_S = (1.0, 5.0)
TRANSCRIPT_KEY = "transcript"

# Episodes in flight per agent server. miles calls run() from one event loop (the rollout
# process), so a plain dict counts them.
_in_flight: dict[str, int] = {}


def agent_urls() -> list[str]:
    """NEMO_GYM_URL's agent servers, in order."""
    raw = os.getenv("NEMO_GYM_URL", "http://127.0.0.1:8200")
    return [url.strip().rstrip("/") for url in raw.split(",") if url.strip()]


async def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON and return the JSON response, retrying transport errors.

    No HTTP timeout here: the caller bounds the whole episode with
    asyncio.wait_for (a /run legitimately takes minutes).
    """
    async with httpx.AsyncClient(timeout=None) as client:
        for attempt in range(_POST_ATTEMPTS):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                if attempt == _POST_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(random.uniform(*_POST_BACKOFF_S))
    raise AssertionError("unreachable")


def _resolve_session_url(base_url: str) -> str:
    """The session's OpenAI-compatible policy URL."""
    return f"{base_url}/v1"


def build_run_body(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild tool_call_agent's ToolCallAgentRunRequest from miles' sample fields.

    Sampling kwargs cross the dialect boundary here: miles speaks chat
    completions (max_tokens), the /run body speaks the Responses API
    (max_output_tokens); tool_call_agent translates back on the engine side.
    """
    params: dict[str, Any] = {
        "input": prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}],
        "tools": metadata.get("tools") or [],
    }
    for key in ("temperature", "top_p"):
        if request_kwargs.get(key) is not None:
            params[key] = request_kwargs[key]
    if request_kwargs.get("max_tokens") is not None:
        params["max_output_tokens"] = request_kwargs["max_tokens"]
    # Flatten every metadata field except tools (consumed above) into the body:
    # the run-request schema is extra="allow", so any environment's grading
    # extras (expected_answer, test_cases, ...) flow through without this
    # adapter naming them. Reserved keys come last so metadata cannot override them.
    extras = {k: v for k, v in metadata.items() if k != "tools"}
    return {
        **extras,
        "responses_create_params": params,
        "policy_base_url": _resolve_session_url(base_url),
    }


async def abort(args: Any) -> None:
    """Miles' agent abort hook (miles.utils.misc.call_agent_abort_hook): DAPO
    oversampling has the groups it needs, and miles waits for every episode
    still in flight before training. Ask the agent server to stop them now
    (a trial server cancels its running trials, no grading); an agent without
    POST /abort (env_loop_agent) leaves its episodes to drain. Every agent server of
    NEMO_GYM_URL is asked."""
    for nemo_gym_url in agent_urls():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{nemo_gym_url}/abort", json={})
            if response.status_code == 404:
                continue
            response.raise_for_status()
            logger.info(f"agent {nemo_gym_url} aborted {response.json().get('aborted')} in-flight episodes")
        except Exception as e:
            logger.warning(f"agent abort at {nemo_gym_url} failed: {e}")


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one episode via the tool_call_agent server.

    Returns the reward dict merged into sample.metadata. On transport
    failure/timeout the recorded session still becomes a sample, the reward
    hook scores it 0.0 (no "reward" key), and env_metrics.run_transport_failed
    marks it so the failure is a wandb env/* metric instead of a silent zero.
    Episodes that never reached the model are ABORTED by the generate layer,
    so a check_no_aborted dynamic-sampling filter drops their group.
    """
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    nemo_gym_url = min(agent_urls(), key=lambda url: _in_flight.get(url, 0))
    timeout_s = float(os.getenv("NEMO_GYM_RUN_TIMEOUT", "600"))

    body = build_run_body(base_url, prompt, request_kwargs, metadata)
    _in_flight[nemo_gym_url] = _in_flight.get(nemo_gym_url, 0) + 1
    try:
        response = await asyncio.wait_for(post_json(f"{nemo_gym_url}/run", body), timeout=timeout_s)
    except TimeoutError:
        logger.error(f"tool_call_agent /run at {nemo_gym_url} timed out after {timeout_s:.0f}s")
        return {"env_metrics": {"run_transport_failed": 1.0}}
    except asyncio.CancelledError:
        logger.warning(f"tool_call_agent /run at {nemo_gym_url} cancelled")
        return {"env_metrics": {"run_transport_failed": 1.0}}
    except Exception as e:
        logger.error(f"tool_call_agent /run at {nemo_gym_url} failed: {e}")
        return {"env_metrics": {"run_transport_failed": 1.0}}
    finally:
        _in_flight[nemo_gym_url] -= 1

    # Every top-level numeric except reward is an agent/env counter (e.g.
    # policy_400_truncated, tool_errors_model_bad_args, tool_errors_env_*):
    # forward generically so new counters reach wandb env/* with no code here.
    env_metrics = {
        key: float(value)
        for key, value in response.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "reward"
    }
    env_metrics["run_transport_failed"] = 0.0
    result: dict[str, Any] = {"reward": response.get("reward", 0.0), "env_metrics": env_metrics}
    # Agents that measure their own overhead (env_loop_agent's env reset/step
    # time) report it under agent_metrics; miles' generate layer reads
    # agent_metrics.total_tool_time into Sample.non_generation_time.
    if isinstance(response.get("agent_metrics"), dict):
        result["agent_metrics"] = {
            key: float(value)
            for key, value in response["agent_metrics"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    if os.getenv("MILES_OPD_DUMP_DIR") and response.get("messages") is not None:
        result[TRANSCRIPT_KEY] = response["messages"]
    return result
