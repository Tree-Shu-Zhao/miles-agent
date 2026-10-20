"""Generic Gymnasium episode driver over chat completions with a per-request policy endpoint.

Counterpart of tool_call_agent for reset/step environments
(resources_servers.gymnasium.GymnasiumServer subclasses, e.g. alfworld_env).
Where tool_call_agent drives the OpenAI tool-calling protocol, this agent
drives the Gymnasium protocol:

- POST /reset before the first model call; the opening observation is
  appended as a ``user`` message. Per-episode identity (``game_file``, ...)
  rides the /run body (extra="allow") into /reset and /step; the resources
  server's signed session cookie is the episode handle thereafter.
- Every assistant message is one action attempt: it goes to POST /step as a
  single-turn Responses object (output_text + function_call items). Actions
  travel as tool calls — the dataset row declares the env's action tool
  (e.g. alfworld's ``act``), the env reads the FIRST tool call, and the step
  observation returns as that call's ``tool`` reply; extra calls in the same
  turn are answered "ignored". A turn with no tool call still costs a step
  (the env counts a format error) and its retry hint returns as a ``user``
  message, since there is no call id to answer.
- Optional ``scripted_prefix`` in the /run body: assistant messages (content
  + tool_calls) replayed through /step BEFORE the first policy call, exactly
  as if the policy had produced them, so the policy resumes a recorded
  episode from its mid-trajectory state with the replayed turns in its
  transcript. Replayed steps count against the env's ``max_episode_steps``
  but not against the agent's ``max_turns``; they are reported as
  ``scripted_*`` metrics and never forwarded to the env body. A prefix that
  ends the episode leaves the policy no turn.
- Strict Gymnasium endings: ``terminated``/``truncated`` from /step are
  authoritative. Agent-side max_turns exhaustion and policy HTTP-400
  (context overflow) count as truncation and POST /close best-effort so the
  env frees the session (its TTL sweeper is the backstop).
- Reward is the sum of step rewards; /verify is never called
  (GymnasiumServer.verify raises NotImplementedError by design).
- Besides the numeric metrics, the /run response carries ``last_observation``,
  the observation after the final action (never shown to the policy). It is a
  string, so the trainer's env_metrics filter ignores it; the R2S takeover
  stage votes on its fingerprint.
- It also carries ``messages``, the structured transcript as built here (prompt
  messages, the reset observation, every assistant message with its tool calls
  as the engine's parser returned them, replayed scripted turns included, and
  the tool/user observations) and ``terminated``; the R2S pilot dump keeps the
  transcript and the takeover stage reads both.

Like tool_call_agent, it honors the per-request ``policy_base_url`` and
speaks ``POST {base}/chat/completions`` — the only dialect recording proxies
(Miles' session server) capture.
"""

import time
from typing import Any

from agent_utils.chat_dialect import (
    response_from_assistant_texts,
    response_from_chat_message,
    to_chat_messages,
    to_chat_sampling,
    to_chat_tools,
)
from agent_utils.disconnect import RunToCompletionOnDisconnect
from fastapi import Body, Request
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.server_utils import request as http_request  # `request` is shadowed by run()'s Request parameter
from pydantic import ConfigDict

# /run body fields that are the agent's own, never forwarded to the env.
_AGENT_ONLY_FIELDS = {"policy_base_url", "policy_api_key", "scripted_prefix"}


class EnvLoopAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Episode turn budget = MODEL CALLS; under strict Gymnasium endings each
    # model call is exactly one env step, so align this with the env's
    # max_episode_steps.
    max_turns: int = 30


class EnvLoopAgentRunRequest(BaseRunRequest):
    # dataset extras (game_file, max_episode_steps, ...) ride along
    model_config = ConfigDict(extra="allow")

    policy_base_url: str | None = None
    policy_api_key: str | None = None
    # Assistant chat messages replayed through /step before the first policy call
    # (see the module docstring). Each is {"role": "assistant", "content": ..., "tool_calls": [...]}.
    scripted_prefix: list[dict[str, Any]] | None = None


def _merge_numeric(into: dict[str, float], info: dict[str, Any]) -> None:
    """Env info -> flat float metrics (bools included: won/lost are the point)."""
    for key, value in (info or {}).items():
        if isinstance(value, bool) or isinstance(value, (int, float)):
            into[key] = float(value)


class _Episode:
    """Per-/run bookkeeping shared by the scripted-prefix and policy phases."""

    def __init__(self, env_body: dict[str, Any], cookies: Any, messages: list[dict[str, Any]], observation: str):
        self.env_body = env_body
        self.cookies = cookies  # the episode handle, re-threaded after every env call
        self.messages = messages
        self.env_metrics: dict[str, float] = {}
        self.env_time = 0.0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.last_observation = observation
        self.extra_tool_calls = 0

    @property
    def ended(self) -> bool:
        return self.terminated or self.truncated


class EnvLoopAgent(RunToCompletionOnDisconnect, SimpleResponsesAPIAgent):
    config: EnvLoopAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:  # noqa: B008
        # Agent-as-model interface; this agent only serves /run.
        raise NotImplementedError

    async def _env_post(self, ep: _Episode, url_path: str, json: dict[str, Any]) -> dict[str, Any]:
        """POST to the resources server under the episode cookie; failures propagate
        (a half-played episode with a made-up reward is worse than a dropped one)."""
        t0 = time.monotonic()
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path=url_path, json=json, cookies=ep.cookies
        )
        await raise_for_status(response)
        data = await get_response_json(response)
        ep.cookies = response.cookies
        ep.env_time += time.monotonic() - t0
        return data

    async def _step(self, ep: _Episode, msg: dict[str, Any], model: str) -> None:
        """One assistant message -> /step -> its observation appended as the tool (or user) reply."""
        step_body = ep.env_body | {"response": response_from_chat_message(msg, model).model_dump(mode="json")}
        step_data = await self._env_post(ep, "/step", step_body)

        ep.total_reward += float(step_data.get("reward") or 0.0)
        _merge_numeric(ep.env_metrics, step_data.get("info"))
        ep.terminated = bool(step_data.get("terminated"))
        ep.truncated = bool(step_data.get("truncated"))

        observation = step_data.get("observation") or ""
        ep.last_observation = observation
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            # Chat convention: every tool_call_id gets a reply. The env
            # executes only the first call; extras are told so.
            ep.messages.append(
                {"role": "tool", "tool_call_id": tool_calls[0].get("id") or "call_0", "content": observation}
            )
            for i, tc in enumerate(tool_calls[1:], start=1):
                ep.extra_tool_calls += 1
                ep.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or f"call_{i}",
                        "content": "Ignored: one action per turn; only the first tool call was executed.",
                    }
                )
        elif observation:
            # No tool call this turn (format fault): the env's retry hint
            # has no call id to answer, so it rides as a user message.
            ep.messages.append({"role": "user", "content": observation})

    async def _replay_scripted_prefix(
        self, ep: _Episode, scripted: list[dict[str, Any]], model: str
    ) -> dict[str, float]:
        """Replay the scripted assistant messages; stop at the first terminal/truncated step."""
        steps = 0
        for msg in scripted:
            ep.messages.append(msg)
            await self._step(ep, msg, model)
            steps += 1
            if ep.ended:
                break
        # Cumulative env counters after the replay = the replay's own faults (the
        # policy has not acted yet); the caller rejects a restore that misfired.
        return {
            "scripted_steps": float(steps),
            "scripted_terminated": float(ep.ended),
            "scripted_invalid_actions": ep.env_metrics.get("invalid_actions", 0.0) if steps else 0.0,
            "scripted_format_errors": ep.env_metrics.get("action_format_errors", 0.0) if steps else 0.0,
        }

    async def run(self, request: Request, body: EnvLoopAgentRunRequest) -> dict[str, Any]:
        params = body.responses_create_params
        # policy_base_url is expected to include /v1; only /chat/completions is appended.
        assert body.policy_base_url, "env_loop_agent requires a per-request policy_base_url in the /run body"
        base = body.policy_base_url.rstrip("/")
        model = params.model or "model"
        sampling = to_chat_sampling(params)
        chat_tools = to_chat_tools(params)  # the env's action tool, declared by the dataset row
        headers = {"Authorization": f"Bearer {body.policy_api_key or 'EMPTY'}"}
        # Never forward the policy endpoint, api key or scripted prefix to the env;
        # dataset extras (game_file, ...) must ride along for reset/step.
        env_body = body.model_dump(exclude=_AGENT_ONLY_FIELDS)

        t0 = time.monotonic()
        reset_response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/reset", json=env_body
        )
        # Reset failures (bad row, missing game) propagate: better to drop the
        # trajectory than to fake an episode.
        await raise_for_status(reset_response)
        reset_data = await get_response_json(reset_response)
        messages = to_chat_messages(params)
        if reset_data.get("observation"):
            messages.append({"role": "user", "content": reset_data["observation"]})
        ep = _Episode(env_body, reset_response.cookies, messages, reset_data.get("observation") or "")
        ep.env_time += time.monotonic() - t0
        _merge_numeric(ep.env_metrics, reset_data.get("info"))

        scripted_metrics = await self._replay_scripted_prefix(ep, body.scripted_prefix or [], model)

        assistant_texts: list[str] = []
        policy_400_truncated = 0
        finished = ep.ended  # a prefix that already ended the episode leaves the policy no turn

        for _ in range(self.config.max_turns):
            if finished:
                break
            payload: dict[str, Any] = {"model": model, "messages": ep.messages, **sampling}
            if chat_tools:
                payload["tools"] = chat_tools
            model_response = await http_request("POST", f"{base}/chat/completions", json=payload, headers=headers)
            if model_response.status == 400:
                # Accumulated transcript no longer fits the policy server (prompt >
                # max_model_len is a hard 400). Same semantics as a truncated
                # training rollout: end the episode with what exists.
                policy_400_truncated = 1
                break
            await raise_for_status(model_response)
            data = await get_response_json(model_response)

            msg = data["choices"][0]["message"]
            # Append verbatim: rebuilding {role, content} drops reasoning_content
            # and breaks stored-prefix matching in recording proxies.
            ep.messages.append(msg)
            assistant_texts.append(msg.get("content") or "")

            await self._step(ep, msg, model)
            if ep.ended:
                finished = True  # the env already closed the session
                break

        if not finished:
            # max_turns exhausted or policy-400: locally truncated. Free the
            # env session now; the TTL sweeper only backstops failures here.
            ep.truncated = True
            try:
                await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/close",
                    json={},
                    cookies=ep.cookies,
                )
            except Exception:
                pass

        resp_obj = response_from_assistant_texts(assistant_texts, model)
        # Top-level numerics flow into gym's aggregate metrics and, via the
        # miles agent function, into sample.metadata["env_metrics"] -> wandb
        # env/*. agent_metrics.total_tool_time feeds Sample.non_generation_time
        # (throughput accounting subtracts env time from generation time).
        return {
            "responses_create_params": params.model_dump(mode="json"),
            "response": resp_obj.model_dump(mode="json"),
            "reward": ep.total_reward,
            **ep.env_metrics,
            **scripted_metrics,
            "turns": float(len(assistant_texts)),
            "env_truncated": float(ep.truncated and not ep.terminated),
            "terminated": float(ep.terminated),
            "policy_400_truncated": policy_400_truncated,
            "extra_tool_calls": float(ep.extra_tool_calls),
            "agent_metrics": {"total_tool_time": ep.env_time},
            # The observation after the final action (never shown to the policy):
            # the R2S takeover stage votes on its fingerprint. Non-numeric, so the
            # trainer's env_metrics filter ignores it.
            "last_observation": ep.last_observation,
            "messages": ep.messages,
        }


if __name__ == "__main__":
    EnvLoopAgent.run_webserver()
