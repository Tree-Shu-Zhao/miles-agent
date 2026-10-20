"""Generic chat-completions tool-calling episode driver with a per-request policy endpoint.

Gym's simple_agent drives the same loop but (a) routes model calls to the
statically configured model server, ignoring ``policy_base_url`` in the /run
body, and (b) speaks /v1/responses — while RL trainers that record rollouts
through a per-episode proxy (e.g. Miles' session server) record ONLY
``POST {base}/v1/chat/completions``. This agent closes both gaps:

- honors per-request ``policy_base_url`` (the pattern NeMo Gym PR #2166 added to
  mini_swe_agent/mini_swe_agent_2);
- runs the episode over chat completions: each structured ``tool_call`` is
  POSTed to the resources-server route of the same name, the result is
  appended as a ``tool`` message, and the final transcript goes to
  POST /verify for the reward.

/run returns a BaseVerifyResponse-shaped dict; a trainer-side adapter
(e.g. miles_agent/gym_bridge.py) forwards its ``reward`` into
``sample.metadata["reward"]``.
"""

import json
import re
from typing import Any

from agent_utils.chat_dialect import (
    response_from_assistant_texts,
    to_chat_messages,
    to_chat_sampling,
    to_chat_tools,
)
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


class ToolCallAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int = 4


class ToolCallAgentRunRequest(BaseRunRequest):
    # dataset extras (expected_answer, ...) ride along
    model_config = ConfigDict(extra="allow")

    policy_base_url: str | None = None
    policy_api_key: str | None = None


class ToolCallAgent(SimpleResponsesAPIAgent):
    config: ToolCallAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:  # noqa: B008
        # Agent-as-model interface; this agent only serves /run.
        raise NotImplementedError

    async def run(self, request: Request, body: ToolCallAgentRunRequest) -> dict[str, Any]:
        params = body.responses_create_params
        messages = to_chat_messages(params)
        chat_tools = to_chat_tools(params)

        # policy_base_url is expected to include /v1; only /chat/completions is appended.
        assert body.policy_base_url, "tool_call_agent requires a per-request policy_base_url in the /run body"
        base = body.policy_base_url.rstrip("/")
        model = params.model or "model"
        sampling = to_chat_sampling(params)
        headers = {"Authorization": f"Bearer {body.policy_api_key or 'EMPTY'}"}

        resources_cookies = None
        assistant_texts: list[str] = []
        policy_400_truncated = 0
        # Calls the engine's parser recovered from the model (bad-args ones
        # included): the structured count rewards read, never the markup.
        tool_call_count = 0
        tool_error_counts: dict[str, int] = {}
        for _ in range(self.config.max_turns):
            payload: dict[str, Any] = {"model": model, "messages": messages, **sampling}
            if chat_tools:
                payload["tools"] = chat_tools
            model_response = await http_request("POST", f"{base}/chat/completions", json=payload, headers=headers)
            if model_response.status == 400:
                # Accumulated transcript no longer fits the policy server (prompt >
                # max_model_len is a hard 400). Same semantics as a truncated training
                # rollout: end the episode and let /verify score what exists.
                policy_400_truncated = 1
                break
            await raise_for_status(model_response)
            data = await get_response_json(model_response)

            msg = data["choices"][0]["message"]
            # Append verbatim: rebuilding {role, content} drops tool_calls/reasoning_content
            # and breaks stored-prefix matching in recording proxies.
            messages.append(msg)
            assistant_texts.append(msg.get("content") or "")

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break
            tool_call_count += len(tool_calls)

            # Serial execution of every call in the turn, like simple_agent: the chat
            # convention requires one tool reply per tool_call_id, and strict endpoints
            # reject a follow-up request with unanswered calls.
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError) as e:
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})
                    # Model-fault bucket: bad args track model format quality,
                    # not environment health — never merge with env failures.
                    key = "tool_errors_model_bad_args"
                    tool_error_counts[key] = tool_error_counts.get(key, 0) + 1
                else:
                    # Resources-server error bodies (unknown route, tool failures) are valid
                    # model-visible observations; never abort the episode on them.
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{name}",
                        json=args,
                        cookies=resources_cookies,
                    )
                    tool_output = (await api_response.content.read()).decode()
                    resources_cookies = api_response.cookies
                    # Generic contract: a resources server that degrades a tool failure
                    # into a model-visible observation flags it with X-Env-Tool-Error,
                    # whose value is a coarse category (sanitized here to bound key
                    # cardinality; full detail lives in the observation text).
                    if category := api_response.headers.get("X-Env-Tool-Error"):
                        key = "tool_errors_env_" + re.sub(r"[^a-z0-9_]", "_", category.lower())[:32]
                        tool_error_counts[key] = tool_error_counts.get(key, 0) + 1

                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": tool_output})

        resp_obj = response_from_assistant_texts(assistant_texts, model)
        # Never forward the policy endpoint or api key to verify; dataset extras
        # (expected_answer, ...) must ride along for grading.
        verify_payload = body.model_dump(exclude={"policy_base_url", "policy_api_key"}) | {
            "response": resp_obj.model_dump(mode="json")
        }
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_payload,
            cookies=resources_cookies,
        )
        # Verify errors propagate: better to drop the trajectory than to fake a reward.
        await raise_for_status(verify_response)
        result = await get_response_json(verify_response)
        # Top-level numerics flow into gym's aggregate metrics and, via the miles
        # agent function, into sample.metadata["env_metrics"] -> wandb env/*.
        return result | {
            "policy_400_truncated": policy_400_truncated,
            "tool_call_count": tool_call_count,
            **tool_error_counts,
        }


if __name__ == "__main__":
    ToolCallAgent.run_webserver()
