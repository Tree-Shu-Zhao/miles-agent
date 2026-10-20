"""Chat-completions dialect helpers shared by the episode-driver agents.

Both drivers in this overlay speak the same wire dialect — Responses-form
/run inputs in, ``POST {policy_base_url}/chat/completions`` out, and a
BaseVerifyResponse-shaped ``response`` object back — while differing in the
episode protocol (tool_call_agent: OpenAI tool calling + /verify;
env_loop_agent: Gymnasium /reset + /step). This module holds the dialect so
the protocol loops stay the only difference between the two agents.
"""

import time
import uuid
from typing import Any

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


def to_chat_messages(params: NeMoGymResponseCreateParamsNonStreaming) -> list[dict[str, Any]]:
    """Responses-form ``input`` (message list or bare string) -> chat-completions messages.

    Keep only role/content: validated input items may carry extra fields that
    some OpenAI-compatible endpoints reject.
    """
    raw = params.input
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    messages: list[dict[str, Any]] = []
    for item in raw:
        d = item if isinstance(item, dict) else item.model_dump()
        messages.append({"role": d["role"], "content": d["content"]})
    return messages


def to_chat_tools(params: NeMoGymResponseCreateParamsNonStreaming) -> list[dict[str, Any]] | None:
    """Responses-form flat tools -> nested chat-completions tools.

    Returns None when there are none so the caller can omit the ``tools`` key
    entirely (some servers reject an empty list).
    """
    if not params.tools:
        return None
    chat_tools: list[dict[str, Any]] = []
    for tool in params.tools:
        t: dict[str, Any] = dict(tool) if isinstance(tool, dict) else tool.model_dump()
        if t.get("type", "function") != "function":
            continue
        chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": t["parameters"],
                },
            }
        )
    return chat_tools


def to_chat_sampling(params: NeMoGymResponseCreateParamsNonStreaming) -> dict[str, Any]:
    """Sampling params across the dialect boundary (max_output_tokens -> max_tokens)."""
    sampling: dict[str, Any] = {}
    if params.temperature is not None:
        sampling["temperature"] = params.temperature
    if params.top_p is not None:
        sampling["top_p"] = params.top_p
    if params.max_output_tokens is not None:
        sampling["max_tokens"] = params.max_output_tokens
    return sampling


def response_from_chat_message(msg: dict[str, Any], model: str) -> NeMoGymResponse:
    """One assistant chat-completions message -> a single-turn Responses object.

    Content becomes an output_text message item; each tool call becomes a
    function_call item (order preserved: content first, calls in list order).
    This is the shape EnvStepRequest.response carries, so a Gymnasium
    resources server can read structured actions instead of parsing text.
    """
    output: list[dict[str, Any]] = []
    text = msg.get("content") or ""
    if text:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "msg_0",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": tc.get("id") or f"call_{i}",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "",
                "status": "completed",
            }
        )
    return NeMoGymResponse.model_validate(
        {
            "id": f"resp_{uuid.uuid4().hex}",
            "created_at": time.time(),
            "model": model,
            "object": "response",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def response_from_assistant_texts(assistant_texts: list[str], model: str) -> NeMoGymResponse:
    """Assistant turns -> the Responses object the /run contract returns
    (and EnvStepRequest carries): one output_text message per non-empty turn."""
    output = [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": f"msg_{i}",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        for i, text in enumerate(assistant_texts)
        if text
    ]
    return NeMoGymResponse.model_validate(
        {
            "id": f"resp_{uuid.uuid4().hex}",
            "created_at": time.time(),
            "model": model,
            "object": "response",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )
