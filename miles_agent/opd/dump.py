"""Per-process jsonl dump of training episodes in token space (the R2S pipeline's input).

One row per finished training sample, REF's keys first (the campaign rows), the v2 fields after:
  {"rollout_id", "index", "status", "reward" (metadata["reward"]: the env score the agent
   returned; raw_reward is written by reward_routing, which runs after the generate hook),
   "metadata" (game_file/split/data_source/max_episode_steps/goal_id/tools... - every
   jsonable field but the bulky and the per-run ones), "env_metrics", "tokens" (prompt +
   response), "response_length", "loss_mask", "turns": [[start, end, delib_end], ...]
   (response-relative, turns.py),
   "v": 2, "prompt" (the row's chat messages), "transcript" (the structured chat
   env_loop_agent produced: prompt messages, the reset observation, assistant messages with
   their tool calls, tool/user observations; None when the agent returned none),
   "tito_model", "chat_template_kwargs" (the launch's render, for a stage that re-renders)}
Stage A reads tokens/loss_mask/turns, stage B replays the transcript's assistant messages as
a scripted_prefix, stage D splices into the prompt tokens; no stage parses chat text.

Files: $MILES_OPD_DUMP_DIR/dump_<hostname>_<pid>.jsonl, appended and flushed per row so a
killed job keeps everything written so far. The env var is the dump's one switch (the
launcher's --opd-dump exports it to the actors).
"""

import json
import os
import socket
import threading
from typing import Any

from miles_agent import gym_bridge

# "tools" is kept: the R2S takeover stage re-declares the env action tool from it.
_SKIP_METADATA_KEYS = {"env_metrics", "agent_metrics", "reward", "raw_reward", gym_bridge.TRANSCRIPT_KEY}
_lock = threading.Lock()
_handles: dict[str, Any] = {}


def dump_dir() -> str | None:
    return os.environ.get("MILES_OPD_DUMP_DIR") or None


def _handle(directory: str):
    if directory not in _handles:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"dump_{socket.gethostname()}_{os.getpid()}.jsonl")
        _handles[directory] = open(path, "a", encoding="utf-8")
    return _handles[directory]


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def dump_row(sample, turns, transcript: list[dict] | None, args) -> None:
    """Append the sample's row (above) when MILES_OPD_DUMP_DIR is set; a no-op otherwise."""
    directory = dump_dir()
    if not directory:
        return
    metadata = sample.metadata or {}
    row = {
        "rollout_id": sample.rollout_id,
        "index": sample.index,
        "status": sample.status.value,
        "reward": metadata.get("reward"),
        "metadata": {
            k: v
            for k, v in metadata.items()
            if k not in _SKIP_METADATA_KEYS and not k.startswith("session_server") and _jsonable(v)
        },
        "env_metrics": metadata.get("env_metrics") or {},
        "tokens": list(sample.tokens),
        "response_length": sample.response_length,
        "loss_mask": list(sample.loss_mask) if sample.loss_mask is not None else None,
        "turns": [[t.start, t.end, t.delib_end] for t in turns],
        "v": 2,
        "prompt": sample.prompt,
        "transcript": transcript,
        "tito_model": args.tito_model,
        "chat_template_kwargs": args.apply_chat_template_kwargs,
    }
    line = json.dumps(row, ensure_ascii=False)
    with _lock:
        f = _handle(directory)
        f.write(line + "\n")
        f.flush()
