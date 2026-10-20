"""The generate function of the opd arm and the pilot dump (launcher.OPD_GENERATE_FN,
--custom-generate-function-path).

Wraps miles' agentic episode (agentic_tool_call.generate: the TITO session trace of one
env_loop_agent /run through gym_bridge.run) and, on TRAINING samples only,
  1. segments the merged token stream into turns at the model's call marker and records the
     deliberation metrics (metadata["env_metrics"]["opd_*"] -> wandb rollout/env/opd_*),
  2. appends the episode to the R2S pilot dump when MILES_OPD_DUMP_DIR is set
     (miles_agent.opd.dump; before scoring, the rows carry no teacher log-probs),
  3. under --use-opd scores the stream on the teacher into Sample.teacher_log_probs
     (miles_agent.opd.teacher_scoring). A sample the teacher could not score is ABORTED, so
     the recipes' no-aborted filter drops its group: no step trains on a half-scored sample.
TRUNCATED samples train in miles and are scored like COMPLETED ones; a non-aborted sample
with no response is neither scored nor aborted (an empty teacher list keeps miles' length
checks happy). Evaluation samples pass through untouched. The structured transcript
gym_bridge forwards under the dump env var rides sample.metadata for this one hop and is
popped from every sample, eval included, before anything else sees it.
"""

import functools
import json
import logging
import time

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.types import Sample

from miles_agent import gym_bridge, tool_call_markup
from miles_agent.opd import dump, teacher_scoring, turns

logger = logging.getLogger(__name__)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    output = await agentic_tool_call.generate(input)
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    transcripts = [sample.metadata.pop(gym_bridge.TRANSCRIPT_KEY, None) for sample in samples]
    if input.evaluation:
        return output
    for sample, transcript in zip(samples, transcripts, strict=True):
        await _annotate(sample, transcript, input.args)
    return output


async def _annotate(sample: Sample, transcript: list[dict] | None, args) -> None:
    """Turn metrics, the dump row and the teacher scoring of one training sample."""
    if sample.status is Sample.Status.ABORTED:
        return
    if sample.response_length == 0:
        if args.use_opd:
            sample.teacher_log_probs = []  # length 0 == response_length: passes the batch checks
        _record(sample, {"opd_empty_response": 1.0})
        return
    response_tokens = list(sample.tokens[-sample.response_length :])
    episode_turns = turns.segment_turns(response_tokens, sample.loss_mask, marker_id(args, _tools(sample)))
    _record(sample, turns.turn_metrics(episode_turns))
    dump.dump_row(sample, episode_turns, transcript, args)
    if args.use_opd:
        _record(sample, await _score(sample, args))


async def _score(sample: Sample, args) -> dict[str, float]:
    t0 = time.monotonic()
    try:
        return await teacher_scoring.score_sample(args, sample)
    except teacher_scoring.TeacherScoringError as e:
        logger.error(f"OPD_TEACHER_SCORING_FAILED sample index={sample.index}: {e}")
        sample.teacher_log_probs = None
        sample.status = Sample.Status.ABORTED
        return {
            "opd_teacher_scoring_failed": 1.0,
            "opd_teacher_ctx_overflow": float(e.ctx_overflow),
            "opd_teacher_latency_s": time.monotonic() - t0,
        }


def _record(sample: Sample, metrics: dict[str, float]) -> None:
    env_metrics = dict(sample.metadata.get("env_metrics") or {})
    env_metrics.update(metrics)
    sample.metadata["env_metrics"] = env_metrics


def _tools(sample: Sample) -> list[dict]:
    tools = sample.metadata.get("tools")
    if not tools:
        raise ValueError(
            f"sample index={sample.index} declares no tools: the call marker comes from the env's action tool"
        )
    return tools


def marker_id(args, tools: list[dict]) -> int:
    """The id of the token that opens a tool call for this launch's policy, derived once per
    process: the launch's family template (--tito-model, --apply-chat-template-kwargs) rendered
    through the policy tokenizer (--hf-checkpoint) around the rows' action tool, never a literal."""
    kwargs = args.apply_chat_template_kwargs
    kwargs_key = kwargs if isinstance(kwargs, str) else json.dumps(kwargs or {}, sort_keys=True)
    return _marker_id(args.tito_model, args.hf_checkpoint, kwargs_key, json.dumps(tools[0], sort_keys=True))


@functools.cache
def _marker_id(tito_model: str, hf_checkpoint: str, chat_template_kwargs: str, tool: str) -> int:
    from transformers import AutoTokenizer  # the actors' image

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    render = tool_call_markup.tito_render_chat(tito_model, tokenizer, chat_template_kwargs)
    try:
        return tool_call_markup.marker_token_id(tokenizer, render, json.loads(tool))
    except ValueError as e:
        raise ValueError(f"--tito-model {tito_model}: {e}") from e


def _add_arguments(parser) -> None:
    """miles registers a generate function's flags through this attribute (arguments.py): the
    wrapped function's, then the teacher scoring knobs at the campaign's values."""
    inner = getattr(agentic_tool_call.generate, "add_arguments", None)
    if callable(inner):
        inner(parser)
    parser.add_argument(
        "--opd-score-concurrency", type=int, default=32, help="Teacher scoring calls in flight per rollout process."
    )
    parser.add_argument(
        "--opd-score-timeout-s", type=float, default=900.0, help="Timeout of one teacher scoring call in seconds."
    )
    parser.add_argument(
        "--opd-score-attempts",
        type=int,
        default=4,
        help="Attempts per teacher scoring call (transport failures, timeouts and 5xx; 2-10 s backoff).",
    )


generate.add_arguments = _add_arguments
