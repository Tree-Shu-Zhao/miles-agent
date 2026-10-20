"""R2S teacher repair (the paper's pipeline) over the pilot dump of an opd arm, in token space.

Stages, each `python -m miles_agent.r2s.<stage>` over plain jsonl (docs/r2s.md section 4):
  A  divergence.py  mine the dump: per turn mean(lp_student - lp_teacher) over the deliberation
                    span, argmax turn per episode, cap per goal, top-N states
  B  takeover.py    replay a state's student turns as env_loop_agent's scripted_prefix and let
                    the base teacher finish the episode K times through the live env
  J  judge.py       WebShop only: the teacher as judge reads each purchase against the instruction
  C  select.py      the per-env table (EnvConfig, ENV_DEFAULTS) and the one selection rule: the
                    takeovers passing the env's check, grouped by how they end; the largest group
                    needs m of them and its shortest takeover is tau
  D  label.py       the student's episode scored by the frozen teacher with tau spliced in as a
                    privileged user message: top-k targets on every assistant token from the
                    divergence turn to the end
  E  teacher_ft.py  full-parameter FT of the teacher on the labels (FSDP under torchrun), promoted
                    only when the held-out privileged CE drops; bf16 HF export for sglang
common.py holds the dump row (Episode), the structured-transcript helpers, jsonl io, summaries
with their DONE marker, the resume primitives and the flags every stage shares. No stage parses
chat text: tool calls arrive as the engine parser's structured messages and every token-space
fact comes from the model's own template (miles_agent.tool_call_markup).
"""
