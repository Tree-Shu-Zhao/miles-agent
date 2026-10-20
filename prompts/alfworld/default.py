"""ALFWorld: one example act call, no demonstration turns (every alfworld campaign so far).

Wording is load-bearing, not style: probed live against Qwen2.5-7B-Instruct at
temperature 1.0 (2026-08-31, sglang). Without a literal example call
the model put the command in the function-name slot or emitted bare JSON without
the tags (0/8 parsed); with it, turn-1 and after-tool-observation turns parsed
12/12 and 6/6. Once a bare-text turn is in history the model self-imitates and
never recovers (0/6), so first-turn compliance is everything. The example is the
policy model's own markup ({example_call}, rendered by its chat template); the
wording names no markup.
"""

SYSTEM = """\
You are an agent in a household environment, acting to complete a task. The \
first user message is the initial observation and includes your task. After \
each action, a tool message returns the result of that action plus the list \
of admissible actions you may take next.

Respond ONLY with a call to the act tool. The function name is always "act" \
(never a command); the command string goes in its command argument, copied \
verbatim from the admissible actions list. Example of a complete, correct \
reply:
{example_call}
A reply without an act tool call is invalid and wastes a turn."""

EXAMPLE_ARGUMENTS = {"command": "go to desk 1"}
