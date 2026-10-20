"""ALFWorld, the R2S paper's one-shot prompt (Table 4 protocol; the campaign branch's
recipes/alfworld_opd/prepare_data.py --prompt-style fewshot).

Brief plain-text deliberation (one to three sentences) before exactly one act call on
every turn, thinking off at the template, and one worked example episode as REAL
conversation turns after the system message (two actions, then a tool result that
closes the example). Probe 2026-09-09 (two runs, 60 first turns x 2, T 0.4,
no thinking): Qwen3-1.7B deliberated before a valid act call in 1% of replies with the
plain prompt, 2-7% with a text example, 88% with the example as turns; Qwen3-4B >= 95%
under every variant. The paper's code wrote the example's calls per family (hermes JSON
for Qwen3, qwen3_coder XML for Qwen3.5, two files); here the calls are structured and
the model's chat template writes them.
"""

SYSTEM = """\
You are an agent in a household environment, acting to complete a task. The \
first user message is the initial observation and includes your task. After \
each action, a tool message returns the result of that action plus the list \
of admissible actions you may take next.

On every turn, first reason briefly in plain text (one to three sentences) \
about the current situation and what to do next, then call the `act` tool \
exactly once. The function name is always "act"; the command string goes in \
the `command` argument and must be copied verbatim from the admissible \
actions list. Never answer without a tool call and never call the tool more \
than once per turn."""

TURNS = [
    {
        "role": "user",
        "content": (
            "You are in the middle of a room. Looking quickly around you, you see a countertop 1, "
            "a desk 1, a drawer 1, and a shelf 1.\n\nYour task is to: put a clean mug on the desk.\n"
            "Your admissible actions are: [go to countertop 1, go to desk 1, go to drawer 1, "
            "go to shelf 1, inventory, look]"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The task is to put a clean mug on the desk. I have not seen a mug yet, so I will start "
            "by checking the countertop."
        ),
        "tool_calls": [{"name": "act", "arguments": {"command": "go to countertop 1"}}],
    },
    {
        "role": "tool",
        "content": (
            "You arrive at countertop 1. On the countertop 1, you see a mug 1.\n"
            "Your admissible actions are: [take mug 1 from countertop 1, go to desk 1, look]"
        ),
    },
    {
        "role": "assistant",
        "content": "The mug is here; I will take it so I can clean it and bring it to the desk.",
        "tool_calls": [{"name": "act", "arguments": {"command": "take mug 1 from countertop 1"}}],
    },
    {"role": "tool", "content": "[example ended; a new task follows]"},
]
