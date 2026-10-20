"""ScienceWorld, the R2S paper's one-shot prompt (Table 4 protocol; the campaign branch's
recipes/scienceworld_opd/prepare_data.py --prompt-style fewshot).

The same system message as `default` (brief plain-text deliberation, then exactly one act
call on every turn, thinking off at the template) followed by one worked example episode
as REAL conversation turns: two actions in TCOD's wording (the fixed hallway -> kitchen
episode of the paper's code), then a tool result that closes the example. The turns are
the shape the alfworld_opd probe found necessary for Qwen3-1.7B to deliberate before a
valid call (prompts/alfworld/r2s_oneshot.py); the Table 4 campaign used it for every
student. The paper's code wrote the example's calls per family (hermes JSON for Qwen3,
qwen3_coder XML for Qwen3.5, --tool-call-format); here the calls are structured and the
model's chat template writes them.
"""

SYSTEM = """\
You are an expert agent operating in the ScienceWorld text environment. The \
first user message states your task, your current observation, the available \
action commands and the objects you can interact with. After each action, a \
tool message returns the environment feedback plus the updated action commands \
and objects.

On every turn, first reason briefly in plain text (one to three sentences) \
about the current situation and what to do next, then call the `act` tool \
exactly once. The function name is always "act"; the `command` argument is one \
valid action formed by combining an action command with appropriate object(s), \
for example "open door to kitchen" or "focus on thermometer". Never answer \
without a tool call and never call the tool more than once per turn."""

_ACTIONS = (
    "'activate OBJ', 'close OBJ', 'focus on OBJ', 'go to OBJ', 'look around', 'open OBJ', 'pick up OBJ', 'use OBJ'"
)

TURNS = [
    {
        "role": "user",
        "content": (
            "Your ScienceWorld task is: Your task is to focus on the thermometer in the kitchen.\n"
            "Your current observation is: This room is called the hallway. In it, you see: the agent, "
            "a substance called air, a picture. You also see: A door to the kitchen (that is closed).\n"
            f"Available action commands: [{_ACTIONS}]\n"
            "Available objects you can interact with: ['air', 'door to kitchen', 'hallway', 'picture']"
        ),
    },
    {
        "role": "assistant",
        "content": "The thermometer is in the kitchen and the kitchen door is closed, so I open it first.",
        "tool_calls": [{"name": "act", "arguments": {"command": "open door to kitchen"}}],
    },
    {
        "role": "tool",
        "content": (
            "The door is now open.\n"
            f"Available action commands: [{_ACTIONS}]\n"
            "Available objects you can interact with: ['air', 'door to kitchen', 'hallway', 'kitchen', 'picture']"
        ),
    },
    {
        "role": "assistant",
        "content": "The door is open; I move into the kitchen to find the thermometer.",
        "tool_calls": [{"name": "act", "arguments": {"command": "go to kitchen"}}],
    },
    {"role": "tool", "content": "[example ended; a new task follows]"},
]
