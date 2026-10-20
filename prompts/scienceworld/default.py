"""ScienceWorld: the paper's plain system prompt, no example call, no demonstration turns
(the campaign branch's recipes/scienceworld_opd/prepare_data.py SYSTEM_PROMPT, its
--prompt-style plain).

TCOD's wording of what the model sees (task, observation, action commands, objects;
arXiv 2604.24005) and the R2S protocol's turn shape: brief plain-text deliberation,
then exactly one act call, thinking off at the template. The command is composed from
an action command and object(s), not copied from an admissible list (ScienceWorld has
none; TCOD checks none), so the prompt shows two commands in words and no example
call: the first turn's act call is what the one-shot variant (r2s_oneshot) demonstrates.
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
