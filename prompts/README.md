# prompts/ — the words of every recipe, one file per prompt

`prompts/<env>/<name>.py` is a prompt for the `<env>` environment. A recipe's
`prepare_data.py` loads one by name and renders it for the policy model
(`miles_agent/prompts.py`); the job selects it with `PROMPT=<name>`
(`--prompt <name>` on the script), and each recipe names its default
(alfworld: `default`, scienceworld: `default`, webshop: `think`). Nothing else in the repo knows a
prompt's words: envs see tool calls, rewards see outcomes, tests check shape.

To try a variant, copy a file and edit the words — no code changes, no
registration. Files are self-contained on purpose (shared paragraphs are
repeated, not imported), so copying one never drags another along.

## What a prompt file holds

```python
SYSTEM = """... Example of a complete, correct reply:
{example_call}
..."""                                       # the system message
EXAMPLE_ARGUMENTS = {"command": "go to desk 1"}  # the example call's arguments
TURNS = [                                    # optional demonstration turns
    {"role": "user", "content": "You are in the middle of a room ..."},
    {"role": "assistant", "content": "I have not seen a mug yet ...",
     "tool_calls": [{"name": "act", "arguments": {"command": "go to countertop 1"}}]},
    {"role": "tool", "content": "You arrive at countertop 1 ..."},
]
```

- `SYSTEM` — the system message. `{example_call}` is replaced by one call of
  the row's action tool with `EXAMPLE_ARGUMENTS`, written as the model's chat
  template writes a call (hermes JSON for Qwen3, the qwen3_coder XML for
  Qwen3.5, ...). A prompt that shows several calls in its text labels them,
  `{example_call:search}` ... with `EXAMPLE_CALLS = {"search": {...}, ...}`.
  The words never name a markup. A prompt whose words show no call
  (scienceworld `default`) has neither `{example_call}` nor `EXAMPLE_ARGUMENTS`.
- `TURNS` — demonstration turns appended after the system message (one-shot /
  few-shot prompts). Calls and tool results are structured here; `render()`
  writes them in the model's markup and proves the result byte-equal to what
  the template renders for the structured turns, in context. The agents
  forward role/content only, which is why the rendering happens at prepare
  time rather than in the session server.

The tool schema stays with the recipe (`ACT_TOOL` in its `prepare_data.py`):
the env server's action tool name is an env contract, not wording.

## Prompts in the repo

| env | name | what |
|---|---|---|
| alfworld | `default` | one example act call, no demonstration (every alfworld campaign so far) |
| alfworld | `r2s_oneshot` | the R2S paper's one-shot prompt: brief deliberation, then one call, one worked example as turns |
| scienceworld | `default` | the paper's plain prompt: TCOD's wording of the lists, brief deliberation, then one call; no example call, no demonstration |
| scienceworld | `r2s_oneshot` | the R2S paper's one-shot prompt: the same system message, one worked example (hallway -> kitchen, two actions) as turns |
| webshop | `think` | TCOD protocol: reason in `<think></think>`, then the call (default) |
| webshop | `none` | the call only (wstrain1 / wstrain2) |
| webshop | `r2s_oneshot` | the R2S paper's one-shot prompt: deliberation, then one call, one example episode in the system text |

The three `r2s_oneshot` files reproduce the paper's rows byte for byte under the
Qwen3 (hermes) and Qwen3.5 (qwen3_coder) templates — the paper's code kept two
renderings per environment for that.

## Where prompts do not live here

- `search_r1` — the rows come from the wiki_search gym data with their own
  messages; `recipes/search_r1/prepare_data.py` shuffles fields only.
- `tmax` — the agent harness (mini-swe-agent, TMax's vanillux) owns its
  prompts; the recipe runs it as a black box.
