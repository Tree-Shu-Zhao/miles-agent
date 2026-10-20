"""WebShop, the R2S paper's one-shot prompt (Table 4 protocol; the campaign branch's
recipes/webshop_opd/prepare_data.py).

Brief plain-text deliberation before exactly one act call on every turn, thinking off at
the template, and ONE worked example episode inside the system message as a format
anchor (the earlier environment code's FEWSHOT=1): page observation -> brief reasoning -> one act call whose
command is copied verbatim from the admissible actions; observations abbreviated with
'...'. The example's four calls are labelled placeholders, written by the model's chat
template (the paper's code wrote them per family: hermes JSON for Qwen3, qwen3_coder XML
for Qwen3.5; a pseudo-call notation was imitated literally by Qwen3-4B and collapsed its
tool calling). The goal instruction is NOT here: it arrives in the first observation.
"""

_TASK = """\
You are an expert autonomous shopping agent operating in the WebShop \
e-commerce environment. The first user message is the initial observation and \
includes your task (the item to buy and its required attributes). After each \
action, a tool message returns the new page plus the list of admissible \
actions you may take next.

On every turn, first reason briefly in plain text (one to three sentences) \
about the page and which admissible action best advances the shopping goal, \
then call the `act` tool exactly once with that command, copied verbatim from \
the admissible actions list. Commands look like search[<query>] (only on a \
page with a search bar) or click[<button or product code>]. A typical episode \
is: search with the key attributes, click the best-matching product code, \
click each required option (size, color, ...), then click[buy now] to finish. \
You MUST finish with click[buy now] before running out of steps; an episode \
that never buys scores zero. Never answer without a tool call and never call \
the tool more than once per turn.
"""

_HEAD = (
    "Here is one example episode illustrating the response format (observations "
    "abbreviated with '...'). Every action is copied exactly from the admissible "
    "actions of that step.\n"
)

# (observation, reasoning, labelled call) per example step; EXAMPLE_CALLS holds the commands.
_STEPS = (
    (
        "Task: i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars.\n"
        "Observation: 'Search'. Admissible actions: ['search[<your query>]'].",
        "The page only offers a search bar, so I should search for the product with its key attributes.",
        "{example_call:search}",
    ),
    (
        "Observation: 'Page 1 (Total results: 50)' [SEP] 'B000VOHH8I' [SEP] 'L'eau "
        "D'issey By Issey Miyake for Men Eau De Toilette Spray' [SEP] '$64.98' [SEP] "
        "'...'. Admissible actions: ['click[b000vohh8i]', 'click[next >]', ...].",
        "B000VOHH8I is a l'eau d'issey eau de toilette at $64.98, within the 100.00 budget. I will open its page.",
        "{example_call:open}",
    ),
    (
        "Observation: '6.76 fl oz (pack of 1)' [SEP] 'Price: $64.98' [SEP] 'Buy Now' "
        "[SEP] '...'. Admissible actions: ['click[6.76 fl oz (pack of 1)]', "
        "'click[buy now]', ...].",
        "This is the right product. The task asks for the 6.76 fl oz size, so I select that option before buying.",
        "{example_call:option}",
    ),
    (
        "Observation: 'You have clicked 6.76 fl oz (pack of 1)' [SEP] '...'. "
        "Admissible actions: ['click[buy now]', ...].",
        "The size matches and the price is under 100.00 dollars, so I finish the purchase.",
        "{example_call:buy}",
    ),
)

SYSTEM = (
    _TASK
    + "\n"
    + _HEAD
    + "\n"
    + "\n\n".join(f"{obs}\nReply: {reasoning}\n{call}" for obs, reasoning, call in _STEPS)
    + "\n\nEnd of example. Now here is your task."
)

EXAMPLE_CALLS = {
    "search": {"command": "search[l'eau d'issey 6.76 fl oz bottle]"},
    "open": {"command": "click[b000vohh8i]"},
    "option": {"command": "click[6.76 fl oz (pack of 1)]"},
    "buy": {"command": "click[buy now]"},
}
