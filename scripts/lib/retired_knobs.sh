#!/bin/bash
# The OPD campaign branches' job knobs that main does not read, refused with their replacement the moment the head
# starts (the STUDENT_MODEL rule of the training entry: never silently ignored). Sourced; it only reads the environment.
retired_knob() {  # <name> <replacement>
  [ -z "${!1:-}" ] || { echo "FATAL: $1 is a retired knob: $2"; exit 1; }
}
retired_knob TEACHER2_MODEL "one teacher per run, TEACHER_MODEL / TEACHER_HF / TEACHER_URL (the two-teacher alpha mixture stayed on the campaign branch)"
retired_knob ONESHOT_PROTOCOL "RUN_SCRIPT=opd.py with PROMPT=r2s_oneshot (prompts/<env>/r2s_oneshot.py, the paper's one-shot rows)"
retired_knob ALFWORLD_OPD_PROMPT "PROMPT=r2s_oneshot (prompts/alfworld/)"
retired_knob SCIW_OPD_PROMPT "PROMPT=r2s_oneshot (prompts/scienceworld/)"
retired_knob OPD_TOOL_CALL_FORMAT "the tool-call markup belongs to the model: miles_agent/launcher.py MODEL_FAMILIES, from POLICY_MODEL's name"
retired_knob TEACHER_TOOL_PARSER "the served model's family names its parser (MODEL_FAMILIES); the OPD teacher scores tokens and parses nothing"
retired_knob WEBSHOP_TOOL_CALL_STYLE "the policy model's own chat template renders the WebShop rows' example call (MODEL_FAMILIES)"
for knob in $(compgen -e | grep -E '_OPD_REFS$' || true); do
  retired_knob "$knob" "the FTB baseline (reference actions) is not in main; docs/r2s.md points at its branch"
done
retired_knob WEBSHOP_OPD_REF_FILE "the FTB baseline (reference actions) is not in main; docs/r2s.md points at its branch"
retired_knob TOOLS_FILE "the R2S pipeline's per-environment config (--env, miles_agent/r2s/select.py) supplies the tools"
retired_knob VOTE_KEY "the R2S pipeline's per-environment config (--env, miles_agent/r2s/select.py) fixes the vote key"
retired_knob TAU_SELECT "the paper's one selection path per environment; R2S_ARGS_C carries a stage's extra flags"
