# Architecture: how an episode runs

How miles-agent wires miles to an environment: the episode flow, the division of labor, the three episode protocols
and where tool-call markup lives. Paths are relative to the repo root.

## The episode flow

```
miles train  ──custom-generate──▶ miles.rollout.generate_hub.agentic_tool_call.generate
                                        │  mints per-episode session URL
                                        ▼
                     miles_agent/gym_bridge.run
                                        │  POST /run {responses_create_params,
                                        │             metadata (tools, task, budgets),
                                        ▼             policy_base_url=session URL}
                     an episode server (a gym agent, or miles_harbor's trial server)
                        │ episode loop over POST {session}/v1/chat/completions
                        │ tool_calls → the env's routes (/search, /step, /sandbox/exec ...)
                        │ done       → the env's verdict → reward
                        ▼
                     Miles session server records every turn (TITO, miles' token-in/token-out protocol: token ids,
                     logprobs, loss masks; obs spans masked 0) → Sample(s)
                     reward rides sample.metadata["reward"] → reward hook
```

Division of labor: **Miles owns tokens** (fixed per-family TITO template,
append-only, drift asserted not healed), **the environment owns the episode**
(tool dispatch, sandboxes, verification), miles-agent owns only the thin
adapters above.

## The three episode protocols

Three episode protocols share this chassis; all speak OpenAI tool calling on the policy side and differ only in how
the episode runs:

- `tool_call_agent` — OpenAI tool-calling protocol: tools declared per row,
  each tool_call POSTs to the resources-server route of the same name, the
  MODEL ends the episode by not calling tools, reward comes from POST /verify.
  (wiki_search / search_r1.)
- `env_loop_agent` — Gymnasium protocol: the ENV speaks first (POST /reset),
  every assistant turn is one action attempt carried as a call to the row's
  action tool (POST /step), the ENV declares terminated/truncated, reward is
  the sum of step rewards; no /verify. (alfworld_env / alfworld, webshop_env /
  webshop, scienceworld_env / scienceworld.) The three env servers subclass
  `gym_root/agent_utils/session_gym.py` and share `procslots.py`, the
  worker-process slot pool. A /run body may carry `scripted_prefix`,
  assistant turns replayed through /step before the policy's first call so
  the policy resumes a recorded episode mid-trajectory; the /run response
  carries `last_observation`, the observation after the final action (the
  R2S takeover stage votes on its fingerprint).
- `miles_harbor.trial_server` — Harbor trials: POST /run runs one Harbor trial (the task's environment is Harbor's
  docker environment, or the backend named by `HARBOR_ENVIRONMENT_IMPORT_PATH`; a harness from
  `miles_harbor.harnesses`, installed from a pack under `HARBOR_PACK_ROOT` built by
  `gym_root/miles_harbor/packs/harbor_packs.sh`; the task's verifier) with the agent pointed at the rollout's
  session server; reward is the verifier's. (`recipes/tmax`; `scripts/harbor_trial_server.sh`,
  `scripts/harbor_run.sh`.) With Harbor's docker environment the pack must be visible inside every trial container:
  build it, then mount it into the containers yourself — this repository has no mount mechanism for it yet
  (`docs/data.md`). `miles_harbor.opensandbox_env` runs the sandboxes on an OpenSandbox server instead
  (`docs/local_setup.md`).

## Tool-call markup and model families

Tool-call markup belongs to the model, never to miles-agent: the policy
model's chat template renders the markup a prompt example shows
(`miles_agent/tool_call_markup.py`), miles' TITO family parses it back, and
envs, agents and rewards see structured calls only. `POLICY_MODEL` selects the
model (default Qwen3.5-4B); `launcher.MODEL_FAMILIES` maps its name prefix to
the TITO template, the engines' tool-call parser and the chat-template
kwargs, so a new family is one table entry (or the explicit `--tito-model` /
`--chat-template-kwargs` / `--megatron-model-type` flags). A mixture of
experts (a family marked `moe`, or a Qwen checkpoint named by its active
parameters, `-A3B`) trains with its experts over a node's GPUs and the
engines' routing replayed.
