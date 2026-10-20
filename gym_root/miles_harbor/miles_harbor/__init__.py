"""Harbor (https://github.com/harbor-framework/harbor) trials, unmodified, for miles' rollouts (trial_server) and
evaluations (`harbor run`), in the environment HARBOR_ENVIRONMENT_IMPORT_PATH names: Harbor's own docker environment
when it is unset, a sandbox backend a site provides (an import path; the hooks it may define are listed in
trial_server.py), or an OpenSandbox server's (opensandbox_env.py: miles_harbor.opensandbox_env:OpenSandboxEnvironment).
The harness packs the agents install from live under HARBOR_PACK_ROOT (default ~/.cache/miles-harness)."""
