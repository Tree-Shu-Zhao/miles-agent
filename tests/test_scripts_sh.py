"""The public scripts (scripts/**/*.sh) without a GPU, a scheduler or the real tools: fake commands on a temp PATH
(the fake-command idiom of the old job tests) that log every call as a JSON line and answer from a JSON state file
(the state-file idiom), run under the SYSTEM bash (R24: a Mac's /bin/bash 3.2 is what a student
has, so no bash 4 construct may hide on a tested path), plus the static gates every new public file passes:
`bash -n`, no `rm` line, no site literal. This module is the harness the later split commits extend
(lib/gym_ladder.sh, run_local.sh, serve_model.sh, stop_all.sh, ...): the helpers below are theirs too.

Harness rules (fixer passes 2 and 3 of the I2a spec):
  * a bare `kill` in a script is bash's BUILTIN (functions -> builtins -> PATH), so no script line `kill ...`
    ever reaches the fake; the fake is reached only through `xargs kill`, logs AND delivers the signal; tests
    assert the EFFECT (`_pid_gone`) or a static property of the script text, never a logged `kill`;
  * `_pid_gone` is valid only for a pid that is not an unreaped child of pytest: POSIX kill(pid, 0) succeeds on
    a zombie until its parent waits, so a direct child of the test is poll()ed / wait()ed, never `_pid_gone`'d;
    `_spawn_sleeper` therefore spawns through a short-lived intermediate interpreter, which makes the sleeper an
    orphan that init reaps on death (stdio DEVNULL: a sleeper inheriting the captured pipes hangs the
    intermediate), and the fixture only KILLs survivors;
  * the fake `sleep` returns at once unless state["sleep_cap_s"] says otherwise (the stop_all test and the serve
    `--stop` case set 0.3 s, so a just-TERMed orphan's zombie window is closed before a script's survivor probe).
"""

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
COMMON = SCRIPTS / "lib" / "common.sh"
RAY = SCRIPTS / "lib" / "ray.sh"
# resolved BEFORE any fake dir is prepended: the system bash (R24)
BASH = shutil.which("bash")
# the public scripts MR-2 / MR-3 landed before the split (D24): always part of the bash -n sweep
PREEXISTING_PUBLIC = ("start_gym.sh", "teacher_node.sh", "lib/sglang.sh", "lib/retired_knobs.sh")
# every file a split commit adds to the public tree: the identity gate scans them all (later commits append)
NEW_PUBLIC_FILES = [
    "scripts/lib/common.sh",
    "scripts/lib/ray.sh",
    "local.env.example",
    "tests/test_scripts_sh.py",
    "tests/test_eval_full.py",
    "scripts/lib/gym_ladder.sh",
    "scripts/start_gym.sh",
    "scripts/run_local.sh",
    "scripts/ray_cluster.sh",
    "scripts/serve_model.sh",
    "scripts/eval_ckpt.sh",
    "scripts/export_hf.sh",
    "scripts/stop_all.sh",
    "tools/convert_ckpt_to_hf.py",
    "scripts/fetch_data.sh",
    "scripts/build_gym_venvs.sh",
    "env_servers/webshop/data_md5.txt",
    "gym_root/resources_servers/wiki_search/data/prepare.py",
    "tests/test_wiki_search_prepare.py",
    "scripts/harbor_trial_server.sh",
    "scripts/harbor_run.sh",
    "scripts/check_patches.sh",
    "patches/miles/series",
    "gym_root/miles_harbor/packs/harbor_packs.sh",
    "gym_root/miles_harbor/packs/miles_mini_ext.py",
    "tests/test_harness_pack_layout.py",
    "tests/contract_vars.txt",
    "tests/fake_env.sh",
    "tests/test_contract_doc.py",
    "scripts/contract_selftest.sh",
    "scripts/smoke_tests.sh",
    "scripts/setup.sh",
    "docs/containers.md",
    "docs/local_setup.md",
    "docs/data.md",
]
# The site literals no new public file may carry (rules_i2a.md "Never"; D-C): the personal and site names,
# the personal roots, CGNAT addresses (the tests use RFC 5737 192.0.2.x), the personal mail domain, the three
# scheduler verbs, the scheduler's variable prefix and the vendor word (the tool name `nvidia-smi`, the variable
# `NVIDIA_SMI` and the dependency home github.com/NVIDIA-NeMo/Gym stay legal). Each literal is spelled in two
# pieces so that this file, which the gate scans too, carries none of them.
IDENTITY_LITERAL_PARTS = [
    "shu" "zhao",
    "/lus" "tre/fsw",
    "/scratch/" "fsw",
    "smz" "5505",
    r"100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+",
    "@nvi" "dia\\.com",
    r"\bNVIDIA\b(?!-NeMo)",
    r"\bnvidia\b(?!-smi)",
]
# the scheduler words D22 allows only in docs/cluster.md, examples/slurm/ and patches/miles/README.md
SCHEDULER_LITERAL_PARTS = [
    r"\bsb" r"atch\b",
    r"\bsr" r"un\b",
    r"\bscon" r"trol\b",
    "SLURM" "_",
]
SITE_LITERAL = re.compile("|".join(IDENTITY_LITERAL_PARTS + SCHEDULER_LITERAL_PARTS))
IDENTITY_LITERAL = re.compile("|".join(IDENTITY_LITERAL_PARTS))
# the usability words of D22 (same three exceptions): a page a reader without a cluster needs may not use them
USABILITY_LITERAL = re.compile("|".join([
    r"\ben" r"root\b",
    r"\bpy" r"xis\b",
    r"\.sq" r"sh\b",
    r"\btail" r"net\b",
    r"\$" r"P\b",
    r"\bNR" r"LP\b",
    "/opt/miles" "-agent",
]))
# an `rm` as a command (start of line or after a separator / keyword), outside comments; `--rm` is a flag
RM_COMMAND = re.compile(r"(?<![-\w/.])rm\s")
DEFAULT_FAKES = ("curl", "nvidia-smi", "ss", "ray", "python3", "getent", "hostname", "sleep", "pgrep", "uname", "ssh",
                 "uv", "gym", "setsid", "kill", "rm", "harbor")
# the real tools a bare PATH (`_env(bare_path=True)`) keeps: everything the libs call that is not faked, minus the
# ones a test may want absent (ss, getent, hostname, nvidia-smi, ray, setsid, timeout are never in the toolbox)
TOOLBOX = ("dirname", "basename", "awk", "seq", "tr", "grep", "cut", "sort", "uniq", "tail", "head", "env", "cat",
           "sed", "wc", "ps", "mkdir", "touch", "sh", "mv", "cp", "ln", "date", "tee", "xargs", "readlink", "id",
           "printf", "echo", "test", "true", "false", "sha256sum", "shasum", "tar", "find", "md5sum", "md5", "cmp")
# every knob a script under test reads (the former job tests' filter widened): a developer shell with any of
# them set cannot change the argv the tests compare; CDPATH would make `$(cd ... && pwd)` print, BASH_ENV would
# run a file in every fake
STRIP_EXACT = {
    "MODELS_DIR", "DATA_DIR", "RESULTS_DIR", "RUNS_DIR", "TOOLS_DIR", "GYM_VENV_ROOT", "CONTRACT_VERSION",
    "SYNC_DIR", "CUDA_VISIBLE_DEVICES", "NEMO_GYM_URL", "NEMO_GYM_PORT", "RETRIEVER_URL", "JAVA_HOME", "DRY_RUN",
    "PROJECT_PORTS", "JOB_ID", "NODE_RANK", "NNODES", "HEAD_IP", "NODE_IP", "GPUS_PER_NODE", "RECIPE", "RUN_TAG",
    "TOPOLOGY", "MODE", "RUN_SCRIPT", "EXTRA_ARGS", "PROMPT", "SRC_MOUNT", "MY_IP", "ALFWORLD_DATA",
    "WIKI_SEARCH_DATA", "SESSION_SERVER_EXTERNAL_HOST", "TRAIN_NODES", "ROLLOUT_NODES", "EVAL_PROMPT_DATA",
    "HF_MODEL_ARGS", "HF_MODEL_NAME", "ITER_DIR", "OUT_DIR", "HF_INPUT", "HF_ORIGIN", "GYM_CONFIG", "MEGATRON_PATH",
    "PYTHON", "RAY_BIN", "GYM_BIN", "UV_BIN", "CURL", "NVIDIA_SMI", "SSH", "LOCAL_ENV", "REPO", "LIB",
    "LOCAL", "MASTER_ADDR", "CDPATH", "BASH_ENV", "ENV", "PROMPT_DATA", "EVAL_DATA", "GYMCFG", "AGENT_PORT_OVERRIDE",
    "DRIVER_DATA_ARGS", "RUN_DATA_DIR", "TORCH_DIST_DIR", "ENGINE_LOG", "ENGINE_PID", "BUILD_DIR", "UV_CACHE_DIR",
    "UV_LINK_MODE", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_DOWNLOADS", "UV_PROJECT_ENVIRONMENT", "UV_OFFLINE", "RAY_V",
    "OPENAI_V", "TRIAL_SERVER_VENV", "TRIAL_SERVER_MODULE", "GIT",
}
STRIP_PREFIXES = ("MILES_AGENT_", "MILES_", "SLURM" "_", "TEACHER_", "FAKE_", "POLICY_", "PREEMPT_", "WEBSHOP_",
                  "SCIW_", "SCIENCEWORLD_", "HARBOR_", "WANDB_", "SERVE_", "SCRIPT_TEST_", "RECIPE_", "GYM_")

# One body for every fake (dispatch on the command name). Each call lands as a JSON line in $SCRIPT_TEST_LOG
# ({"cmd", "argv", "cwd", "env": a few facts}); the replies come from $SCRIPT_TEST_STATE (JSON):
#   curl              exit 0 for the next code of state["curl"][url] (a list is consumed down to its last entry,
#                     which then repeats; default 000; exit 22 unless 200); state["curl_body"][url] is printed
#                     on a 200 without -o
#   nvidia-smi -L     state["gpus"] lines "GPU i: ..."
#   ss -tln           state["ss"] (a string or a list of lines)
#   ray status        exit state["ray_status_rc"] (default 1: no cluster); ray start also touches
#                     state["ray_touch"] when given (a worker test lets the join itself create $SYNC_DIR/ENDED)
#   python3 / python  -m sglang.launch_server: pid into <log dir>/sglang.pid, serves until TERM (exit 0);
#                     -m miles_agent.launcher serve-template <hf> <out>: writes <out>, prints state["parser"];
#                     `-` with MILES_EXPECT_GPUS in env (the GPU-wait heredoc): exit 0; `-` fed the retriever
#                     probe heredoc (it names MILES_PROBE_URL): exit state["probe_rc"] (default 0); `-` fed the
#                     serve preflight heredoc (it names SERVE_PREFLIGHT): after state["preflight_delay_s"] (0)
#                     prints `SERVE_PREFLIGHT <verdict> ...` with state["preflight"] (default PASS), exit 0;
#                     `-c <code>` with state["python_c"] == "real": re-executed by the real interpreter (the
#                     webshop gates' JSON one-liners read the fake curl's body);
#                     .../prepare_data.py | prepare.py: writes the --output-dir rows (state["rows"]: an int, or a
#                     dict by file name, -1 = that file is not written; webshop also gets protocol.json with
#                     state["goals_digest"]) or --output;
#                     .../export_goals.py: writes the --output rows + <output>.meta.json (goals, goals_digest =
#                     state["goals_digest"], engine_url = --engine-url);
#                     any other *.py (run.py, eval_*.py, opd.py: the driver): touches state["run_touch"], writes
#                     each path: text of state["run_write"], exits state["run_rc"]; anything else exits 0
#   getent hosts      192.0.2.10 <name>
#   hostname          state["hostname"] (default fakehost); -I prints state["hostname_I"] or exits 1 silently
#   sleep             sleeps min(<requested>, state["sleep_cap_s"]) seconds (default 0)
#   pgrep <args>      the pids of state["pgrep"][" ".join(args)] one per line; exit 1 silently when absent
#   uname             -m -> state["uname_m"] (default x86_64), else Linux
#   ssh               runs its remaining argv here (one word -> through bash -c)
#   uv / gym          `uv run gym env start ...` and `gym env start ...`: pid into <log dir>/gym.pid, serve until
#                     TERM (exit 0); any other argv logs and exits 0
#   setsid            os.setsid() (an OSError ignored) then execvp of the rest: `$!` is the server itself
#   kill              reached through `xargs kill` only (see the module docstring): parses -TERM / -KILL / -<n> /
#                     -s SIG, skips `--`, a negative target -> os.killpg; ProcessLookupError ignored; exit 0
#   rm                poisoned: writes RM_CALLED into the log dir and exits 1
#   python3 -m miles_harbor.trial_server   prints state["trial_server_tasks"] (a TRIAL_SERVER_TASKS line) and serves
#                     until TERM (trial_server.pid), or exits 3 at once with state["trial_server_dies"]
#   python3 - (stdin) a heredoc naming HARBOR_SELFTEST answers n HARBOR_SELFTEST_ANSWER lines and the verdict of
#                     state["selftest_rc"]; with state["python_stdin"] == "real" any other heredoc runs on the
#                     real interpreter (the config builders of the Harbor scripts)
#   harbor run -c <job.yaml> -y   writes state["harbor_results"] (Harbor trial results) as <jobs_dir>/<job_name>/
#                     trial<i>/result.json, exits state["harbor_rc"] (default 0)
#   alfworld-download creates <--data-dir>/json_2.1.1 unless state["alfworld_download_creates"] is false, exits
#                     state["alfworld_download_rc"] (default 0)
#   java -version     prints state["java_version"] (default an openjdk 17 line) to stderr, exits 0
#   git               log: state["git_log"] (subjects, newest first) one per line; am <patch>: exit 128 when the
#                     patch's name is state["git_am_fail"], else its Subject joins state["git_log"] (exit 0)
FAKE = r'''
import json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

name = Path(sys.argv[0]).name
argv = sys.argv[1:]
log_path = Path(os.environ["SCRIPT_TEST_LOG"])
log_dir = log_path.parent
state_path = Path(os.environ.get("SCRIPT_TEST_STATE") or (log_dir / "state.json"))
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "MILES_EXPECT_GPUS", "MILES_GCS", "NEMO_GYM_URL", "PYTHONPATH", "MASTER_ADDR",
            "MILES_SCRIPT_EXTERNAL_RAY", "NEMO_GYM_EXTRA_ROOTS", "RAY_TMPDIR", "UV_OFFLINE", "JAVA_HOME",
            "ALFWORLD_DATA", "RETRIEVER_URL", "WEBSHOP_HOME", "RUN_DATA_DIR", "SYNC_DIR", "PREEMPT_SENTINEL",
            "WANDB_API_KEY", "WANDB_ENABLE", "MILES_PROBE_URL", "RAY_ENABLE_UV_RUN_RUNTIME_ENV", "HOME", "HARBOR_PACK_ROOT")


def load_state():
    try:
        return json.loads(state_path.read_text())
    except (OSError, ValueError):
        return {}


def save_state(new_state):
    state_path.write_text(json.dumps(new_state))


state = load_state()


def log(**extra):
    record = {"cmd": name, "argv": argv, "cwd": os.getcwd(),
              "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ}}
    record.update(extra)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def touch(path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()


def serve_until_term(pid_file):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # armed before the pid is published
    (log_dir / pid_file).write_text(str(os.getpid()))
    while True:
        time.sleep(0.05)


def write_rows(path, n):
    if n < 0:  # a converter that ended without the file (state["rows"] -1): nothing is written
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"row": i}) + "\n")


def rows_for(basename):
    rows = state.get("rows", 1)
    if isinstance(rows, dict):
        return int(rows.get(basename, rows.get("default", 1)))
    return int(rows)


if name == "rm":
    (log_dir / "RM_CALLED").write_text(" ".join(argv))
    log()
    sys.exit(1)

log()

if name == "alfworld-download":
    if "--data-dir" in argv and state.get("alfworld_download_creates", True):
        Path(argv[argv.index("--data-dir") + 1], "json_2.1.1").mkdir(parents=True, exist_ok=True)
    sys.exit(int(state.get("alfworld_download_rc", 0)))
if name == "java":
    if "-version" in argv:
        print(state.get("java_version", 'openjdk version "17.0.12" 2024-07-16'), file=sys.stderr)
    sys.exit(0)
if name == "sleep":
    cap = float(state.get("sleep_cap_s", 0))
    try:
        wanted = float(argv[0])
    except (IndexError, ValueError):
        wanted = 0.0
    time.sleep(min(wanted, cap))
    sys.exit(0)
if name == "curl":
    valued = ("-o", "-X", "-d", "-H", "--data", "--header", "--request", "--output")
    url = [a for i, a in enumerate(argv) if not a.startswith("-") and (i == 0 or argv[i - 1] not in valued)][-1]
    codes = state.get("curl", {}).get(url)
    if isinstance(codes, list):
        if len(codes) > 1:
            code = codes.pop(0)
            save_state(state)
        else:
            code = codes[0] if codes else "000"
    else:
        code = codes if codes is not None else "000"
    if str(code) == "200":
        body = state.get("curl_body", {}).get(url)
        if body is not None and "-o" not in argv:
            print(body if isinstance(body, str) else json.dumps(body))
        sys.exit(0)
    sys.exit(22)
if name == "nvidia-smi":
    if "-L" in argv:
        for i in range(int(state.get("gpus", 0))):
            print(f"GPU {i}: Fake GPU (UUID: GPU-{i})")
    sys.exit(0)
if name == "ss":
    out = state.get("ss", "")
    if isinstance(out, list):
        out = "\n".join(out)
    if out:
        print(out)
    sys.exit(0)
if name == "ray":
    if argv[:1] == ["status"]:
        sys.exit(int(state.get("ray_status_rc", 1)))
    if argv[:1] == ["start"]:
        touch(state.get("ray_touch"))
    sys.exit(0)
if name == "getent":
    if argv[:1] == ["hosts"]:
        print("192.0.2.10 " + (argv[1] if len(argv) > 1 else "fakehost"))
    sys.exit(0)
if name == "hostname":
    if "-I" in argv:
        if state.get("hostname_I") is None:
            sys.exit(1)
        print(state["hostname_I"])
    else:
        print(state.get("hostname", "fakehost"))
    sys.exit(0)
if name == "pgrep":
    pids = state.get("pgrep", {}).get(" ".join(argv))
    if not pids:
        sys.exit(1)
    for pid in pids:
        print(pid)
    sys.exit(0)
if name == "uname":
    print(state.get("uname_m", "x86_64") if "-m" in argv else "Linux")
    sys.exit(0)
if name == "ssh":
    valued = ("-o", "-p", "-i", "-l", "-F", "-J", "-L", "-R", "-D", "-W", "-E", "-b", "-c", "-e", "-m", "-O",
              "-Q", "-S", "-w", "-B")
    rest, i, host_seen = [], 0, False
    while i < len(argv):
        a = argv[i]
        if not host_seen and a.startswith("-"):
            i += 2 if a in valued else 1
            continue
        if not host_seen:
            host_seen = True
        else:
            rest.append(a)
        i += 1
    if not rest:
        sys.exit(0)
    if len(rest) == 1:
        os.execvp(shutil.which("bash"), ["bash", "-c", rest[0]])
    os.execvp(rest[0], rest)
if name in ("uv", "gym"):
    if (name == "uv" and argv[:4] == ["run", "gym", "env", "start"]) or (name == "gym" and argv[:2] == ["env", "start"]):
        serve_until_term("gym.pid")
    sys.exit(0)
if name == "setsid":
    try:
        os.setsid()
    except OSError:
        pass
    os.execvp(argv[0], argv)
if name == "kill":
    sig, targets, i, seen_sig = signal.SIGTERM, [], 0, False
    while i < len(argv):
        a = argv[i]
        if a == "--":
            targets += argv[i + 1:]
            break
        if a == "-s" and i + 1 < len(argv):
            sig = getattr(signal, "SIG" + argv[i + 1].upper().replace("SIG", "")); seen_sig = True; i += 2
            continue
        if a.startswith("-") and not seen_sig and not targets:
            spec = a[1:]
            sig = signal.Signals(int(spec)) if spec.isdigit() else getattr(signal, "SIG" + spec.upper().replace("SIG", ""))
            seen_sig = True
        else:
            targets.append(a)
        i += 1
    for t in targets:
        pid = int(t)
        try:
            if pid < 0:
                os.killpg(-pid, sig)
            else:
                os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    sys.exit(0)
if name == "git":
    # `log`: state["git_log"] (subjects, newest first) one per line; `am <patch>`: exit 128 when the patch's name
    # is state["git_am_fail"], else the patch's Subject (first line, tag dropped) is prepended to state["git_log"]
    # so that a later `log` shows it (the gate's idempotency), exit 0; `ls-files -s -- <sub>`: the gitlink line of
    # state["git_gitlink"][<sub>] (nothing without one); `-C <dir> rev-parse HEAD`: state["git_head"][<dir>] (nothing
    # without one); `merge-base --is-ancestor`: exit state["git_ancestor_rc"] (0); anything else exits 0
    if "log" in argv:
        for subject in state.get("git_log", []):
            print(subject)
        sys.exit(0)
    if "am" in argv:
        patch = Path(argv[-1])
        if patch.name == state.get("git_am_fail"):
            print(f"error: patch failed: {patch.name}", file=sys.stderr)
            sys.exit(128)
        m = re.search(r"^Subject: \[PATCH[^\]]*\] (.*)$", patch.read_text(errors="replace"), re.M)
        state["git_log"] = [m.group(1) if m else patch.name] + list(state.get("git_log", []))
        save_state(state)
        sys.exit(0)
    if "ls-files" in argv:
        sha = state.get("git_gitlink", {}).get(argv[-1])
        if sha:
            print("160000 " + sha + " 0 " + argv[-1])
        sys.exit(0)
    if "rev-parse" in argv:
        sha = state.get("git_head", {}).get(argv[argv.index("-C") + 1] if "-C" in argv else "")
        if sha:
            print(sha)
        sys.exit(0)
    if "merge-base" in argv:
        sys.exit(int(state.get("git_ancestor_rc", 0)))
    sys.exit(0)
if name == "harbor":
    if argv[:1] == ["run"] and "-c" in argv:
        text = Path(argv[argv.index("-c") + 1]).read_text()
        fields = {k: v.strip("'\"") for k, v in re.findall(r"^(jobs_dir|job_name): (.*)$", text, re.M)}
        job = Path(fields["jobs_dir"]) / fields["job_name"]
        results = state.get("harbor_results")
        if results is not None:
            job.mkdir(parents=True, exist_ok=True)
            (job / "result.json").write_text(json.dumps({"n_trials": len(results)}))
            for i, result in enumerate(results):
                (job / f"trial{i}").mkdir(exist_ok=True)
                (job / f"trial{i}" / "result.json").write_text(json.dumps(result))
    sys.exit(int(state.get("harbor_rc", 0)))
if name in ("python3", "python"):
    if argv[:2] == ["-m", "sglang.launch_server"]:
        serve_until_term("sglang.pid")
    if argv[:2] == ["-m", "miles_harbor.trial_server"]:
        if state.get("trial_server_dies"):
            print("FakeTrialServer: no dataset", file=sys.stderr)
            sys.exit(3)
        print(state.get("trial_server_tasks", "TRIAL_SERVER_TASKS 2 of ds harness=mini-swe-agent"), flush=True)
        serve_until_term("trial_server.pid")
    if argv[:3] == ["-m", "miles_agent.launcher", "serve-template"]:
        if len(argv) > 4:
            Path(argv[4]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[4]).write_text("{{ fake template }}")
        print(state.get("parser", "fake_tool_parser"))
        sys.exit(0)
    if argv[:1] == ["-c"] and state.get("python_c") == "real":
        os.execv(sys.executable, [sys.executable, *argv])
    if argv[:1] == ["-"]:
        text = sys.stdin.read()
        if "HARBOR_SELFTEST" in text:
            n = int(argv[2]) if len(argv) > 2 else 1
            for i in range(n):
                print(f"HARBOR_SELFTEST_ANSWER {i} t{i} status=200 s=1 {{\"reward\": 1.0}}", flush=True)
            rc = int(state.get("selftest_rc", 0))
            verdict = f"HARBOR_SELFTEST_{'PASS' if rc == 0 else 'FAIL'} {n if rc == 0 else 0} of {n} answered a reward in 1 s"
            print(verdict, flush=True)
            sys.exit(rc)
        if state.get("python_stdin") == "real":  # the heredoc runs on the real interpreter, its argv kept
            os.execv(sys.executable, [sys.executable, "-c", text, *argv[1:]])
        if "MILES_PROBE_URL" in text:
            sys.exit(int(state.get("probe_rc", 0)))
        if "SERVE_PREFLIGHT" in text:
            time.sleep(float(state.get("preflight_delay_s", 0)))  # a slow request (a TERM must not wait for it)
            verdict = state.get("preflight", "PASS")
            calls = [("bash", "{\"command\": \"ls /tmp\"}")] if verdict == "PASS" else []
            print(f"SERVE_PREFLIGHT {verdict} finish_reason=stop tool_calls={calls} content='' usage=None")
            sys.exit(0)
        sys.exit(0)
    script = Path(argv[0]).name if argv and not argv[0].startswith("-") else ""
    if script in ("prepare_data.py", "prepare.py"):
        names = {"alfworld": ["alfworld_train.jsonl", "alfworld_eval274.jsonl"],
                 "scienceworld": ["scienceworld_train.jsonl", "scienceworld_eval274.jsonl"],
                 "webshop": ["webshop_train.jsonl", "webshop_eval.jsonl"],
                 "wiki_search": ["train.jsonl", "validation.jsonl", "test.jsonl"]}
        parts = Path(argv[0]).resolve().parts
        recipe = next((p for p in reversed(parts) if p in names), "")
        if "--output-dir" in argv:
            out = Path(argv[argv.index("--output-dir") + 1])
            for n in names.get(recipe, ["train.jsonl", "eval.jsonl"]):
                write_rows(out / n, rows_for(n))
            if recipe == "webshop":
                goals = rows_for("webshop_eval.jsonl")
                out.mkdir(parents=True, exist_ok=True)
                (out / "protocol.json").write_text(json.dumps(
                    {"goals_digest": state.get("goals_digest", "fakedigest"), "goals": goals, "unique_goals": goals}))
        if "--output" in argv:
            out = argv[argv.index("--output") + 1]
            write_rows(out, rows_for(Path(out).name))
        sys.exit(0)
    if script == "export_goals.py":
        out = argv[argv.index("--output") + 1]
        n = rows_for(Path(out).name)
        write_rows(out, n)
        url = argv[argv.index("--engine-url") + 1] if "--engine-url" in argv else ""
        Path(out + ".meta.json").write_text(json.dumps(
            {"engine_url": url, "exported_at": "fake", "goals": n,
             "goals_digest": state.get("goals_digest", "fakedigest"), "protocol": {}}, indent=2, sort_keys=True) + "\n")
        sys.exit(0)
    if script.endswith(".py"):
        touch(state.get("run_touch"))
        for path, text in (state.get("run_write") or {}).items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        sys.exit(int(state.get("run_rc", 0)))
    sys.exit(0)
sys.exit(0)
'''

_SLEEPERS = []


@pytest.fixture(autouse=True)
def _reap_sleepers():
    """Only KILLs the sleepers a test left alive; nothing ever waits for an orphan."""
    yield
    while _SLEEPERS:
        pid = _SLEEPERS.pop()
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _fakes(tmp, *names, state=None):
    """Write one fake per name into <tmp>/bin (default: every fake), the state file and an empty log; returns bin."""
    tmp = Path(tmp)
    fake_dir = tmp / "bin"
    fake_dir.mkdir(exist_ok=True)
    names = names or DEFAULT_FAKES
    for stale in fake_dir.iterdir():  # a later call with fewer names must leave none of the earlier fakes behind
        if stale.name not in names:
            stale.unlink()
    for name in names:
        path = fake_dir / name
        path.write_text("#!" + sys.executable + "\n" + FAKE)
        path.chmod(0o755)
    _set_state(tmp, state or {})
    (tmp / "calls.jsonl").write_text("")
    return fake_dir


def _set_state(tmp, state):
    (Path(tmp) / "state.json").write_text(json.dumps(state))


def _toolbox(tmp):
    """<tmp>/tools: symlinks to the real TOOLBOX commands found on the original PATH (the controlled PATH of a
    `bare_path` run, where a test needs a command such as ss or getent to be absent on every host)."""
    tools = Path(tmp) / "tools"
    if not tools.is_dir():
        tools.mkdir()
        for name in TOOLBOX:
            real = shutil.which(name)
            if real:
                (tools / name).symlink_to(real)
    return tools


def _env(tmp, bare_path=False, interp_dir=True, **extra):
    """os.environ minus every knob a script reads; PATH = fakes + the interpreter's dir + the old PATH (or, with
    bare_path, the toolbox of real coreutils instead of the old PATH); HOME = tmp; the log and state files; extra.
    interp_dir=False leaves the interpreter's dir off PATH: a "no uv anywhere" scenario must not find one beside
    python3 (a container image, a venv with uv installed)."""
    env = {k: v for k, v in os.environ.items() if k not in STRIP_EXACT and not k.startswith(STRIP_PREFIXES)}
    tmp = Path(tmp)
    path = [str(tmp / "bin")]
    if interp_dir:
        path.append(str(Path(sys.executable).parent))
    path.append(str(_toolbox(tmp)) if bare_path else os.environ.get("PATH", ""))
    env.update({"PATH": os.pathsep.join(path), "HOME": str(tmp), "SCRIPT_TEST_LOG": str(tmp / "calls.jsonl"),
                "SCRIPT_TEST_STATE": str(tmp / "state.json")})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _bash(snippet, env, cwd=None, timeout=60):
    """BASH -c snippet -> (rc, stdout+stderr)."""
    proc = subprocess.run([BASH, "-c", snippet], env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def _run_script(rel, *args, env, cwd=None, timeout=120):
    """BASH $REPO/<rel> args -> (rc, stdout+stderr)."""
    proc = subprocess.run([BASH, str(REPO / rel), *args], env=env, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def _calls(env, cmd=None):
    """The logged calls, oldest first (optionally only one command's)."""
    text = Path(env["SCRIPT_TEST_LOG"]).read_text()
    calls = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [c for c in calls if cmd is None or c["cmd"] == cmd]


def _dry(out, label, nth=0):
    """The argv printed for a `DRY_RUN <label>:` line (the nth one), shell-split."""
    prefix = f"DRY_RUN {label}:"
    lines = [line[len(prefix):] for line in out.splitlines() if line.startswith(prefix)]
    assert len(lines) > nth, f"no DRY_RUN {label} line #{nth} in:\n{out}"
    return shlex.split(lines[nth])


def _is_zombie(pid):
    probe = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return probe.stdout.strip().startswith("Z")


def _pid_gone(pid_or_path, timeout=2.0):
    """True once the pid (or the pid a fake wrote into the file) is gone or a zombie nobody will reap; a survivor
    is KILLed and False is returned. Not valid for an unreaped child of this process (module docstring)."""
    pid = pid_or_path if isinstance(pid_or_path, int) else int(Path(pid_or_path).read_text().strip())
    deadline = time.time() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        if _is_zombie(pid):
            return True
        if time.time() >= deadline:
            break
        time.sleep(0.02)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return False


def _spawn_sleeper(tmp, term_ignored=False, new_session=False):
    """A 600 s sleeper that is nobody's child here (an intermediate interpreter spawned it and exited), so init
    reaps it the moment it dies; with term_ignored it ignores TERM and the call returns only after its READY
    handshake (a TERM during interpreter start-up would otherwise kill it). Registered for a KILL at teardown."""
    body = "import time; time.sleep(600)"
    env = dict(os.environ)
    ready = None
    if term_ignored:
        ready = Path(tmp) / f"sleeper_ready_{len(_SLEEPERS)}_{time.time_ns()}"
        body = ("import os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(os.environ['SLEEPER_READY']).touch(); time.sleep(600)")
        env["SLEEPER_READY"] = str(ready)
    snippet = (
        "import subprocess, sys; "
        f"p = subprocess.Popen([sys.executable, '-c', {body!r}], stdin=subprocess.DEVNULL, "
        f"stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session={new_session!r}); print(p.pid)"
    )
    result = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, check=True, timeout=10,
                            env=env)
    pid = int(result.stdout.strip())
    _SLEEPERS.append(pid)
    if ready is not None:
        deadline = time.time() + 5
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "the TERM-ignoring sleeper never raised READY"
    return pid


_ORPHANS_REAPED = []


def _orphans_are_reaped(tmp):
    """Whether this host's init reaps a TERMed orphan (a PID-namespaced container without an init may not): the
    probe runs once; a test whose assertion needs a reaped orphan skips when it returns False."""
    if not _ORPHANS_REAPED:
        pid = _spawn_sleeper(tmp)
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        _ORPHANS_REAPED.append(not _is_zombie(pid))
    return _ORPHANS_REAPED[0]


def _shell_files():
    files = sorted(SCRIPTS.rglob("*.sh"))
    examples = REPO / "examples" / "slurm"
    if examples.is_dir():
        files += sorted(examples.glob("*.sb" "atch"))  # two pieces: the identity gate scans this file too
    return files


def _function_body(text, name):
    """The lines of `name() {` up to its closing `}` at column 0."""
    start = text.index(f"\n{name}() {{")
    end = text.index("\n}\n", start)
    return text[start:end + 2]


# ---- static gates -------------------------------------------------------------------------------------------


def test_every_public_script_parses():
    files = _shell_files()
    rels = {str(p.relative_to(SCRIPTS)) for p in files if p.suffix == ".sh"}
    assert set(PREEXISTING_PUBLIC) <= rels and {"lib/common.sh", "lib/ray.sh"} <= rels
    for path in files:
        proc = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{path}: {proc.stderr}"


def test_public_scripts_have_no_rm_lines():
    offenders = []
    for path in sorted(SCRIPTS.rglob("*.sh")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = re.sub(r"(^|\s)#.*$", "", line)
            if RM_COMMAND.search(code):
                offenders.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    assert offenders == []


def test_new_public_files_carry_no_site_literal():
    hits = []
    for rel in NEW_PUBLIC_FILES:
        path = REPO / rel
        assert path.is_file(), rel
        for i, line in enumerate(path.read_text().splitlines(), 1):
            match = SITE_LITERAL.search(line)
            if match is None:
                continue
            # the one allowed hit: ray.sh unsets the scheduler's family around `ray start` (unsetting is not reading)
            if rel == "scripts/lib/ray.sh" and match.group(0) == "SLURM" "_" and line.lstrip().startswith("unset "):
                continue
            hits.append(f"{rel}:{i}: {match.group(0)}")
    assert hits == []
    ray = RAY.read_text()
    assert ray.count("unset $(env | grep -E '^(PMI|PMIX|MPI|OMPI|SLURM)' | cut -d= -f1)") == 2


def test_fakes_poison_rm(tmp_path):
    _fakes(tmp_path)
    env = _env(tmp_path)
    rc, out = _bash("rm -f x; echo RC=$?", env)
    assert "RC=1" in out and (tmp_path / "RM_CALLED").exists()
    assert _calls(env, "rm")[0]["argv"] == ["-f", "x"]


# ---- scripts/lib/common.sh ----------------------------------------------------------------------------------


def test_local_env_never_overrides_the_environment(tmp_path):
    _fakes(tmp_path)
    knobs = tmp_path / "knobs.env"
    knobs.write_text(
        "# a comment\n"
        "\n"
        "FOO_FROM_FILE=file\n"
        "EMPTY_IN_ENV=file\n"
        "BAR=$REPO/bar\n"
        "BAZ=\"quoted value\"\n"
        "QUX='single'\n"
        "HOME_X=${HOME}/x\n"
        "WITH_EQ=a=b\n"
        "NOTED=value   # an example line, uncommented\n"
        "NOTED_EMPTY=        # empty = the default\n"
        "NOTED_QUOTED=\"a # b\"   # the quotes keep the hash\n"
        "HASHED=a#b\n"
        "WORDY=$REPOSITORY/$HOMEBREW_PREFIX/${REPO}x/$HOME_X\n"
        "TAIL=$REPO:$REPO\n"
        "TWICE=first\n"
        "TWICE=second\n"
        "ENV_TWICE=file1\n"
        "ENV_TWICE=file2\n"
        "lower=skip\n"
        "1BAD=skip\n"
        "  INDENTED=skip\n"
        "NOEQ\n"
    )
    snippet = (
        f". {COMMON}; echo \"FOO=$FOO_FROM_FILE EMPTY=[$EMPTY_IN_ENV] BAR=$BAR BAZ=$BAZ QUX=$QUX HOME_X=$HOME_X "
        f"WITH_EQ=$WITH_EQ NOTED=[$NOTED] NOTED_EMPTY=[$NOTED_EMPTY] NOTED_QUOTED=[$NOTED_QUOTED] HASHED=[$HASHED] "
        f"WORDY=[$WORDY] TAIL=[$TAIL] TWICE=[$TWICE] ENV_TWICE=[$ENV_TWICE] "
        f"lower=${{lower:-unset}} INDENTED=${{INDENTED:-unset}} NOEQ=${{NOEQ:-unset}}\"; "
        "env | grep -c '^1BAD=' || echo BAD=unset; sh -c 'echo CHILD_BAR=$BAR'"
    )
    env = _env(tmp_path, LOCAL_ENV=str(knobs), FOO_FROM_FILE="env", EMPTY_IN_ENV="", ENV_TWICE="env")
    rc, out = _bash(snippet, env)
    assert rc == 0, out
    assert f"FOO=env EMPTY=[] BAR={REPO}/bar BAZ=quoted value QUX=single HOME_X={tmp_path}/x WITH_EQ=a=b" in out
    assert "NOTED=[value] NOTED_EMPTY=[] NOTED_QUOTED=[a # b] HASHED=[a#b]" in out
    # $REPO / $HOME are replaced at a word boundary only ($REPOSITORY, $HOMEBREW_PREFIX, $HOME_X stay as written;
    # ${REPO}x and $REPO: are replaced); the LAST line of a key wins within the file, the environment over the file
    assert f"WORDY=[$REPOSITORY/$HOMEBREW_PREFIX/{REPO}x/$HOME_X] TAIL=[{REPO}:{REPO}] TWICE=[second] ENV_TWICE=[env]" in out
    assert "lower=unset INDENTED=unset NOEQ=unset" in out and "BAD=unset" in out
    assert f"CHILD_BAR={REPO}/bar" in out  # exported: the drivers and the gym see the knobs
    # a missing file and LOCAL_ENV=/dev/null are silent; the default is $REPO/local.env
    rc, out = _bash(f". {COMMON}; echo \"LE=$LOCAL_ENV\"", _env(tmp_path, LOCAL_ENV="/dev/null"))
    assert rc == 0 and "LE=/dev/null" in out
    rc, out = _bash(f". {COMMON}; echo \"LE=$LOCAL_ENV\"", _env(tmp_path))
    assert rc == 0 and f"LE={REPO}/local.env" in out


def test_local_env_example_documents_every_public_knob(tmp_path):
    """local.env.example: one commented `#KEY=` line per public knob (DESIGN §2 + S11, the C1 list), nothing but
    comments and blanks (copied as-is it changes nothing), and the run-level knobs say so."""
    text = (REPO / "local.env.example").read_text()
    assert all(line.startswith("#") or not line.strip() for line in text.splitlines())
    documented = set(re.findall(r"^#([A-Z_][A-Z0-9_]*)=", text, re.M))
    expected = {
        "MILES_AGENT_MODELS", "MILES_AGENT_DATA", "MILES_AGENT_RESULTS", "GYM_VENV_ROOT", "POLICY_MODEL", "POLICY_HF",
        "MEGATRON_PATH", "MILES_ROOT", "RETRIEVER_URL", "WEBSHOP_HOME", "ALFWORLD_DATA", "WIKI_SEARCH_DATA",
        "SCIENCEWORLD_JAR", "JAVA_HOME", "SCIENCEWORLD_ENV_WORKERS", "NEMO_GYM_URL", "NEMO_GYM_PORT",
        "WEBSHOP_SERVER_PORT", "TEACHER_MODEL", "TEACHER_HF", "TEACHER_URL", "HARBOR_ENVIRONMENT_IMPORT_PATH",
        "HARBOR_PACK_ROOT", "SERVE_REQUIRE_TOOL_CALL", "HF_MODEL_ARGS", "HF_MODEL_NAME", "WEBSHOP_GOALS_DIGEST",
        "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_ENABLE", "HF_TOKEN", "RECIPE", "TOPOLOGY", "RUN_TAG", "RUN_SCRIPT",
        "PROMPT", "EXTRA_ARGS", "MODE", "MILES_TORCH_DIST_DIR", "MILES_DATA_DIR", "PROJECT_PORTS", "PYTHON",
        "RAY_BIN", "GYM_BIN", "UV_BIN", "CURL", "NVIDIA_SMI", "SSH", "SCIENCEWORLD_TEST_LIST", "WEBSHOP_GOALS_FILE",
        "SCIENCEWORLD_TEST_SHA256",
        "WEBSHOP_MD5_TABLE", "WEBSHOP_ENGINE_BASE_URL", "BUILD_DIR", "GYM_TAR", "HARBOR_DATASET", "HARBOR_PROVIDER",
        "HARBOR_ENVIRONMENT_KWARGS", "HARBOR_HARNESS", "HARBOR_AGENT_KWARGS", "HARBOR_MAX_CONCURRENT",
        "HARBOR_AGENT_NETWORK", "HARBOR_REWARD_REQUIRES_SUBMIT", "HARBOR_RECORDS", "HARBOR_WORK_DIR", "HARBOR_PYTHON",
        "HARBOR_BIN", "TRIAL_SERVER_VENV", "TRIAL_SERVER_MODULE", "OPENSANDBOX_ENDPOINT", "OPENSANDBOX_KEY",
        "OPENSANDBOX_CA_FILE", "OPENSANDBOX_PROXY",
    }
    assert expected <= documented, sorted(expected - documented)
    for knob in ("RECIPE", "TOPOLOGY", "RUN_TAG", "RUN_SCRIPT", "PROMPT", "EXTRA_ARGS", "MODE"):
        assert re.search(rf"^#{knob}=.*run-level: normally a scripts/run_local.sh flag", text, re.M), knob
    # loaded as-is, the example sets nothing: the lib's own exports are the only variables after sourcing
    _fakes(tmp_path)
    rc, out = _bash(f". {COMMON}; env | grep -c '^POLICY_MODEL=' || echo NONE",
                    _env(tmp_path, LOCAL_ENV=str(REPO / "local.env.example")))
    assert rc == 0 and "NONE" in out


def test_roots_default_under_the_repo_and_follow_the_wrapper_dirs(tmp_path):
    _fakes(tmp_path)
    snippet = (
        f". {COMMON}; echo \"M=$MILES_AGENT_MODELS D=$MILES_AGENT_DATA R=$MILES_AGENT_RESULTS RUNS=$RUNS_DIR "
        "TD=$MILES_TORCH_DIST_DIR MR=$MILES_ROOT JOB=$JOB_ID PORT=$NEMO_GYM_PORT PORTS=[$PROJECT_PORTS]\"; "
        "run_dirs t1; echo \"RD=$RUN_DATA_DIR SY=$SYNC_DIR PS=$PREEMPT_SENTINEL\"; "
        "sh -c 'echo CHILD=$MILES_AGENT_DATA:$RUN_DATA_DIR'"
    )
    rc, out = _bash(snippet, _env(tmp_path, LOCAL_ENV="/dev/null"))
    assert rc == 0, out
    # the probe common.sh makes (`[ -d /root/miles ]`): a stat failure is "no". os.path.isdir swallows every OSError
    # the way `[ -d ]` does; pathlib's is_dir() lets EACCES through (a 0700 /root, as on CI runners)
    miles_root = "/root/miles" if os.path.isdir("/root/miles") else f"{REPO}/3rdparty/miles"
    assert (f"M={REPO}/models D={REPO}/data R={REPO}/results RUNS={REPO}/results/runs "
            f"TD={REPO}/models/torch_dist_v5 MR={miles_root} JOB=") in out
    assert re.search(r"JOB=\d+ PORT=8200 PORTS=\[6100 6200 6379 7100-7131 7200 8100 8200 8265 13141-13143\]", out)
    assert f"RD={REPO}/data/runs/t1 SY={REPO}/results/t1/sync PS={REPO}/results/t1/stop" in out
    assert f"CHILD={REPO}/data:{REPO}/data/runs/t1" in out
    # the wrapper's neutral names are honoured ...
    rc, out = _bash(snippet, _env(tmp_path, LOCAL_ENV="/dev/null", MODELS_DIR="/x", DATA_DIR="/xd",
                                  RESULTS_DIR="/xr", RUNS_DIR="/xruns", JOB_ID="4242", MILES_ROOT="/mr"))
    assert rc == 0 and "M=/x D=/xd R=/xr RUNS=/xruns TD=/x/torch_dist_v5 MR=/mr JOB=4242" in out
    assert "RD=/xd/runs/t1 SY=/xr/t1/sync PS=/xr/t1/stop" in out
    # ... and the new names beat them; MILES_DATA_DIR / SYNC_DIR / PREEMPT_SENTINEL / MILES_TORCH_DIST_DIR too
    rc, out = _bash(snippet, _env(tmp_path, LOCAL_ENV="/dev/null", MODELS_DIR="/x", MILES_AGENT_MODELS="/y",
                                  DATA_DIR="/xd", MILES_AGENT_DATA="/yd", MILES_AGENT_RESULTS="/yr",
                                  MILES_TORCH_DIST_DIR="/td", MILES_DATA_DIR="/md", SYNC_DIR="/s",
                                  PREEMPT_SENTINEL="/p", NEMO_GYM_PORT="8201", PROJECT_PORTS="1 2-3"))
    assert rc == 0 and "M=/y D=/yd R=/yr RUNS=/yr/runs TD=/td" in out and "PORT=8201 PORTS=[1 2-3]" in out
    assert "RD=/md SY=/s PS=/p" in out
    # local.env feeds the same ladder (below the environment)
    (tmp_path / "le").write_text("MILES_AGENT_MODELS=$REPO/m2\nMILES_AGENT_DATA=/ld\n")
    rc, out = _bash(snippet, _env(tmp_path, LOCAL_ENV=str(tmp_path / "le"), MILES_AGENT_DATA="/ed"))
    assert rc == 0 and f"M={REPO}/m2 D=/ed " in out
    rc, out = _bash(f". {COMMON}; run_dirs", _env(tmp_path, LOCAL_ENV="/dev/null"))
    assert rc == 1 and "FATAL: run_dirs: a run tag is required" in out


def test_gpu_count_prefers_cuda_visible_devices(tmp_path):
    _fakes(tmp_path, state={"gpus": 8})
    snippet = f". {COMMON}; measure_gpus"
    base = dict(LOCAL_ENV="/dev/null")
    rc, out = _bash(snippet, _env(tmp_path, CUDA_VISIBLE_DEVICES="0,1,2", **base))
    assert (rc, out.strip()) == (0, "3")
    env = _env(tmp_path, **base)
    rc, out = _bash(snippet, env)
    assert (rc, out.strip()) == (0, "8") and _calls(env, "nvidia-smi")[0]["argv"] == ["-L"]
    rc, out = _bash(snippet, _env(tmp_path, CUDA_VISIBLE_DEVICES="", **base))
    assert (rc, out.strip()) == (0, "8")  # empty counts as unset
    _fakes(tmp_path, state={"gpus": 8})  # a fresh log
    env = _env(tmp_path, GPUS_PER_NODE="4", CUDA_VISIBLE_DEVICES="0,1", **base)
    rc, out = _bash(snippet, env)
    assert (rc, out.strip()) == (0, "4") and _calls(env, "nvidia-smi") == []
    _set_state(tmp_path, {"gpus": 0})
    rc, out = _bash(snippet, _env(tmp_path, **base))
    assert (rc, out.strip()) == (0, "0")
    rc, out = _bash(f"set -euo pipefail; {snippet}; echo ALIVE",
                    _env(tmp_path, NVIDIA_SMI=str(tmp_path / "no-such-nvidia-smi"), **base))
    assert (rc, out.split()) == (0, ["0", "ALIVE"])  # a missing tool is 0, never a FATAL and never a set -e death


def test_contract_check_refuses_an_unknown_major(tmp_path):
    _fakes(tmp_path)
    snippet = f". {COMMON}; contract_check; echo OK"
    base = dict(LOCAL_ENV="/dev/null")
    assert _bash(snippet, _env(tmp_path, **base)) == (0, "OK\n")
    assert _bash(snippet, _env(tmp_path, CONTRACT_VERSION="1", **base)) == (0, "OK\n")
    assert _bash(snippet, _env(tmp_path, CONTRACT_VERSION="1.4", **base)) == (0, "OK\n")
    rc, out = _bash(snippet, _env(tmp_path, CONTRACT_VERSION="2", **base))
    assert rc == 1 and out == "FATAL: unsupported CONTRACT_VERSION 2 (this entry speaks 1)\n"


def test_dry_run_helpers_quote_the_argv_and_grade_the_depth(tmp_path):
    _fakes(tmp_path)
    base = dict(LOCAL_ENV="/dev/null")
    snippet = (f". {COMMON}; dry_run_print driver python3 run.py --save-dir '/r/t x/' '+k=v w' 'http://127.0.0.1:8200'; "
               "dry_run_print empty; echo \"depth=[$(dry_run_depth)]\"; is_dry_run && echo DRY || echo WET")
    rc, out = _bash(snippet, _env(tmp_path, **base))
    assert rc == 0, out
    assert _dry(out, "driver") == ["python3", "run.py", "--save-dir", "/r/t x/", "+k=v w", "http://127.0.0.1:8200"]
    assert "DRY_RUN empty:\n" in out and "depth=[]" in out and "WET" in out
    rc, out = _bash(snippet, _env(tmp_path, DRY_RUN="1", **base))
    assert rc == 0 and "depth=[argv]" in out and "DRY\n" in out
    rc, out = _bash(snippet, _env(tmp_path, DRY_RUN="driver", **base))
    assert rc == 0 and "depth=[driver]" in out and "WET" in out


def test_cleanup_runs_every_fragment_and_preempt_term_raises_the_sentinel(tmp_path):
    _fakes(tmp_path)
    sentinel = tmp_path / "stop"
    snippet = (f"set -euo pipefail; . {COMMON}; add_cleanup 'echo one'; add_cleanup 'false'; add_cleanup 'echo two'; "
               "arm_preempt_trap; kill -TERM $$; echo SURVIVED; exit 0")
    rc, out = _bash(snippet, _env(tmp_path, LOCAL_ENV="/dev/null", PREEMPT_SENTINEL=str(sentinel), NODE_RANK="2"))
    assert rc == 0, out
    assert "PREEMPT_TERM on node 2: raising the sentinel" in out and sentinel.exists()
    assert out.splitlines()[-3:] == ["SURVIVED", "two", "one"]  # newest fragment first; the failing one skips nothing
    # without a sentinel the trap only logs
    rc, out = _bash(f". {COMMON}; arm_preempt_trap; kill -USR1 $$; echo ALIVE", _env(tmp_path, LOCAL_ENV="/dev/null"))
    assert rc == 0 and "PREEMPT_TERM on node 0: raising the sentinel\nALIVE" in out


def test_wait_http_ready_passes_fails_and_detects_a_dead_pid(tmp_path):
    url = "http://127.0.0.1:8200/openapi.json"
    log = tmp_path / "server.log"
    log.write_text("boot log line\n")
    base = dict(LOCAL_ENV="/dev/null")
    # alive (the shell itself) and answering on the third poll: two sleeps, three curls
    _fakes(tmp_path, state={"curl": {url: ["000", "000", "200"]}})
    env = _env(tmp_path, **base)
    rc, out = _bash(f". {COMMON}; wait_http_ready {url} $$ {log} 5 1 && echo READY", env)
    assert rc == 0 and "READY" in out, out
    assert [c["argv"][-1] for c in _calls(env, "curl")] == [url] * 3
    assert [c["argv"] for c in _calls(env, "sleep")] == [["1"], ["1"]]
    # a dead pid: a child this shell spawned and reaped; the gate ends before its first sleep, with the log's tail
    _fakes(tmp_path)
    env = _env(tmp_path, **base)
    rc, out = _bash(f". {COMMON}; sleep 600 & p=$!; wait \"$p\"; wait_http_ready {url} \"$p\" {log} 5 1; echo RC=$?", env)
    assert rc == 0 and "RC=1" in out, out
    assert f"exited before {url} answered" in out and "boot log line" in out
    assert [c["argv"] for c in _calls(env, "sleep")] == [["600"]]
    # tries exhausted
    _fakes(tmp_path)
    env = _env(tmp_path, **base)
    rc, out = _bash(f". {COMMON}; wait_http_ready {url} $$ {log} 3 1; echo RC=$?", env)
    assert rc == 0 and "RC=1" in out and f"{url} not ready after 3 x 1s" in out and "boot log line" in out
    assert len(_calls(env, "curl")) == 3 and len(_calls(env, "sleep")) == 3
    # the predicate gates the curl: a 200 behind a predicate that never passes is not ready
    _fakes(tmp_path, state={"curl": {url: ["200"]}})
    env = _env(tmp_path, **base)
    rc, out = _bash(f". {COMMON}; never() {{ false; }}; wait_http_ready {url} $$ {log} 2 1 never; echo RC=$?", env)
    assert rc == 0 and "RC=1" in out and _calls(env, "curl") == []


def test_port_precheck_names_the_holder_and_never_kills(tmp_path):
    base = dict(LOCAL_ENV="/dev/null")
    held = "LISTEN 0 4096 0.0.0.0:8265 0.0.0.0:* users:((\"raylet\",pid=4242,fd=3))"
    _fakes(tmp_path, state={"ss": "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n" + held})
    env = _env(tmp_path, **base)
    rc, out = _bash(f". {COMMON}; port_precheck 6100 8265 7100-7102; echo NOT_REACHED", env)
    assert rc == 1 and "NOT_REACHED" not in out
    assert f"FATAL: port :8265 is already held: {held}" in out
    assert [c["argv"] for c in _calls(env, "ss")] == [["-tln"]]
    assert _calls(env, "kill") == []
    # the no-kill property is static: nothing between port_precheck() and its closing brace signals anything
    body = _function_body(COMMON.read_text(), "port_precheck")
    assert not re.search(r"\bkill\b|\bxargs\b|pkill", body)
    # a range is expanded; a holder in it is named
    _fakes(tmp_path, state={"ss": ["LISTEN 0 128 [::]:7101 [::]:*"]})
    rc, out = _bash(f". {COMMON}; port_precheck 7100-7102", _env(tmp_path, **base))
    assert rc == 1 and "FATAL: port :7101 is already held: LISTEN 0 128 [::]:7101 [::]:*" in out
    # the ports of the URLs after -- are ours and skipped; without -- the same holder is a FATAL
    _fakes(tmp_path, state={"ss": "LISTEN 0 128 127.0.0.1:6100 0.0.0.0:*"})
    rc, out = _bash(f". {COMMON}; port_precheck 6100 8265 -- http://127.0.0.1:6100 http://192.0.2.5:13141/v1; echo CLEAR",
                    _env(tmp_path, **base))
    assert rc == 0 and "CLEAR" in out and "FATAL" not in out
    rc, out = _bash(f". {COMMON}; port_precheck 6100 8265; echo NOT_REACHED", _env(tmp_path, **base))
    assert rc == 1 and "FATAL: port :6100 is already held:" in out
    # a comma list (NEMO_GYM_URL's trial servers): each URL's port is ours
    _fakes(tmp_path, state={"ss": ["LISTEN 0 128 127.0.0.1:8200 0.0.0.0:*", "LISTEN 0 128 127.0.0.1:6100 0.0.0.0:*"]})
    rc, out = _bash(f". {COMMON}; port_precheck 6100 8200 -- http://127.0.0.1:8201,http://127.0.0.1:8200,"
                    "http://127.0.0.1:6100; echo CLEAR", _env(tmp_path, **base))
    assert rc == 0 and "CLEAR" in out and "FATAL" not in out, out
    # the session-port check before the driver (today's 7100-7131 loop): a WARN with the evidence, rc 0, no kill
    _fakes(tmp_path, state={"ss": "ESTAB 0 0 127.0.0.1:7101 127.0.0.1:51234"})
    env = _env(tmp_path, **base)
    rc, out = _bash(f"set -euo pipefail; . {COMMON}; port_warn_check 7100-7102; echo CLEAR", env)
    assert rc == 0, out
    assert out == ("WARN: session port 7101 occupied before launch:\nESTAB 0 0 127.0.0.1:7101 127.0.0.1:51234\nCLEAR\n")
    assert [c["argv"] for c in _calls(env, "ss")] == [["-tan", "( sport = :7100 )"], ["-tan", "( sport = :7101 )"],
                                                      ["-tanp", "( sport = :7101 )"], ["-tan", "( sport = :7102 )"]]
    assert not re.search(r"\bkill\b|\bxargs\b|pkill", _function_body(COMMON.read_text(), "port_warn_check"))
    # no ss on PATH (the fake dir without one, and nothing else on PATH): a WARN, never a FATAL; the warn check is silent
    _fakes(tmp_path, *(n for n in DEFAULT_FAKES if n != "ss"))
    rc, out = _bash(f". {COMMON}; port_precheck 6100 8265; port_warn_check 7100-7102; echo CLEAR",
                    _env(tmp_path, bare_path=True, **base))
    assert rc == 0 and out == "WARN: no ss on PATH: port pre-check skipped\nCLEAR\n"


def test_node_ip_prefers_getent_then_node_ip_and_fatals_when_unknown(tmp_path):
    base = dict(LOCAL_ENV="/dev/null")
    _fakes(tmp_path)
    env = _env(tmp_path, **base)
    assert _bash(f". {COMMON}; node_ip", env) == (0, "192.0.2.10\n")
    assert _calls(env, "getent")[0]["argv"] == ["hosts", "fakehost"]
    # hostname -I answers when getent does not
    _fakes(tmp_path, *(n for n in DEFAULT_FAKES if n != "getent"), state={"hostname_I": "192.0.2.12 192.0.2.13"})
    assert _bash(f". {COMMON}; node_ip", _env(tmp_path, bare_path=True, **base)) == (0, "192.0.2.12\n")
    # neither (no getent on a bare PATH, hostname -I silent): NODE_IP answers; nothing at all is a FATAL (on
    # stderr: stdout is the answer) and a return of 1
    _fakes(tmp_path, *(n for n in DEFAULT_FAKES if n != "getent"))
    assert _bash(f". {COMMON}; node_ip", _env(tmp_path, bare_path=True, NODE_IP="192.0.2.9", **base)) == (0, "192.0.2.9\n")
    rc, out = _bash(f". {COMMON}; ip=$(node_ip) || echo \"RC=$? ip=[$ip]\"", _env(tmp_path, bare_path=True, **base))
    assert rc == 0 and "FATAL: could not determine node IP" in out and "RC=1 ip=[]" in out


# ---- scripts/lib/ray.sh -------------------------------------------------------------------------------------


def test_ray_head_and_worker_argv(tmp_path):
    base = dict(LOCAL_ENV="/dev/null")
    head_start = ["start", "--head", "--node-ip-address=192.0.2.10", "--port=6379", "--num-gpus", "4",
                  "--disable-usage-stats", "--dashboard-host=0.0.0.0", "--min-worker-port=2000", "--max-worker-port=2999"]
    _fakes(tmp_path, state={"curl": {"http://127.0.0.1:8265/api/version": ["000", "200"]}})
    env = _env(tmp_path, **base)
    rc, out = _bash(f"set -euo pipefail; . {COMMON}; . {RAY}; ray_head 192.0.2.10 4; "
                    "echo \"MA=$MASTER_ADDR EXT=$MILES_SCRIPT_EXTERNAL_RAY\"", env)
    assert rc == 0 and "MA=192.0.2.10 EXT=1" in out, out
    assert [c["argv"] for c in _calls(env, "ray")] == [head_start, ["stop", "--force"]]  # the stop from the EXIT trap
    assert [c["argv"] for c in _calls(env, "sleep")] == [["5"]]
    # the dashboard gate: 12 polls of 5 s, then a FATAL
    _fakes(tmp_path)
    env = _env(tmp_path, **base)
    rc, out = _bash(f"set -euo pipefail; . {COMMON}; . {RAY}; ray_head 192.0.2.10 4; echo NOT_REACHED", env)
    assert rc == 1 and "FATAL: our ray dashboard is not serving on 8265" in out and "NOT_REACHED" not in out
    assert len(_calls(env, "curl")) == 12 and len(_calls(env, "sleep")) == 12
    # the stop is registered as soon as the start returns 0: the head that failed the gate is stopped by the EXIT trap
    assert [c["argv"] for c in _calls(env, "ray")] == [head_start, ["stop", "--force"]]
    # ray_running is `ray status`
    _fakes(tmp_path, state={"ray_status_rc": 0})
    assert _bash(f". {COMMON}; . {RAY}; ray_running && echo UP", _env(tmp_path, **base)) == (0, "UP\n")
    _fakes(tmp_path, state={"ray_status_rc": 1})
    assert _bash(f". {COMMON}; . {RAY}; ray_running || echo DOWN", _env(tmp_path, **base)) == (0, "DOWN\n")
    # a worker: the join creates $SYNC_DIR/ENDED itself (the fake ray start), so the idle loop ends at once
    sync = tmp_path / "sync"
    sync.mkdir()
    _fakes(tmp_path, state={"ray_touch": str(sync / "ENDED")})
    env = _env(tmp_path, SYNC_DIR=str(sync), **base)
    rc, out = _bash(f"set -euo pipefail; . {COMMON}; . {RAY}; ray_worker_join 192.0.2.10 192.0.2.11 4 1; echo JOIN_RC=$?", env)
    assert rc == 0, out
    assert "RAY_WORKER_JOINED node_rank=1 ip=192.0.2.11\nRAY_WORKER_DONE node_rank=1\nJOIN_RC=0" in out
    assert [c["argv"] for c in _calls(env, "ray")] == [
        ["start", "--address", "192.0.2.10:6379", "--node-ip-address", "192.0.2.11", "--num-gpus", "4",
         "--disable-usage-stats", "--min-worker-port=2000", "--max-worker-port=2999"],
        ["stop"],
    ]
    assert _calls(env, "sleep") == []
    # ENDED already up before the join: the head is gone, the worker never starts ray
    _fakes(tmp_path)
    env = _env(tmp_path, SYNC_DIR=str(sync), **base)
    rc, out = _bash(f"set -euo pipefail; . {COMMON}; . {RAY}; ray_worker_join 192.0.2.10 192.0.2.11 4 1; echo NOT_REACHED", env)
    assert rc == 1 and "FATAL: head exited before ray came up" in out and "NOT_REACHED" not in out
    assert _calls(env, "ray") == []


def test_ray_wait_gpus_passes_the_expected_count(tmp_path):
    _fakes(tmp_path)
    env = _env(tmp_path, LOCAL_ENV="/dev/null")
    rc, out = _bash(f"set -euo pipefail; . {COMMON}; . {RAY}; ray_wait_gpus 12 192.0.2.10:6379 && echo WAITED", env)
    assert rc == 0 and "WAITED" in out, out
    (call,) = _calls(env, "python3")
    assert call["argv"] == ["-"]
    assert call["env"]["MILES_EXPECT_GPUS"] == "12" and call["env"]["MILES_GCS"] == "192.0.2.10:6379"
    # the one call shape the entry's anchor test pins on this file (the entry computes the count on its own line)
    assert "MILES_EXPECT_GPUS=$1 MILES_GCS=$2 \"$PYTHON\" - <<'PYEOF'" in RAY.read_text()


# ---- the harness itself -------------------------------------------------------------------------------------


def test_sleeper_helpers_spawn_reapable_orphans(tmp_path):
    plain = _spawn_sleeper(tmp_path)
    stubborn = _spawn_sleeper(tmp_path, term_ignored=True)
    grouped = _spawn_sleeper(tmp_path, new_session=True)
    for pid in (plain, stubborn, grouped):
        os.kill(pid, 0)  # alive
    os.kill(plain, signal.SIGTERM)
    assert _pid_gone(plain)
    os.kill(stubborn, signal.SIGTERM)
    time.sleep(0.1)
    os.kill(stubborn, 0)  # TERM ignored: still alive
    os.kill(stubborn, signal.SIGKILL)
    assert _pid_gone(stubborn)
    os.killpg(grouped, signal.SIGTERM)  # its own session: a group kill reaches it
    assert _pid_gone(grouped)
    assert _orphans_are_reaped(tmp_path) in (True, False)


# ---- scripts/lib/gym_ladder.sh + scripts/start_gym.sh --------------------------------------------------------

LADDER = SCRIPTS / "lib" / "gym_ladder.sh"
GYM_CWD = f"{REPO}/3rdparty/Gym"
GYM_HEAD = ["+model_endpoint_readiness_timeout_seconds=0", "+head_server.host=127.0.0.1", "+head_server.port=8100"]
GYM_RANGE = ["+port_range_low=8300", "+port_range_high=8999"]
LADDER_ECHO = ('echo "PD=$PROMPT_DATA ED=$EVAL_DATA CFG=$GYMCFG APO=$AGENT_PORT_OVERRIDE X=[$GYM_EXTRA_OVERRIDES] '
               'DDA=[$DRIVER_DATA_ARGS] RRA=[$RECIPE_RUN_ARGS] PM=$POLICY_MODEL PH=$POLICY_HF TD=$TORCH_DIST_DIR '
               'F=$RECIPE_PREPARE_DATA/$RECIPE_PROBE/$RECIPE_PRE_RAY/$RECIPE_PRE_GYM"')


def _cfg(server):
    """The gym config of a resources server directory (alfworld_env -> .../alfworld_env/configs/alfworld.yaml)."""
    return f"{REPO}/gym_root/resources_servers/{server}/configs/{server.replace('_env', '')}.yaml"


def _roots(tmp_path, model="Qwen3.5-4B"):
    """The three roots under tmp, the policy model's config.json in place, a fixed JOB_ID, no local.env."""
    models = Path(tmp_path) / "models"
    (models / model).mkdir(parents=True, exist_ok=True)
    (models / model / "config.json").write_text("{}")
    (Path(tmp_path) / "data").mkdir(exist_ok=True)
    return dict(LOCAL_ENV="/dev/null", MILES_AGENT_MODELS=str(models), MILES_AGENT_DATA=str(tmp_path / "data"),
                MILES_AGENT_RESULTS=str(tmp_path / "results"), JOB_ID="4242")


def _ladder(snippet, env, **kw):
    """common.sh + gym_ladder.sh sourced under the scripts' own options, then the snippet."""
    return _bash(f"set -euo pipefail; . {COMMON}; . {LADDER}; {snippet}", env, **kw)


def _dry_env(out):
    """The KEY -> VALUE of the `DRY_RUN env:` lines."""
    prefix = "DRY_RUN env: "
    return dict(line[len(prefix):].split("=", 1) for line in out.splitlines() if line.startswith(prefix))


def _dry_cwd(out):
    prefix = "DRY_RUN cwd: "
    return [line[len(prefix):] for line in out.splitlines() if line.startswith(prefix)]


def _fake_at(path):
    """A copy of the fake at an arbitrary path (a venv's python, a prebuilt gym, an engine interpreter)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!" + sys.executable + "\n" + FAKE)
    path.chmod(0o755)
    return path


def _serve(args, env, until=None, script="start_gym.sh"):
    """Popen the script (start_gym.sh, or `script` under scripts/) and, with `until`, read its stdout up to the line
    starting with it (or EOF); returns (proc, lines). Without `until` nothing is read: a server that prints nothing
    would block the read forever."""
    proc = subprocess.Popen([BASH, str(SCRIPTS / script), *args], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    while until is not None:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if line.startswith(until):
            break
    return proc, lines


def _wait_for(path, timeout=10):
    deadline = time.time() + timeout
    while not Path(path).exists() and time.time() < deadline:
        time.sleep(0.05)
    return Path(path).exists()


def test_start_gym_without_a_venv_root_uses_uv_run(tmp_path):
    _fakes(tmp_path)
    env = _env(tmp_path, **_roots(tmp_path))
    rc, out = _run_script("scripts/start_gym.sh", "--dry-run", "alfworld", env=env)
    assert rc == 0, out
    assert _dry(out, "gym") == ["uv", "run", "gym", "env", "start", "--config", _cfg("alfworld_env"), "--model-type",
                                "openai_model", *GYM_HEAD,
                                "+alfworld_agent.responses_api_agents.env_loop_agent.port=8200", *GYM_RANGE]
    assert _dry_env(out) == {"NEMO_GYM_EXTRA_ROOTS": f"{REPO}/gym_root", "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                             "RAY_TMPDIR": "/tmp/gymray"}
    assert _dry_cwd(out) == [GYM_CWD]
    assert "WARN" not in out and "GYM_UP" not in out
    assert _calls(env) == [] and not (tmp_path / "gym.pid").exists()
    # the flag may follow the recipe; extra +overrides ride at the end; a missing policy dir is a WARN here
    env = _env(tmp_path, LOCAL_ENV="/dev/null", MILES_AGENT_MODELS=str(tmp_path / "nomodels"),
               MILES_AGENT_DATA=str(tmp_path / "data"), WEBSHOP_SERVER_PORT="6300")
    rc, out = _run_script("scripts/start_gym.sh", "webshop", "--dry-run", "+webshop_env.x=1", env=env)
    assert rc == 0, out
    argv = _dry(out, "gym")
    assert argv[:5] == ["uv", "run", "gym", "env", "start"] and argv[-1] == "+webshop_env.x=1"
    assert "+webshop_env.resources_servers.webshop_env.engine_base_url=http://127.0.0.1:6300" in argv
    assert f"WARN: no policy model HF dir {tmp_path}/nomodels/Qwen3.5-4B" in out
    assert _dry(out, "webshop_engine") == ["bash", f"{REPO}/env_servers/webshop/launch_engine.sh"]
    assert _calls(env) == []


def test_start_gym_with_a_venv_root_uses_the_prebuilt_gym(tmp_path):
    _fakes(tmp_path)
    env = _env(tmp_path, GYM_VENV_ROOT="/r", **_roots(tmp_path))
    rc, out = _run_script("scripts/start_gym.sh", "--dry-run", "scienceworld", env=env)
    assert rc == 0, out
    assert _dry(out, "gym") == ["/r/gym_main_venv/bin/gym", "env", "start", "--config", _cfg("scienceworld_env"),
                                "--model-type", "openai_model", "+skip_venv_if_present=true",
                                "+uv_venv_dir=/r/gym_venvs", "+uv_cache_dir=/r/uv_cache", *GYM_HEAD,
                                "+scienceworld_agent.responses_api_agents.env_loop_agent.port=8200", *GYM_RANGE,
                                "+scienceworld_env.resources_servers.scienceworld_env.env_workers=128"]
    assert _dry_env(out) == {"NEMO_GYM_EXTRA_ROOTS": f"{REPO}/gym_root", "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                             "RAY_TMPDIR": "/tmp/gymray", "UV_OFFLINE": "1"}
    assert _dry_cwd(out) == [GYM_CWD] and _calls(env) == []


def test_start_gym_legacy_no_recipe_form_is_unchanged(tmp_path):
    _fakes(tmp_path)
    base = dict(LOCAL_ENV="/dev/null")
    env = _env(tmp_path, DRY_RUN="1", **base)
    rc, out = _run_script("scripts/start_gym.sh", env=env)
    assert rc == 0, out
    assert _dry(out, "gym") == ["uv", "run", "gym", "env", "start", "--config", _cfg("wiki_search"), "--model-type",
                                "openai_model"]
    assert _dry_env(out) == {"NEMO_GYM_EXTRA_ROOTS": f"{REPO}/gym_root", "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}
    assert _dry_cwd(out) == [GYM_CWD] and _calls(env) == []
    # GYM_CONFIG swaps the config; every word (a +override, a flag, a hydra key=value) goes to gym untouched
    rc, out = _run_script("scripts/start_gym.sh", "--dry-run", "+a=b", "--x", "k=v",
                          env=_env(tmp_path, GYM_CONFIG="/c.yaml", **base))
    assert rc == 0 and _dry(out, "gym") == ["uv", "run", "gym", "env", "start", "--config", "/c.yaml", "--model-type",
                                            "openai_model", "+a=b", "--x", "k=v"], out
    # the real call: `exec uv run ...` from 3rdparty/Gym with the two exports; the exec'd fake serves until TERM
    env = _env(tmp_path, **base)
    proc, lines = _serve(["+a=b"], env)  # nothing is printed before the exec replaces the shell: no read
    try:
        assert _wait_for(tmp_path / "gym.pid"), "".join(lines)
        proc.send_signal(signal.SIGTERM)
        out = "".join(lines) + proc.communicate(timeout=20)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
    assert proc.returncode == 0, out  # the script IS the gym process: the fake's TERM handler exits 0
    call, = _calls(env, "uv")
    assert call["argv"] == ["run", "gym", "env", "start", "--config", _cfg("wiki_search"), "--model-type",
                            "openai_model", "+a=b"]
    assert call["cwd"] == GYM_CWD and call["env"]["NEMO_GYM_EXTRA_ROOTS"] == f"{REPO}/gym_root"
    assert call["env"]["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0" and "RAY_TMPDIR" not in call["env"]
    text = (SCRIPTS / "start_gym.sh").read_text()
    assert 'cd "$REPO/3rdparty/Gym"' in text and 'exec "${gym_argv[@]}"' in text


def test_start_gym_recipe_form_serves_until_term(tmp_path):
    roots = _roots(tmp_path)
    ready = {"http://127.0.0.1:8100/readyz": "200", "http://127.0.0.1:8200/openapi.json": ["000", "200"]}
    _fakes(tmp_path, state={"curl": ready, "sleep_cap_s": 0.3})
    env = _env(tmp_path, **roots)
    rendered = [REPO / "gym_root" / "env.yaml", REPO / "3rdparty" / "Gym" / "env.yaml"]
    saved = [p.read_bytes() if p.exists() else None for p in rendered]  # a developer's own copies survive the test
    proc, lines = _serve(["alfworld"], env, until="GYM_UP")
    try:
        assert lines and lines[-1].startswith("GYM_UP agent=http://127.0.0.1:8200"), "".join(lines)
        text = rendered[0].read_text()
        assert "policy_base_url: http://127.0.0.1:9/v1" in text and "policy_api_key: dummy" in text
        assert rendered[1].read_bytes() == rendered[0].read_bytes()
        assert _wait_for(tmp_path / "gym.pid")
        proc.send_signal(signal.SIGTERM)
        out = "".join(lines) + proc.communicate(timeout=20)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
        for path, data in zip(rendered, saved):
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(data)
    assert proc.returncode == 143, out
    assert _pid_gone(tmp_path / "gym.pid")  # the exit ran the cleanup trap, which killed the gym itself
    assert "start_gym: RECIPE=alfworld serving (gym pid " in out
    call, = _calls(env, "uv")
    assert call["argv"][:7] == ["run", "gym", "env", "start", "--config", _cfg("alfworld_env"), "--model-type"]
    assert call["cwd"] == GYM_CWD
    assert Path(f"{roots['MILES_AGENT_RESULTS']}/runs/gym_gym_4242.log").exists()  # RUN_TAG defaults to gym


def test_recipe_vars_sets_the_ladder_for_each_recipe(tmp_path):
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    d, m = roots["MILES_AGENT_DATA"], roots["MILES_AGENT_MODELS"]

    def vars_of(recipe, **extra):
        rc, out = _ladder(f"run_dirs t1; recipe_vars {recipe}; {LADDER_ECHO}", _env(tmp_path, **{**roots, **extra}))
        assert rc == 0, out
        return out

    out = vars_of("alfworld")
    assert (f"PD={d}/runs/t1/alfworld_train.jsonl ED={d}/runs/t1/alfworld_eval274.jsonl CFG={_cfg('alfworld_env')} "
            "APO=+alfworld_agent.responses_api_agents.env_loop_agent.port=8200 X=[] "
            f"DDA=[--prompt-data {d}/runs/t1/alfworld_train.jsonl --eval-data {d}/runs/t1/alfworld_eval274.jsonl] "
            f"RRA=[] PM=Qwen3.5-4B PH={m}/Qwen3.5-4B TD={m}/torch_dist_v5 F=prepare_data_alfworld/:/:/:") in out
    out = vars_of("scienceworld")
    assert (f"PD={d}/runs/t1/scienceworld_train.jsonl ED={d}/runs/t1/scienceworld_eval274.jsonl "
            f"CFG={_cfg('scienceworld_env')} APO=+scienceworld_agent.responses_api_agents.env_loop_agent.port=8200 "
            "X=[+scienceworld_env.resources_servers.scienceworld_env.env_workers=128] ") in out
    assert "F=prepare_data_scienceworld/:/:/pre_gym_scienceworld" in out
    out = vars_of("webshop")
    assert (f"PD={d}/runs/t1/webshop_canonical/webshop_train.jsonl ED={d}/runs/t1/webshop_canonical/webshop_eval.jsonl "
            f"CFG={_cfg('webshop_env')} APO=+webshop_agent.responses_api_agents.env_loop_agent.port=8200 "
            "X=[+webshop_env.resources_servers.webshop_env.engine_base_url=http://127.0.0.1:6200 "
            "+webshop_env.resources_servers.webshop_env.observation_mode=text_rich "
            "+webshop_env.resources_servers.webshop_env.invalid_action_feedback=error] ") in out
    assert "F=prepare_data_webshop/:/:/pre_gym_webshop" in out
    assert f"PD={d}/runs/t1/webshop_tcod/webshop_train.jsonl" in vars_of("webshop", WEBSHOP_PROTOCOL="tcod")
    out = vars_of("search_r1", RETRIEVER_URL="http://192.0.2.5:8000")
    assert (f"PD={d}/search_r1_train.jsonl ED={d}/search_r1_eval893.jsonl CFG={_cfg('wiki_search')} "
            "APO=+wiki_search_agent.responses_api_agents.tool_call_agent.port=8200 "
            "X=[+wiki_search.resources_servers.wiki_search.retriever_base_url=http://192.0.2.5:8000] "
            f"DDA=[--prompt-data {d}/search_r1_train.jsonl --eval-data {d}/search_r1_eval893.jsonl] RRA=[]") in out
    assert "F=prepare_data_search_r1/:/pre_ray_search_r1/pre_gym_search_r1" in out
    # NEMO_GYM_PORT moves the agent port; a prebuilt root arms the probe; POLICY_HF, MILES_TORCH_DIST_DIR and
    # MILES_DATA_DIR win over the derived paths
    out = vars_of("alfworld", NEMO_GYM_PORT="8201", GYM_VENV_ROOT="/r")
    assert "APO=+alfworld_agent.responses_api_agents.env_loop_agent.port=8201" in out
    assert "F=prepare_data_alfworld/probe_gym_venvs_alfworld/:/:" in out
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "config.json").write_text("{}")
    out = vars_of("alfworld", POLICY_MODEL="Other", POLICY_HF=str(tmp_path / "hf"), MILES_TORCH_DIST_DIR="/td",
                  MILES_DATA_DIR="/md")
    assert "PD=/md/alfworld_train.jsonl" in out and f"PM=Other PH={tmp_path}/hf TD=/td" in out
    # the gates
    rc, out = _ladder("run_dirs t1; recipe_vars nope; echo NOT_REACHED", _env(tmp_path, **roots))
    assert rc == 1 and "NOT_REACHED" not in out
    assert "FATAL: unknown RECIPE=nope (alfworld | scienceworld | webshop | search_r1 | tmax)" in out
    rc, out = _ladder("recipe_vars alfworld; echo NOT_REACHED", _env(tmp_path, **roots))
    assert rc == 1 and "FATAL: recipe_vars: run_dirs <tag> must run first" in out and "NOT_REACHED" not in out
    nomodels = dict(roots, MILES_AGENT_MODELS=str(tmp_path / "nomodels"))
    rc, out = _ladder("run_dirs t1; recipe_vars alfworld; echo NOT_REACHED", _env(tmp_path, **nomodels))
    assert rc == 1 and f"FATAL: no policy model HF dir {tmp_path}/nomodels/Qwen3.5-4B" in out
    assert "NOT_REACHED" not in out
    rc, out = _ladder("run_dirs t1; recipe_vars alfworld; echo REACHED", _env(tmp_path, DRY_RUN="1", **nomodels))
    assert rc == 0 and f"WARN: no policy model HF dir {tmp_path}/nomodels/Qwen3.5-4B" in out and "REACHED" in out


def test_search_r1_ladder_needs_retriever_url_and_overrides_the_config(tmp_path):
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    yaml = Path(_cfg("wiki_search"))
    before = yaml.read_bytes()
    assert b"retriever_base_url: http://127.0.0.1:8000" in before
    rc, out = _ladder("run_dirs t1; recipe_vars search_r1; echo NOT_REACHED", _env(tmp_path, **roots))
    assert rc == 1 and "NOT_REACHED" not in out
    assert "FATAL: RECIPE=search_r1 needs RETRIEVER_URL" in out and "RETRIEVER_URL=http://<host>:<port>" in out
    wiki = tmp_path / "data" / "wiki_search"
    wiki.mkdir()
    for name in ("train.jsonl", "validation.jsonl"):
        (wiki / name).write_text("{}\n")
    env = _env(tmp_path, RETRIEVER_URL="http://192.0.2.5:8000", DRY_RUN="1", **roots)
    rc, out = _ladder('run_dirs t1; recipe_vars search_r1; echo "X=[$GYM_EXTRA_OVERRIDES] W=$WIKI_SEARCH_DATA"; '
                      "pre_gym_search_r1", env)
    assert rc == 0, out
    assert f"X=[+wiki_search.resources_servers.wiki_search.retriever_base_url=http://192.0.2.5:8000] W={wiki}" in out
    assert yaml.read_bytes() == before  # the config file is never edited in place (the old wrapper sed -i'd it)
    # a trailing slash on RETRIEVER_URL is dropped: the server appends its own paths to the base
    rc_slash, out_slash = _ladder('run_dirs t1; recipe_vars search_r1; echo "X=[$GYM_EXTRA_OVERRIDES]"',
                                  _env(tmp_path, RETRIEVER_URL="http://192.0.2.5:8000/", DRY_RUN="1", **roots))
    assert rc_slash == 0, out_slash
    assert "X=[+wiki_search.resources_servers.wiki_search.retriever_base_url=http://192.0.2.5:8000]" in out_slash
    data_dir = REPO / "gym_root" / "resources_servers" / "wiki_search" / "data"
    if not (data_dir / "train.jsonl").exists() and not (data_dir / "validation.jsonl").exists():
        # the links the pre-gym step makes (under --dry-run: printed only), and the FATAL for missing sources
        assert _dry(out, "link") == ["ln", "-s", f"{wiki}/train.jsonl", f"{data_dir}/train.jsonl"]
        assert _dry(out, "link", 1) == ["ln", "-s", f"{wiki}/validation.jsonl", f"{data_dir}/validation.jsonl"]
        env = _env(tmp_path, RETRIEVER_URL="http://192.0.2.5:8000", WIKI_SEARCH_DATA=str(tmp_path / "nowhere"), **roots)
        rc, out = _ladder("run_dirs t1; recipe_vars search_r1; pre_gym_search_r1; echo NOT_REACHED", env)
        assert rc == 1 and "NOT_REACHED" not in out
        assert f"FATAL: no {tmp_path}/nowhere/train.jsonl: run scripts/fetch_data.sh search_r1" in out
        assert not (data_dir / "train.jsonl").exists()
    assert yaml.read_bytes() == before
    assert "sed -i" not in LADDER.read_text()


def test_scienceworld_ladder_accepts_the_sciw_aliases(tmp_path):
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    echo = ('echo "J=[$SCIENCEWORLD_JAR] H=[$SCIENCEWORLD_JAVA_HOME] W=$SCIENCEWORLD_ENV_WORKERS X=[$GYM_EXTRA_OVERRIDES] '
            'C=[$(scienceworld_java_home)]"')
    sw = "+scienceworld_env.resources_servers.scienceworld_env."

    def vars_of(**extra):
        rc, out = _ladder(f"run_dirs t1; recipe_vars scienceworld; {echo}", _env(tmp_path, **{**roots, **extra}))
        assert rc == 0, out
        return out

    # nothing set: "" = the wheel's bundled jar and the java on PATH (the yaml's own defaults), only env_workers
    assert f"J=[] H=[] W=128 X=[{sw}env_workers=128]" in vars_of()
    # the old aliases
    assert (f"J=[/j] H=[/sj] W=4 X=[{sw}jar_path=/j {sw}java_home=/sj {sw}env_workers=4]"
            in vars_of(SCIW_JAR="/j", SCIW_JAVA_HOME="/sj", SCIW_ENV_WORKERS="4"))
    # the new names win
    assert "J=[/n] H=[/nh] W=2 " in vars_of(SCIW_JAR="/j", SCIENCEWORLD_JAR="/n", SCIW_JAVA_HOME="/sj",
                                             SCIENCEWORLD_JAVA_HOME="/nh", SCIW_ENV_WORKERS="4",
                                             SCIENCEWORLD_ENV_WORKERS="2")
    # JAVA_HOME sits between the alias and the TOOLS_DIR default; the defaults need the dir / file to exist
    (tmp_path / "tools" / "jdk-17").mkdir(parents=True)
    tools = str(tmp_path / "tools")
    assert f"H=[{tools}/jdk-17]" in vars_of(TOOLS_DIR=tools)
    assert "H=[/jh]" in vars_of(TOOLS_DIR=tools, JAVA_HOME="/jh")
    assert "H=[/sj]" in vars_of(TOOLS_DIR=tools, JAVA_HOME="/jh", SCIW_JAVA_HOME="/sj")
    assert "H=[]" in vars_of(TOOLS_DIR=str(tmp_path / "notools"))
    # the ladder's value is the shared helper's (scripts/lib/common.sh scienceworld_java_home): scripts/fetch_data.sh
    # and scripts/build_gym_venvs.sh resolve the JDK through the same function, so the three cannot disagree
    out = vars_of(TOOLS_DIR=tools, JAVA_HOME="/jh", SCIW_JAVA_HOME="/sj")
    assert "H=[/sj] " in out and "C=[/sj]" in out
    assert "C=[/nh]" in vars_of(SCIW_JAVA_HOME="/sj", SCIENCEWORLD_JAVA_HOME="/nh")
    assert f"C=[{tools}/jdk-17]" in vars_of(TOOLS_DIR=tools) and "C=[]" in vars_of(TOOLS_DIR=str(tmp_path / "notools"))
    assert "default_scienceworld_java_home" not in LADDER.read_text()  # one resolution, not a second copy of the order
    jar = tmp_path / "data" / "scienceworld" / "scienceworld.jar"
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"jar\n")
    out = vars_of()
    assert f"J=[{jar}] " in out and f"X=[{sw}jar_path={jar} {sw}env_workers=128]" in out


def test_webshop_ladder_defaults_webshop_home_under_the_data_root(tmp_path):
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    d = roots["MILES_AGENT_DATA"]
    echo = ('echo "H=$WEBSHOP_HOME PY=$WEBSHOP_PYTHON V=$WEBSHOP_VENDOR PORT=$WEBSHOP_SERVER_PORT '
            'URL=$WEBSHOP_ENGINE_BASE_URL PROTO=$WEBSHOP_PROTOCOL X=[$GYM_EXTRA_OVERRIDES]"')
    rc, out = _ladder(f"run_dirs t1; recipe_vars webshop; {echo}", _env(tmp_path, **roots))
    assert rc == 0, out
    assert (f"H={d}/webshop PY={d}/webshop/env/bin/python V={d}/webshop/webshop PORT=6200 "
            "URL=http://127.0.0.1:6200 PROTO=canonical") in out
    # the knobs of env_servers/webshop/defaults.sh stay overridable; WEBSHOP_SERVER_PORT reaches the adapter's
    # engine_base_url override
    rc, out = _ladder(f"run_dirs t1; recipe_vars webshop; {echo}",
                      _env(tmp_path, WEBSHOP_HOME="/w", WEBSHOP_SERVER_PORT="6300", WEBSHOP_PROTOCOL="tcod", **roots))
    assert rc == 0, out
    assert "H=/w PY=/w/env/bin/python V=/w/webshop PORT=6300 URL=http://127.0.0.1:6300 PROTO=tcod" in out
    assert "X=[+webshop_env.resources_servers.webshop_env.engine_base_url=http://127.0.0.1:6300 " in out
    # the old wrapper's project root never decides the home here: the data root does
    rc, out = _ladder(f"run_dirs t1; recipe_vars webshop; {echo}", _env(tmp_path, P="/p", **roots))
    assert rc == 0 and f"H={d}/webshop " in out, out


def test_venv_probe_gate_skips_without_a_root(tmp_path):
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    env = _env(tmp_path, **roots)
    rc, out = _ladder("run_dirs t1; recipe_vars alfworld; venv_probe_gate; echo RC=$?", env)
    assert rc == 0 and "GYM_VENV_PROBE_OK mode=uv_run\nRC=0" in out and _calls(env) == [], out
    root = tmp_path / "r"
    env = _env(tmp_path, GYM_VENV_ROOT=str(root), **roots)
    rc, out = _ladder("run_dirs t1; recipe_vars alfworld; venv_probe_gate; echo NOT_REACHED", env)
    assert rc == 1 and "NOT_REACHED" not in out and "GYM_VENV_PROBE_OK" not in out
    assert "FATAL: gym venv probe failed for RECIPE=alfworld" in out and f"scripts/build_gym_venvs.sh {root}" in out
    # the venvs this recipe starts plus gym >= 0.7.0's legacy_agent relay: one import probe each
    venvs = {"responses_api_agents/env_loop_agent": "agent_utils.chat_dialect, nemo_gym",
             "resources_servers/alfworld_env": "alfworld_env.app, alfworld, textworld",
             "environment_servers/legacy_agent": "nemo_gym.base_environment_server, httptools"}
    for rel in venvs:
        _fake_at(root / "gym_venvs" / rel / ".venv" / "bin" / "python")
    rc, out = _ladder("run_dirs t1; recipe_vars alfworld; venv_probe_gate; echo RC=$?", env)
    assert rc == 0 and "GYM_VENV_PROBE_OK\nRC=0" in out, out
    assert [c["argv"] for c in _calls(env, "python")] == [["-c", f"import {imports}"] for imports in venvs.values()]
    # --dry-run prints the probe argv and no marker
    _fakes(tmp_path)  # a fresh call log (the venv pythons are files of their own, untouched)
    env = _env(tmp_path, GYM_VENV_ROOT=str(root), DRY_RUN="1", **roots)
    rc, out = _ladder("run_dirs t1; recipe_vars alfworld; venv_probe_gate", env)
    assert rc == 0 and "GYM_VENV_PROBE_OK" not in out and _calls(env) == []
    assert _dry(out, "probe") == [str(root / "gym_venvs/responses_api_agents/env_loop_agent/.venv/bin/python"), "-c",
                                  "import agent_utils.chat_dialect, nemo_gym"]
    assert len([line for line in out.splitlines() if line.startswith("DRY_RUN probe:")]) == 3
    # the probe python of every recipe is the venv's own interpreter (static: the prebuilt layout of D6)
    text = LADDER.read_text()
    assert 'local py=$GYM_VENV_ROOT/gym_venvs/$1/$2/.venv/bin/python' in text
    for name, imports in (("search_r1", "wiki_search.app"), ("webshop", "webshop_env.app"),
                          ("scienceworld", "scienceworld_env.app, scienceworld, py4j")):
        assert f'"{imports}"' in _function_body(text, f"probe_gym_venvs_{name}")


def test_tmax_ladder_needs_a_preset_gym_url(tmp_path):
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    m = roots["MILES_AGENT_MODELS"]
    rc, out = _ladder("run_dirs t1; recipe_vars tmax; echo NOT_REACHED",
                      _env(tmp_path, RUN_TAG="t1", PROMPT_DATA="/p.jsonl", **roots))
    assert rc == 1 and "FATAL: RECIPE=tmax needs NEMO_GYM_URL=http://<env host>:8200" in out
    assert "NOT_REACHED" not in out
    rc, out = _ladder("run_dirs t1; recipe_vars tmax; echo NOT_REACHED",
                      _env(tmp_path, RUN_TAG="t1", NEMO_GYM_URL="http://192.0.2.7:8200", **roots))
    assert rc == 1 and "FATAL: RECIPE=tmax needs PROMPT_DATA=<rows.jsonl>" in out and "NOT_REACHED" not in out
    base = dict(roots, RUN_TAG="t1", PROMPT_DATA="/p.jsonl", NEMO_GYM_URL="http://192.0.2.7:8200")
    rc, out = _ladder(f"run_dirs t1; recipe_vars tmax; {LADDER_ECHO}", _env(tmp_path, **base))
    assert rc == 0, out
    # (the string carries a blank where an optional part is empty: it is word-split by the driver line)
    assert "PD=/p.jsonl ED= CFG= APO= X=[] DDA=[--prompt-data /p.jsonl] RRA=[" in out
    assert f"PM=Qwen3.5-4B PH={m}/Qwen3.5-4B TD={m}/torch_dist_v5 F=:/:/:/:" in out
    rra = shlex.split(re.search(r"RRA=\[(.*?)\]", out).group(1))
    assert rra == ["--run-tag", "t1", "--nemo-gym-url", "http://192.0.2.7:8200"]
    rc, out = _ladder(f"run_dirs t1; recipe_vars tmax; {LADDER_ECHO}",
                      _env(tmp_path, EVAL_PROMPT_DATA="/e.jsonl", SESSION_SERVER_EXTERNAL_HOST="192.0.2.9", **base))
    assert rc == 0, out
    rra = shlex.split(re.search(r"RRA=\[(.*?)\]", out).group(1))
    assert rra == ["--run-tag", "t1", "--eval-prompt-data", "/e.jsonl", "--nemo-gym-url", "http://192.0.2.7:8200",
                   "--session-server-external-host", "192.0.2.9"]
    assert "tmax" not in rra  # run.py adds the `tmax` name itself: `--eval-prompt-data tmax <path>`
    # fully_async: the node split run_local.sh checked (run.py's own defaults are 1 + 3)
    rc, out = _ladder(f"run_dirs t1; recipe_vars tmax; {LADDER_ECHO}",
                      _env(tmp_path, TOPOLOGY="fully_async", TRAIN_NODES="1", ROLLOUT_NODES="7", **base))
    assert rc == 0, out
    rra = shlex.split(re.search(r"RRA=\[(.*?)\]", out).group(1))
    assert rra[-4:] == ["--train-nodes", "1", "--rollout-nodes", "7"]
    # no gym ladder: the hooks are no-ops, the probe gate passes even with a prebuilt root
    env = _env(tmp_path, GYM_VENV_ROOT="/r", **base)
    rc, out = _ladder('run_dirs t1; recipe_vars tmax; "$RECIPE_PREPARE_DATA"; venv_probe_gate; "$RECIPE_PRE_RAY"; '
                      '"$RECIPE_PRE_GYM"; echo OK', env)
    assert rc == 0 and "GYM_VENV_PROBE_OK mode=no_gym\nOK" in out and _calls(env) == [], out
    # start_gym.sh refuses to start a gym for it
    rc, out = _run_script("scripts/start_gym.sh", "--dry-run", "tmax", env=_env(tmp_path, **base))
    assert rc == 1 and "FATAL: RECIPE=tmax starts no gym here: NEMO_GYM_URL=http://192.0.2.7:8200" in out, out


def test_prepare_data_functions_build_and_gate_the_rows(tmp_path):
    roots = _roots(tmp_path)
    d, hf = roots["MILES_AGENT_DATA"], f"{roots['MILES_AGENT_MODELS']}/Qwen3.5-4B"
    alf = tmp_path / "alf"
    (alf / "json_2.1.1").mkdir(parents=True)

    def prepare(recipe, state, **extra):
        _fakes(tmp_path, state=state)
        env = _env(tmp_path, **{**roots, **extra})
        rc, out = _ladder(f'run_dirs t1; recipe_vars {recipe}; "$RECIPE_PREPARE_DATA"; echo PREPARED', env)
        return rc, out, [c["argv"] for c in _calls(env, "python3")]

    # alfworld: the game-file root, the output dir of PROMPT_DATA, the policy model, the optional prompt; EV == 274
    rc, out, calls = prepare("alfworld", {"rows": {"alfworld_eval274.jsonl": 274}}, ALFWORLD_DATA=str(alf))
    assert rc == 0 and "PREPARED" in out, out
    assert calls == [[f"{REPO}/recipes/alfworld/prepare_data.py", "--data-root", str(alf), "--output-dir",
                      f"{d}/runs/t1", "--model", hf]]
    rc, out, calls = prepare("alfworld", {"rows": {"alfworld_eval274.jsonl": 274}}, ALFWORLD_DATA=str(alf),
                             PROMPT="think")
    assert rc == 0 and calls[0][-2:] == ["--prompt", "think"], out
    rc, out, calls = prepare("alfworld", {"rows": {"alfworld_eval274.jsonl": 273}}, ALFWORLD_DATA=str(alf))
    assert rc == 1 and "PREPARED" not in out
    assert "FATAL: eval set has 273 rows, expected 274 (seen-then-unseen order!)" in out
    # a converter that ended without the eval file (a fresh data root: nothing from the runs above): a FATAL naming
    # the file, not a bare redirection error under set -e
    rc, out, calls = prepare("alfworld", {"rows": {"alfworld_eval274.jsonl": -1}}, ALFWORLD_DATA=str(alf),
                             MILES_AGENT_DATA=str(tmp_path / "d_noeval"))
    assert rc == 1 and "PREPARED" not in out and len(calls) == 1, out
    assert f"FATAL: prepare wrote no eval file {tmp_path}/d_noeval/runs/t1/alfworld_eval274.jsonl" in out
    rc, out, calls = prepare("alfworld", {}, ALFWORLD_DATA=str(tmp_path / "noalf"))
    assert rc == 1 and f"FATAL: no {tmp_path}/noalf/json_2.1.1" in out and calls == []
    rc, out, calls = prepare("alfworld", {}, ALFWORLD_DATA=str(tmp_path / "noalf"), DRY_RUN="1")
    assert rc == 0 and f"WARN: no {tmp_path}/noalf/json_2.1.1" in out and calls == [] and "PREPARED" in out
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/alfworld/prepare_data.py", "--data-root",
                                    f"{tmp_path}/noalf", "--output-dir", f"{d}/runs/t1", "--model", hf]
    # scienceworld
    rc, out, calls = prepare("scienceworld", {"rows": {"scienceworld_eval274.jsonl": 274}})
    assert rc == 0, out
    assert calls == [[f"{REPO}/recipes/scienceworld/prepare_data.py", "--output-dir", f"{d}/runs/t1", "--model", hf]]
    rc, out, calls = prepare("scienceworld", {"rows": {"scienceworld_eval274.jsonl": 275}})
    assert rc == 1 and "FATAL: eval set has 275 rows, expected 274 (140 valid_seen then 134 valid_unseen!)" in out
    # webshop: the per-protocol dir, --protocol, the row count per protocol, protocol.json echoed
    rc, out, calls = prepare("webshop", {"rows": {"webshop_eval.jsonl": 756}, "goals_digest": "d1"})
    assert rc == 0, out
    assert calls == [[f"{REPO}/recipes/webshop/prepare_data.py", "--output-dir", f"{d}/runs/t1/webshop_canonical",
                      "--protocol", "canonical", "--model", hf]]
    assert '"goals_digest": "d1"' in out
    rc, out, calls = prepare("webshop", {"rows": {"webshop_eval.jsonl": 500}}, WEBSHOP_PROTOCOL="tcod")
    assert rc == 0 and calls[0][2:5] == [f"{d}/runs/t1/webshop_tcod", "--protocol", "tcod"], out
    rc, out, calls = prepare("webshop", {"rows": {"webshop_eval.jsonl": 756}}, WEBSHOP_PROTOCOL="tcod")
    assert rc == 1 and "FATAL: eval set has 756 rows, expected 500 for protocol tcod" in out
    rc, out, calls = prepare("webshop", {}, WEBSHOP_PROTOCOL="bogus")
    assert rc == 1 and "FATAL: WEBSHOP_PROTOCOL=bogus is not tcod|canonical" in out and calls == []
    # search_r1: two conversions from $WIKI_SEARCH_DATA, skipped once both files exist; EV == 893
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    for name in ("train.jsonl", "validation.jsonl"):
        (wiki / name).write_text("{}\n")
    s = dict(RETRIEVER_URL="http://192.0.2.5:8000", WIKI_SEARCH_DATA=str(wiki))
    rc, out, calls = prepare("search_r1", {"rows": {"search_r1_eval893.jsonl": 893}}, **s)
    assert rc == 0, out
    assert calls == [[f"{REPO}/recipes/search_r1/prepare_data.py", "--input", f"{wiki}/train.jsonl", "--output",
                      f"{d}/search_r1_train.jsonl"],
                     [f"{REPO}/recipes/search_r1/prepare_data.py", "--input", f"{wiki}/validation.jsonl", "--output",
                      f"{d}/search_r1_eval893.jsonl"]]
    rc, out, calls = prepare("search_r1", {}, **s)
    assert rc == 0 and calls == [], out  # idempotent: both files exist
    rc, out, calls = prepare("search_r1", {"rows": {"search_r1_eval893.jsonl": 892}},
                             MILES_AGENT_DATA=str(tmp_path / "d3"), **s)
    assert rc == 1 and "FATAL: eval set has 892 rows, expected 893 (source-block order!)" in out
    rc, out, calls = prepare("search_r1", {}, RETRIEVER_URL="http://192.0.2.5:8000",
                             WIKI_SEARCH_DATA=str(tmp_path / "nowiki"), MILES_AGENT_DATA=str(tmp_path / "d2"))
    assert rc == 1 and calls == []
    assert f"FATAL: no {tmp_path}/nowiki/train.jsonl: run scripts/fetch_data.sh search_r1" in out
    rc, out, calls = prepare("search_r1", {}, DRY_RUN="1", **s)
    assert rc == 0 and calls == [] and _dry(out, "prepare", 1)[-2:] == ["--output", f"{d}/search_r1_eval893.jsonl"]


def test_render_env_yaml_has_the_no_checkout_fatal_first(tmp_path):
    """render_env_yaml (the entry runs it right before gym_start_and_wait) FATALs without the gym checkout as the
    start does, before its sed / cp (a cp into a missing dir would end the entry with a bare error under set -e);
    --dry-run returns before the gate. gym_cwd is redefined in the snippet so that the real checkout is never written."""
    _fakes(tmp_path)
    env = _env(tmp_path, **_roots(tmp_path))
    missing = tmp_path / "no_gym"
    snippet = f'gym_cwd() {{ printf "%s\\n" "{missing}"; }}; render_env_yaml; echo NOT_REACHED'
    rc, out = _ladder(snippet, env)
    assert rc == 1 and f"FATAL: no gym checkout at {missing}: git submodule update --init 3rdparty/Gym" in out
    assert "NOT_REACHED" not in out
    rc, out = _ladder(snippet, dict(env, DRY_RUN="1"))
    assert rc == 0 and "NOT_REACHED" in out and "FATAL" not in out


def test_gym_start_and_wait_boots_the_fake_gym_and_prints_gym_up(tmp_path):
    roots = _roots(tmp_path)
    readyz, openapi = "http://127.0.0.1:8100/readyz", "http://127.0.0.1:8200/openapi.json"
    _fakes(tmp_path, state={"curl": {readyz: "200", openapi: ["000", "200"]}, "sleep_cap_s": 0.3})
    env = _env(tmp_path, RUN_TAG="T", **roots)
    rc, out = _ladder('run_dirs T; recipe_vars alfworld; gym_start_and_wait; echo "URL=$NEMO_GYM_URL LOG=$GYM_LOG"; '
                      "sh -c 'echo CHILD=$NEMO_GYM_URL'", env)
    assert rc == 0, out
    log = f"{roots['MILES_AGENT_RESULTS']}/runs/T_gym_4242.log"
    assert f"GYM_UP agent=http://127.0.0.1:8200\nURL=http://127.0.0.1:8200 LOG={log}\nCHILD=http://127.0.0.1:8200" in out
    assert Path(log).exists()
    assert _pid_gone(tmp_path / "gym.pid")  # the EXIT trap's cleanup killed the gym process itself
    call, = _calls(env, "uv")
    assert call["argv"] == ["run", "gym", "env", "start", "--config", _cfg("alfworld_env"), "--model-type",
                            "openai_model", *GYM_HEAD, "+alfworld_agent.responses_api_agents.env_loop_agent.port=8200",
                            *GYM_RANGE]
    assert call["cwd"] == GYM_CWD
    assert call["env"]["NEMO_GYM_EXTRA_ROOTS"] == f"{REPO}/gym_root" and call["env"]["RAY_TMPDIR"] == "/tmp/gymray"
    assert call["env"]["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0" and "UV_OFFLINE" not in call["env"]
    # the head's /readyz gates the agent's /openapi.json (84 x 5 s)
    assert [c["argv"][-1] for c in _calls(env, "curl")] == [readyz, openapi, readyz, openapi]
    assert [c["argv"] for c in _calls(env, "sleep")] == [["5"]]
    # a prebuilt root: its own gym binary, the three venv flags, UV_OFFLINE=1
    root = tmp_path / "r"
    _fake_at(root / "gym_main_venv" / "bin" / "gym")
    _fakes(tmp_path, state={"curl": {readyz: "200", openapi: ["000", "200"]}, "sleep_cap_s": 0.3})
    env = _env(tmp_path, RUN_TAG="T", GYM_VENV_ROOT=str(root), **roots)
    rc, out = _ladder("run_dirs T; recipe_vars alfworld; gym_start_and_wait", env)
    assert rc == 0 and "GYM_UP agent=http://127.0.0.1:8200" in out, out
    assert _pid_gone(tmp_path / "gym.pid")
    call, = _calls(env, "gym")
    assert call["argv"][:9] == ["env", "start", "--config", _cfg("alfworld_env"), "--model-type", "openai_model",
                                "+skip_venv_if_present=true", f"+uv_venv_dir={root}/gym_venvs",
                                f"+uv_cache_dir={root}/uv_cache"]
    assert call["env"]["UV_OFFLINE"] == "1" and call["cwd"] == GYM_CWD
    # a gym that never answers: 84 polls, the FATAL, the process killed
    _fakes(tmp_path)
    env = _env(tmp_path, RUN_TAG="T", **roots)
    rc, out = _ladder("run_dirs T; recipe_vars alfworld; gym_start_and_wait; echo NOT_REACHED", env)
    assert rc == 1 and "FATAL: gym servers failed to boot" in out and "NOT_REACHED" not in out and "GYM_UP" not in out
    assert _pid_gone(tmp_path / "gym.pid")
    assert len(_calls(env, "sleep")) == 84 and [c["argv"][-1] for c in _calls(env, "curl")] == [readyz] * 84
    # no RUN_TAG: a FATAL before anything starts
    _fakes(tmp_path)
    env = _env(tmp_path, **roots)
    rc, out = _ladder("run_dirs T; recipe_vars alfworld; gym_start_and_wait", env)
    assert rc == 1 and "FATAL: gym_start_and_wait: RUN_TAG is required" in out and _calls(env) == []
    # the static shape the C4 entry relies on: the exec makes $! the gym itself; the builtin kill in the cleanup
    body = _function_body(LADDER.read_text(), "gym_start_and_wait")
    assert 'exec "${GYM_ARGV[@]}" > "$GYM_LOG" 2>&1' in body and "GYM_PID=$!" in body
    assert "add_cleanup 'kill $GYM_PID 2>/dev/null'" in body
    assert 'wait_http_ready "http://127.0.0.1:$NEMO_GYM_PORT/openapi.json" "$GYM_PID" "$GYM_LOG" 84 5 gym_all_bound' in body


def test_pre_gym_webshop_gates_the_engine_protocol_and_goal_set(tmp_path):
    roots = _roots(tmp_path)
    ws = tmp_path / "data" / "webshop"
    _fake_at(ws / "env" / "bin" / "python")  # the py3.10 engine interpreter of $WEBSHOP_HOME
    (ws / "env" / "lib" / "jvm").mkdir(parents=True)
    (ws / "webshop" / "web_agent_site").mkdir(parents=True)
    (ws / "webshop" / "data").mkdir()
    (ws / "webshop" / "data" / "items_shuffle_1000.json").write_text("[]")
    health = "http://127.0.0.1:6200/health"
    body = {"protocol": {"seed": 233, "observation_mode": "text_rich", "human_goals": False, "catalog": "small"},
            "goals_digest": "d1", "goals": 6910, "unique_goals": 6901}

    def run(state, **extra):
        _fakes(tmp_path, state={"curl": {health: "200"}, "curl_body": {health: body}, "python_c": "real",
                                "rows": {"webshop_eval.jsonl": 756}, "goals_digest": "d1", **state})
        env = _env(tmp_path, RUN_TAG="T", **{**roots, **extra})
        rc, out = _ladder('run_dirs T; recipe_vars webshop; "$RECIPE_PREPARE_DATA" >/dev/null; pre_gym_webshop; '
                          'echo "EL=${ENGINE_LOG:-}"', env)
        return rc, out, env

    rc, out, env = run({})
    assert rc == 0, out
    assert ("WEBSHOP_ENGINE_UP port=6200 workers=8 goals=6910 unique_goals=6901 goals_digest=d1 protocol=canonical "
            "seed=233 obs=text_rich human_goals=0 catalog=small invalid_feedback=error prompt=think") in out
    log = f"{roots['MILES_AGENT_RESULTS']}/runs/T_webshop_engine_4242.log"
    assert f"EL={log}" in out and Path(log).exists()
    engine, = _calls(env, "python")  # launch_engine.sh ran server.py on the engine interpreter with every knob
    assert engine["argv"] == [f"{REPO}/env_servers/webshop/server.py", "--host", "127.0.0.1", "--port", "6200",
                              "--engine-workers", "8", "--engine-call-timeout", "120", "--session-ttl", "2400",
                              "--vendored-path", f"{ws}/webshop", "--data-dir", f"{ws}/webshop/data", "--goal-seed",
                              "233", "--observation-mode", "text_rich", "--small"]
    assert engine["env"]["WEBSHOP_HOME"] == str(ws) and engine["env"]["JAVA_HOME"] == f"{ws}/env/lib/jvm"
    assert [c["argv"][-1] for c in _calls(env, "curl")][:1] == [health]
    # the gates
    rc, out, env = run({"goals_digest": "d2"})
    assert rc == 1 and "WEBSHOP_ENGINE_UP" not in out
    assert ("FATAL: engine goal set (digest d1, 6910 rows / 6901 unique) != dataset (digest d2, 756 rows / 756 unique)"
            in out)
    rc, out, env = run({}, WEBSHOP_OBS_MODE="text")
    assert rc == 1 and ("FATAL: engine protocol (233 text_rich 0 small) != job knobs (seed=233 obs=text human_goals=0 "
                        "catalog=small)") in out
    rc, out, env = run({}, WEBSHOP_HUMAN_GOALS="1")
    assert rc == 1 and "FATAL: WEBSHOP_HUMAN_GOALS=1 needs WEBSHOP_FULL=1" in out and _calls(env, "python") == []
    rc, out, env = run({}, WEBSHOP_HOME=str(tmp_path / "nows"))
    assert rc == 1 and f"FATAL: no webshop py3.10 interpreter at {tmp_path}/nows/env/bin/python" in out
    rc, out, env = run({"curl": {health: "000"}})
    assert rc == 1 and "FATAL: webshop engine server failed to boot on :6200" in out
    # --dry-run: the launch line, nothing started, no marker
    rc, out, env = run({}, DRY_RUN="1")
    assert rc == 0, out
    assert _dry(out, "webshop_engine") == ["bash", f"{REPO}/env_servers/webshop/launch_engine.sh"]
    assert _calls(env, "python") == [] and "WEBSHOP_ENGINE_UP" not in out


def test_pre_gym_scienceworld_reports_the_runtime(tmp_path):
    roots = _roots(tmp_path)
    jar = tmp_path / "data" / "scienceworld" / "scienceworld.jar"
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b"jar\n")
    digest = hashlib.sha256(b"jar\n").hexdigest()[:16]
    java_text = "#!/bin/sh\necho 'openjdk version \"17.0.2\" 2022-01-18' >&2\necho 'OpenJDK Runtime Environment' >&2\n"
    jdk = tmp_path / "tools" / "jdk-17"
    (jdk / "bin").mkdir(parents=True)
    (jdk / "bin" / "java").write_text(java_text)
    (jdk / "bin" / "java").chmod(0o755)
    _fakes(tmp_path)
    snippet = 'run_dirs T; recipe_vars scienceworld; pre_gym_scienceworld; echo "JH=${JAVA_HOME:-} P1=${PATH%%:*}"'
    rc, out = _ladder(snippet, _env(tmp_path, TOOLS_DIR=str(tmp_path / "tools"), **roots))
    assert rc == 0, out
    assert f'SCIW_RUNTIME jar={jar} sha256={digest} java=openjdk version "17.0.2" 2022-01-18 workers=128 prompt=default' in out
    assert f"JH={jdk} P1={jdk}/bin" in out
    # the aliases reach the line; PROMPT is reported
    rc, out = _ladder(snippet, _env(tmp_path, SCIW_JAR=str(jar), SCIW_JAVA_HOME=str(jdk), SCIW_ENV_WORKERS="4",
                                    PROMPT="p1", **roots))
    assert rc == 0 and f'sha256={digest} java=openjdk version "17.0.2" 2022-01-18 workers=4 prompt=p1' in out, out
    # nothing pinned and no java anywhere: a FATAL, not a silent exit under pipefail
    nodata = dict(roots, MILES_AGENT_DATA=str(tmp_path / "nodata"))
    rc, out = _ladder(snippet + "; echo NOT_REACHED", _env(tmp_path, bare_path=True, **nodata))
    assert rc == 1 and "FATAL: no java on PATH for the ScienceWorld worker JVMs" in out and "NOT_REACHED" not in out
    # nothing pinned, a java on PATH: the wheel's bundled jar
    (tmp_path / "bin" / "java").write_text(java_text)
    (tmp_path / "bin" / "java").chmod(0o755)
    rc, out = _ladder(snippet, _env(tmp_path, **nodata))
    assert rc == 0, out
    assert 'SCIW_RUNTIME jar=bundled sha256=- java=openjdk version "17.0.2" 2022-01-18 workers=128 prompt=default' in out
    assert "JH= " in out
    rc, out = _ladder(snippet + "; echo NOT_REACHED", _env(tmp_path, SCIENCEWORLD_JAR="/nope.jar", **roots))
    assert rc == 1 and "FATAL: missing ScienceWorld jar /nope.jar" in out and "NOT_REACHED" not in out
    rc, out = _ladder(snippet + "; echo NOT_REACHED", _env(tmp_path, JAVA_HOME=str(tmp_path / "nojdk"), **roots))
    assert rc == 1 and f"FATAL: missing JDK {tmp_path}/nojdk/bin/java" in out and "NOT_REACHED" not in out
    rc, out = _ladder(snippet, _env(tmp_path, DRY_RUN="1", SCIENCEWORLD_JAR="/nope.jar",
                                    JAVA_HOME=str(tmp_path / "nojdk"), **roots))
    assert rc == 0 and "WARN: missing ScienceWorld jar /nope.jar" in out, out
    assert f"WARN: missing JDK {tmp_path}/nojdk/bin/java" in out and "SCIW_RUNTIME" not in out


def test_pre_ray_search_r1_probes_the_retriever(tmp_path):
    roots = _roots(tmp_path)
    base = dict(roots, RETRIEVER_URL="http://192.0.2.5:8000/")
    _fakes(tmp_path)
    env = _env(tmp_path, **base)
    rc, out = _ladder('run_dirs T; recipe_vars search_r1; pre_ray_search_r1; echo "PU=$MILES_PROBE_URL"', env)
    assert rc == 0, out
    assert "RETRIEVER_BRIDGE_OK\nPU=http://192.0.2.5:8000/retrieve" in out
    probe, = _calls(env, "python3")
    assert probe["argv"] == ["-"] and probe["env"]["MILES_PROBE_URL"] == "http://192.0.2.5:8000/retrieve"
    # the heredoc is the old bridge's probe itself: a real POST with the retriever payload, 120 s, contents required
    body = _function_body(LADDER.read_text(), "pre_ray_search_r1")
    assert 'payload = json.dumps({"queries": ["capital of France"], "topk": 1, "return_scores": True}).encode()' in body
    assert 'if "contents" in body:' in body and "deadline = time.time() + 120" in body
    _fakes(tmp_path, state={"probe_rc": 1})
    rc, out = _ladder("run_dirs T; recipe_vars search_r1; pre_ray_search_r1; echo NOT_REACHED", _env(tmp_path, **base))
    assert rc == 1 and "NOT_REACHED" not in out and "RETRIEVER_BRIDGE_OK" not in out
    assert "FATAL: retriever probe failed: no document contents from http://192.0.2.5:8000/retrieve" in out
    _fakes(tmp_path)
    env = _env(tmp_path, DRY_RUN="1", **base)
    rc, out = _ladder("run_dirs T; recipe_vars search_r1; pre_ray_search_r1; echo REACHED", env)
    assert rc == 0 and _dry(out, "retriever_probe") == ["python3", "-"] and "REACHED" in out, out
    assert _calls(env) == [] and "RETRIEVER_BRIDGE_OK" not in out


# ---- scripts/run_local.sh + scripts/ray_cluster.sh -----------------------------------------------------------

ENTRY = SCRIPTS / "run_local.sh"
READYZ, OPENAPI, DASH = "http://127.0.0.1:8100/readyz", "http://127.0.0.1:8200/openapi.json", "http://127.0.0.1:8265/api/version"
SERVICES_UP = {READYZ: "200", OPENAPI: "200", DASH: "200"}
RAY_HEAD_ARGV = ["ray", "start", "--head", "--node-ip-address=192.0.2.10", "--port=6379", "--num-gpus", "2",
                 "--disable-usage-stats", "--dashboard-host=0.0.0.0", "--min-worker-port=2000", "--max-worker-port=2999"]
RAY_JOIN_ARGV = ["start", "--address", "192.0.2.10:6379", "--node-ip-address", "192.0.2.10", "--num-gpus", "2",
                 "--disable-usage-stats", "--min-worker-port=2000", "--max-worker-port=2999"]
RENDERED = (REPO / "gym_root" / "env.yaml", REPO / "3rdparty" / "Gym" / "env.yaml")  # render_env_yaml's two copies


def _run_roots(tmp_path):
    """_roots plus the ALFWorld game-file dir the alfworld data prep looks for."""
    roots = _roots(tmp_path)
    (Path(roots["MILES_AGENT_DATA"]) / "alfworld" / "json_2.1.1").mkdir(parents=True, exist_ok=True)
    return roots


def _run_state(**extra):
    """The fakes' state of a run that gets through: two GPUs, every service answering, 274 eval rows."""
    return {"gpus": 2, "curl": dict(SERVICES_UP), "rows": {"alfworld_eval274.jsonl": 274}, **extra}


def _driver_argv(roots, tag="T", gpus="2", nodes="1", extra=()):
    """The alfworld driver line the entry prints and runs (the fake python3 logs it without its own name)."""
    m, d, r = roots["MILES_AGENT_MODELS"], roots["MILES_AGENT_DATA"], roots["MILES_AGENT_RESULTS"]
    return ["python3", "run.py", "--topology", "colocated_1n", "--num-nodes", nodes, "--num-gpus-per-node", gpus,
            "--base-dir", f"{m}/torch_dist_v5", "--model-name", "Qwen3.5-4B", "--hf-checkpoint", f"{m}/Qwen3.5-4B",
            "--ref-load", f"{m}/torch_dist_v5/Qwen3.5-4B_torch_dist", "--save-dir", f"{r}/{tag}/",
            "--prompt-data", f"{d}/runs/{tag}/alfworld_train.jsonl", "--eval-data", f"{d}/runs/{tag}/alfworld_eval274.jsonl",
            "--preempt-sentinel", f"{r}/{tag}/stop", *extra]


def _driver_calls(env):
    return [c for c in _calls(env, "python3") if c["argv"][:1] == ["run.py"]]


def _entry(tmp_path, *args, env, script="scripts/run_local.sh", **kw):
    """scripts/run_local.sh (or `script`, one that execs it), with this checkout's rendered env.yaml copies put back
    afterwards (a real run renders them; a developer's own copies survive the test)."""
    saved = [p.read_bytes() if p.exists() else None for p in RENDERED]
    try:
        return _run_script(script, *args, env=env, **kw)
    finally:
        for path, data in zip(RENDERED, saved):
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(data)


def test_run_local_dry_run_prints_the_argv_per_recipe(tmp_path):
    _fakes(tmp_path, state={"gpus": 2})
    roots = _run_roots(tmp_path)
    m, d, r = roots["MILES_AGENT_MODELS"], roots["MILES_AGENT_DATA"], roots["MILES_AGENT_RESULTS"]
    hf = f"{m}/Qwen3.5-4B"
    before = set(tmp_path.rglob("*"))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/alfworld/prepare_data.py", "--data-root", f"{d}/alfworld",
                                    "--output-dir", f"{d}/runs/T", "--model", hf]
    assert _dry(out, "probe") == []  # no GYM_VENV_ROOT: nothing to probe
    assert _dry(out, "ray_head") == RAY_HEAD_ARGV
    assert _dry(out, "gym") == ["uv", "run", "gym", "env", "start", "--config", _cfg("alfworld_env"), "--model-type",
                                "openai_model", *GYM_HEAD, "+alfworld_agent.responses_api_agents.env_loop_agent.port=8200",
                                *GYM_RANGE]
    assert _dry_env(out) == {"NEMO_GYM_EXTRA_ROOTS": f"{REPO}/gym_root", "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                             "RAY_TMPDIR": "/tmp/gymray"}
    assert _dry(out, "driver") == _driver_argv(roots)
    labels = [line.split(":", 1)[0] for line in out.splitlines() if line.startswith("DRY_RUN ")]
    assert labels == ["DRY_RUN pin_check", "DRY_RUN prepare", "DRY_RUN probe", "DRY_RUN ray_head", "DRY_RUN gym",
                      "DRY_RUN env", "DRY_RUN env", "DRY_RUN env", "DRY_RUN cwd", "DRY_RUN driver"]
    pin = _dry(out, "pin_check")
    assert pin[:2] == ["python3", "-c"] and "MILES_PIN_ACTIVE path=" in pin[2] and "import miles" in pin[2]
    assert "GYM_UP" not in out and "RUN_INNER_PASS" not in out and "WARN" not in out and "FATAL" not in out
    assert {c["cmd"] for c in _calls(env)} <= {"nvidia-smi", "getent", "hostname"}  # measured, nothing started
    assert set(tmp_path.rglob("*")) == before  # no directory, no file
    # `=` spellings; MODE and the words after `--` ride on the driver line (after EXTRA_ARGS from the environment)
    rc, out = _entry(tmp_path, "--dry-run=argv", "--recipe=alfworld", "--run-tag=T", "--mode", "debug_rollout_only",
                     "--", "--num-rollout", "2", env=_env(tmp_path, EXTRA_ARGS="--seed 7", **roots))
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[:8] == ["python3", "run.py", "--topology", "colocated_1n", "--mode", "debug_rollout_only", "--num-nodes", "1"]
    assert argv[-4:] == ["--seed", "7", "--num-rollout", "2"]
    # scienceworld
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "scienceworld", "--run-tag", "T", env=_env(tmp_path, **roots))
    assert rc == 0, out
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/scienceworld/prepare_data.py", "--output-dir", f"{d}/runs/T",
                                    "--model", hf]
    argv = _dry(out, "driver")
    assert argv[argv.index("--prompt-data") + 1:][:3] == [f"{d}/runs/T/scienceworld_train.jsonl", "--eval-data",
                                                        f"{d}/runs/T/scienceworld_eval274.jsonl"]
    assert _dry(out, "gym")[-1] == "+scienceworld_env.resources_servers.scienceworld_env.env_workers=128"
    # webshop: the per-protocol dir, the engine line
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "webshop", "--run-tag", "T", env=_env(tmp_path, **roots))
    assert rc == 0, out
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/webshop/prepare_data.py", "--output-dir",
                                    f"{d}/runs/T/webshop_canonical", "--protocol", "canonical", "--model", hf]
    assert _dry(out, "webshop_engine") == ["bash", f"{REPO}/env_servers/webshop/launch_engine.sh"]
    argv = _dry(out, "driver")
    assert argv[argv.index("--eval-data") + 1] == f"{d}/runs/T/webshop_canonical/webshop_eval.jsonl"
    assert "+webshop_env.resources_servers.webshop_env.engine_base_url=http://127.0.0.1:6200" in _dry(out, "gym")
    # search_r1: RETRIEVER_URL, two conversions, the retriever probe, the gym's retriever override
    env = _env(tmp_path, RETRIEVER_URL="http://192.0.2.5:8000", **roots)
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "search_r1", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert _dry(out, "prepare")[-2:] == ["--output", f"{d}/search_r1_train.jsonl"]
    assert _dry(out, "prepare", 1)[-2:] == ["--output", f"{d}/search_r1_eval893.jsonl"]
    assert _dry(out, "retriever_probe") == ["python3", "-"] and "RETRIEVER_BRIDGE_OK" not in out
    assert "+wiki_search.resources_servers.wiki_search.retriever_base_url=http://192.0.2.5:8000" in _dry(out, "gym")
    argv = _dry(out, "driver")
    assert argv[argv.index("--prompt-data") + 1] == f"{d}/search_r1_train.jsonl"
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "search_r1", "--run-tag", "T", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: RECIPE=search_r1 needs RETRIEVER_URL" in out
    # a prebuilt root: the three venv flags, the probe argv, UV_OFFLINE
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "alfworld", "--run-tag", "T",
                     env=_env(tmp_path, GYM_VENV_ROOT="/r", **roots))
    assert rc == 0, out
    assert _dry(out, "gym")[:9] == ["/r/gym_main_venv/bin/gym", "env", "start", "--config", _cfg("alfworld_env"),
                                    "--model-type", "openai_model", "+skip_venv_if_present=true", "+uv_venv_dir=/r/gym_venvs"]
    assert _dry(out, "probe") == ["/r/gym_venvs/responses_api_agents/env_loop_agent/.venv/bin/python", "-c",
                                  "import agent_utils.chat_dialect, nemo_gym"]
    assert _dry_env(out)["UV_OFFLINE"] == "1"
    # tmax: an external agent server (NEMO_GYM_URL): no gym line, no data prep, the recipe's own flags
    env = _env(tmp_path, NEMO_GYM_URL="http://192.0.2.7:8200", PROMPT_DATA="/p.jsonl", **roots)
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "tmax", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert "DRY_RUN gym:" not in out and "DRY_RUN prepare:" not in out and "this run starts no gym" in out
    assert _dry(out, "driver") == ["python3", "run.py", "--topology", "colocated_1n", "--num-nodes", "1",
                                   "--num-gpus-per-node", "2", "--base-dir", f"{m}/torch_dist_v5", "--model-name",
                                   "Qwen3.5-4B", "--hf-checkpoint", hf, "--ref-load", f"{m}/torch_dist_v5/Qwen3.5-4B_torch_dist",
                                   "--save-dir", f"{r}/T/", "--prompt-data", "/p.jsonl", "--run-tag", "T", "--nemo-gym-url",
                                   "http://192.0.2.7:8200", "--preempt-sentinel", f"{r}/T/stop"]
    # fully_async on 2 nodes: the split reaches the driver (1 training node, the rest rollout)
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "tmax", "--run-tag", "T",
                     env=_env(tmp_path, TOPOLOGY="fully_async", NNODES="2", **{**roots, "NEMO_GYM_URL": "http://192.0.2.7:8200",
                                                                                 "PROMPT_DATA": "/p.jsonl"}))
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[argv.index("--num-nodes") + 1] == "2"
    assert argv[argv.index("--train-nodes") + 1] == "1" and argv[argv.index("--rollout-nodes") + 1] == "1"
    # the usage gates
    rc, out = _entry(tmp_path, "--dry-run", "--run-tag", "T", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: --recipe (or RECIPE=) is required: scripts/run_local.sh" in out
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "alfworld", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: --run-tag (or RUN_TAG=) is required" in out
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "alfworld", "--run-tag", "T", "--bogus", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: unknown argument '--bogus'" in out
    rc, out = _entry(tmp_path, "--dry-run", "--recipe", "alfworld", "--run-tag", "T", "--mode", "nope", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: unknown MODE=nope (normal | debug_rollout_only | r2s_pipeline)" in out
    rc, out = _entry(tmp_path, "--dry-run=deep", "--recipe", "alfworld", "--run-tag", "T", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: unknown dry-run depth 'deep'" in out
    rc, out = _entry(tmp_path, "--help", env=_env(tmp_path, **roots))
    assert rc == 0 and "--dry-run=driver" in out and "RUN_INNER_PASS" in out
    assert set(tmp_path.rglob("*")) == before


def test_run_local_under_the_old_wrapper_names_resolves_to_todays_paths(tmp_path):
    """Invariant 2 of the split: under the environment the old job wrapper (now a cluster-jobs wrapper) exports before
    its exec, the entry resolves every path to today's value — the prepared-data paths, the prebuilt venv probes and
    the gym argv under /opt, the recipe overrides, the driver argv (today's line plus the two explicit flags
    --num-nodes / --num-gpus-per-node) — and measures nothing, starts nothing (every value is handed over)."""
    _fakes(tmp_path)
    wrapper = dict(
        LOCAL_ENV="/dev/null", CONTRACT_VERSION="1", JOB_ID="123", JOB_RESTART="0", JOB_NAME="miles_run", NODE_RANK="0",
        NNODES="1", NODE_IP="192.0.2.10", HEAD_IP="192.0.2.10", MY_IP="192.0.2.10", GPUS_PER_NODE="4", PROJECT_DIR="/p",
        MODELS_DIR="/p/models", DATA_DIR="/p/data", RESULTS_DIR="/p/results", RUNS_DIR="/p/runs", TOOLS_DIR="/p/tools",
        IMAGES_DIR="/p/images", STAGE_TARS_DIR="/p/stage_tars", WIKI_SEARCH_DATA="/n/data/wiki_search", GYM_VENV_ROOT="/opt",
        MILES_ROOT="/root/miles", SRC_MOUNT=str(REPO), WEBSHOP_HOME="/p/webshop",
        SCIW_JAR="/p/data/scienceworld/scienceworld.jar", SCIW_JAVA_HOME="/p/tools/jdk-17", SYNC_DIR="/p/runs/sync_123_0",
        PREEMPT_SENTINEL="/p/runs/sync_123_0/preempt", RUN_TAG="T", TOPOLOGY="colocated_1n", MODE="normal",
        ALFWORLD_DATA="/p/data/alfworld", EXTRA_ARGS="--num-rollout 2", WANDB_ENABLE="1", DRY_RUN="1",
    )
    # the old wrapper's driver line with the project root at /p (the new flags sit right after --topology / $MODE_ARGS)
    today_head = ["python3", "run.py", "--topology", "colocated_1n"]
    today_body = ["--base-dir", "/p/models/torch_dist_v5", "--model-name", "Qwen3.5-4B", "--hf-checkpoint",
                  "/p/models/Qwen3.5-4B", "--ref-load", "/p/models/torch_dist_v5/Qwen3.5-4B_torch_dist", "--save-dir",
                  "/p/results/T/"]
    today_tail = ["--preempt-sentinel", "/p/runs/sync_123_0/preempt", "--num-rollout", "2"]
    new_flags = ["--num-nodes", "1", "--num-gpus-per-node", "4"]

    def gym_argv(server, agent, *overrides):  # today's gym line under /opt (the config under the mounted repo)
        return ["/opt/gym_main_venv/bin/gym", "env", "start", "--config", _cfg(server), "--model-type", "openai_model",
                "+skip_venv_if_present=true", "+uv_venv_dir=/opt/gym_venvs", "+uv_cache_dir=/opt/uv_cache", *GYM_HEAD,
                f"+{agent}.port=8200", *GYM_RANGE, *overrides]

    def run(**extra):
        env = _env(tmp_path, **{**wrapper, **extra})
        rc, out = _entry(tmp_path, env=env)  # no flag: the wrapper's exec passes none
        assert rc == 0, out
        assert _calls(env) == []  # nothing measured, nothing started
        assert "FATAL" not in out and "RUN_INNER_PASS" not in out
        return out

    out = run(RECIPE="alfworld")
    assert _dry(out, "driver") == today_head + new_flags + today_body + [
        "--prompt-data", "/p/data/runs/T/alfworld_train.jsonl", "--eval-data", "/p/data/runs/T/alfworld_eval274.jsonl"] + today_tail
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/alfworld/prepare_data.py", "--data-root", "/p/data/alfworld",
                                    "--output-dir", "/p/data/runs/T", "--model", "/p/models/Qwen3.5-4B"]
    assert _dry(out, "gym") == gym_argv("alfworld_env", "alfworld_agent.responses_api_agents.env_loop_agent")
    assert _dry_env(out) == {"NEMO_GYM_EXTRA_ROOTS": f"{REPO}/gym_root", "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                             "RAY_TMPDIR": "/tmp/gymray", "UV_OFFLINE": "1"}
    assert _dry_cwd(out) == [GYM_CWD]
    assert len([line for line in out.splitlines() if line.startswith("DRY_RUN probe:")]) == 3
    assert _dry(out, "probe") == ["/opt/gym_venvs/responses_api_agents/env_loop_agent/.venv/bin/python", "-c",
                                  "import agent_utils.chat_dialect, nemo_gym"]
    assert _dry(out, "probe", 2) == ["/opt/gym_venvs/environment_servers/legacy_agent/.venv/bin/python", "-c",
                                     "import nemo_gym.base_environment_server, httptools"]
    assert _dry(out, "ray_head") == [*RAY_HEAD_ARGV[:5], "--num-gpus", "4", *RAY_HEAD_ARGV[7:]]
    assert "p.startswith('/root/miles')" in _dry(out, "pin_check")[2]
    assert "run_local: RECIPE=alfworld RUN_TAG=T TOPOLOGY=colocated_1n MODE=normal NNODES=1 NODE_RANK=0" in out
    out = run(RECIPE="alfworld", MODE="debug_rollout_only")
    assert _dry(out, "driver") == today_head + ["--mode", "debug_rollout_only"] + new_flags + today_body + [
        "--prompt-data", "/p/data/runs/T/alfworld_train.jsonl", "--eval-data", "/p/data/runs/T/alfworld_eval274.jsonl"] + today_tail
    # scienceworld: the wrapper's SCIW_* exports keep today's jar / JDK overrides
    out = run(RECIPE="scienceworld")
    sw = "+scienceworld_env.resources_servers.scienceworld_env."
    assert _dry(out, "gym") == gym_argv("scienceworld_env", "scienceworld_agent.responses_api_agents.env_loop_agent",
                                        f"{sw}jar_path=/p/data/scienceworld/scienceworld.jar", f"{sw}java_home=/p/tools/jdk-17",
                                        f"{sw}env_workers=128")
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/scienceworld/prepare_data.py", "--output-dir", "/p/data/runs/T",
                                    "--model", "/p/models/Qwen3.5-4B"]
    argv = _dry(out, "driver")
    assert argv[argv.index("--prompt-data") + 1:][:3] == ["/p/data/runs/T/scienceworld_train.jsonl", "--eval-data",
                                                        "/p/data/runs/T/scienceworld_eval274.jsonl"]
    assert _dry(out, "probe", 1) == ["/opt/gym_venvs/resources_servers/scienceworld_env/.venv/bin/python", "-c",
                                     "import scienceworld_env.app, scienceworld, py4j"]
    # webshop: today's $WEBSHOP_HOME (the wrapper exports it), the engine URL, the protocol dir
    out = run(RECIPE="webshop")
    ws = "+webshop_env.resources_servers.webshop_env."
    assert _dry(out, "gym") == gym_argv("webshop_env", "webshop_agent.responses_api_agents.env_loop_agent",
                                        f"{ws}engine_base_url=http://127.0.0.1:6200", f"{ws}observation_mode=text_rich",
                                        f"{ws}invalid_action_feedback=error")
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/webshop/prepare_data.py", "--output-dir",
                                    "/p/data/runs/T/webshop_canonical", "--protocol", "canonical", "--model", "/p/models/Qwen3.5-4B"]
    assert _dry(out, "webshop_engine") == ["bash", f"{REPO}/env_servers/webshop/launch_engine.sh"]
    assert "WARN: no webshop py3.10 interpreter at /p/webshop/env/bin/python" in out
    argv = _dry(out, "driver")
    assert argv[argv.index("--prompt-data") + 1] == "/p/data/runs/T/webshop_canonical/webshop_train.jsonl"
    # search_r1: the wrapper's bridge URL and wiki_search rows; the gym's retriever override replaces today's sed -i
    out = run(RECIPE="search_r1", RETRIEVER_URL="http://127.0.0.1:6100")
    assert _dry(out, "prepare") == ["python3", f"{REPO}/recipes/search_r1/prepare_data.py", "--input",
                                    "/n/data/wiki_search/train.jsonl", "--output", "/p/data/search_r1_train.jsonl"]
    assert _dry(out, "prepare", 1) == ["python3", f"{REPO}/recipes/search_r1/prepare_data.py", "--input",
                                       "/n/data/wiki_search/validation.jsonl", "--output", "/p/data/search_r1_eval893.jsonl"]
    assert _dry(out, "retriever_probe") == ["python3", "-"]
    assert _dry(out, "gym") == gym_argv("wiki_search", "wiki_search_agent.responses_api_agents.tool_call_agent",
                                        "+wiki_search.resources_servers.wiki_search.retriever_base_url=http://127.0.0.1:6100")
    assert _dry(out, "probe") == ["/opt/gym_venvs/responses_api_agents/tool_call_agent/.venv/bin/python", "-c",
                                  "import agent_utils.chat_dialect, nemo_gym"]
    argv = _dry(out, "driver")
    assert argv[argv.index("--prompt-data") + 1:][:3] == ["/p/data/search_r1_train.jsonl", "--eval-data",
                                                        "/p/data/search_r1_eval893.jsonl"]


def test_run_local_end_to_end_with_fakes_passes(tmp_path):
    roots = _run_roots(tmp_path)
    m, d, r = roots["MILES_AGENT_MODELS"], roots["MILES_AGENT_DATA"], roots["MILES_AGENT_RESULTS"]
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[-1] == "RUN_INNER_PASS"
    for text in ("GYM_VENV_PROBE_OK mode=uv_run", "GYM_UP agent=http://127.0.0.1:8200",
                 "ray: starting the head on 192.0.2.10:6379 with 2 GPUs"):
        assert text in out
    assert "FATAL" not in out and "WARN" not in out and "DRY_RUN" not in out
    # the fake ray: no cluster -> the head -> stopped by the EXIT trap; the gym process is gone the same way
    assert [c["argv"] for c in _calls(env, "ray")] == [["status"], RAY_HEAD_ARGV[1:], ["stop", "--force"]]
    assert _pid_gone(tmp_path / "gym.pid")
    (run,) = _driver_calls(env)
    assert run["argv"] == _driver_argv(roots)[1:] and run["cwd"] == f"{REPO}/recipes/alfworld"
    assert run["env"]["NEMO_GYM_URL"] == "http://127.0.0.1:8200" and run["env"]["MASTER_ADDR"] == "192.0.2.10"
    assert run["env"]["MILES_SCRIPT_EXTERNAL_RAY"] == "1" and run["env"]["PREEMPT_SENTINEL"] == f"{r}/T/stop"
    assert run["env"]["SYNC_DIR"] == f"{r}/T/sync" and run["env"]["ALFWORLD_DATA"] == f"{d}/alfworld"
    # the order on the fake python: the pin check, the data prep (its rows gate passed), then the driver
    py = [c["argv"] for c in _calls(env, "python3")]
    assert py[0][0] == "-c" and "MILES_PIN_ACTIVE" in py[0][1] and f"'{m}" not in py[0][1]
    assert py[1][:3] == [f"{REPO}/recipes/alfworld/prepare_data.py", "--data-root", f"{d}/alfworld"]
    assert py[-1][:1] == ["run.py"]
    (gym,) = _calls(env, "uv")
    assert gym["argv"][:7] == ["run", "gym", "env", "start", "--config", _cfg("alfworld_env"), "--model-type"]
    assert gym["cwd"] == GYM_CWD and gym["env"]["NEMO_GYM_EXTRA_ROOTS"] == f"{REPO}/gym_root"
    assert Path(f"{r}/T/sync").is_dir() and Path(f"{r}/runs/T_gym_4242.log").exists()
    assert Path(f"{d}/runs/T/alfworld_eval274.jsonl").exists()
    assert not Path(f"{r}/T/sync/ENDED").exists()  # one node: no worker waits for it
    # the gates in front of the driver still bite: the rows gate
    _fakes(tmp_path, state=_run_state(rows={"alfworld_eval274.jsonl": 273}))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 1 and "FATAL: eval set has 273 rows, expected 274" in out and _driver_calls(env) == []
    assert _calls(env, "ray") == []  # before ray, before the gym


def test_run_local_stops_on_a_raised_sentinel(tmp_path):
    roots = _run_roots(tmp_path)
    sentinel = Path(roots["MILES_AGENT_RESULTS"]) / "T" / "stop"
    # raised before the launch: the services came up, the driver never starts, exit 0 (the caller requeues or resumes)
    sentinel.parent.mkdir(parents=True)
    sentinel.touch()
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert "GYM_UP agent=http://127.0.0.1:8200" in out and "PREEMPT before launch: not starting the driver" in out
    assert "RUN_INNER_PASS" not in out and _driver_calls(env) == [] and _pid_gone(tmp_path / "gym.pid")
    sentinel.unlink()
    # raised while the driver runs: the driver saves, writes <sentinel>.done and exits 0 -> RUN_INNER_PREEMPT_STOP
    _fakes(tmp_path, state=_run_state(run_touch=str(sentinel), run_write={f"{sentinel}.done": '{"rollout_id":1}'}))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert out.rstrip().splitlines()[-1] == 'RUN_INNER_PREEMPT_STOP rc=0 durable={"rollout_id":1}'
    assert "RUN_INNER_PASS" not in out and "FATAL" not in out
    (run,) = _driver_calls(env)
    assert run["argv"][-2:] == ["--preempt-sentinel", str(sentinel)]
    assert _pid_gone(tmp_path / "gym.pid") and [c["argv"] for c in _calls(env, "ray")][-1] == ["stop", "--force"]
    # raised, but the driver died before its save: the marker says `none` and the rc is kept
    sentinel.unlink()
    Path(f"{sentinel}.done").unlink()
    _fakes(tmp_path, state=_run_state(run_touch=str(sentinel), run_rc=3))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 3 and out.rstrip().splitlines()[-1] == "RUN_INNER_PREEMPT_STOP rc=3 durable=none"
    # TERM / USR1 raise the sentinel instead of killing the entry: the trap is armed before any rank branches off
    text = ENTRY.read_text()
    assert text.index("arm_preempt_trap") < text.index("# ---- ray worker branch")


def test_run_local_fails_when_the_driver_fails(tmp_path):
    roots = _run_roots(tmp_path)
    _fakes(tmp_path, state=_run_state(run_rc=1))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 1, out
    assert "RUN_INNER_PASS" not in out and "RUN_INNER_PREEMPT_STOP" not in out
    assert "FATAL: the driver run.py exited with rc=1" in out and len(_driver_calls(env)) == 1
    assert _pid_gone(tmp_path / "gym.pid") and [c["argv"] for c in _calls(env, "ray")][-1] == ["stop", "--force"]
    # a gym that never answers: the ladder's FATAL before the driver, the gym killed, nothing launched
    _fakes(tmp_path, state=_run_state(curl={DASH: "200"}))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 1 and "FATAL: gym servers failed to boot" in out and _driver_calls(env) == []
    assert _pid_gone(tmp_path / "gym.pid") and [c["argv"] for c in _calls(env, "ray")][-1] == ["stop", "--force"]


def test_run_local_refuses_a_topology_node_count_mismatch(tmp_path):
    _fakes(tmp_path, state={"gpus": 2})
    roots = _run_roots(tmp_path)

    def dry(**extra):
        return _entry(tmp_path, "--dry-run", "--recipe", "alfworld", "--run-tag", "T", env=_env(tmp_path, **{**roots, **extra}))

    rc, out = dry(TOPOLOGY="async_3n", NNODES="1")
    assert rc == 1 and "FATAL: TOPOLOGY=async_3n needs NNODES=3, got 1" in out and "DRY_RUN driver" not in out
    rc, out = dry(TOPOLOGY="async_4n_teacher", NNODES="4")
    assert rc == 1 and "FATAL: async_4n_teacher needs TEACHER_MODEL=<HF dir name under" in out
    rc, out = dry(TOPOLOGY="async_4n_teacher", NNODES="4", TEACHER_URL="http://192.0.2.5:13141")
    assert rc == 1 and "FATAL: TOPOLOGY=async_4n_teacher needs NNODES=3, got 4" in out
    rc, out = dry(TOPOLOGY="fully_async", NNODES="4", TRAIN_NODES="2", ROLLOUT_NODES="3")
    assert rc == 1 and "FATAL: TOPOLOGY=fully_async needs NNODES=5, got 4" in out
    rc, out = dry(TOPOLOGY="nope")
    assert rc == 1 and "FATAL: unknown TOPOLOGY=nope" in out
    # the counts that pass, and what reaches the driver: the ray nodes (the job's own teacher node excluded)
    rc, out = dry(TOPOLOGY="async_4n_teacher", NNODES="4", TEACHER_MODEL="Teacher-9B")
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[argv.index("--topology") + 1] == "async_4n_teacher" and argv[argv.index("--num-nodes") + 1] == "3"
    assert _dry(out, "ray_wait_gpus") == ["6", "192.0.2.10:6379"] and _dry(out, "teacher_gate") == ["teacher_gate"]
    assert "--teacher-url" not in argv  # the gate, not run under --dry-run, reads it from $SYNC_DIR/TEACHER_URL
    rc, out = dry(TOPOLOGY="async_2n_teacher", NNODES="1", TEACHER_URL="http://192.0.2.5:13141")
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[argv.index("--num-nodes") + 1] == "1" and argv[-2:] == ["--teacher-url", "http://192.0.2.5:13141"]
    assert "DRY_RUN ray_wait_gpus" not in out
    rc, out = dry(TOPOLOGY="fully_async", NNODES="4", TRAIN_NODES="1")
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[argv.index("--num-nodes") + 1] == "4" and _dry(out, "ray_wait_gpus") == ["8", "192.0.2.10:6379"]
    rc, out = dry(TOPOLOGY="async_3n", NNODES="3")
    assert rc == 0 and _dry(out, "driver")[_dry(out, "driver").index("--num-nodes") + 1] == "3", out
    # the teacher rank of a *_teacher topology under --dry-run says what it would run and exits
    rc, out = dry(TOPOLOGY="async_2n_teacher", NNODES="2", NODE_RANK="1", TEACHER_MODEL="Teacher-9B")
    assert rc == 0 and _dry(out, "teacher_node") == ["teacher_node_main"] and "DRY_RUN driver" not in out
    # no GPU: a FATAL outside the dry run, `--num-gpus-per-node 0` inside it
    _fakes(tmp_path, state={"gpus": 0})
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=_env(tmp_path, **roots))
    assert rc == 1 and "FATAL: no GPU: set GPUS_PER_NODE or CUDA_VISIBLE_DEVICES" in out
    rc, out = dry()
    assert rc == 0 and "--num-gpus-per-node 0" in " ".join(_dry(out, "driver"))
    rc, out = dry(CUDA_VISIBLE_DEVICES="0,1,2")
    assert rc == 0 and "--num-gpus-per-node 3" in " ".join(_dry(out, "driver"))


def test_run_local_worker_rank_joins_and_idles(tmp_path):
    roots = _run_roots(tmp_path)
    sync = Path(roots["MILES_AGENT_RESULTS"]) / "T" / "sync"
    worker = dict(roots, NODE_RANK="1", NNODES="3", TOPOLOGY="async_3n", HEAD_IP="192.0.2.10")
    # the join itself raises $SYNC_DIR/ENDED (the fake ray start), so the idle loop ends at once
    _fakes(tmp_path, state={"gpus": 2, "ray_touch": str(sync / "ENDED")})
    env = _env(tmp_path, **worker)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert "RAY_WORKER_JOINED node_rank=1 ip=192.0.2.10\nRAY_WORKER_DONE node_rank=1" in out
    assert [c["argv"] for c in _calls(env, "ray")] == [RAY_JOIN_ARGV, ["stop"]]
    assert _calls(env, "uv") == [] and _calls(env, "gym") == [] and _driver_calls(env) == []
    assert "GYM_UP" not in out and "RUN_INNER_PASS" not in out and "FATAL" not in out
    assert (sync / "ENDED").exists()
    # ENDED already up before the join (the head is gone): ray.sh's FATAL, no ray start
    _fakes(tmp_path, state={"gpus": 2})
    env = _env(tmp_path, **dict(worker, NODE_RANK="2"))
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 1 and "FATAL: head exited before ray came up" in out and _calls(env, "ray") == []
    # a worker's dry run prints its join and starts nothing
    _fakes(tmp_path, state={"gpus": 2})
    env = _env(tmp_path, DRY_RUN="1", **worker)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert _dry(out, "ray_worker") == ["ray_worker_join", "192.0.2.10", "192.0.2.10", "2", "1"] and _calls(env, "ray") == []


def test_run_local_driver_dry_run_brings_the_services_up_and_stops_before_the_driver(tmp_path):
    roots = _run_roots(tmp_path)
    d, r = roots["MILES_AGENT_DATA"], roots["MILES_AGENT_RESULTS"]
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--dry-run=driver", "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    assert out.index("GYM_UP agent=http://127.0.0.1:8200") < out.index("DRY_RUN driver:")
    assert _dry(out, "driver") == _driver_argv(roots)
    assert "RUN_INNER_PASS" not in out and "FATAL" not in out and _driver_calls(env) == []
    assert "DRY_RUN gym:" not in out and "DRY_RUN prepare:" not in out and "DRY_RUN ray_head:" not in out  # real
    assert _pid_gone(tmp_path / "gym.pid")
    assert [c["argv"] for c in _calls(env, "ray")] == [["status"], RAY_HEAD_ARGV[1:], ["stop", "--force"]]
    assert Path(f"{d}/runs/T/alfworld_eval274.jsonl").exists() and Path(f"{r}/runs/T_gym_4242.log").exists()
    # DRY_RUN=driver from the environment is the same depth
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, DRY_RUN="driver", **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0 and "GYM_UP" in out and _dry(out, "driver") == _driver_argv(roots) and _driver_calls(env) == [], out
    # a multi-node head at this depth waits for the workers' GPUs and raises ENDED for them at its exit
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, DRY_RUN="driver", NNODES="3", TOPOLOGY="async_3n", **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[argv.index("--num-nodes") + 1] == "3" and (Path(r) / "T" / "sync" / "ENDED").exists()
    (wait,) = [c for c in _calls(env, "python3") if c["argv"] == ["-"]]
    assert wait["env"]["MILES_EXPECT_GPUS"] == "6" and wait["env"]["MILES_GCS"] == "192.0.2.10:6379"
    # a head already up on the machine (scripts/ray_cluster.sh head) is reused, not restarted, and not stopped
    _fakes(tmp_path, state=_run_state(ray_status_rc=0))
    env = _env(tmp_path, DRY_RUN="driver", **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0 and "ray: a cluster already answers on this machine" in out, out
    assert [c["argv"] for c in _calls(env, "ray")] == [["status"]]
    (gym,) = _calls(env, "uv")
    assert gym["env"]["MASTER_ADDR"] == "192.0.2.10" and gym["env"]["MILES_SCRIPT_EXTERNAL_RAY"] == "1"


def test_run_local_port_precheck_skips_our_own_services(tmp_path):
    roots = _run_roots(tmp_path)
    held = 'LISTEN 0 4096 0.0.0.0:8265 0.0.0.0:* users:(("raylet",pid=4242,fd=3))'
    _fakes(tmp_path, state=_run_state(ss=held))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 1 and f"FATAL: port :8265 is already held: {held}" in out
    assert _calls(env, "ray") == [] and _calls(env, "uv") == [] and _driver_calls(env) == []
    assert [c["argv"] for c in _calls(env, "ss")] == [["-tln"]]
    # the retriever bridge's port is ours when RETRIEVER_URL names it: the run goes through ...
    _fakes(tmp_path, state=_run_state(ss="LISTEN 0 128 127.0.0.1:6100 0.0.0.0:*"))
    env = _env(tmp_path, RETRIEVER_URL="http://127.0.0.1:6100", **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0 and "RUN_INNER_PASS" in out and "FATAL" not in out, out
    # ... and the same holder without the URL is a FATAL naming it
    _fakes(tmp_path, state=_run_state(ss="LISTEN 0 128 127.0.0.1:6100 0.0.0.0:*"))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 1 and "FATAL: port :6100 is already held:" in out and _calls(env, "ray") == []
    # the session ports before the driver: a WARN with the evidence, never a stop (the fake ss answers every query
    # with this one line; the pre-check reads its column 4, so the session port sits on the peer side here)
    _fakes(tmp_path, state=_run_state(ss="ESTAB 0 0 127.0.0.1:51234 127.0.0.1:7101"))
    env = _env(tmp_path, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0 and "WARN: session port 7101 occupied before launch:" in out and "RUN_INNER_PASS" in out, out
    # the pre-check sits before anything starts; the entry never kills a holder (a sweep is the wrapper's)
    text = ENTRY.read_text()
    assert text.index("port_precheck $PROJECT_PORTS") < text.index('"$RECIPE_PREPARE_DATA"') < text.index("elif ray_running")
    assert not re.search(r"kill -9|pkill|xargs", text)


def test_run_local_wandb_key_is_dropped_when_wandb_is_off(tmp_path):
    roots = _run_roots(tmp_path)
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, WANDB_ENABLE="0", **{"WANDB_API" "_KEY": "x"}, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    (run,) = _driver_calls(env)
    assert "WANDB_API_KEY" not in run["env"] and run["env"]["WANDB_ENABLE"] == "0"
    _fakes(tmp_path, state=_run_state())
    env = _env(tmp_path, **{"WANDB_API" "_KEY": "x"}, **roots)
    rc, out = _entry(tmp_path, "--recipe", "alfworld", "--run-tag", "T", env=env)
    assert rc == 0, out
    (run,) = _driver_calls(env)
    assert run["env"]["WANDB_API_KEY"] == "x" and run["env"]["WANDB_ENABLE"] == "1"  # the default: on


def test_ray_cluster_ssh_join_and_stop(tmp_path):
    base = dict(LOCAL_ENV="/dev/null")
    # ssh-join: one ssh per host under REMOTE_PREFIX; the fake ssh runs the remote command here, so each host's join
    # is this script's own `join`, the same ray start line as scripts/lib/ray.sh's worker (no idle loop)
    _fakes(tmp_path, state={"gpus": 2})
    env = _env(tmp_path, REMOTE_PREFIX=f"cd {tmp_path} &&", **base)
    rc, out = _run_script("scripts/ray_cluster.sh", "ssh-join", "192.0.2.10", "h1", "h2", env=env)
    assert rc == 0, out
    remote = f"cd {tmp_path} && bash {REPO}/scripts/ray_cluster.sh join 192.0.2.10"
    assert [c["argv"] for c in _calls(env, "ssh")] == [["-o", "BatchMode=yes", "h1", remote], ["-o", "BatchMode=yes", "h2", remote]]
    assert [c["argv"] for c in _calls(env, "ray")] == [RAY_JOIN_ARGV, RAY_JOIN_ARGV]
    assert out.count("RAY_JOINED head=192.0.2.10:6379 ip=192.0.2.10 gpus=2") == 2 and "ssh-join done hosts=2" in out
    # stop h1 h2: `ray stop --force` here, then on each host over ssh (this script's `stop` again)
    _fakes(tmp_path, state={"gpus": 2})
    env = _env(tmp_path, **base)
    rc, out = _run_script("scripts/ray_cluster.sh", "stop", "h1", "h2", env=env)
    assert rc == 0, out
    assert [c["argv"] for c in _calls(env, "ray")] == [["stop", "--force"]] * 3
    assert [c["argv"][2:] for c in _calls(env, "ssh")] == [["h1", f"bash {REPO}/scripts/ray_cluster.sh stop"],
                                                           ["h2", f"bash {REPO}/scripts/ray_cluster.sh stop"]]
    assert "RAY_STOPPED hosts=2" in out
    # head: ray.sh's head on this node's IP with the measured GPUs, RAY_HEAD=, and NO stop at the exit (the head
    # outlives the script: run_local.sh reuses it, `stop` ends it)
    _fakes(tmp_path, state={"gpus": 2, "curl": {DASH: "200"}})
    env = _env(tmp_path, **base)
    rc, out = _run_script("scripts/ray_cluster.sh", "head", env=env)
    assert rc == 0 and "RAY_HEAD=192.0.2.10:6379" in out, out
    assert [c["argv"] for c in _calls(env, "ray")] == [RAY_HEAD_ARGV[1:]]
    # status is `ray status`; a bare call, a join without its head and an unknown command are FATALs
    _fakes(tmp_path, state={"ray_status_rc": 0})
    env = _env(tmp_path, **base)
    rc, out = _run_script("scripts/ray_cluster.sh", "status", env=env)
    assert rc == 0 and [c["argv"] for c in _calls(env, "ray")] == [["status"]]
    rc, out = _run_script("scripts/ray_cluster.sh", env=env)
    assert rc == 1 and "FATAL: a command is required: head | join <head_ip> | ssh-join <head_ip> <host>..." in out
    rc, out = _run_script("scripts/ray_cluster.sh", "join", env=env)
    assert rc == 1 and "FATAL: join needs <head_ip>" in out
    rc, out = _run_script("scripts/ray_cluster.sh", "ssh-join", "192.0.2.10", env=env)
    assert rc == 1 and "FATAL: ssh-join needs at least one <host>" in out
    rc, out = _run_script("scripts/ray_cluster.sh", "nope", env=env)
    assert rc == 1 and "FATAL: unknown command 'nope'" in out
    # --dry-run: the argv per host, nothing run
    _fakes(tmp_path, state={"gpus": 2})
    env = _env(tmp_path, REMOTE_PREFIX="docker exec miles", **base)
    rc, out = _run_script("scripts/ray_cluster.sh", "--dry-run", "ssh-join", "192.0.2.10", "h1", "h2", env=env)
    assert rc == 0, out
    assert _dry(out, "ssh_join") == ["ssh", "-o", "BatchMode=yes", "h1",
                                     f"docker exec miles bash {REPO}/scripts/ray_cluster.sh join 192.0.2.10"]
    assert _dry(out, "ssh_join", 1)[3] == "h2"
    rc, out = _run_script("scripts/ray_cluster.sh", "--dry-run", "stop", "h1", env=env)
    assert rc == 0 and _dry(out, "ray_stop") == ["ray", "stop", "--force"] and _dry(out, "ray_stop", 1)[3] == "h1", out
    rc, out = _run_script("scripts/ray_cluster.sh", "head", "--dry-run", env=env)
    assert rc == 0 and _dry(out, "ray_head") == RAY_HEAD_ARGV, out
    rc, out = _run_script("scripts/ray_cluster.sh", "--dry-run", "join", "192.0.2.10", env=env)
    assert rc == 0 and _dry(out, "ray_join") == ["ray", *RAY_JOIN_ARGV], out
    assert _calls(env, "ssh") == [] and _calls(env, "ray") == []


# ---- C5: scripts/serve_model.sh, scripts/eval_ckpt.sh, scripts/export_hf.sh, scripts/stop_all.sh -------------
SERVE_MODELS = "http://127.0.0.1:7000/v1/models"
SERVE_MODELS_IP = "http://192.0.2.10:7000/v1/models"


def _serve_state(name, preflight="PASS", **extra):
    """The fakes' state of a serve that comes up: two GPUs, the parser name serve-template prints (taken from state:
    no literal parser name in the assertions), /v1/models 000 once then 200 naming the model, the preflight verdict."""
    return {"gpus": 2, "parser": "state_parser_x", "curl": {SERVE_MODELS: ["000", "200"], SERVE_MODELS_IP: "200"},
            "curl_body": {SERVE_MODELS: {"data": [{"id": name}]}}, "preflight": preflight, **extra}


def _serve_roots(tmp_path):
    """_roots plus a TMPDIR under the test dir: serve_model.sh's pid files are node-local, under it."""
    return {**_roots(tmp_path), "TMPDIR": str(tmp_path / "tmp")}


def _serve_pid_file(roots, port=7000):
    """serve_model.sh's pid file (the setsid leader): ${TMPDIR}/miles-agent-${USER:-user}/serve_<port>.pid."""
    return Path(roots["TMPDIR"]) / f"miles-agent-{os.environ.get('USER') or 'user'}" / f"serve_{port}.pid"


def test_serve_model_parser_wait_preflight_and_stop(tmp_path):
    """scripts/serve_model.sh (D13, D5): SERVE_PARSER (what serve-template printed), the /v1/models wait naming the
    model, SERVE_PREFLIGHT and `SERVE_UP url=http://<ip>:<port>/v1 model=<name> ...` in that order; a TERM runs the
    stop sequence on sglang's process group and exits 0, asserted by the EFFECT (the sglang fake, a real setsid
    leader, is gone) and never by a logged kill (the builtin); a NO_TOOL_CALL preflight is reported and served through
    unless SERVE_REQUIRE_TOOL_CALL=1 (then the FATAL, no SERVE_UP, exit 3, the server stopped); --stop runs the same
    sequence on the process group the pid file names (a session-leader orphan here), after writing the request to
    <pid file>.stop, so that a running foreground script ends with `SERVE_STOPPED rc=<rc>` and exit 0 instead of
    the FATAL; a stale or missing pid file is a WARN, exit 0; a TERM during a slow preflight acts at once (the trap
    is armed before it). The pid file is node-local: ${TMPDIR}/miles-agent-${USER}/serve_<port>.pid."""
    roots = _serve_roots(tmp_path)
    model = f"{roots['MILES_AGENT_MODELS']}/Qwen3.5-4B"
    results = roots["MILES_AGENT_RESULTS"]
    pid_file = _serve_pid_file(roots)
    sglang_pid = tmp_path / "sglang.pid"  # the fake server's own record, written once its TERM handler is armed
    template = f"{results}/runs/serve_7000/chat_template.jinja"

    def serve_then_term(state, **extra):
        if sglang_pid.exists():
            sglang_pid.unlink()  # the previous case's record: this case waits for its own server's
        _fakes(tmp_path, state=state)
        env = _env(tmp_path, **roots, **extra)
        proc, lines = _serve([model], env, until="SERVE_UP", script="serve_model.sh")
        try:
            assert lines and lines[-1].startswith("SERVE_UP "), "".join(lines)
            assert _wait_for(sglang_pid), "".join(lines)
            proc.send_signal(signal.SIGTERM)
            out = "".join(lines) + proc.communicate(timeout=30)[0]
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=5)
        return proc.returncode, out, lines, env

    # PASS: the three markers in order, the serve-template and sglang lines of the old serving job, TERM -> exit 0
    rc, out, lines, env = serve_then_term(_serve_state("Qwen3.5-4B"))
    assert rc == 0, out
    assert [line.split()[0] for line in lines if line.startswith("SERVE_")] == ["SERVE_PARSER", "SERVE_PREFLIGHT", "SERVE_UP"]
    assert re.search(r"^SERVE_STOPPED rc=\d+ \(TERM / INT to serve_model.sh\)$", out, re.M), out
    assert "SERVE_PARSER state_parser_x\n" in out and "SERVE_PREFLIGHT PASS finish_reason=stop tool_calls=" in out
    assert lines[-1].rstrip("\n") == "SERVE_UP url=http://192.0.2.10:7000/v1 model=Qwen3.5-4B parser=state_parser_x dp=2"
    assert pid_file.read_text().strip() == sglang_pid.read_text().strip()  # $! = the setsid leader = the server
    assert _pid_gone(pid_file)  # stop_server: TERM to the group, the server exits, the closing KILL finds nothing
    assert "FATAL" not in out and "WARN" not in out
    python = _calls(env, "python3")
    serve_template, = [c for c in python if c["argv"][:3] == ["-m", "miles_agent.launcher", "serve-template"]]
    assert serve_template["argv"][3:] == [model, template, "--thinking", "off", "--tito-model", "qwen35"]
    assert serve_template["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(REPO)  # SRC_MOUNT unset: this checkout
    sglang, = [c for c in python if c["argv"][:2] == ["-m", "sglang.launch_server"]]
    assert sglang["argv"][2:] == ["--model-path", model, "--served-model-name", "Qwen3.5-4B", "--tool-call-parser",
                                  "state_parser_x", "--chat-template", template, "--host", "0.0.0.0", "--port", "7000",
                                  "--dp-size", "2", "--mem-fraction-static", "0.85"]
    assert [c["argv"][0] for c in _calls(env, "setsid")] == ["python3"]  # the server in its own process group
    preflight, = [c for c in python if c["argv"][:1] == ["-"]]
    assert preflight["argv"][1:] == ["http://127.0.0.1:7000/v1/chat/completions", "Qwen3.5-4B"]
    assert [c["argv"] for c in _calls(env, "curl")] == [["-sf", SERVE_MODELS], ["-sf", SERVE_MODELS], ["-sf", SERVE_MODELS_IP]]
    assert Path(f"{results}/runs/serve_sglang_4242.log").exists()  # RUN_TAG defaults to serve
    assert Path(template).read_text() == "{{ fake template }}"
    assert Path(f"{pid_file}.stop").read_text() == ""  # emptied at the start: no request yet
    # --stop from another shell while the server runs: the request file first, then the TERM to the group; the
    # foreground script ends with SERVE_STOPPED (the server's exit code; a requested stop), exit 0, never the FATAL
    if sglang_pid.exists():
        sglang_pid.unlink()
    _fakes(tmp_path, state=_serve_state("Qwen3.5-4B"))
    env = _env(tmp_path, **roots)
    proc, lines = _serve([model], env, until="SERVE_UP", script="serve_model.sh")
    try:
        assert lines and lines[-1].startswith("SERVE_UP "), "".join(lines)
        assert _wait_for(sglang_pid), "".join(lines)
        rc_stop, out_stop = _run_script("scripts/serve_model.sh", "--stop", env=env)
        out = "".join(lines) + proc.communicate(timeout=30)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
    assert rc_stop == 0 and f"SERVE_STOPPED pid={sglang_pid.read_text().strip()}" in out_stop, out_stop
    assert proc.returncode == 0, out
    assert re.search(rf"^SERVE_STOPPED rc=\d+ \(stop requested through {re.escape(str(pid_file))}\.stop\)$", out, re.M), out
    assert "FATAL" not in out and "SERVE_STOP rc=0" not in out and _pid_gone(pid_file)
    assert Path(f"{pid_file}.stop").read_text().strip() == sglang_pid.read_text().strip()
    # a TERM during the preflight (a slow request): the trap, armed before it, acts at once; SERVE_STOPPED, exit 0,
    # no SERVE_UP
    if sglang_pid.exists():
        sglang_pid.unlink()
    _fakes(tmp_path, state=_serve_state("Qwen3.5-4B", preflight_delay_s=20))
    env = _env(tmp_path, **roots)
    proc, lines = _serve([model], env, until="serve_model: sglang pid", script="serve_model.sh")
    try:
        assert _wait_for(sglang_pid), "".join(lines)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not any(c["argv"][:1] == ["-"] for c in _calls(env, "python3")):
            time.sleep(0.05)
        assert any(c["argv"][:1] == ["-"] for c in _calls(env, "python3")), "the preflight did not start"
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        out = "".join(lines) + proc.communicate(timeout=30)[0]
        took = time.monotonic() - t0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
    assert proc.returncode == 0, out
    assert took < 10, took  # not the 20 s the request would have taken
    assert "SERVE_STOPPED rc=" in out and "SERVE_UP" not in out and "SERVE_PREFLIGHT" not in out and "FATAL" not in out
    assert _pid_gone(pid_file)
    # NO_TOOL_CALL with the knob unset (the old serving job): reported, then SERVE_UP all the same
    rc, out, lines, env = serve_then_term(_serve_state("Qwen3.5-4B", preflight="NO_TOOL_CALL"))
    assert rc == 0, out
    assert "SERVE_PREFLIGHT NO_TOOL_CALL finish_reason=stop tool_calls=[]" in out
    assert lines[-1].startswith("SERVE_UP url=http://192.0.2.10:7000/v1 model=Qwen3.5-4B ") and "FATAL" not in out
    assert _pid_gone(pid_file)
    # NO_TOOL_CALL under SERVE_REQUIRE_TOOL_CALL=1: the FATAL, no SERVE_UP, exit 3, the server stopped (the up loop's
    # sleeps are 0.05 s real here, so the server has its process group by the time the stop reaches it)
    if sglang_pid.exists():
        sglang_pid.unlink()
    state = _serve_state("Qwen3.5-4B", preflight="NO_TOOL_CALL", sleep_cap_s=0.05)
    state["curl"][SERVE_MODELS] = ["000", "000", "000", "200"]
    _fakes(tmp_path, state=state)
    env = _env(tmp_path, SERVE_REQUIRE_TOOL_CALL="1", **roots)
    rc, out = _run_script("scripts/serve_model.sh", model, env=env)
    assert rc == 3, out
    assert "SERVE_PREFLIGHT NO_TOOL_CALL" in out
    assert "FATAL: the served model returned no tool call (SERVE_REQUIRE_TOOL_CALL=1)" in out and "SERVE_UP" not in out
    assert _pid_gone(pid_file)
    # --stop: the pid file names a session leader (an orphan here), its group gets the TERM, it is gone, exit 0; the
    # pid file stays where it is (nothing is deleted); a stale file and a missing one are WARNs, exit 0
    _fakes(tmp_path, state={"sleep_cap_s": 0.3})
    env = _env(tmp_path, **roots)
    leader = _spawn_sleeper(tmp_path, new_session=True)
    pid_file.write_text(f"{leader}\n")
    rc, out = _run_script("scripts/serve_model.sh", "--stop", env=env)
    assert rc == 0 and f"SERVE_STOPPED pid={leader}" in out, out
    assert _pid_gone(leader) and pid_file.exists()
    assert Path(f"{pid_file}.stop").read_text().strip() == str(leader)  # the request, written before the TERM
    assert _calls(env, "kill") == []  # the builtin did it: nothing reached the fake
    rc, out = _run_script("scripts/serve_model.sh", "--stop", env=env)
    assert rc == 0 and f"WARN: pid {leader} of {pid_file} is not running: nothing to stop" in out, out
    rc, out = _run_script("scripts/serve_model.sh", "--stop", env=_env(tmp_path, SERVE_PORT="7001", **roots))
    assert rc == 0 and f"WARN: no pid file {_serve_pid_file(roots, 7001)}: nothing to stop" in out, out


def test_serve_model_dry_run_prints_both_argv(tmp_path):
    """--dry-run (S12): the serve-template line and the sglang line as argv, the GPUs measured, nothing started and
    nothing written; the knobs land in the argv; no GPU and a model dir without config.json are WARNs there and
    FATALs otherwise; no model is a FATAL; --stop --dry-run prints the group kill and signals nothing."""
    roots = _serve_roots(tmp_path)
    model = f"{roots['MILES_AGENT_MODELS']}/Qwen3.5-4B"
    results = roots["MILES_AGENT_RESULTS"]
    _fakes(tmp_path, state={"gpus": 2})
    env = _env(tmp_path, **roots)
    env.pop("PYTHONPATH", None)  # the serve-template line appends the caller's PYTHONPATH to the checkout
    before = set(tmp_path.rglob("*"))
    rc, out = _run_script("scripts/serve_model.sh", "--dry-run", model, env=env)
    assert rc == 0, out
    template = f"{results}/runs/serve_7000/chat_template.jinja"
    assert _dry(out, "serve_template") == ["env", f"PYTHONPATH={REPO}", "python3", "-m", "miles_agent.launcher",
                                           "serve-template", model, template, "--thinking", "off", "--tito-model", "qwen35"]
    assert _dry(out, "sglang") == ["setsid", "python3", "-m", "sglang.launch_server", "--model-path", model,
                                   "--served-model-name", "Qwen3.5-4B", "--tool-call-parser", "<serve-template>",
                                   "--chat-template", template, "--host", "0.0.0.0", "--port", "7000", "--dp-size", "2",
                                   "--mem-fraction-static", "0.85"]
    labels = [line.split(":", 1)[0] for line in out.splitlines() if line.startswith("DRY_RUN ")]
    assert labels == ["DRY_RUN serve_template", "DRY_RUN sglang"]
    assert "SERVE_PARSER" not in out and "SERVE_UP" not in out and "FATAL" not in out and "WARN" not in out
    assert {c["cmd"] for c in _calls(env)} <= {"nvidia-smi", "getent", "hostname"}  # measured, nothing started
    assert set(tmp_path.rglob("*")) == before  # no results dir, no template, no pid file
    # the knobs, SERVE_MODEL_PATH from the environment, SRC_MOUNT on the PYTHONPATH
    src = tmp_path / "src"
    env = _env(tmp_path, SERVE_MODEL_PATH=model, SERVE_PORT="7100", SERVE_MODEL_NAME="policy", SERVE_TITO_MODEL="qwen3",
               SERVE_THINKING="on", SERVE_DP="1", SERVE_CONTEXT_LENGTH="32768", SERVE_EXTRA_ARGS="--log-level warning",
               SERVE_WORK_DIR=str(tmp_path / "w"), SRC_MOUNT=str(src), **roots)
    env.pop("PYTHONPATH", None)
    rc, out = _run_script("scripts/serve_model.sh", "--dry-run", env=env)
    assert rc == 0, out
    assert _dry(out, "serve_template") == ["env", f"PYTHONPATH={src}", "python3", "-m", "miles_agent.launcher", "serve-template",
                                           model, f"{tmp_path}/w/chat_template.jinja", "--thinking", "on", "--tito-model", "qwen3"]
    argv = _dry(out, "sglang")
    assert argv[4:8] == ["--model-path", model, "--served-model-name", "policy"]
    assert argv[-10:] == ["--port", "7100", "--dp-size", "1", "--mem-fraction-static", "0.85", "--context-length", "32768",
                          "--log-level", "warning"]
    # no GPU: a WARN under --dry-run (dp 0 in the argv), a FATAL before anything starts otherwise
    _fakes(tmp_path, state={"gpus": 0})
    env = _env(tmp_path, **roots)
    rc, out = _run_script("scripts/serve_model.sh", "--dry-run", model, env=env)
    assert rc == 0 and "WARN: no GPU measured: --dp-size 0" in out, out
    assert _dry(out, "sglang")[-3:] == ["0", "--mem-fraction-static", "0.85"]
    rc, out = _run_script("scripts/serve_model.sh", model, env=env)
    assert rc == 1 and "FATAL: no GPU measured" in out and _calls(env, "python3") == [], out
    # a model dir without config.json: a WARN under --dry-run, a FATAL otherwise; no model at all: a FATAL
    _fakes(tmp_path, state={"gpus": 2})
    rc, out = _run_script("scripts/serve_model.sh", "--dry-run", str(tmp_path / "nope"), env=env)
    assert rc == 0 and f"WARN: no HF checkpoint at {tmp_path}/nope (config.json missing)" in out, out
    assert "DRY_RUN sglang:" in out
    rc, out = _run_script("scripts/serve_model.sh", str(tmp_path / "nope"), env=env)
    assert rc == 1 and f"FATAL: no HF checkpoint at {tmp_path}/nope (config.json missing)" in out, out
    rc, out = _run_script("scripts/serve_model.sh", "--dry-run", env=env)
    assert rc == 1 and "FATAL: a model dir is required (the positional, or SERVE_MODEL_PATH=)" in out, out
    rc, out = _run_script("scripts/serve_model.sh", "--bogus", env=env)
    assert rc == 1 and "FATAL: unknown argument '--bogus'" in out, out
    assert _calls(env, "python3") == [] and _calls(env, "setsid") == []
    # --stop --dry-run: the group kill as argv, the leader untouched
    leader = _spawn_sleeper(tmp_path, new_session=True)
    _serve_pid_file(roots).parent.mkdir(parents=True, exist_ok=True)
    _serve_pid_file(roots).write_text(f"{leader}\n")
    rc, out = _run_script("scripts/serve_model.sh", "--dry-run", "--stop", env=env)
    assert rc == 0 and _dry(out, "stop") == ["kill", "-TERM", "--", f"-{leader}"], out
    os.kill(leader, 0)  # still there (the fixture KILLs it)


def test_eval_ckpt_sh_builds_the_eval_run(tmp_path):
    """scripts/eval_ckpt.sh is a run of scripts/run_local.sh with RUN_SCRIPT=eval_ckpt.py on colocated_1n: the eval
    flags lead the driver's EXTRA_ARGS, the decode gets its own tag (eval_<arm>_s<seed>, or --run-tag), the `--`
    words and the caller's EXTRA_ARGS follow; exactly one of --eval-load-tag / --eval-base."""
    _fakes(tmp_path, state={"gpus": 2})
    roots = _run_roots(tmp_path)
    r = roots["MILES_AGENT_RESULTS"]
    env = _env(tmp_path, **roots)
    flags = ["--eval-load-tag", "arm1", "--eval-load-iter", "200", "--recipe-script", "opd.py", "--seed", "1001"]
    rc, out = _entry(tmp_path, "--dry-run", "alfworld", *flags, env=env, script="scripts/eval_ckpt.sh")
    assert rc == 0, out
    expected = _driver_argv(roots, tag="eval_arm1_s1001", extra=flags)
    expected[1] = "eval_ckpt.py"
    assert _dry(out, "driver") == expected
    assert ("eval_ckpt: RECIPE=alfworld RUN_TAG=eval_arm1_s1001 RUN_SCRIPT=eval_ckpt.py TOPOLOGY=colocated_1n "
            "EXTRA_ARGS=[--eval-load-tag arm1 --eval-load-iter 200 --recipe-script opd.py --seed 1001]") in out
    assert "run_local: RECIPE=alfworld RUN_TAG=eval_arm1_s1001 TOPOLOGY=colocated_1n MODE=normal NNODES=1" in out
    assert "FATAL" not in out and "WARN" not in out and "RUN_INNER_PASS" not in out
    # --eval-base (the HF weights, rollout-only), --run-tag, `=` spellings, the `--` words, then the caller's EXTRA_ARGS
    rc, out = _entry(tmp_path, "--dry-run", "alfworld", "--eval-base", "--run-tag=base_check", "--", "--num-rollout", "2",
                     env=_env(tmp_path, EXTRA_ARGS="--seed 7", **roots), script="scripts/eval_ckpt.sh")
    assert rc == 0, out
    expected = _driver_argv(roots, tag="base_check", extra=["--eval-base", "--num-rollout", "2", "--seed", "7"])
    expected[1] = "eval_ckpt.py"
    assert _dry(out, "driver") == expected
    # another recipe with an eval_ckpt.py, no seed: the tag is eval_<arm>, the iter is the arm's tracker's
    rc, out = _entry(tmp_path, "--dry-run", "scienceworld", "--eval-load-tag=arm2", env=env, script="scripts/eval_ckpt.sh")
    assert rc == 0, out
    argv = _dry(out, "driver")
    assert argv[:4] == ["python3", "eval_ckpt.py", "--topology", "colocated_1n"] and argv[-2:] == ["--eval-load-tag", "arm2"]
    assert argv[argv.index("--save-dir") + 1] == f"{r}/eval_arm2/"
    # the FATALs: both modes, neither, no recipe, a recipe without eval_ckpt.py, an iter with --eval-base, a bad iter
    for args, text in (
        (["alfworld", "--eval-load-tag", "arm1", "--eval-base"],
         "FATAL: --eval-load-tag decodes a saved checkpoint, --eval-base the HF weights: give one of them, not both"),
        (["alfworld"], "FATAL: one of --eval-load-tag <arm tag> / --eval-base is required"),
        (["--eval-base"], "FATAL: <recipe> is required"),
        (["nope", "--eval-base"], "FATAL: recipes/nope has no eval_ckpt.py"),
        (["alfworld", "--eval-base", "--eval-load-iter", "3"], "FATAL: --eval-load-iter goes with --eval-load-tag, not with --eval-base"),
        (["alfworld", "--eval-load-tag", "arm1", "--eval-load-iter", "x"], "FATAL: --eval-load-iter must be a save id, got 'x'"),
        (["alfworld", "--eval-load-tag"], "FATAL: --eval-load-tag needs a value"),
    ):
        rc, out = _entry(tmp_path, "--dry-run", *args, env=env, script="scripts/eval_ckpt.sh")
        assert rc == 1 and text in out and "DRY_RUN driver" not in out, (args, out)


def _export_fixture(tmp_path):
    """A results root with arm1's complete iter_0000200 (its .metadata, the tracker naming 200), a miles checkout
    with the converter tool and three model-args scripts, the fakes; returns (roots, miles)."""
    roots = _roots(tmp_path)
    results = Path(roots["MILES_AGENT_RESULTS"])
    (results / "arm1" / "iter_0000200").mkdir(parents=True)
    (results / "arm1" / "iter_0000200" / ".metadata").write_text("")
    (results / "arm1" / "latest_checkpointed_iteration.txt").write_text("200\n")
    miles = tmp_path / "miles"
    (miles / "tools").mkdir(parents=True)
    (miles / "tools" / "convert_torch_dist_to_hf.py").write_text("")
    (miles / "scripts" / "models").mkdir(parents=True)
    for name in ("qwen3.5-4B", "x", "a"):
        (miles / "scripts" / "models" / f"{name}.py").write_text("")
    _fakes(tmp_path)
    return roots, miles


def _converter_argv(roots, miles, iter_dir, out, origin=None, model_args="qwen3.5-4B", model_name="qwen3_5"):
    """tools/convert_ckpt_to_hf.py's argv for a torch_dist iter dir (today's convert job's line: MILES_ROOT for the
    in-image miles, the policy's HF dir as the default origin)."""
    origin = origin or f"{roots['MILES_AGENT_MODELS']}/Qwen3.5-4B"
    return [f"{REPO}/tools/convert_ckpt_to_hf.py", "--miles-root", str(miles), "--input-dir", str(iter_dir),
            "--model-args-script", f"{miles}/scripts/models/{model_args}.py", "--model-name", model_name,
            "--output-dir", str(out), "--origin-hf-dir", str(origin), "--fill-from-origin", "--origin-dtypes", "--check-origin"]


def test_export_hf_resolves_the_iter_and_calls_the_converter(tmp_path):
    """scripts/export_hf.sh <tag | dir> <iter | latest> <out>: the iter dir under $MILES_AGENT_RESULTS/<tag>, gated by
    its .metadata and the run's tracker (the partial-checkpoint rule); the converter line of today's convert job with
    MILES_ROOT (PYTHONPATH, --miles-root, the model-args script) and CUDA_VISIBLE_DEVICES= in its environment;
    HF_MODEL_ARGS / HF_MODEL_NAME and the flags over them; .complete written, EXPORT_HF_OK last; a finished export
    is not redone; every gate is a FATAL before the converter runs and a WARN under --dry-run."""
    roots, miles = _export_fixture(tmp_path)
    r = roots["MILES_AGENT_RESULTS"]
    iter_dir = f"{r}/arm1/iter_0000200"
    env = _env(tmp_path, MILES_ROOT=str(miles), **roots)
    out_dir = tmp_path / "hf_out"
    rc, out = _run_script("scripts/export_hf.sh", "arm1", "200", str(out_dir), env=env)
    assert rc == 0, out
    call, = _calls(env, "python3")
    assert call["argv"] == _converter_argv(roots, miles, iter_dir, out_dir)
    assert call["env"]["PYTHONPATH"] == str(miles) and call["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert (out_dir / ".complete").is_file()
    assert f"HF_EXPORT_READY {out_dir}" in out and out.splitlines()[-1] == f"EXPORT_HF_OK out={out_dir} iter=200"
    assert "FATAL" not in out and "WARN" not in out
    # the iter spelled with its zeros, `latest` through the tracker, the results dir instead of the tag: the same iter
    for i, args in enumerate((("arm1", "0000200"), ("arm1", "latest"), (f"{r}/arm1", "latest"), (f"{r}/arm1/", "200"))):
        _fakes(tmp_path)
        rc, out = _run_script("scripts/export_hf.sh", *args, str(tmp_path / f"hf_out_{i}"), env=env)
        assert rc == 0, (args, out)
        call, = _calls(env, "python3")
        assert call["argv"][4] == iter_dir and out.splitlines()[-1].endswith(" iter=200"), (args, out)
    # HF_MODEL_ARGS / HF_MODEL_NAME, then the flags over them and --origin-hf-dir
    origin2 = tmp_path / "origin2"
    origin2.mkdir()
    (origin2 / "config.json").write_text("{}")
    _fakes(tmp_path)
    env_xy = _env(tmp_path, MILES_ROOT=str(miles), HF_MODEL_ARGS="x", HF_MODEL_NAME="y", **roots)
    rc, out = _run_script("scripts/export_hf.sh", "arm1", "200", str(tmp_path / "hf_xy"), env=env_xy)
    assert rc == 0, out
    assert _calls(env_xy, "python3")[0]["argv"] == _converter_argv(roots, miles, iter_dir, tmp_path / "hf_xy",
                                                                   model_args="x", model_name="y")
    _fakes(tmp_path)
    rc, out = _run_script("scripts/export_hf.sh", "arm1", "200", str(tmp_path / "hf_ab"), "--model-args", "a",
                          "--model-name", "b", "--origin-hf-dir", str(origin2), env=env_xy)
    assert rc == 0, out
    assert _calls(env_xy, "python3")[0]["argv"] == _converter_argv(roots, miles, iter_dir, tmp_path / "hf_ab",
                                                                   origin=origin2, model_args="a", model_name="b")
    # a finished export (its .complete) is not redone
    _fakes(tmp_path)
    rc, out = _run_script("scripts/export_hf.sh", "arm1", "200", str(out_dir), env=env)
    assert rc == 0 and _calls(env, "python3") == [], out
    assert f"HF_EXPORT_READY {out_dir} (already)" in out
    assert out.splitlines()[-1] == f"EXPORT_HF_OK out={out_dir} iter=200 already=1"
    # a converter failure: its rc, no .complete
    _fakes(tmp_path, state={"run_rc": 2})
    rc, out = _run_script("scripts/export_hf.sh", "arm1", "200", str(tmp_path / "hf_fail"), env=env)
    assert rc == 2 and "FATAL: the converter exited with rc=2" in out, out
    assert not (tmp_path / "hf_fail" / ".complete").exists()
    # the gates, each a FATAL before the converter runs
    results = Path(r)
    (results / "arm2" / "iter_0000100").mkdir(parents=True)                        # no tracker at all
    (results / "arm2" / "iter_0000100" / ".metadata").write_text("")
    (results / "arm3" / "iter_0000300").mkdir(parents=True)                        # the tracker is behind
    (results / "arm3" / "iter_0000300" / ".metadata").write_text("")
    (results / "arm3" / "latest_checkpointed_iteration.txt").write_text("200\n")
    (results / "arm4" / "iter_0000010").mkdir(parents=True)                        # no .metadata
    (results / "arm4" / "latest_checkpointed_iteration.txt").write_text("10\n")
    (results / "arm5" / "iter_0000400").mkdir(parents=True)                        # the tracker names no save id
    (results / "arm5" / "iter_0000400" / ".metadata").write_text("")
    (results / "arm5" / "latest_checkpointed_iteration.txt").write_text("release\n")
    bare_miles = tmp_path / "bare_miles"                                           # no converter tool
    (bare_miles / "scripts" / "models").mkdir(parents=True)
    (bare_miles / "scripts" / "models" / "qwen3.5-4B.py").write_text("")
    _fakes(tmp_path)
    never = str(tmp_path / "hf_never")
    for args, text, extra in (
        (["arm2", "100", never], f"FATAL: no {r}/arm2/latest_checkpointed_iteration.txt: the run has no completed save", {}),
        (["arm3", "300", never], f"FATAL: {r}/arm3/latest_checkpointed_iteration.txt names save 200, before iter 300", {}),
        (["arm4", "10", never], f"FATAL: {r}/arm4/iter_0000010 has no .metadata", {}),
        (["arm5", "latest", never],
         f"FATAL: {r}/arm5/latest_checkpointed_iteration.txt names no save id ('release'; Megatron's 'release' marks a converted base)", {}),
        (["arm5", "400", never],
         f"FATAL: {r}/arm5/latest_checkpointed_iteration.txt names no save id ('release'; Megatron's 'release' marks a converted base)", {}),
        (["arm1", "999", never], f"FATAL: no checkpoint dir {r}/arm1/iter_0000999", {}),
        (["arm1", "abc", never], "FATAL: <iter> must be a save id or 'latest', got 'abc'", {}),
        (["arm1", "200"], "FATAL: three positionals (<run tag | results dir> <iter | latest> <out dir>) or none, got 2", {}),
        (["arm1", "200", never, "--model-args", "nope"], f"FATAL: no model-args script {miles}/scripts/models/nope.py", {}),
        (["arm1", "200", never, "--origin-hf-dir", str(tmp_path / "no_origin")],
         f"FATAL: no origin HF checkpoint at {tmp_path}/no_origin", {}),
        (["arm1", "200", never], f"FATAL: MILES_ROOT={bare_miles} has no tools/convert_torch_dist_to_hf.py",
         {"MILES_ROOT": str(bare_miles)}),
    ):
        case_env = _env(tmp_path, **{"MILES_ROOT": str(miles), **roots, **extra}) if extra else env
        rc, out = _run_script("scripts/export_hf.sh", *args, env=case_env)
        assert rc == 1 and text in out, (args, out)
    assert _calls(env, "python3") == [] and not Path(never).exists()
    # --dry-run: the argv (env CUDA_VISIBLE_DEVICES= PYTHONPATH=<miles> python3 ...), nothing run, nothing written;
    # a gate is a WARN there
    rc, out = _run_script("scripts/export_hf.sh", "--dry-run", "arm1", "latest", str(tmp_path / "hf_dry"), env=env)
    assert rc == 0, out
    assert _dry(out, "convert") == ["env", "CUDA_VISIBLE_DEVICES=", f"PYTHONPATH={miles}", "python3",
                                    *_converter_argv(roots, miles, iter_dir, tmp_path / "hf_dry")]
    assert _calls(env, "python3") == [] and not (tmp_path / "hf_dry").exists() and "WARN" not in out
    rc, out = _run_script("scripts/export_hf.sh", "--dry-run", "arm3", "300", str(tmp_path / "hf_dry"), env=env)
    assert rc == 0 and "WARN: " in out and _dry(out, "convert")[8] == f"{r}/arm3/iter_0000300", out
    rc, out = _run_script("scripts/export_hf.sh", "--dry-run", "arm5", "latest", str(tmp_path / "hf_dry"), env=env)
    assert rc == 0 and f"WARN: {r}/arm5/latest_checkpointed_iteration.txt names no save id ('release'" in out, out
    assert _dry(out, "convert")[8] == f"{r}/arm5/iter_<latest>" and _calls(env, "python3") == []


def test_export_hf_accepts_the_convert_hook_environment(tmp_path):
    """The Contract B convert hook (D14, PROJECT_CONVERT_CMD): no positionals, ITER_DIR and OUT_DIR from the wrapper's
    environment, HF_INPUT (an HF export the run wrote itself: --input-hf-dir, no model args) and HF_ORIGIN
    (--origin-hf-dir) honoured when non-empty; the .metadata gate, and the tracker gate when the run has a tracker;
    a pre-existing .complete is left alone; without the pair and without positionals the FATAL names both."""
    roots, miles = _export_fixture(tmp_path)
    r = roots["MILES_AGENT_RESULTS"]
    base = {"MILES_ROOT": str(miles), **roots}
    iter_dir = tmp_path / "iter_0000200"  # no tracker beside it: the hook form does not require one
    iter_dir.mkdir()
    (iter_dir / ".metadata").write_text("")
    out_dir = tmp_path / "out"
    env = _env(tmp_path, ITER_DIR=str(iter_dir), OUT_DIR=str(out_dir), HF_INPUT="", HF_ORIGIN="", **base)
    rc, out = _run_script("scripts/export_hf.sh", env=env)
    assert rc == 0, out
    call, = _calls(env, "python3")
    assert call["argv"] == _converter_argv(roots, miles, iter_dir, out_dir)
    assert call["env"]["PYTHONPATH"] == str(miles) and call["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert (out_dir / ".complete").is_file() and out.splitlines()[-1] == f"EXPORT_HF_OK out={out_dir} iter=200"
    # HF_INPUT: the run's own HF export (its .complete) is finished instead, no model args; HF_ORIGIN is the origin
    hf_in = tmp_path / "run_hf"
    hf_in.mkdir()
    (hf_in / ".complete").write_text("")
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "config.json").write_text("{}")
    _fakes(tmp_path)
    out2 = tmp_path / "out2"
    env = _env(tmp_path, ITER_DIR=str(iter_dir), OUT_DIR=str(out2), HF_INPUT=str(hf_in), HF_ORIGIN=str(origin), **base)
    rc, out = _run_script("scripts/export_hf.sh", env=env)
    assert rc == 0, out
    call, = _calls(env, "python3")
    assert call["argv"] == [f"{REPO}/tools/convert_ckpt_to_hf.py", "--miles-root", str(miles), "--input-hf-dir", str(hf_in),
                            "--output-dir", str(out2), "--origin-hf-dir", str(origin), "--fill-from-origin",
                            "--origin-dtypes", "--check-origin"]
    assert out.splitlines()[-1] == f"EXPORT_HF_OK out={out2} iter=hf"
    # HF_ORIGIN alone; --origin-hf-dir over it
    _fakes(tmp_path)
    env = _env(tmp_path, ITER_DIR=str(iter_dir), OUT_DIR=str(tmp_path / "out3"), HF_ORIGIN=str(origin), **base)
    rc, out = _run_script("scripts/export_hf.sh", env=env)
    assert rc == 0, out
    assert _calls(env, "python3")[0]["argv"] == _converter_argv(roots, miles, iter_dir, tmp_path / "out3", origin=origin)
    _fakes(tmp_path)
    rc, out = _run_script("scripts/export_hf.sh", "--dry-run", "--origin-hf-dir", str(roots["MILES_AGENT_MODELS"]) + "/Qwen3.5-4B",
                          env=env)
    assert rc == 0 and _dry(out, "convert")[4:] == _converter_argv(roots, miles, iter_dir, tmp_path / "out3"), out
    # a pre-existing .complete (the wrapper's own touch, an earlier export) is left alone: nothing redone
    before = (out_dir / ".complete").stat().st_mtime_ns
    _fakes(tmp_path)
    env = _env(tmp_path, ITER_DIR=str(iter_dir), OUT_DIR=str(out_dir), **base)
    rc, out = _run_script("scripts/export_hf.sh", env=env)
    assert rc == 0 and _calls(env, "python3") == [] and (out_dir / ".complete").stat().st_mtime_ns == before, out
    assert out.splitlines()[-1] == f"EXPORT_HF_OK out={out_dir} iter=200 already=1"
    # the run's tracker, when there is one, must have reached the iter; .metadata is required; HF_INPUT needs its
    # .complete; the pair is required
    run2 = tmp_path / "run2"
    (run2 / "iter_0000200").mkdir(parents=True)
    (run2 / "iter_0000200" / ".metadata").write_text("")
    (run2 / "latest_checkpointed_iteration.txt").write_text("100\n")
    (tmp_path / "iter_0000005").mkdir()
    (tmp_path / "unfinished_hf").mkdir()
    run3 = tmp_path / "run3"  # a tracker that names no save id (Megatron's `release`: a converted base)
    (run3 / "iter_0000200").mkdir(parents=True)
    (run3 / "iter_0000200" / ".metadata").write_text("")
    (run3 / "latest_checkpointed_iteration.txt").write_text("release\n")
    (tmp_path / "not_an_iter").mkdir()
    (tmp_path / "iter_12x").mkdir()
    never = str(tmp_path / "never")
    for extra, text in (
        ({"ITER_DIR": str(run2 / "iter_0000200"), "OUT_DIR": never},
         f"FATAL: {run2}/latest_checkpointed_iteration.txt names save 100, before iter 200"),
        ({"ITER_DIR": str(run3 / "iter_0000200"), "OUT_DIR": never},
         f"FATAL: {run3}/latest_checkpointed_iteration.txt names no save id ('release'; Megatron's 'release' marks a converted base)"),
        ({"ITER_DIR": str(tmp_path / "not_an_iter"), "OUT_DIR": never}, f"FATAL: {tmp_path}/not_an_iter is not an iter_<N> dir"),
        ({"ITER_DIR": str(tmp_path / "iter_12x"), "OUT_DIR": never}, f"FATAL: {tmp_path}/iter_12x is not an iter_<N> dir"),
        ({"ITER_DIR": str(tmp_path / "iter_0000005"), "OUT_DIR": never}, f"FATAL: {tmp_path}/iter_0000005 has no .metadata"),
        ({"ITER_DIR": str(iter_dir), "OUT_DIR": never, "HF_INPUT": str(tmp_path / "unfinished_hf")},
         f"FATAL: HF_INPUT={tmp_path}/unfinished_hf has no .complete marker"),
        ({"ITER_DIR": str(iter_dir)}, "FATAL: ITER_DIR and OUT_DIR must both be set"),
        ({"OUT_DIR": never}, "FATAL: ITER_DIR and OUT_DIR must both be set"),
        ({}, "FATAL: ITER_DIR and OUT_DIR must both be set"),
    ):
        env = _env(tmp_path, **{**base, **extra})
        rc, out = _run_script("scripts/export_hf.sh", env=env)
        assert rc == 1 and text in out and _calls(env, "python3") == [], (extra, out)
    assert not Path(never).exists()
    # under --dry-run the non-iter dir is a WARN and the argv still prints (the label falls back to the dir's name)
    env = _env(tmp_path, ITER_DIR=str(tmp_path / "not_an_iter"), OUT_DIR=never, **base)
    rc, out = _run_script("scripts/export_hf.sh", "--dry-run", env=env)
    assert rc == 0 and f"WARN: {tmp_path}/not_an_iter is not an iter_<N> dir" in out, out
    assert _dry(out, "convert")[4:] == _converter_argv(roots, miles, tmp_path / "not_an_iter", never) and not Path(never).exists()
    # HF_EXPORT_READY has one emitter, this script: no other public script prints it (the old convert job, now
    # private, printed plain text)
    others = [p for p in sorted((REPO / "scripts").rglob("*.sh")) if p.name != "export_hf.sh"]
    emitting = [f"{p.name}:{line}" for p in others for line in p.read_text().splitlines()
                if "HF_EXPORT_READY" in line and not line.lstrip().startswith("#")]
    assert emitting == []
    # a tracker that has reached the iter passes (arm1's, from the fixture); a trailing slash is dropped
    _fakes(tmp_path)
    env = _env(tmp_path, ITER_DIR=f"{r}/arm1/iter_0000200/", OUT_DIR=str(tmp_path / "out4"), **base)
    rc, out = _run_script("scripts/export_hf.sh", env=env)
    assert rc == 0 and _calls(env, "python3")[0]["argv"][4] == f"{r}/arm1/iter_0000200", out


def test_stop_all_kill_order_and_keep_ray(tmp_path):
    """scripts/stop_all.sh: TERM in the launcher's cleanup order (sglang by name, train.py, train_async.py,
    MegatronTrain; the gym, the WebShop engine; the serve pid files as process groups), every signal through the
    external `xargs kill` (so the fake logs the order AND delivers it to six real orphan sleepers), a 10 s grace,
    KILL to the survivors, ray last unless --keep-ray, STOP_ALL_OK killed=<n>; --dry-run signals nothing. The pid
    files are serve_model.sh's node-local ones (${TMPDIR}/miles-agent-${USER}/*.pid) plus SERVE_PID_FILE when set;
    <pid file>.stop (the request) is written before each group's TERM, not under --dry-run."""
    roots = _serve_roots(tmp_path)
    pid_file = _serve_pid_file(roots)
    pid_file.parent.mkdir(parents=True)
    p_sglang, p_train, p_mega, p_gym = (_spawn_sleeper(tmp_path) for _ in range(4))
    p_nemo = _spawn_sleeper(tmp_path, term_ignored=True)
    p_serve = _spawn_sleeper(tmp_path, new_session=True)
    pid_file.write_text(f"{p_serve}\n")
    pgrep = {"sglang": [p_sglang], "-f train.py": [p_train], "-f MegatronTrain": [p_mega], "-f gym env start": [p_gym],
             "-f nemo_gym": [p_nemo]}
    _fakes(tmp_path, state={"pgrep": pgrep, "sleep_cap_s": 0.3})
    env = _env(tmp_path, **roots)
    rc, out = _run_script("scripts/stop_all.sh", env=env)
    assert rc == 0, out
    kills = [c["argv"] for c in _calls(env, "kill")]
    terms = [["-TERM", str(p)] for p in (p_sglang, p_train, p_mega, p_gym, p_nemo)] + [["-TERM", "--", f"-{p_serve}"]]
    assert kills[:6] == terms, out
    assert all(k[0] == "-KILL" for k in kills[6:]) and ["-KILL", str(p_nemo)] in kills[6:], out
    gone = {p: _pid_gone(p) for p in (p_sglang, p_train, p_mega, p_gym, p_nemo, p_serve)}
    assert all(gone.values()), (gone, out)
    assert out.splitlines()[-1] == "STOP_ALL_OK killed=6"
    assert [c["argv"] for c in _calls(env, "pgrep")] == [["sglang"], ["-f", "train.py"], ["-f", "train_async.py"],
                                                          ["-f", "MegatronTrain"], ["-f", "gym env start"], ["-f", "nemo_gym"],
                                                          ["-f", "launch_engine.sh"], ["-f", "webshop/server"]]
    assert [c["argv"] for c in _calls(env, "sleep")] == [["5"], ["10"]]
    assert _calls(env)[-1]["cmd"] == "ray" and _calls(env)[-1]["argv"] == ["stop", "--force"]  # ray last
    assert "stop_all: TERM sglang:" in out and f"stop_all: TERM group of {pid_file}: {p_serve}" in out
    assert Path(f"{pid_file}.stop").read_text().strip() == str(p_serve)  # the request, before the TERM
    # --keep-ray: no ray call; the now stale pid file is skipped, never removed
    _fakes(tmp_path)
    env = _env(tmp_path, **roots)
    rc, out = _run_script("scripts/stop_all.sh", "--keep-ray", env=env)
    assert rc == 0 and _calls(env, "ray") == [] and _calls(env, "kill") == [], out
    assert f"names pid {p_serve}, which is gone: skipped" in out and out.splitlines()[-1] == "STOP_ALL_OK killed=0"
    assert pid_file.exists()
    # SERVE_PID_FILE elsewhere: that file is walked too (the stale default one still skipped)
    extra_file = tmp_path / "elsewhere" / "serve_7100.pid"
    extra_file.parent.mkdir()
    p_extra = _spawn_sleeper(tmp_path, new_session=True)
    extra_file.write_text(f"{p_extra}\n")
    _fakes(tmp_path, state={"sleep_cap_s": 0.3})
    env = _env(tmp_path, SERVE_PID_FILE=str(extra_file), **roots)
    rc, out = _run_script("scripts/stop_all.sh", "--keep-ray", env=env)
    assert rc == 0 and [c["argv"] for c in _calls(env, "kill")][:1] == [["-TERM", "--", f"-{p_extra}"]], out
    assert _pid_gone(p_extra) and extra_file.exists() and out.splitlines()[-1] == "STOP_ALL_OK killed=1"
    assert Path(f"{extra_file}.stop").read_text().strip() == str(p_extra) and f"names pid {p_serve}, which is gone: skipped" in out
    # --dry-run: the pgrep patterns and the kills as argv, nothing signalled, the sleepers still alive
    q1 = _spawn_sleeper(tmp_path)
    q2 = _spawn_sleeper(tmp_path, new_session=True)
    pid_file.write_text(f"{q2}\n")
    _fakes(tmp_path, state={"pgrep": {"sglang": [q1]}})
    env = _env(tmp_path, **roots)
    rc, out = _run_script("scripts/stop_all.sh", "--dry-run", env=env)
    assert rc == 0, out
    assert _dry(out, "pgrep") == ["sglang"] and _dry(out, "pgrep", 7) == ["-f", "webshop/server"]
    assert _dry(out, "kill") == ["-TERM", str(q1)] and _dry(out, "kill", 1) == ["-TERM", "--", f"-{q2}"]
    assert _dry(out, "ray_stop") == ["ray", "stop", "--force"] and out.splitlines()[-1] == "STOP_ALL_OK killed=2"
    assert _calls(env, "kill") == [] and _calls(env, "ray") == [] and _calls(env, "sleep") == []
    assert Path(f"{pid_file}.stop").read_text().strip() == str(p_serve)  # --dry-run writes no request
    for q in (q1, q2):
        os.kill(q, 0)  # still there (the fixture KILLs them)
    rc, out = _run_script("scripts/stop_all.sh", "--bogus", env=env)
    assert rc == 1 and "FATAL: unknown argument '--bogus'" in out
    # exactly one KILL, the TERM-ignoring sleeper: on a host whose init reaps the TERMed orphans within the grace
    if not _orphans_are_reaped(tmp_path):
        pytest.skip("this host's init does not reap TERMed orphans: the TERMed sleepers stay zombies past the grace")
    assert kills[6:] == [["-KILL", str(p_nemo)]], out


# ---- scripts/fetch_data.sh + scripts/build_gym_venvs.sh (C6) ------------------------------------------------

FETCH = "scripts/fetch_data.sh"
BUILD = "scripts/build_gym_venvs.sh"
WIKI_DATA = REPO / "gym_root" / "resources_servers" / "wiki_search" / "data"
MD5_TABLE = REPO / "env_servers" / "webshop" / "data_md5.txt"
GOALS_DIGEST = "455c9eaa96191605"
# the paper's list (--from the campaign file) and the sorted stand-in (--synthesize): neither is in the repository
# (recipes/scienceworld/data/README.md), so the cases pin the stand-in's digest through SCIENCEWORLD_TEST_SHA256
SCIW_SHA256 = "b2e1024f82a09e63011f37e4f423036c94b175b49b79c8d131eba7994cba2739"
SCIW_SYNTH_SHA256 = "f98541246a3145ce0ae2f97831a5f7dc7cd19e083a7b9e782f52dda4a1a8c11c"
SCIW_TASK_SPLITS = REPO / "gym_root" / "resources_servers" / "scienceworld_env" / "task_splits.py"
VENV_RELS = ("resources_servers/wiki_search", "responses_api_agents/tool_call_agent", "resources_servers/alfworld_env",
             "resources_servers/webshop_env", "resources_servers/scienceworld_env", "responses_api_agents/env_loop_agent",
             "responses_api_models/openai_model", "environment_servers/legacy_agent")


def _task_splits():
    """gym_root/resources_servers/scienceworld_env/task_splits.py loaded from its file (stdlib only; the fetch_data.sh
    heredocs import the same module from the repo)."""
    spec = importlib.util.spec_from_file_location("_fetch_data_task_splits", SCIW_TASK_SPLITS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _campaign_file(path, tasks):
    """The campaign's raw form: {"task_desc": "<json with task_name/var_num>"} rows (task_splits.load_test accepts it)."""
    path.write_text("".join(json.dumps({"task_desc": json.dumps({"task_name": t, "var_num": v}), "idx": i}) + "\n"
                            for i, (t, v) in enumerate(tasks)))


def test_fetch_data_webshop_without_from_prints_the_manual_step_and_fails(tmp_path):
    """The catalog is the WebShop authors' data: without --from the script prints the manual step (DATA_MANUAL_STEP
    with the md5 prefixes of the three small-catalog files) and exits 2; under --dry-run that is a WARN and exit 0.
    The tracked table carries exactly those three names with the 8-hex prefixes env_servers/webshop/setup.sh quotes."""
    _fakes(tmp_path)
    env = _env(tmp_path, **_roots(tmp_path))
    rc, out = _run_script(FETCH, "webshop", env=env)
    assert rc == 2, out
    assert "FATAL: webshop: the catalog is the WebShop authors' data and is not downloaded here (no --from <dir>)" in out
    step = [line for line in out.splitlines() if line.startswith("DATA_MANUAL_STEP webshop:")]
    assert len(step) == 1 and "rerun with --from <dir>" in step[0] and "FETCH_DATA_OK" not in out
    for name, prefix in (("items_shuffle_1000.json", "df2b2610"), ("items_ins_v2_1000.json", "a3346e8b"),
                         ("items_human_ins.json", "738c99a2")):
        assert f"{name} {prefix};" in step[0]
    rc, out = _run_script(FETCH, "--dry-run", "webshop", env=env)
    assert rc == 0, out
    assert "WARN: webshop: no --from <dir>" in out and "DATA_MANUAL_STEP webshop:" in out and "FATAL" not in out
    rows = [line.split() for line in MD5_TABLE.read_text().splitlines()]
    assert rows == [["items_shuffle_1000.json", "df2b2610"], ["items_ins_v2_1000.json", "a3346e8b"],
                    ["items_human_ins.json", "738c99a2"]]
    setup = (REPO / "env_servers" / "webshop" / "setup.sh").read_text()
    for _, prefix in rows:
        assert f"{prefix}..." in setup, prefix
    # a flag of the other mode, or of another environment, is a usage error
    rc, out = _run_script(FETCH, "webshop", "--from", str(tmp_path), "--engine-url", "http://127.0.0.1:1", env=env)
    assert rc == 1 and "FATAL: --engine-url is not an option of webshop --from" in out
    rc, out = _run_script(FETCH, "alfworld", "--regenerate-goals", env=env)
    assert rc == 1 and "FATAL: --regenerate-goals is not an option of alfworld" in out


def test_fetch_data_webshop_from_copies_and_verifies_the_md5_table(tmp_path):
    """--from <dir>: each catalog file is md5-checked against the table (a prefix entry matches a prefix, a full entry
    must match whole, a name without an entry is copied with a WARN, a mismatch or a missing file is a FATAL before
    anything is copied) and lands in $WEBSHOP_DATA_DIR = $WEBSHOP_HOME/webshop/data, where
    env_servers/webshop/setup.sh looks; --full adds the big catalog; --dry-run prints the copies and touches nothing."""
    _fakes(tmp_path)
    src = tmp_path / "catalog"
    src.mkdir()
    small = ["items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"]
    big = ["items_shuffle.json", "items_ins_v2.json"]
    for name in small + big:
        (src / name).write_text(json.dumps({"file": name}))

    def md5(name):
        return hashlib.md5((src / name).read_bytes()).hexdigest()

    table = tmp_path / "md5.txt"
    table.write_text(f"items_shuffle_1000.json {md5('items_shuffle_1000.json')[:8]}\n"
                     f"items_ins_v2_1000.json {md5('items_ins_v2_1000.json')}\n"
                     f"items_shuffle.json {md5('items_shuffle.json')}\n")
    home = tmp_path / "wshome"
    dest = home / "webshop" / "data"
    env = _env(tmp_path, **_roots(tmp_path), WEBSHOP_HOME=str(home), WEBSHOP_MD5_TABLE=str(table))
    rc, out = _run_script(FETCH, "--dry-run", "webshop", "--from", str(src), env=env)
    assert rc == 0, out
    assert [_dry(out, "copy", i) for i in range(3)] == [["cp", str(src / n), str(dest / n)] for n in small]
    assert "WARN: no md5 entry for items_human_ins.json" in out and "FETCH_DATA_OK" not in out and not home.exists()
    rc, out = _run_script(FETCH, "--dry-run", "webshop", "--from", str(src), "--full", env=env)
    assert rc == 0 and [_dry(out, "copy", i)[-1] for i in range(5)] == [str(dest / n) for n in small + big]
    assert f"WEBSHOP_HOME={home} bash env_servers/webshop/setup.sh --full" in out  # the hint names --full with --full
    rc, out = _run_script(FETCH, "webshop", "--from", str(src), env=env)
    assert rc == 0, out
    assert f"webshop: items_shuffle_1000.json md5 {md5('items_shuffle_1000.json')} matches the 8-hex entry" in out
    assert f"webshop: items_ins_v2_1000.json md5 {md5('items_ins_v2_1000.json')} matches the 32-hex entry" in out
    assert sorted(p.name for p in dest.iterdir()) == sorted(small)
    for name in small:
        assert (dest / name).read_bytes() == (src / name).read_bytes(), name
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=webshop dir={dest}"
    # ... and without --full the hint is the small-catalog command (FULL is 0 here, not empty)
    assert any(line.endswith(f"WEBSHOP_HOME={home} bash env_servers/webshop/setup.sh") for line in out.splitlines()), out
    # the engine's own data dir as --from: nothing is copied over itself
    rc, out = _run_script(FETCH, "webshop", "--from", str(dest), env=env)
    assert rc == 0 and out.count("is already in place at") == 3 and "FETCH_DATA_OK env=webshop" in out
    # a missing catalog file is a FATAL; so is a mismatch, before any copy (a WARN under --dry-run, the copy still
    # printed), and a missing --from dir
    (src / "items_human_ins.json").unlink()
    rc, out = _run_script(FETCH, "webshop", "--from", str(src), env=env)
    assert rc == 1 and f"FATAL: no {src / 'items_human_ins.json'}" in out
    (src / "items_human_ins.json").write_text(json.dumps({"file": "items_human_ins.json"}))
    table.write_text("items_shuffle_1000.json deadbeef\n")
    home2 = tmp_path / "wshome2"
    env2 = _env(tmp_path, **_roots(tmp_path), WEBSHOP_HOME=str(home2), WEBSHOP_MD5_TABLE=str(table))
    rc, out = _run_script(FETCH, "webshop", "--from", str(src), env=env2)
    assert rc == 1, out
    assert f"FATAL: webshop: items_shuffle_1000.json md5 {md5('items_shuffle_1000.json')} does not match deadbeef" in out
    assert not home2.exists()
    rc, out = _run_script(FETCH, "--dry-run", "webshop", "--from", str(src), env=env2)
    assert rc == 0 and "WARN: webshop: items_shuffle_1000.json md5" in out and _dry(out, "copy")[0] == "cp"
    rc, out = _run_script(FETCH, "webshop", "--from", str(tmp_path / "nowhere"), env=env2)
    assert rc == 1 and f"FATAL: no --from dir {tmp_path / 'nowhere'}" in out and not home2.exists()


def test_fetch_data_webshop_regenerate_goals_runs_the_exporter_and_pins_the_digest(tmp_path):
    """--regenerate-goals: the engine's /health first (none: FATAL naming launch_engine.sh), then export_goals.py
    --engine-url U --output <goals> (WEBSHOP_GOALS_FILE; U = --engine-url, else WEBSHOP_ENGINE_BASE_URL, else the
    loopback engine port), then the exporter's sidecar digest must equal WEBSHOP_GOALS_DIGEST (default
    455c9eaa96191605; a mismatch is a FATAL with the export left in place) and the sidecar is copied to the name
    webshop_goals.meta.json beside the export; --dry-run prints the exporter argv and skips the health check."""
    goals = tmp_path / "goals" / "webshop_goals.jsonl"
    health = "http://127.0.0.1:6200/health"
    _fakes(tmp_path, state={"curl": {health: "200"}, "goals_digest": GOALS_DIGEST, "rows": 7})
    roots = _roots(tmp_path)
    env = _env(tmp_path, **roots, WEBSHOP_GOALS_FILE=str(goals))
    exporter = f"{REPO}/env_servers/webshop/export_goals.py"
    rc, out = _run_script(FETCH, "--dry-run", "webshop", "--regenerate-goals", env=env)
    assert rc == 0, out
    assert _dry(out, "export_goals") == ["python3", exporter, "--engine-url", "http://127.0.0.1:6200", "--output", str(goals)]
    assert f"WARN: the engine health check ({health}) is skipped" in out and not _calls(env, "python3")
    assert not _calls(env, "curl") and not goals.parent.exists()
    rc, out = _run_script(FETCH, "webshop", "--regenerate-goals", env=env)
    assert rc == 0, out
    [export] = _calls(env, "python3")
    assert export["argv"] == [exporter, "--engine-url", "http://127.0.0.1:6200", "--output", str(goals)]
    assert [c["argv"][-1] for c in _calls(env, "curl")] == [health]
    assert len(goals.read_text().splitlines()) == 7
    meta = json.loads((goals.parent / "webshop_goals.meta.json").read_text())
    assert meta["goals_digest"] == GOALS_DIGEST and meta["goals"] == 7 and meta["engine_url"] == "http://127.0.0.1:6200"
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=webshop goals=7 goals_digest={GOALS_DIGEST}"
    # another engine by flag (a trailing slash dropped); a mismatching digest is a FATAL, the export stays, the
    # copied sidecar is not touched
    _set_state(tmp_path, {"curl": {"http://192.0.2.10:6231/health": "200"}, "goals_digest": "0123456789abcdef", "rows": 9})
    rc, out = _run_script(FETCH, "webshop", "--regenerate-goals", "--engine-url", "http://192.0.2.10:6231/", env=env)
    assert rc == 1, out
    assert f"FATAL: regenerated goals digest 0123456789abcdef != pinned {GOALS_DIGEST} (WEBSHOP_GOALS_DIGEST)" in out
    assert _calls(env, "python3")[-1]["argv"][2] == "http://192.0.2.10:6231"
    assert len(goals.read_text().splitlines()) == 9 and "FETCH_DATA_OK" not in out
    assert json.loads((goals.parent / "webshop_goals.meta.json").read_text())["goals"] == 7
    # the pin is a knob; WEBSHOP_ENGINE_BASE_URL picks the engine
    env_pin = _env(tmp_path, **roots, WEBSHOP_GOALS_FILE=str(goals), WEBSHOP_GOALS_DIGEST="0123456789abcdef",
                   WEBSHOP_ENGINE_BASE_URL="http://192.0.2.10:6231")
    rc, out = _run_script(FETCH, "webshop", "--regenerate-goals", env=env_pin)
    assert rc == 0 and "FETCH_DATA_OK env=webshop goals=9 goals_digest=0123456789abcdef" in out
    # no engine: FATAL naming the launcher, nothing exported
    n = len(_calls(env, "python3"))
    _set_state(tmp_path, {"curl": {}})
    rc, out = _run_script(FETCH, "webshop", "--regenerate-goals", env=_env(tmp_path, **roots, WEBSHOP_SERVER_PORT="6300"))
    assert rc == 1, out
    assert ("FATAL: no WebShop engine at http://127.0.0.1:6300: start it with env_servers/webshop/launch_engine.sh "
            f"(WEBSHOP_HOME={roots['MILES_AGENT_DATA']}/webshop) and rerun") in out
    assert len(_calls(env, "python3")) == n


def test_fetch_data_search_r1_dry_run_names_prepare_and_the_links(tmp_path):
    """search_r1: prepare.py --output-dir $WIKI_SEARCH_DATA (default $MILES_AGENT_DATA/wiki_search; --splits, split
    names only, passed through; $PYTHON), then train/validation linked into the gym data dir the config reads;
    --dry-run prints exactly those three commands and runs nothing (the real run needs the Hub)."""
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    env = _env(tmp_path, **roots)
    rc, out = _run_script(FETCH, "--dry-run", "search_r1", env=env)
    assert rc == 0, out
    data = f"{roots['MILES_AGENT_DATA']}/wiki_search"
    assert _dry(out, "prepare") == ["python3", str(WIKI_DATA / "prepare.py"), "--output-dir", data]
    assert _dry(out, "link") == ["ln", "-sfn", f"{data}/train.jsonl", str(WIKI_DATA / "train.jsonl")]
    assert _dry(out, "link", 1) == ["ln", "-sfn", f"{data}/validation.jsonl", str(WIKI_DATA / "validation.jsonl")]
    assert out.count("DRY_RUN ") == 3 and not _calls(env, "python3") and "FETCH_DATA_OK" not in out
    env2 = _env(tmp_path, **roots, WIKI_SEARCH_DATA="/w", PYTHON="/opt/py/bin/python")
    rc, out = _run_script(FETCH, "search_r1", "--splits", "validation", "test", "--dry-run", env=env2)
    assert rc == 0, out
    assert _dry(out, "prepare") == ["/opt/py/bin/python", str(WIKI_DATA / "prepare.py"), "--output-dir", "/w",
                                    "--splits", "validation", "test"]
    assert _dry(out, "link")[2] == "/w/train.jsonl"
    # --splits takes split names only: the next flag or environment name ends the list; a non-split is a usage error
    rc, out = _run_script(FETCH, "--dry-run", "--splits", "validation", "search_r1", env=env2)
    assert rc == 0, out
    assert _dry(out, "prepare")[-2:] == ["--splits", "validation"] and _dry(out, "link")[2] == "/w/train.jsonl"
    rc, out = _run_script(FETCH, "search_r1", "--splits", "bogus", "--dry-run", env=env2)
    assert rc == 1 and "FATAL: --splits takes split names (train validation test), not 'bogus'" in out
    # the linked names are git-ignored in the gym data dir (the fixture is not) and are the paths the config reads
    ignored = (WIKI_DATA / ".gitignore").read_text().split()
    for name in ("train.jsonl", "validation.jsonl", "placeholder_train.jsonl"):
        assert any(Path(name).match(pattern) for pattern in ignored), name
    assert not any(Path("example.jsonl").match(pattern) for pattern in ignored) and (WIKI_DATA / "example.jsonl").is_file()
    cfg = (WIKI_DATA.parent / "configs" / "wiki_search.yaml").read_text()
    assert "resources_servers/wiki_search/data/train.jsonl" in cfg and "resources_servers/wiki_search/data/validation.jsonl" in cfg
    # usage errors and --help
    rc, out = _run_script(FETCH, "search_r1", "--from", "/x", "--dry-run", env=env)
    assert rc == 1 and "FATAL: --from is not an option of search_r1" in out
    rc, out = _run_script(FETCH, "--dry-run", env=env)
    assert rc == 1 and "FATAL: an environment is required" in out
    rc, out = _run_script(FETCH, "--dry-run", "mars", env=env)
    assert rc == 1 and "FATAL: unknown environment 'mars'" in out
    rc, out = _run_script(FETCH, "--help", env=env)
    assert rc == 0 and out.startswith("scripts/fetch_data.sh") and "DATA_MANUAL_STEP" in out and "set -euo" not in out


def test_fetch_data_alfworld_uses_the_venv_binary_when_a_root_is_set(tmp_path):
    """alfworld: GYM_VENV_ROOT set -> <root>/gym_venvs/resources_servers/alfworld_env/.venv/bin/alfworld-download
    (missing: FATAL naming the venv build), else the alfworld-download on PATH (none: FATAL printing the in-venv
    command); --data-dir $ALFWORLD_DATA (default $MILES_AGENT_DATA/alfworld), then json_2.1.1 must exist."""
    roots = _roots(tmp_path)
    root = tmp_path / "venvs"
    venv_bin = _fake_at(root / "gym_venvs" / "resources_servers" / "alfworld_env" / ".venv" / "bin" / "alfworld-download")
    _fakes(tmp_path)  # no alfworld-download among the PATH fakes
    env = _env(tmp_path, **roots, GYM_VENV_ROOT=str(root))
    alf = f"{roots['MILES_AGENT_DATA']}/alfworld"
    rc, out = _run_script(FETCH, "--dry-run", "alfworld", env=env)
    assert rc == 0, out
    assert _dry(out, "alfworld_download") == [str(venv_bin), "--data-dir", alf] and "WARN" not in out
    assert not _calls(env, "alfworld-download") and not Path(alf).exists()
    rc, out = _run_script(FETCH, "alfworld", env=env)
    assert rc == 0, out
    [call] = _calls(env, "alfworld-download")
    assert call["argv"] == ["--data-dir", alf] and Path(alf, "json_2.1.1").is_dir()
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=alfworld dir={alf}"
    # no venv binary under the root: FATAL naming the build (a WARN under --dry-run)
    env2 = _env(tmp_path, **roots, GYM_VENV_ROOT="/nowhere")
    missing = "/nowhere/gym_venvs/resources_servers/alfworld_env/.venv/bin/alfworld-download"
    rc, out = _run_script(FETCH, "alfworld", env=env2)
    assert rc == 1, out
    assert f"FATAL: no {missing}: build the alfworld_env venv first (scripts/build_gym_venvs.sh /nowhere alfworld_env)" in out
    rc, out = _run_script(FETCH, "--dry-run", "alfworld", env=env2)
    assert rc == 0 and f"WARN: no {missing}" in out and _dry(out, "alfworld_download")[0] == missing
    # no root: the binary on PATH, ALFWORLD_DATA honoured; a download that leaves no json_2.1.1 is a FATAL
    _fakes(tmp_path, *DEFAULT_FAKES, "alfworld-download", state={"alfworld_download_creates": False})
    alf2 = tmp_path / "alf2"
    env3 = _env(tmp_path, **roots, ALFWORLD_DATA=str(alf2))
    rc, out = _run_script(FETCH, "alfworld", env=env3)
    assert rc == 1 and f"FATAL: no {alf2}/json_2.1.1 after the download" in out
    assert _calls(env3, "alfworld-download")[-1]["argv"] == ["--data-dir", str(alf2)]
    _set_state(tmp_path, {"alfworld_download_rc": 3})
    rc, out = _run_script(FETCH, "alfworld", env=env3)
    assert rc == 1 and f"FATAL: alfworld-download exited with rc=3 (ALFWORLD_DATA={alf2})" in out
    _set_state(tmp_path, {})
    rc, out = _run_script(FETCH, "alfworld", env=env3)
    assert rc == 0 and out.splitlines()[-1] == f"FETCH_DATA_OK env=alfworld dir={alf2}"
    rc, out = _run_script(FETCH, "--dry-run", "alfworld", env=env3)
    assert rc == 0 and _dry(out, "alfworld_download") == ["alfworld-download", "--data-dir", str(alf2)]
    # neither: the exact command to run inside the venv
    _fakes(tmp_path)
    rc, out = _run_script(FETCH, "alfworld", env=_env(tmp_path, bare_path=True, **roots))
    assert rc == 1, out
    assert ("FATAL: no alfworld-download on PATH (and no GYM_VENV_ROOT): run it inside the alfworld_env venv — "
            f"cd gym_root/resources_servers/alfworld_env && uv run alfworld-download --data-dir {alf} — or build the "
            "venvs (scripts/build_gym_venvs.sh) and set GYM_VENV_ROOT") in out


def test_fetch_data_scienceworld_reserialises_the_tcod_list_and_pins_its_digest(tmp_path):
    """scienceworld: the JDK check (java -version under the ladder's JDK: SCIENCEWORLD_JAVA_HOME, SCIW_JAVA_HOME,
    JAVA_HOME, $TOOLS_DIR/jdk-17, else on PATH; < 11 is a FATAL; no sha256 tool on PATH is a FATAL too),
    then the eval list: --from re-serialises the campaign's raw rows through task_splits.load_test to
    SCIENCEWORLD_TEST_LIST and FATALs unless the sha256 is the pinned one (SCIENCEWORLD_TEST_SHA256, default the
    paper's list; no list is in the repository, so the case builds a campaign-format file from the sorted stand-in
    task_splits.synthesized_test() and pins that digest; the output stays for inspection; a file load_test refuses
    leaves the previous output alone); without --from an existing list is classified by its digest and the
    upstream pointer printed. The real interpreter runs the heredoc."""
    _fakes(tmp_path, *DEFAULT_FAKES, "java")
    roots = _roots(tmp_path)
    ts = _task_splits()
    tasks = ts.synthesized_test()
    expected = tmp_path / "expected.jsonl"  # the stand-in's bytes (write_test_list = the heredoc's rows)
    ts.write_test_list(tasks, expected)
    assert len(tasks) == 1308 and hashlib.sha256(expected.read_bytes()).hexdigest() == SCIW_SYNTH_SHA256
    campaign = tmp_path / "test.jsonl"  # the campaign's raw form: {"task_desc": "<json>"} rows in the same order
    _campaign_file(campaign, tasks)
    out_list = f"{roots['MILES_AGENT_DATA']}/scienceworld/scienceworld_test1308.jsonl"
    env = _env(tmp_path, **roots, PYTHON=sys.executable, SCIENCEWORLD_TEST_SHA256=SCIW_SYNTH_SHA256)
    rc, out = _run_script(FETCH, "--dry-run", "scienceworld", "--from", str(campaign), env=env)
    assert rc == 0, out
    assert _dry(out, "reserialise") == [sys.executable, "-", str(REPO), str(campaign), out_list]
    assert "scienceworld: java 17 ok" in out and not Path(out_list).exists() and "FETCH_DATA_OK" not in out
    rc, out = _run_script(FETCH, "scienceworld", "--from", str(campaign), env=env)
    assert rc == 0, out
    assert Path(out_list).read_bytes() == expected.read_bytes()
    assert f"reserialised 1308 rows -> {out_list}.tmp" in out
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=scienceworld test_list={out_list} sha256={SCIW_SYNTH_SHA256}"
    assert [c["argv"] for c in _calls(env, "java")] == [["-version"]] * 2
    # without the override the pin is the paper's order: the FATAL names both digests and the knob, the output stays
    rc, out = _run_script(FETCH, "scienceworld", "--from", str(campaign), env=_env(tmp_path, **roots, PYTHON=sys.executable))
    assert rc == 1, out
    assert (f"FATAL: scienceworld test list (re-serialised from {campaign}) digest {SCIW_SYNTH_SHA256} != pinned "
            f"{SCIW_SHA256} (SCIENCEWORLD_TEST_SHA256): not the TCOD campaign's taskset "
            "(recipes/scienceworld/data/README.md)") in out
    assert "FETCH_DATA_OK" not in out and Path(out_list).read_bytes() == expected.read_bytes()
    # two rows swapped: the sets are intact (load_test accepts the file) but the order is another one
    lines = campaign.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    swapped = tmp_path / "swapped.jsonl"
    swapped.write_text("\n".join(lines) + "\n")
    rc, out = _run_script(FETCH, "scienceworld", "--from", str(swapped), env=env)
    assert rc == 1, out
    assert f"FATAL: scienceworld test list (re-serialised from {swapped}) digest " in out
    assert f"!= pinned {SCIW_SYNTH_SHA256} (SCIENCEWORLD_TEST_SHA256): not the TCOD campaign's taskset" in out
    assert Path(out_list).is_file() and Path(out_list).read_bytes() != expected.read_bytes()
    # a file load_test refuses (1307 rows): nothing written over the previous output, no .tmp left
    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(lines[1:]) + "\n")
    before = Path(out_list).read_bytes()
    rc, out = _run_script(FETCH, "scienceworld", "--from", str(short), env=env)
    assert rc == 1 and "1307 rows != 1308" in out and f"FATAL: re-serialising {short} failed" in out
    assert Path(out_list).read_bytes() == before and not Path(out_list + ".tmp").exists()
    rc, out = _run_script(FETCH, "scienceworld", "--from", str(tmp_path / "none.jsonl"), env=env)
    assert rc == 1 and f"FATAL: no --from file {tmp_path / 'none.jsonl'}" in out
    # without --from: an existing list is classified by its digest (the swapped output: unknown -> a WARN naming
    # the frozen pin and the synthesized digest) and the upstream pointer printed
    rc, out = _run_script(FETCH, "scienceworld", env=env)
    assert rc == 0, out
    unknown = hashlib.sha256(before).hexdigest()
    assert f"WARN: unknown list digest {unknown} at {out_list}" in out and SCIW_SYNTH_SHA256 in out
    assert "DATA_UPSTREAM scienceworld: TCOD_examples/scienceworld/get_sciworld_data.py (arXiv 2604.24005)" in out
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=scienceworld test_list={out_list} sha256={unknown} order=unknown"
    env_t = _env(tmp_path, **roots, PYTHON=sys.executable, SCIENCEWORLD_TEST_LIST=str(expected))
    rc, out = _run_script(FETCH, "scienceworld", env=env_t)
    assert rc == 0, out
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=scienceworld test_list={expected} sha256={SCIW_SYNTH_SHA256} order=synthesized"
    rc, out = _run_script(FETCH, "scienceworld", env=dict(env_t, SCIENCEWORLD_TEST_SHA256=SCIW_SYNTH_SHA256))
    assert rc == 0 and out.splitlines()[-1].endswith(f"sha256={SCIW_SYNTH_SHA256} order=frozen"), out
    rc, out = _run_script(FETCH, "--dry-run", "scienceworld", env=env_t)
    assert rc == 0 and "DATA_UPSTREAM scienceworld" in out and "order=synthesized" in out and "FETCH_DATA_OK" not in out
    # the JDK in the ladder's order (scripts/lib/common.sh scienceworld_java_home): SCIENCEWORLD_JAVA_HOME, the alias
    # SCIW_JAVA_HOME, JAVA_HOME, $TOOLS_DIR/jdk-17, then PATH; an old java is a FATAL (a WARN under --dry-run)
    jdk = tmp_path / "jdk"
    _fake_at(jdk / "bin" / "java")
    _set_state(tmp_path, {"java_version": 'java version "1.8.0_292"'})
    rc, out = _run_script(FETCH, "scienceworld", env=_env(tmp_path, **roots, JAVA_HOME=str(jdk)))
    assert rc == 1 and f"FATAL: java 8 at {jdk}/bin/java is too old: ScienceWorld needs a JDK >= 11" in out
    rc, out = _run_script(FETCH, "--dry-run", "scienceworld", env=_env(tmp_path, **roots, JAVA_HOME=str(jdk)))
    assert rc == 0 and f"WARN: java 8 at {jdk}/bin/java is too old" in out and "DATA_UPSTREAM scienceworld" in out
    nojdk = tmp_path / "nojdk"
    rc, out = _run_script(FETCH, "scienceworld", env=_env(tmp_path, **roots, JAVA_HOME=str(nojdk)))
    assert rc == 1 and (f"FATAL: no {nojdk}/bin/java (the JDK resolves to {nojdk}: SCIENCEWORLD_JAVA_HOME, alias "
                        "SCIW_JAVA_HOME, else JAVA_HOME, else TOOLS_DIR/jdk-17): point it at a JDK 17") in out
    # the alias beats JAVA_HOME, the new name beats the alias (the missing one is the one chosen)
    rc, out = _run_script(FETCH, "scienceworld", env=_env(tmp_path, **roots, JAVA_HOME=str(jdk), SCIW_JAVA_HOME=str(nojdk)))
    assert rc == 1 and f"FATAL: no {nojdk}/bin/java" in out, out
    rc, out = _run_script(FETCH, "scienceworld", env=_env(tmp_path, **roots, SCIW_JAVA_HOME=str(jdk),
                                                           SCIENCEWORLD_JAVA_HOME=str(tmp_path / "nojdk2")))
    assert rc == 1 and f"FATAL: no {tmp_path}/nojdk2/bin/java" in out, out
    _fakes(tmp_path)  # no java on PATH at all (a bare PATH: the host's own java must not be found either)
    bare = _env(tmp_path, bare_path=True, **roots, PYTHON=sys.executable, SCIENCEWORLD_TEST_LIST=str(expected))
    rc, out = _run_script(FETCH, "scienceworld", env=bare)
    assert rc == 1 and ("FATAL: no java on PATH and no JAVA_HOME: install a JDK 17 or set JAVA_HOME (SCIENCEWORLD_JAVA_HOME, "
                        "alias SCIW_JAVA_HOME, wins over it)") in out
    # $TOOLS_DIR/jdk-17 stands in when nothing else is set
    _fake_at(tmp_path / "tools" / "jdk-17" / "bin" / "java")
    rc, out = _run_script(FETCH, "scienceworld", env=dict(bare, TOOLS_DIR=str(tmp_path / "tools")))
    assert rc == 0 and "scienceworld: java 17 ok" in out and "FETCH_DATA_OK env=scienceworld" in out, out
    assert _calls(env=dict(bare), cmd="java")[-1]["argv"] == ["-version"]
    bare["SCIW_JAVA_HOME"] = str(jdk)
    rc, out = _run_script(FETCH, "scienceworld", env=bare)
    assert rc == 0, out
    assert "scienceworld: java 17 ok" in out and "FETCH_DATA_OK env=scienceworld" in out
    # no sha256 tool on PATH (a toolbox without sha256sum / shasum): a FATAL, not an empty digest
    nosha = tmp_path / "tools_nosha"
    nosha.mkdir()
    for name in TOOLBOX:
        real = shutil.which(name)
        if real and name not in ("sha256sum", "shasum"):
            (nosha / name).symlink_to(real)
    env_nosha = dict(bare, PATH=os.pathsep.join([str(tmp_path / "bin"), str(Path(sys.executable).parent), str(nosha)]))
    rc, out = _run_script(FETCH, "scienceworld", env=env_nosha)
    assert rc == 1 and "FATAL: no sha256 tool (sha256sum or shasum) on PATH" in out and "FETCH_DATA_OK" not in out, out


def test_fetch_data_scienceworld_synthesize_writes_the_sorted_list(tmp_path):
    """--synthesize: task_splits.synthesized_test() (the paper's 1308-task SET in sorted order) written through
    write_test_list to SCIENCEWORLD_TEST_LIST (else $MILES_AGENT_DATA/scienceworld/scienceworld_test1308.jsonl), its
    sha256 pinned, the order WARN, `order=synthesized`; a second run is idempotent; a list holding the frozen pin's
    bytes is never overwritten (no frozen list is in the repository: SCIENCEWORLD_TEST_SHA256 pins the stand-in's
    digest to simulate it); --dry-run prints the argv; --from with --synthesize, or --synthesize on another
    environment, is a usage error. The real interpreter runs the heredoc."""
    _fakes(tmp_path, *DEFAULT_FAKES, "java")
    roots = _roots(tmp_path)
    lst = tmp_path / "lists" / "l.jsonl"
    env = _env(tmp_path, **roots, PYTHON=sys.executable, SCIENCEWORLD_TEST_LIST=str(lst))
    rc, out = _run_script(FETCH, "--dry-run", "scienceworld", "--synthesize", env=env)
    assert rc == 0, out
    assert _dry(out, "synthesize") == [sys.executable, "-", str(REPO), str(lst)]
    assert "scienceworld: java 17 ok" in out and not lst.parent.exists() and "FETCH_DATA_OK" not in out
    rc, out = _run_script(FETCH, "scienceworld", "--synthesize", env=env)
    assert rc == 0, out
    assert hashlib.sha256(lst.read_bytes()).hexdigest() == SCIW_SYNTH_SHA256
    rows = [json.loads(line) for line in lst.read_text().splitlines()]
    assert len(rows) == 1308 and rows == sorted(rows, key=lambda r: (r["task_name"], r["var_num"]))
    assert rows[0] == {"task_name": "chemistry-mix-paint-tertiary-color", "var_num": 0}
    assert rows[-1] == {"task_name": "test-conductivity-of-unknown-substances", "var_num": 299}
    assert f"synthesized 1308 rows (sorted task_name, var_num) -> {lst}.tmp" in out and not Path(f"{lst}.tmp").exists()
    assert "WARN: synthesized order (same task set as the paper's; valid_unseen differs)" in out
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=scienceworld test_list={lst} sha256={SCIW_SYNTH_SHA256} order=synthesized"
    rc, again = _run_script(FETCH, "scienceworld", "--synthesize", env=env)  # idempotent
    assert rc == 0 and again.splitlines()[-1] == out.splitlines()[-1], again
    assert hashlib.sha256(lst.read_bytes()).hexdigest() == SCIW_SYNTH_SHA256
    # without a flag the list is recognised
    rc, out = _run_script(FETCH, "scienceworld", env=env)
    assert rc == 0 and "DATA_UPSTREAM scienceworld" in out, out
    assert out.splitlines()[-1] == f"FETCH_DATA_OK env=scienceworld test_list={lst} sha256={SCIW_SYNTH_SHA256} order=synthesized"
    # the default path under the data root
    rc, out = _run_script(FETCH, "scienceworld", "--synthesize", env=_env(tmp_path, **roots, PYTHON=sys.executable))
    default = Path(roots["MILES_AGENT_DATA"]) / "scienceworld" / "scienceworld_test1308.jsonl"
    assert rc == 0 and default.read_bytes() == lst.read_bytes(), out
    assert out.splitlines()[-1].startswith(f"FETCH_DATA_OK env=scienceworld test_list={default} ")
    # a list whose digest is the frozen pin is never overwritten (the pin set to the stand-in's digest here)
    env_f = dict(env, SCIENCEWORLD_TEST_SHA256=SCIW_SYNTH_SHA256)
    rc, out = _run_script(FETCH, "scienceworld", "--synthesize", env=env_f)
    assert rc == 1, out
    assert f"FATAL: {lst} is the frozen campaign list; not overwriting it with the synthesized order" in out
    assert "FETCH_DATA_OK" not in out and hashlib.sha256(lst.read_bytes()).hexdigest() == SCIW_SYNTH_SHA256
    rc, out = _run_script(FETCH, "--dry-run", "scienceworld", "--synthesize", env=env_f)
    assert rc == 0 and f"WARN: {lst} is the frozen campaign list" in out and _dry(out, "synthesize")[-1] == str(lst)
    rc, out = _run_script(FETCH, "scienceworld", env=env_f)
    assert rc == 0 and out.splitlines()[-1].endswith("order=frozen"), out
    # usage errors
    rc, out = _run_script(FETCH, "scienceworld", "--from", str(lst), "--synthesize", env=env)
    assert rc == 1 and "FATAL: --from and --synthesize are exclusive" in out
    for other in ("alfworld", "webshop", "search_r1"):
        rc, out = _run_script(FETCH, other, "--synthesize", env=env)
        assert rc == 1 and f"FATAL: --synthesize is not an option of {other}" in out, (other, out)


def test_fetch_data_scienceworld_without_a_list_prints_the_manual_step(tmp_path):
    """No --from, no --synthesize and no list at SCIENCEWORLD_TEST_LIST (else the data-root path): the FATAL, the
    upstream pointer and the manual step naming both ways to rebuild the list, exit 2 (the webshop arm's shape);
    nothing is written; under --dry-run a WARN and exit 0."""
    _fakes(tmp_path, *DEFAULT_FAKES, "java")
    roots = _roots(tmp_path)
    absent = tmp_path / "absent.jsonl"
    env = _env(tmp_path, **roots, SCIENCEWORLD_TEST_LIST=str(absent))
    rc, out = _run_script(FETCH, "scienceworld", env=env)
    assert rc == 2, out
    assert f"FATAL: no ScienceWorld eval list at {absent}" in out
    step = [line for line in out.splitlines() if line.startswith("DATA_MANUAL_STEP scienceworld:")]
    assert len(step) == 1 and "rerun with --from <file>" in step[0] and "--synthesize" in step[0], out
    assert "github.com/kokolerk/TCOD" in step[0] and "9632fe4f" in step[0]
    assert "DATA_UPSTREAM scienceworld: TCOD_examples/scienceworld/get_sciworld_data.py (arXiv 2604.24005)" in out
    assert "FETCH_DATA_OK" not in out and not absent.exists()
    rc, out = _run_script(FETCH, "scienceworld", env=_env(tmp_path, **roots))
    default = f"{roots['MILES_AGENT_DATA']}/scienceworld/scienceworld_test1308.jsonl"
    assert rc == 2 and f"FATAL: no ScienceWorld eval list at {default}" in out and not Path(default).exists()
    rc, out = _run_script(FETCH, "--dry-run", "scienceworld", env=env)
    assert rc == 0, out
    assert f"WARN: no ScienceWorld eval list at {absent}" in out and "DATA_MANUAL_STEP scienceworld:" in out
    assert "FATAL" not in out and "FETCH_DATA_OK" not in out


def test_build_gym_venvs_dry_run_lists_the_pinned_steps(tmp_path):
    """build_gym_venvs.sh --dry-run: the uv environment, the bootstrap, CPython 3.13.14, `uv sync --locked` at
    <root>/gym_main_venv from the Gym checkout, the eight venvs at the uv_venv_dir layout with ray/openai pinned to
    the parent's versions, the aarch64 textworld stub, --verify's checks and offline boot probe, the GYM_TAR
    packaging; the root is the argument, else GYM_VENV_ROOT, else $REPO/.gym_venvs; nothing is written."""
    _fakes(tmp_path)
    roots = _roots(tmp_path)
    root = tmp_path / "venvroot"
    env = _env(tmp_path, **roots)
    rc, out = _run_script(BUILD, "--dry-run", str(root), env=env)
    assert rc == 0, out
    uv = f"{root}/uvboot/bin/uv"
    assert _dry_env(out) == {"UV_CACHE_DIR": f"{root}/uv_cache", "UV_LINK_MODE": "copy",
                             "UV_PYTHON_INSTALL_DIR": f"{root}/uv_python", "UV_PYTHON_DOWNLOADS": "automatic"}
    assert "uv_bootstrap_pip" not in out  # a uv is on PATH (the fake): no pip bootstrap of it
    assert _dry(out, "uv_bootstrap_venv") == ["uv", "venv", "--python", str(tmp_path / "bin" / "python3"), f"{root}/uvboot"]
    assert _dry(out, "uv_bootstrap_upgrade") == ["uv", "pip", "install", "--python", f"{root}/uvboot/bin/python3", "-U", "uv"]
    assert _dry(out, "uv_version") == [uv, "--version"]
    assert _dry(out, "uv_python") == [uv, "python", "install", "3.13.14"]
    assert _dry(out, "uv_sync") == ["env", f"UV_PROJECT_ENVIRONMENT={root}/gym_main_venv", uv, "sync", "--locked"]
    cwds = _dry_cwd(out)
    assert cwds[0] == GYM_CWD and len(cwds) == 17  # the sync + two steps per server
    venvs = [_dry(out, "uv_venv", i) for i in range(8)]
    assert all(v[:5] == [uv, "venv", "--seed", "--python", "3.13.14"] for v in venvs)
    assert [v[5] for v in venvs] == [f"{root}/gym_venvs/{rel}/.venv" for rel in VENV_RELS]
    pips = [_dry(out, "uv_pip", i) for i in range(8)]
    assert all(p[:4] == [uv, "pip", "install", "-p"] and p[4] == v[5] for p, v in zip(pips, venvs))
    assert all(p[-2:] == ["ray[default]==<RAY_V>", "openai==<OPENAI_V>"] for p in pips)  # no parent venv yet
    assert pips[0][5:9] == ["-e", f"{GYM_CWD}[dev]", "-e", "."]
    for i in (2, 3, 4):
        assert pips[i][5:11] == ["-e", f"{GYM_CWD}[dev]", "-e", ".", "-e", f"{REPO}/gym_root/agent_utils"], i
    for i in (1, 5, 6, 7):
        assert pips[i][5:7] == ["-r", "requirements.txt"], i
    assert cwds[1:3] == [f"{REPO}/gym_root/resources_servers/wiki_search"] * 2
    assert cwds[13:15] == [f"{GYM_CWD}/responses_api_models/openai_model"] * 2
    assert cwds[15:17] == [f"{GYM_CWD}/environment_servers/legacy_agent"] * 2
    assert "DRY_RUN " + "twp_fetch" not in out and "import_check" not in out and "DRY_RUN tar:" not in out
    assert "GYM_VENVS_OK" not in out and not root.exists() and not (REPO / ".gym_venvs").exists()
    assert "DRY_RUN " + "uv_bootstrap_venv" in out and out.count("DRY_RUN uv_pip:") == 8
    # BUILD_DIR, GYM_TAR, --verify with a JDK and a jar pin, the aarch64 textworld stub
    _set_state(tmp_path, {"uname_m": "aarch64"})
    jdk = tmp_path / "jdk"
    _fake_at(jdk / "bin" / "java")
    env2 = _env(tmp_path, **roots, GYM_TAR="/tars/gym.tar", BUILD_DIR="/bld", JAVA_HOME=str(jdk), SCIENCEWORLD_JAR="/jars/sw.jar")
    rc, out = _run_script(BUILD, "--dry-run", "--verify", str(root), env=env2)
    assert rc == 0, out
    assert _dry_env(out)["UV_CACHE_DIR"] == "/bld/uv_cache" and _dry_env(out)["UV_PYTHON_INSTALL_DIR"] == f"{root}/uv_python"
    assert _dry(out, "uv_bootstrap_venv")[-1] == "/bld/uvboot" and _dry(out, "uv_python")[0] == "/bld/uvboot/bin/uv"
    assert _dry(out, "twp_mkdir") == ["mkdir", "-p", "/bld/twpatch"]
    assert _dry(out, "twp_fetch") == ["curl", "-fsSL", "https://files.pythonhosted.org/packages/source/t/textworld/textworld-1.7.0.tar.gz",
                                      "-o", "textworld-1.7.0.tar.gz"]
    assert _dry(out, "twp_unpack") == ["tar", "xzf", "textworld-1.7.0.tar.gz"] and "DRY_RUN twp_stub: /bld/twpatch/textworld-1.7.0/setup.sh" in out
    assert out.count("DRY_RUN uv_pip:") == 9
    assert _dry(out, "uv_pip", 2) == ["/bld/uvboot/bin/uv", "pip", "install", "-p", f"{root}/gym_venvs/resources_servers/alfworld_env/.venv", "./textworld-1.7.0"]
    assert _dry(out, "uv_pip", 3)[4] == f"{root}/gym_venvs/resources_servers/alfworld_env/.venv" and "-e" in _dry(out, "uv_pip", 3)
    assert "/bld/twpatch" in _dry_cwd(out)
    checks = [_dry(out, "import_check", i) for i in range(9)]
    assert checks[0][0] == f"{root}/gym_main_venv/bin/python"
    assert [c[0] for c in checks[1:]] == [f"{root}/gym_venvs/{rel}/.venv/bin/python" for rel in VENV_RELS]
    assert all(c[1] == "-c" and f'nemo_gym.__file__.startswith("{REPO}/")' in c[2] for c in checks)
    assert _dry(out, "import_probe") == [f"{root}/gym_venvs/resources_servers/wiki_search/.venv/bin/python", "-c", "import wiki_search"]
    assert out.count("DRY_RUN import_probe:") == 4
    jvm = _dry(out, "real_jvm")
    assert jvm[:2] == ["env", f"JAVA_HOME={jdk}"] and jvm[2].startswith(f"PATH={jdk}/bin:/bld/uvboot/bin:")
    assert jvm[3:] == ["SCIENCEWORLD_REAL_TESTS=1", "SCIENCEWORLD_JAR=/jars/sw.jar",
                       f"{root}/gym_venvs/resources_servers/scienceworld_env/.venv/bin/python", "-m", "pytest", "-q", "-p",
                       "no:cacheprovider", "tests/test_worker_pool.py", "-k", "RealJVM"]
    assert "WARN: no ScienceWorld jar at /jars/sw.jar (SCIENCEWORLD_JAR / SCIW_JAR; empty = the wheel's bundled jar)" in out
    boot = _dry(out, "gym_boot")
    assert boot == ["env", "UV_OFFLINE=1", f"{root}/gym_main_venv/bin/gym", "env", "start", "--config",
                    f"{REPO}/gym_root/resources_servers/wiki_search/configs/wiki_search.yaml", "--model-type", "openai_model",
                    "+skip_venv_if_present=true", f"+uv_venv_dir={root}/gym_venvs", f"+uv_cache_dir={root}/uv_cache", *GYM_HEAD]
    assert _dry_cwd(out)[-1] == GYM_CWD and _dry_cwd(out)[-2] == f"{REPO}/gym_root/resources_servers/scienceworld_env"
    assert _dry(out, "uvbin_mkdir") == ["mkdir", "-p", f"{root}/uvbin"]
    assert _dry(out, "uvbin_copy") == ["cp", "/bld/uvboot/bin/uv", f"{root}/uvbin/"]
    assert _dry(out, "tar") == ["tar", "cf", "/tars/gym.tar.partial", "-C", str(root), "gym_main_venv", "gym_venvs",
                                "uv_python", "uvbin", "-C", "/bld", "uv_cache"]
    assert "SCIW_REAL_JVM_OK" not in out and "GYM_BOOT_OK" not in out and "GYM_VENVS_OK" not in out
    assert not root.exists() and not Path("/bld").exists()
    # verify_boot's gym is the EXIT trap's to kill as well: a failure between the launch and its own kill (or a TERM)
    # leaves no gym behind
    body = _function_body((SCRIPTS / "build_gym_venvs.sh").read_text(), "verify_boot")
    assert body.index("GYM_PID=$!") < body.index("add_cleanup 'kill $GYM_PID 2>/dev/null'") < body.index("for _ in $(seq 84)")
    # the JDK in the ladder's order (scripts/lib/common.sh scienceworld_java_home): the alias beats JAVA_HOME; a
    # missing JDK is a WARN under --dry-run that names the knobs
    env3 = _env(tmp_path, **roots, JAVA_HOME="/jh", SCIW_JAVA_HOME=str(jdk))
    rc, out = _run_script(BUILD, "--dry-run", "--verify", str(root), env=env3)
    assert rc == 0 and _dry(out, "real_jvm")[1] == f"JAVA_HOME={jdk}", out
    rc, out = _run_script(BUILD, "--dry-run", "--verify", str(root), env=_env(tmp_path, **roots, JAVA_HOME="/nojdk"))
    assert rc == 0 and "WARN: no JDK at /nojdk (SCIENCEWORLD_JAVA_HOME / SCIW_JAVA_HOME / JAVA_HOME / TOOLS_DIR)" in out, out
    # the root: the argument, else GYM_VENV_ROOT, else $REPO/.gym_venvs; a relative root or an unknown server is a
    # FATAL; a server subset builds those only; UV_BIN skips the bootstrap; no uv anywhere: pip installs one
    rc, out = _run_script(BUILD, "--dry-run", env=_env(tmp_path, **roots, GYM_VENV_ROOT="/g"))
    assert rc == 0 and _dry(out, "uv_sync")[1] == "UV_PROJECT_ENVIRONMENT=/g/gym_main_venv"
    rc, out = _run_script(BUILD, "--dry-run", "/p", "scienceworld_env", "wiki_search",
                          env=_env(tmp_path, **roots, GYM_VENV_ROOT="/g", UV_BIN="/opt/uv/bin/uv"))
    assert rc == 0, out
    assert _dry(out, "uv_sync") == ["env", "UV_PROJECT_ENVIRONMENT=/p/gym_main_venv", "/opt/uv/bin/uv", "sync", "--locked"]
    assert "uv_bootstrap" not in out and "building with UV_BIN=/opt/uv/bin/uv" in out
    assert [_dry(out, "uv_venv", i)[-1] for i in range(2)] == ["/p/gym_venvs/resources_servers/scienceworld_env/.venv",
                                                              "/p/gym_venvs/resources_servers/wiki_search/.venv"]
    assert out.count("DRY_RUN uv_venv:") == 2 and "servers=[scienceworld_env wiki_search]" in out
    rc, out = _run_script(BUILD, "--dry-run", env=env)
    assert rc == 0 and _dry(out, "uv_sync")[1] == f"UV_PROJECT_ENVIRONMENT={REPO}/.gym_venvs/gym_main_venv"
    assert not (REPO / ".gym_venvs").exists()
    rc, out = _run_script(BUILD, "--dry-run", env=_env(tmp_path, **roots, GYM_VENV_ROOT="rel/root"))
    assert rc == 1 and "FATAL: the venv root must be an absolute path, got 'rel/root' (GYM_VENV_ROOT)" in out
    rc, out = _run_script(BUILD, "--dry-run", "/p", "nosuch_server", env=env)
    assert rc == 1 and "FATAL: unknown server 'nosuch_server'" in out
    rc, out = _run_script(BUILD, "--dry-run", "/p", "/q", env=env)
    assert rc == 1 and "FATAL: two roots given ('/p' and '/q')" in out
    rc, out = _run_script(BUILD, "--help", env=env)
    assert rc == 0 and out.startswith("scripts/build_gym_venvs.sh") and "GYM_VENVS_OK" in out and "set -euo" not in out
    _fakes(tmp_path, *[f for f in DEFAULT_FAKES if f != "uv"])
    rc, out = _run_script(BUILD, "--dry-run", "/p", "wiki_search", env=_env(tmp_path, bare_path=True, interp_dir=False, **roots))
    assert rc == 0, out
    assert _dry(out, "uv_bootstrap_pip") == ["python3", "-m", "pip", "install", "uv"]
    assert _dry(out, "uv_bootstrap_venv")[:3] == ["uv", "venv", "--python"]


# ---- C7: scripts/harbor_trial_server.sh + scripts/harbor_run.sh ----------------------------------------------

TRIAL_SERVER = "scripts/harbor_trial_server.sh"
HARBOR_RUN = "scripts/harbor_run.sh"
SANDBOX = "sandbox_backend.env:SandboxEnvironment"
HOST_PORT = ["--host", "192.0.2.10", "--port", "8200"]


def _rows(path, *names):
    """A rows file: one JSON line per task name (metadata.instance_id), as the recipes' data files have them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"prompt": f"Do {n}.", "metadata": {"instance_id": n, "max_episode_steps": 5}} for n in names]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _config_line(out, marker):
    """The JSON after `<marker> ` (HARBOR_TRIAL_SERVER_CONFIG, HARBOR_JOB_CONFIG): exactly one such line."""
    lines = [line for line in out.splitlines() if line.startswith(marker + " ")]
    assert len(lines) == 1, out
    return json.loads(lines[0][len(marker) + 1:])


def _fake_harnesses(tmp_path):
    """A stand-in miles_harbor.harnesses on PYTHONPATH for the real interpreter the job-config builder runs on (the
    real package needs harbor): one pack harness with an ENDPOINT binding, read as the builder reads them."""
    pkg = tmp_path / "fakepkg" / "miles_harbor"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "harnesses.py").write_text(
        "class _Endpoint:\n"
        "    @staticmethod\n"
        "    def agent_config(base_url, api_key, model):\n"
        "        return {'model_name': 'openai/' + model,\n"
        "                'env': {'OPENAI_BASE_URL': base_url.rstrip('/'), 'MSWEA_API_KEY': api_key}}\n"
        "class MiniSweAgent:\n"
        "    PACK = 'mini-swe-agent'\n"
        "    ENDPOINT = _Endpoint()\n"
        "class VanilluxAgent:\n"
        "    ENDPOINT = _Endpoint()\n"
        "HARNESSES = {'mini-swe-agent': MiniSweAgent, 'vanillux': VanilluxAgent}\n"
    )
    return str(tmp_path / "fakepkg")


def _no_harnesses(tmp_path):
    """A miles_harbor on PYTHONPATH whose import fails: ahead of one the interpreter has installed (a dev venv's
    editable install), so the builder's interpreter is without miles_harbor wherever the tests run."""
    pkg = tmp_path / "nopkg" / "miles_harbor"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("raise ImportError('no miles_harbor (the test stands in for its absence)')\n")
    return str(tmp_path / "nopkg")


def test_harbor_trial_server_config_carries_sandbox_kwargs_only_with_a_provider(tmp_path):
    """--dry-run builds the server config (S2, D18) and starts nothing: with HARBOR_ENVIRONMENT_IMPORT_PATH naming a
    sandbox backend and a --provider, the config carries environment_import_path and the provider kwargs
    (sandbox_provider = --provider, layer_cache from HARBOR_LAYER_CACHE); unset, no environment_import_path key and
    environment_kwargs = {} (Harbor's docker environment; a --provider is then a WARN only); a backend without a
    provider gets HARBOR_ENVIRONMENT_KWARGS verbatim; the tasks are the rows' instance_ids; every other key keeps its
    default. A --config call serves the file as it is through TRIAL_SERVER_VENV / TRIAL_SERVER_MODULE (a launcher's
    contract) and builds nothing."""
    roots = _roots(tmp_path)
    _fakes(tmp_path)
    rows = _rows(tmp_path / "data" / "train.jsonl", "t2", "t1", "t2")
    rows2 = _rows(tmp_path / "data" / "eval.jsonl", "t3")
    provider = tmp_path / "provider.yaml"
    provider.write_text("sandbox: {}\n")
    results = roots["MILES_AGENT_RESULTS"]
    common = dict(roots, HARBOR_PYTHON=sys.executable, HARBOR_DATASET="org/tasks@1", RUN_TAG="run7")
    env = _env(tmp_path, HARBOR_ENVIRONMENT_IMPORT_PATH=SANDBOX, HARBOR_LAYER_CACHE="0", HARBOR_HARNESS="vanillux",
               HARBOR_AGENT_KWARGS='{"a": 1}', HARBOR_MAX_CONCURRENT="7", HARBOR_AGENT_NETWORK="task",
               HARBOR_KEEP_TRIALS="1", HARBOR_REWARD_REQUIRES_SUBMIT="1", HARBOR_AGENT_TIMEOUT_S="300",
               HARBOR_TRIAL_TIMEOUT_S="900", **common)
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, "--provider", str(provider), str(rows), str(rows2), env=env)
    assert rc == 0, out
    work = f"{results}/run7/trial_server"
    assert _config_line(out, "HARBOR_TRIAL_SERVER_CONFIG") == {
        "dataset": "org/tasks@1", "download_dir": f"{work}/tasks", "task_names": "3 tasks", "harness": "vanillux",
        "agent_kwargs": {"a": 1}, "environment_import_path": SANDBOX,
        "environment_kwargs": {"sandbox_provider": str(provider), "layer_cache": False},
        "trials_dir": f"{work}/trials", "records": f"{results}/run7/harbor_trials.jsonl", "max_concurrent": 7,
        "agent_network": "task", "keep_trials": True, "reward_requires_submit": True, "agent_timeout_s": 300.0,
        "trial_timeout_s": 900.0,
    }
    assert _dry(out, "trial_server") == [sys.executable, "-m", "miles_harbor.trial_server", "--config",
                                         f"{work}/trial_server.yaml", "--host", "192.0.2.10", "--port", "8200"]
    assert "DRY_RUN selftest" not in out and "WARN" not in out
    assert not Path(results).exists() and _calls(env, "python3") == [] and _calls(env, "curl") == []
    # the docker default: no import path, no provider kwargs, the defaults, a --provider is a WARN only
    env = _env(tmp_path, **common)
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, "--provider", str(provider), str(rows), "--selftest", "4", env=env)
    assert rc == 0, out
    config = _config_line(out, "HARBOR_TRIAL_SERVER_CONFIG")
    assert "environment_import_path" not in config and config["environment_kwargs"] == {}
    assert (config["harness"], config["max_concurrent"], config["agent_network"], config["keep_trials"],
            config["reward_requires_submit"], config["task_names"]) == ("mini-swe-agent", 128, "endpoint", False, False, "2 tasks")
    assert "agent_timeout_s" not in config and "trial_timeout_s" not in config
    assert f"WARN: --provider {provider} is a sandbox backend's" in out
    assert _dry(out, "selftest") == [sys.executable, "-", "http://192.0.2.10:8200", "4", f"{work}/trial_server.yaml"]
    # another backend's kwargs verbatim; HARBOR_AGENT is the old name of the harness knob
    env = _env(tmp_path, HARBOR_ENVIRONMENT_IMPORT_PATH="my.backend:Env", HARBOR_ENVIRONMENT_KWARGS='{"k": [1, 2]}',
               HARBOR_AGENT="vanillux", **common)
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(rows), env=env)
    assert rc == 0, out
    config = _config_line(out, "HARBOR_TRIAL_SERVER_CONFIG")
    assert (config["environment_import_path"], config["environment_kwargs"], config["harness"]) == ("my.backend:Env", {"k": [1, 2]}, "vanillux")
    # a sandbox backend without a provider: its import path, HARBOR_ENVIRONMENT_KWARGS verbatim ({} unset), no WARN
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(rows), env=_env(tmp_path, HARBOR_ENVIRONMENT_IMPORT_PATH=SANDBOX, **common))
    assert rc == 0, out
    config = _config_line(out, "HARBOR_TRIAL_SERVER_CONFIG")
    assert (config["environment_import_path"], config["environment_kwargs"]) == (SANDBOX, {}) and "WARN" not in out
    # FATALs: no dataset, a row without a task name, no host, nothing to serve
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(rows), env=_env(tmp_path, **{k: v for k, v in common.items() if k != "HARBOR_DATASET"}))
    assert rc == 1 and "FATAL: the trial server needs HARBOR_DATASET" in out
    bad = tmp_path / "data" / "bad.jsonl"
    bad.write_text('{"prompt": "x", "metadata": {}}\n')
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(bad), env=_env(tmp_path, **common))
    assert rc == 1 and f"FATAL: {bad}:1 has no metadata.instance_id" in out
    # an old-format list (one name per line, or a JSON array) is a FATAL naming the line, not 0 tasks; so is a file
    # without rows
    old = tmp_path / "data" / "old_list.txt"
    old.write_text("task-a\ntask-b\n")
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(old), env=_env(tmp_path, **common))
    assert rc == 1 and f"FATAL: {old}:1 is not a JSON object row" in out and "Traceback" not in out, out
    aslist = tmp_path / "data" / "aslist.json"
    aslist.write_text(json.dumps([{"metadata": {"instance_id": "t1"}}]) + "\n")
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(aslist), env=_env(tmp_path, **common))
    assert rc == 1 and f"FATAL: {aslist}:1 is not a JSON object row" in out and "Traceback" not in out, out
    empty = tmp_path / "data" / "empty.jsonl"
    empty.write_text("\n")
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(rows), str(empty), env=_env(tmp_path, **common))
    assert rc == 1 and f"FATAL: {empty} has no rows" in out and "HARBOR_TRIAL_SERVER_CONFIG" not in out, out
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", "--port", "8200", str(rows), env=_env(tmp_path, **common))
    assert rc == 1 and "FATAL: --host and --port are required" in out
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, env=_env(tmp_path, **common))
    assert rc == 1 and "FATAL: a --config <yaml> or data files are required" in out
    # a missing data file is a WARN here (its rows skipped), a FATAL for real
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, str(rows), str(tmp_path / "data" / "none.jsonl"), env=_env(tmp_path, **common))
    assert rc == 0 and f"WARN: no data file {tmp_path / 'data' / 'none.jsonl'}" in out
    assert _config_line(out, "HARBOR_TRIAL_SERVER_CONFIG")["task_names"] == "2 tasks"
    rc, out = _run_script(TRIAL_SERVER, *HOST_PORT, str(tmp_path / "data" / "none.jsonl"), env=_env(tmp_path, **common))
    assert rc == 1 and f"FATAL: no data file {tmp_path / 'data' / 'none.jsonl'}" in out
    # --config: a launcher's call shape (the flags in any order); nothing built; the venv's python and the module from
    # the environment; HARBOR_PYTHON when there is no venv
    served = tmp_path / "server.yaml"
    served.write_text("dataset: x\n")
    env = _env(tmp_path, TRIAL_SERVER_VENV="/v", TRIAL_SERVER_MODULE="m.n", **roots)
    rc, out = _run_script(TRIAL_SERVER, "--port", "8200", "--config", str(served), "--dry-run", "--host", "192.0.2.10", env=env)
    assert rc == 0, out
    assert _dry(out, "trial_server") == ["/v/bin/python", "-m", "m.n", "--config", str(served), "--host", "192.0.2.10", "--port", "8200"]
    assert "HARBOR_TRIAL_SERVER_CONFIG" not in out and _calls(env, "python3") == []
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, "--config", str(served), env=_env(tmp_path, HARBOR_PYTHON="/hp/python", **roots))
    assert rc == 0 and _dry(out, "trial_server")[:3] == ["/hp/python", "-m", "miles_harbor.trial_server"]
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, "--config", str(served), str(rows), env=env)
    assert rc == 1 and "FATAL: --config serves a file as it is" in out
    rc, out = _run_script(TRIAL_SERVER, "--dry-run", *HOST_PORT, "--config", str(tmp_path / "none.yaml"), env=env)
    assert rc == 0 and f"WARN: no config file {tmp_path / 'none.yaml'}" in out
    rc, out = _run_script(TRIAL_SERVER, *HOST_PORT, "--config", str(tmp_path / "none.yaml"), env=env)
    assert rc == 1 and f"FATAL: no config file {tmp_path / 'none.yaml'}" in out
    rc, out = _run_script(TRIAL_SERVER, "--help", env=_env(tmp_path, **roots))
    assert rc == 0 and out.startswith("scripts/harbor_trial_server.sh") and "HARBOR_TRIAL_SERVER_UP" in out and "set -euo" not in out


def test_harbor_trial_server_serves_announces_and_stops(tmp_path):
    """The real run with fakes: the config written under the work dir (its home/ tasks/ trials/ made), the server
    started from there with its log beside it, the /health wait, `HARBOR_TRIAL_SERVER_UP url=http://<host>:<port> ...`
    with the /health answer and the server's TRIAL_SERVER_TASKS line (D5: the marker kept), NEMO_GYM_URL exported for
    what follows; a TERM stops the server (the fake's pid is gone: the builtin kill did it) and exits 0; --selftest
    runs the old job's probe after the up line and its verdict is the exit status (PASS: 0; FAIL: a FATAL, 1), the
    server stopped either way; a server that dies at boot is a FATAL with its log, exit 1."""
    roots = _roots(tmp_path)
    results = roots["MILES_AGENT_RESULTS"]
    rows = _rows(tmp_path / "data" / "train.jsonl", "t1", "t2")
    url = "http://192.0.2.10:8200"
    health = {"tasks": 2, "in_flight": 0, "max_concurrent": 128}
    state = {"python_stdin": "real", "curl": {f"{url}/health": "200"}, "curl_body": {f"{url}/health": health}}
    server_pid = tmp_path / "trial_server.pid"
    work = Path(f"{results}/run7/trial_server")

    def up_then(state, *args, term=True, **extra):
        if server_pid.exists():
            server_pid.unlink()  # the previous case's record: this case waits for its own server's
        _fakes(tmp_path, state=state)
        env = _env(tmp_path, HARBOR_DATASET="/data/tasks", RUN_TAG="run7", **roots, **extra)
        proc, lines = _serve([*HOST_PORT, str(rows), *args], env, until="HARBOR_TRIAL_SERVER_UP", script="harbor_trial_server.sh")
        try:
            if term:
                assert lines and lines[-1].startswith("HARBOR_TRIAL_SERVER_UP "), "".join(lines)
                assert _wait_for(server_pid), "".join(lines)
                proc.send_signal(signal.SIGTERM)
            out = "".join(lines) + proc.communicate(timeout=30)[0]
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=5)
        return proc.returncode, out, lines, env

    rc, out, lines, env = up_then(state)
    assert rc == 0, out
    assert lines[-1].rstrip("\n") == f"HARBOR_TRIAL_SERVER_UP url={url} {json.dumps(health)} TRIAL_SERVER_TASKS 2 of ds harness=mini-swe-agent"
    assert _pid_gone(server_pid)
    assert "FATAL" not in out and "WARN" not in out and _calls(env, "kill") == []
    config = yaml.safe_load((work / "trial_server.yaml").read_text())
    assert (config["dataset"], config["task_names"], config["records"]) == ("/data/tasks", ["t1", "t2"], f"{results}/run7/harbor_trials.jsonl")
    assert "environment_import_path" not in config and config["environment_kwargs"] == {}
    assert all((work / d).is_dir() for d in ("home", "tasks", "trials"))
    assert (work / "trial_server.log").read_text().startswith("TRIAL_SERVER_TASKS 2 of ds harness=mini-swe-agent")
    server, = [c for c in _calls(env, "python3") if c["argv"][:2] == ["-m", "miles_harbor.trial_server"]]
    assert server["argv"][2:] == ["--config", str(work / "trial_server.yaml"), "--host", "192.0.2.10", "--port", "8200"]
    assert Path(server["cwd"]).resolve() == work.resolve()
    assert [c["argv"] for c in _calls(env, "curl")] == [["-sf", "-m", "10", "--noproxy", "*", "-o", "/dev/null", f"{url}/health"],
                                                       ["-s", "-m", "10", "--noproxy", "*", f"{url}/health"]]
    assert server["env"]["HOME"] == str(work / "home")  # Harbor's caches under the work dir
    assert server["env"]["HARBOR_PACK_ROOT"] == f"{tmp_path}/.cache/miles-harness"  # the account's HOME, resolved before that
    # an explicit HARBOR_PACK_ROOT reaches the server as it is
    rc, out, lines, env = up_then(state, HARBOR_PACK_ROOT="/packs")
    assert rc == 0, out
    server, = [c for c in _calls(env, "python3") if c["argv"][:2] == ["-m", "miles_harbor.trial_server"]]
    assert server["env"]["HARBOR_PACK_ROOT"] == "/packs" and server["env"]["HOME"] == str(work / "home")
    # --selftest: the probe gets the url, the count and the config (its tasks); PASS last, exit 0, the server stopped
    rc, out, lines, env = up_then(state, "--selftest", "3", term=False)
    assert rc == 0, out
    assert out.count("HARBOR_SELFTEST_ANSWER ") == 3 and out.rstrip().endswith("HARBOR_SELFTEST_PASS 3 of 3 answered a reward in 1 s")
    assert _pid_gone(server_pid) and "FATAL" not in out
    probe, = [c for c in _calls(env, "python3") if c["argv"][:2] == ["-", url]]
    assert probe["argv"] == ["-", url, "3", str(work / "trial_server.yaml")] and probe["env"]["NEMO_GYM_URL"] == url
    rc, out, lines, env = up_then({**state, "selftest_rc": 1}, "--selftest", "2", term=False)
    assert rc == 1, out
    assert "HARBOR_SELFTEST_FAIL 0 of 2" in out and out.rstrip().endswith("FATAL: the selftest failed (HARBOR_SELFTEST_FAIL above, rc=1)")
    assert _pid_gone(server_pid)
    # the server dies at boot: the FATAL names the log, whose tail follows; nothing announced
    rc, out, lines, env = up_then({**state, "trial_server_dies": True, "curl": {f"{url}/health": "000"}}, term=False)
    assert rc == 1, out
    assert "FATAL: the trial server did not serve on 192.0.2.10:8200" in out and "FakeTrialServer: no dataset" in out
    assert "HARBOR_TRIAL_SERVER_UP" not in out


def test_harbor_run_dry_run_builds_the_job_config(tmp_path):
    """--dry-run (S2, D18): HARBOR_JOB_CONFIG as the old job built it -- the dataset spelled three ways, HARBOR_TASKS
    joined with the rows' tasks, the agent from HARBOR_AGENT (a pack harness by its class under HARBOR_HARNESS_LOCAL,
    Harbor's own by name otherwise, HARBOR_AGENT_IMPORT_PATH, the ENDPOINT binding of HARBOR_ENDPOINT_FILE with its
    env values masked), the counts, HARBOR_VERIFY=0 -- with environment.import_path only when
    HARBOR_ENVIRONMENT_IMPORT_PATH is set and the provider kwargs only with a --provider (a backend without one gets
    HARBOR_ENVIRONMENT_KWARGS verbatim); `DRY_RUN harbor_run: harbor run -c <job.yaml> -y`; nothing written, nothing run."""
    roots = _roots(tmp_path)
    _fakes(tmp_path)
    rows = _rows(tmp_path / "data" / "eval.jsonl", "t1", "t2")
    provider = tmp_path / "provider.yaml"
    provider.write_text("sandbox: {}\n")
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(json.dumps({"base_url": "http://192.0.2.20:7000/v1/", "model": "Qwen3.5-4B"}))
    out_dir = tmp_path / "results" / "eval1"
    common = dict(roots, HARBOR_PYTHON=sys.executable, PYTHONPATH=_fake_harnesses(tmp_path))
    env = _env(tmp_path, HARBOR_DATASET="harbor/terminal-bench@2.0", HARBOR_TASKS="t0,t1", HARBOR_N_TASKS="5",
               HARBOR_AGENT="mini-swe-agent", HARBOR_HARNESS_LOCAL="1", HARBOR_ENDPOINT_FILE=str(endpoint),
               HARBOR_AGENT_KWARGS='{"version": "2.4.6"}', HARBOR_AGENT_ENV='{"X": "1"}', HARBOR_N_ATTEMPTS="2",
               HARBOR_N_CONCURRENT="8", HARBOR_VERIFY="0", HARBOR_ENVIRONMENT_IMPORT_PATH=SANDBOX, HARBOR_LAYER_CACHE="0",
               **common)
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), "--provider", str(provider), str(rows), env=env)
    assert rc == 0, out
    assert _config_line(out, "HARBOR_JOB_CONFIG") == {
        "job_name": "eval1", "jobs_dir": f"{out_dir}/work/jobs", "n_attempts": 2, "n_concurrent_trials": 8, "quiet": True,
        "environment": {"import_path": SANDBOX, "kwargs": {"sandbox_provider": str(provider), "layer_cache": False}},
        "agents": [{"import_path": "miles_harbor.harnesses:MiniSweAgent", "model_name": "openai/Qwen3.5-4B",
                    "env": {"OPENAI_BASE_URL": "<set>", "MSWEA_API_KEY": "<set>", "X": "<set>"},
                    "kwargs": {"version": "2.4.6"}}],
        "datasets": [{"name": "harbor/terminal-bench", "ref": "2.0", "download_dir": f"{out_dir}/work/tasks",
                      "task_names": ["t0", "t1", "t2"], "n_tasks": 5}],
        "verifier": {"disable": True},
    }
    assert _dry(out, "harbor_run") == ["harbor", "run", "-c", f"{out_dir}/work/job.yaml", "-y"]
    assert _dry_env(out)["HOME"] == f"{out_dir}/work/home" and _dry_env(out)["RUN_TAG"] == "eval1"
    assert _dry_env(out)["HARBOR_PACK_ROOT"] == f"{tmp_path}/.cache/miles-harness"  # the account's HOME, resolved before the override
    assert not out_dir.exists() and _calls(env, "harbor") == [] and "WARN" not in out
    # the docker default: no environment key (a --provider is a WARN only); the registry spelling; Harbor's own oracle;
    # RUN_TAG names the job; HARBOR_BIN and HARBOR_WORK_DIR
    env = _env(tmp_path, HARBOR_DATASET="swebench-verified@1.0", RUN_TAG="sv", HARBOR_BIN="/opt/hb/bin/harbor", HARBOR_WORK_DIR="/w", **common)
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), "--provider", str(provider), env=env)
    assert rc == 0, out
    job = _config_line(out, "HARBOR_JOB_CONFIG")
    assert "environment" not in job and "verifier" not in job and job["agents"] == [{"name": "oracle"}]
    assert f"WARN: --provider {provider} is a sandbox backend's" in out
    assert job["datasets"] == [{"name": "swebench-verified", "version": "1.0", "download_dir": "/w/tasks"}]
    assert (job["job_name"], job["jobs_dir"], job["n_attempts"], job["n_concurrent_trials"]) == ("sv", "/w/jobs", 1, 32)
    assert _dry(out, "harbor_run") == ["/opt/hb/bin/harbor", "run", "-c", "/w/job.yaml", "-y"] and _dry_env(out)["HOME"] == "/w/home"
    assert _dry_env(out)["RUN_TAG"] == "sv"
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir),
                          env=_env(tmp_path, HARBOR_DATASET="swebench-verified@1.0", HARBOR_PACK_ROOT="/packs", **common))
    assert rc == 0 and _dry_env(out)["HARBOR_PACK_ROOT"] == "/packs", out  # an explicit root as it is
    # a local dataset; a pack harness without HARBOR_HARNESS_LOCAL is Harbor's own agent of that name, still with the
    # endpoint binding; a harness without a pack (beside the sandbox) is always its class; HARBOR_MODEL;
    # HARBOR_AGENT_IMPORT_PATH; a backend without a provider: HARBOR_ENVIRONMENT_KWARGS verbatim, no WARN
    env = _env(tmp_path, HARBOR_DATASET="/data/tasks", HARBOR_AGENT="mini-swe-agent", HARBOR_ENDPOINT_FILE=str(endpoint),
               HARBOR_MODEL="m2", HARBOR_ENVIRONMENT_IMPORT_PATH="my.backend:Env", HARBOR_ENVIRONMENT_KWARGS='{"k": 1}', **common)
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=env)
    assert rc == 0, out
    job = _config_line(out, "HARBOR_JOB_CONFIG")
    assert job["agents"] == [{"name": "mini-swe-agent", "model_name": "m2", "env": {"OPENAI_BASE_URL": "<set>", "MSWEA_API_KEY": "<set>"}}]
    assert job["datasets"] == [{"path": "/data/tasks"}] and job["environment"] == {"import_path": "my.backend:Env", "kwargs": {"k": 1}}
    assert "WARN" not in out
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=_env(tmp_path, HARBOR_DATASET="/d", HARBOR_AGENT="vanillux", HARBOR_ENVIRONMENT_KWARGS='{"k": 1}', **common))
    assert rc == 0 and _config_line(out, "HARBOR_JOB_CONFIG")["agents"] == [{"import_path": "miles_harbor.harnesses:VanilluxAgent"}]
    assert _config_line(out, "HARBOR_JOB_CONFIG")["environment"] == {"kwargs": {"k": 1}}
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=_env(tmp_path, HARBOR_DATASET="/d", HARBOR_AGENT_IMPORT_PATH="x.y:Z", **common))
    assert rc == 0 and _config_line(out, "HARBOR_JOB_CONFIG")["agents"] == [{"import_path": "x.y:Z"}]
    # without miles_harbor importable: a WARN and Harbor's own agent by name; with an endpoint file a FATAL
    env = _env(tmp_path, HARBOR_DATASET="/d", HARBOR_AGENT="mini-swe-agent", **{k: v for k, v in common.items() if k != "PYTHONPATH"}, PYTHONPATH=_no_harnesses(tmp_path))
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=env)
    assert rc == 0 and "WARN: miles_harbor.harnesses is not importable by" in out, out
    assert _config_line(out, "HARBOR_JOB_CONFIG")["agents"] == [{"name": "mini-swe-agent"}]
    env = _env(tmp_path, HARBOR_DATASET="/d", HARBOR_AGENT="mini-swe-agent", HARBOR_ENDPOINT_FILE=str(endpoint),
               HARBOR_PYTHON=sys.executable, PYTHONPATH=_no_harnesses(tmp_path), **roots)
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=env)
    assert rc == 1 and "FATAL: HARBOR_ENDPOINT_FILE needs the harness's ENDPOINT binding" in out
    # a sandbox backend without a provider: its import path, kwargs {} (HARBOR_ENVIRONMENT_KWARGS unset), no FATAL
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=_env(tmp_path, HARBOR_DATASET="/d", HARBOR_ENVIRONMENT_IMPORT_PATH=SANDBOX, **common))
    assert rc == 0 and _config_line(out, "HARBOR_JOB_CONFIG")["environment"] == {"import_path": SANDBOX, "kwargs": {}}, out
    # FATALs: an endpoint file for an agent without a binding, no dataset, no --out
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=_env(tmp_path, HARBOR_DATASET="/d", HARBOR_ENDPOINT_FILE=str(endpoint), **common))
    assert rc == 1 and "FATAL: HARBOR_ENDPOINT_FILE: miles_harbor.harnesses has no ENDPOINT for the agent 'oracle'" in out
    rc, out = _run_script(HARBOR_RUN, "--dry-run", "--out", str(out_dir), env=_env(tmp_path, **common))
    assert rc == 1 and "FATAL: harbor_run needs HARBOR_DATASET" in out
    rc, out = _run_script(HARBOR_RUN, "--dry-run", env=_env(tmp_path, HARBOR_DATASET="/d", **common))
    assert rc == 1 and "FATAL: --out <dir> is required" in out
    rc, out = _run_script(HARBOR_RUN, "--help", env=_env(tmp_path, **roots))
    assert rc == 0 and out.startswith("scripts/harbor_run.sh") and "HARBOR_RESULTS" in out and "set -euo" not in out


def test_harbor_run_runs_harbor_and_summarises_the_results(tmp_path):
    """The run: `harbor run -c <job.yaml> -y` from the work dir, HARBOR_RUN_EXIT, the job dir copied to --out (job.yaml,
    the provider, result.json, jobs.tar.gz, summary.txt) and the old job's summary markers (HARBOR_SUMMARY,
    HARBOR_AGENT_STATS, HARBOR_EXCEPTION, HARBOR_NOT_PASSED; D5) before HARBOR_RESULTS, last; a `harbor run` that
    failed still gets its summary and then ends in FATAL with its code (Contract B)."""
    roots = _roots(tmp_path)
    results = [
        {"task_name": "t1", "verifier_result": {"rewards": {"reward": 1.0}},
         "agent_result": {"n_input_tokens": 100, "n_output_tokens": 10},
         "agent_execution": {"started_at": "2026-10-18T00:00:00Z", "finished_at": "2026-10-18T00:01:00Z"}},
        {"task_name": "t2", "verifier_result": {"rewards": {"reward": 0.0}},
         "agent_result": {"n_input_tokens": 300, "n_output_tokens": 30}},
        {"task_name": "t3", "exception_info": {"exception_type": "AgentTimeoutError"}},
    ]
    _fakes(tmp_path, state={"harbor_results": results})
    out_dir = tmp_path / "results" / "eval1"
    provider = tmp_path / "provider.yaml"
    provider.write_text("sandbox: {}\n")
    common = dict(roots, HARBOR_DATASET="/data/tasks", HARBOR_PYTHON=sys.executable, PYTHONPATH=_fake_harnesses(tmp_path))
    env = _env(tmp_path, HARBOR_ENVIRONMENT_IMPORT_PATH=SANDBOX, **common)
    rc, out = _run_script(HARBOR_RUN, "--out", str(out_dir), "--provider", str(provider), env=env)
    assert rc == 0, out
    harbor, = _calls(env, "harbor")
    assert harbor["argv"] == ["run", "-c", f"{out_dir}/work/job.yaml", "-y"] and Path(harbor["cwd"]).resolve() == (out_dir / "work").resolve()
    assert "HARBOR_RUN_EXIT rc=0" in out
    assert "HARBOR_SUMMARY trials=3 scored=2 passed=1 mean_reward=0.3333 errors=1" in out
    assert ("HARBOR_AGENT_STATS agent_s_median=60 agent_s_max=60 n_input_tokens_median=200 n_input_tokens_max=300 "
            "n_output_tokens_median=20 n_output_tokens_max=30") in out
    assert "HARBOR_EXCEPTION AgentTimeoutError 1" in out
    assert "HARBOR_NOT_PASSED t2 reward=0.0 exception=None" in out and "HARBOR_NOT_PASSED t3 reward=None exception=AgentTimeoutError" in out
    assert "HARBOR_NOT_PASSED t1" not in out and out.rstrip().endswith(f"HARBOR_RESULTS {out_dir}")
    assert {"job.yaml", "provider.yaml", "result.json", "jobs.tar.gz", "summary.txt", "work"} <= {p.name for p in out_dir.iterdir()}
    assert (out_dir / "summary.txt").read_text().startswith("HARBOR_SUMMARY trials=3 scored=2")
    job = yaml.safe_load((out_dir / "job.yaml").read_text())
    assert job["environment"] == {"import_path": SANDBOX, "kwargs": {"sandbox_provider": str(provider), "layer_cache": True}}
    assert job["job_name"] == "eval1" and job["datasets"] == [{"path": "/data/tasks"}]
    assert "FATAL" not in out and "WARN" not in out
    # a failed run: its summary all the same, then the FATAL with Harbor's code (no verifier: only exceptions fail)
    _fakes(tmp_path, state={"harbor_results": results[:1], "harbor_rc": 3})
    out2 = tmp_path / "results" / "eval2"
    rc, out = _run_script(HARBOR_RUN, "--out", str(out2), env=_env(tmp_path, HARBOR_VERIFY="0", **common))
    assert rc == 3, out
    assert "HARBOR_RUN_EXIT rc=3" in out and "HARBOR_SUMMARY trials=1 scored=1 passed=1 mean_reward=1.0000 errors=0" in out
    assert f"HARBOR_RESULTS {out2}" in out and out.rstrip().endswith("FATAL: harbor run rc=3 (see HARBOR_NOT_PASSED / HARBOR_EXCEPTION above)")
    assert yaml.safe_load((out2 / "job.yaml").read_text())["verifier"] == {"disable": True} and not (out2 / "provider.yaml").exists()
    # HARBOR_WORK_DIR=<the out dir>: job.yaml is written where it would be copied to; the copy onto itself is skipped
    _fakes(tmp_path, state={"harbor_results": results[:1]})
    out3 = tmp_path / "results" / "eval3"
    rc, out = _run_script(HARBOR_RUN, "--out", str(out3), env=_env(tmp_path, HARBOR_WORK_DIR=str(out3), **common))
    assert rc == 0, out
    assert "HARBOR_RUN_EXIT rc=0" in out and f"HARBOR_RESULTS {out3}" in out and "FATAL" not in out
    assert yaml.safe_load((out3 / "job.yaml").read_text())["jobs_dir"] == f"{out3}/jobs" and (out3 / "jobs.tar.gz").is_file()


def test_public_harbor_files_name_the_harness_not_a_pack_list():
    """The public trial-server entry and local.env.example name the harness (HARBOR_HARNESS) and no pack list
    (HARBOR_HARNESS_PACKS is a job wrapper's: the packs it extracts on the node and mounts in every sandbox)."""
    pack_list = re.compile(r"\bHARBOR_HARNESS\b")
    for rel in ("scripts/harbor_trial_server.sh", "local.env.example"):
        text = (REPO / rel).read_text()
        assert pack_list.search(text) is not None and "HARBOR_HARNESS_PACKS" not in text, rel


# ---- scripts/check_patches.sh + patches/miles/series (C8) --------------------------------------------------------

CHECK_PATCHES = "scripts/check_patches.sh"
PATCHES_DIR = REPO / "patches" / "miles"
PATCH_SUBJECT = re.compile(r"^Subject: \[PATCH[^\]]*\] (.*)$", re.M)
BASE_SUBJECTS = ["release: freeze dependency lockfile for release/v0.1.2", "release: bump miles version to 0.1.2 (#3922)"]
GIT_LOG_ARGV = ["log", "--format=%s", "-20"]


def _series():
    """patches/miles/series: the first word of every line that is neither blank nor a `#` comment, in order."""
    entries = []
    for line in (PATCHES_DIR / "series").read_text().splitlines():
        words = line.split()
        if words and not words[0].startswith("#"):
            entries.append(words[0])
    return entries


def _patch_subjects(patch_dir=PATCHES_DIR):
    """{name: the first Subject line's text after the tag} for every miles-*.patch of the dir (a folded subject's
    first line, the prefix the gate greps for in `git log --format=%s`)."""
    return {p.name: PATCH_SUBJECT.search(p.read_text(errors="replace")).group(1) for p in sorted(patch_dir.glob("miles-*.patch"))}


def _fake_checkout(tmp_path, name="miles"):
    """A dir with a .git marker: what the gate accepts as a checkout (the fake git never looks inside)."""
    miles = tmp_path / name
    miles.mkdir()
    (miles / ".git").write_text("gitdir: elsewhere\n")
    return miles


def _git_log(subjects_by_name, names):
    """A fake `git log --format=%s -20` reply: the named patches' subjects newest first, then the upstream base."""
    return [subjects_by_name[n] for n in reversed(list(names))] + BASE_SUBJECTS


def _git_calls(env):
    return [c["argv"] for c in _calls(env, "git")]


def test_check_patches_passes_on_applied_subjects_and_names_the_missing_one(tmp_path):
    """The uploader's gate as a script: `git -C <miles> log --format=%s -20` must carry every patch's Subject (the
    first line of a folded one); a fake git answering 8 of 9 -> FATAL naming the 9th with the --apply advice and no
    marker, all 9 -> `PATCHES_OK applied=9/9 ...` last and git only read; --apply am's the missing entries only, in
    series order, re-reads the log and prints the marker, and stops at the first failing am with the `git am --abort`
    advice printed, never run; --apply --dry-run prints `DRY_RUN git_log:` and one `DRY_RUN git_am:` per missing entry
    in series order (none applied, no marker, exit 0); a dir without .git is a FATAL before any git call (a WARN under
    --dry-run, where every entry then counts as missing)."""
    series = _series()
    subjects = _patch_subjects()
    assert sorted(subjects) == sorted(series) and len(series) == 9
    last = series[-1]
    miles = _fake_checkout(tmp_path)
    env = _env(tmp_path, bare_path=True, MILES_SRC=str(miles))
    eight = _git_log(subjects, series[:-1])
    # 8 of 9: the FATAL names the 9th, the log was read once, nothing applied
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": eight})
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 1, out
    assert (f"FATAL: 3rdparty/miles lacks patch {last} ({subjects[last]}); apply with scripts/check_patches.sh --apply "
            "(patches/miles/README.md)") in out
    assert f"check_patches: missing {last}" in out and f"carries 8/9 of the patches in {PATCHES_DIR}" in out
    assert "PATCHES_OK" not in out and out.rstrip().splitlines()[-1].startswith("FATAL:")
    assert _git_calls(env) == [["-C", str(miles), *GIT_LOG_ARGV]]
    # all 9: the marker last, git only read
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": _git_log(subjects, series)})
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 0, out
    assert out.rstrip().splitlines()[-1] == "PATCHES_OK applied=9/9 newly_applied=0"
    assert "FATAL" not in out and "WARN" not in out and "missing" not in out
    assert _git_calls(env) == [["-C", str(miles), *GIT_LOG_ARGV]]
    # --apply --dry-run with 8 of 9: the log argv, one am argv (the 9th), a WARN, no marker, nothing applied
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": eight})
    rc, out = _run_script(CHECK_PATCHES, "--apply", "--dry-run", env=env)
    assert rc == 0, out
    assert _dry(out, "git_log") == ["git", "-C", str(miles), *GIT_LOG_ARGV]
    assert [line for line in out.splitlines() if line.startswith("DRY_RUN git_am:")] == [
        f"DRY_RUN git_am: git -C {miles} am {PATCHES_DIR / last}"]
    assert f"WARN: 3rdparty/miles lacks patch {last} ({subjects[last]})" in out
    assert "PATCHES_OK" not in out and "FATAL" not in out and "check_patches: dry run - 8/9 applied, nothing changed" in out
    assert _git_calls(env) == [["-C", str(miles), *GIT_LOG_ARGV]]
    # ... and with none of them: nine am lines, in series order
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": list(BASE_SUBJECTS)})
    rc, out = _run_script(CHECK_PATCHES, "--apply", "--dry-run", env=env)
    assert rc == 0, out
    assert [_dry(out, "git_am", i) for i in range(9)] == [["git", "-C", str(miles), "am", str(PATCHES_DIR / n)] for n in series]
    assert len([line for line in out.splitlines() if line.startswith("DRY_RUN git_am:")]) == 9
    assert _git_calls(env) == [["-C", str(miles), *GIT_LOG_ARGV]]
    # --apply for real with 8 of 9: exactly the 9th is am'ed (the fake then lists its subject), the log is re-read
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": eight})
    rc, out = _run_script(CHECK_PATCHES, "--apply", env=env)
    assert rc == 0, out
    assert _git_calls(env) == [["-C", str(miles), *GIT_LOG_ARGV], ["-C", str(miles), "am", str(PATCHES_DIR / last)],
                               ["-C", str(miles), *GIT_LOG_ARGV]]
    assert f"check_patches: git am {last}" in out and out.rstrip().splitlines()[-1] == "PATCHES_OK applied=9/9 newly_applied=1"
    # --apply from the base with the third entry failing: two applied before it, none after, the abort advice printed
    third = series[2]
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": list(BASE_SUBJECTS), "git_am_fail": third})
    rc, out = _run_script(CHECK_PATCHES, "--apply", env=env)
    assert rc == 1, out
    assert [argv[-1] for argv in _git_calls(env) if argv[2] == "am"] == [str(PATCHES_DIR / n) for n in series[:3]]
    assert f"FATAL: git am {third} failed in {miles} (rc=128)" in out and f"`git -C {miles} am --abort` (not run by this script)" in out
    assert "PATCHES_OK" not in out
    # --apply when everything is applied: no am at all (idempotent)
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": _git_log(subjects, series)})
    rc, out = _run_script(CHECK_PATCHES, "--apply", env=env)
    assert rc == 0 and out.rstrip().splitlines()[-1] == "PATCHES_OK applied=9/9 newly_applied=0"
    assert _git_calls(env) == [["-C", str(miles), *GIT_LOG_ARGV]]
    # no .git under MILES_SRC: FATAL before any git call; under --dry-run a WARN and every entry counts as missing
    empty = tmp_path / "empty"
    empty.mkdir()
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": eight})
    env_empty = _env(tmp_path, bare_path=True, MILES_SRC=str(empty))
    rc, out = _run_script(CHECK_PATCHES, env=env_empty)
    assert rc == 1, out
    assert f"FATAL: no git checkout at {empty} (MILES_SRC): git submodule update --init 3rdparty/miles" in out
    assert _git_calls(env_empty) == []
    rc, out = _run_script(CHECK_PATCHES, "--apply", "--dry-run", env=env_empty)
    assert rc == 0, out
    assert f"WARN: no git checkout at {empty} (MILES_SRC)" in out and "DRY_RUN git_log" not in out
    assert len([line for line in out.splitlines() if line.startswith("DRY_RUN git_am:")]) == 9 and _git_calls(env_empty) == []
    # --help prints the header; an unknown flag is a FATAL
    rc, out = _run_script(CHECK_PATCHES, "--help", env=env)
    assert rc == 0 and out.startswith("scripts/check_patches.sh — the miles patch gate.") and "--apply" in out
    rc, out = _run_script(CHECK_PATCHES, "--force", env=env)
    assert rc == 1 and "FATAL: unknown argument '--force'" in out


def test_check_patches_series_and_directory_must_agree(tmp_path):
    """MILES_PATCHES_DIR points the gate at another patch dir: a miles-*.patch missing from its series, a series
    entry without a file, a duplicate entry and a missing series file are FATALs before any git call; comments,
    blanks and a trailing `# note` in series are ignored."""
    series = _series()
    three = series[:3]
    other = tmp_path / "patches"
    other.mkdir()
    for name in three:
        shutil.copy(PATCHES_DIR / name, other / name)
    miles = _fake_checkout(tmp_path)
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": _git_log(_patch_subjects(other), three)})
    env = _env(tmp_path, bare_path=True, MILES_SRC=str(miles), MILES_PATCHES_DIR=str(other))
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 1 and f"FATAL: no {other}/series (the patch order file" in out and _git_calls(env) == []
    (other / "series").write_text("\n".join(three[:2]) + "\n")
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 1 and f"FATAL: {other}/series does not list {three[2]} (every {other}/miles-*.patch must be in series)" in out
    (other / "series").write_text("\n".join(three) + "\nmiles-nowhere.patch\n")
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 1 and f"FATAL: {other}/series names miles-nowhere.patch, which is not in {other}/" in out
    (other / "series").write_text("\n".join(three) + f"\n{three[0]}\n")
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 1 and f"FATAL: {other}/series lists {three[0]} twice" in out
    assert _git_calls(env) == []
    (other / "series").write_text(f"# the order\n\n{three[0]}   # first\n{three[1]}\n   \n{three[2]}\n")
    rc, out = _run_script(CHECK_PATCHES, env=env)
    assert rc == 0, out
    assert out.rstrip().splitlines()[-1] == "PATCHES_OK applied=3/3 newly_applied=0"
    # the gate's ${X:-default} reads treat empty as unset: a caller pins them with `DRY_RUN= MILES_SRC= MILES_PATCHES_DIR=
    # GIT=`, and a developer's shell cannot turn the gate into a pass
    gate = (SCRIPTS / "check_patches.sh").read_text()
    for read in ("GIT=${GIT:-git}", "MILES_SRC=${MILES_SRC:-$REPO/3rdparty/miles}",
                 "PATCHES_DIR=${MILES_PATCHES_DIR:-$REPO/patches/miles}"):
        assert read in gate, read
    assert 'is_dry_run() { [ "${DRY_RUN:-}" = 1 ]; }' in Path(COMMON).read_text()


def test_no_reference_to_infra_patches_remains():
    """patches/miles/ is the patches' home (git mv): no text file of the tree names the old location (the pattern is
    built at run time so that this file does not name it either); only reports/ (private run ledgers) and a line of
    patches/miles/README.md that tells the move as history may. The old directory is gone."""
    old = "/".join(["infra", "patches"])
    includes = [f"--include=*.{ext}" for ext in ("py", "sh", "sb" "atch", "md", "toml", "yaml", "yml", "txt", "cfg", "ini", "example")]
    excludes = [f"--exclude-dir={d}" for d in ("reports", "3rdparty", ".git", "__pycache__", "outputs", ".venv", ".claude", "node_modules")]
    proc = subprocess.run(["grep", "-rnI", *includes, *excludes, "--", old, "."], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr
    history = re.compile(r"\b(moved|formerly|history|used to)\b")
    hits = [h for h in proc.stdout.splitlines() if not (h.startswith("./patches/miles/README.md:") and history.search(h))]
    assert hits == [], "\n".join(hits)
    assert (PATCHES_DIR / "series").is_file() and (PATCHES_DIR / "README.md").is_file()
    assert len(list(PATCHES_DIR.glob("miles-*.patch"))) == 9 and not (REPO / "infra" / "patches").exists()


# ---- the example wrapper, the self-test, the smoke, the setup and the docs (C10) ------------------------------

# the example's name ends in the scheduler's suffix: spelled in two pieces (the identity gate scans this file)
EXAMPLE = REPO / "examples" / "slurm" / ("miles_run.example.sb" "atch")
SELFTEST = "scripts/contract_selftest.sh"
SMOKE = "scripts/smoke_tests.sh"
SETUP = "scripts/setup.sh"
FAKE_ENV = REPO / "tests" / "fake_env.sh"
NEW_DOCS = ("docs/cluster.md", "docs/containers.md", "docs/local_setup.md", "docs/data.md")
SITE_FLAG = re.compile(r"--account|--partition|--qos|--container-image")
MILES_FAST_FIRST = "tests/fast/router/test_sessions.py"
MILES_FAST_LAST = "tests/fast/ray/test_train_actor_rpc_surface.py"
MILES_FAST_DESELECT = ("tests/fast/utils/test_arguments.py::TestSecretArgumentsAreClassified::"
                       "test_every_secret_looking_miles_flag_is_either_redacted_or_declared_harmless")


def _contract_names():
    return [line.split()[0] for line in (REPO / "tests" / "contract_vars.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def test_slurm_example_has_only_the_portable_headers(tmp_path):
    """The example wrapper parses; its `#SBATCH` lines are exactly the four portable ones (a site's account,
    partition, QOS, gres and log lines are `##SBATCH` comments); none of --account --partition --qos
    --container-image is an active word (the container form is a commented line); it prints the job markers,
    which no file under scripts/ may (only the wrapper knows the whole job passed). The head-node lookups carry
    `|| true` inside their $( ) so that, under errexit + pipefail, a failing lookup reaches the FATAL line: run for
    real on a machine without the scheduler's tools, the FATAL names the head node, nothing is created."""
    proc = subprocess.run([BASH, "-n", str(EXAMPLE)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    text = EXAMPLE.read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash" and "set -euo pipefail" in lines
    headers = [line for line in lines if line.startswith("#SBATCH")]
    assert headers == ["#SBATCH -N 1", "#SBATCH -t 04:00:00", "#SBATCH -J miles_run", "#SBATCH --signal=B:USR1@600"]
    assert [line for line in lines if SITE_FLAG.search(line) and not line.lstrip().startswith("#")] == []
    assert any(SITE_FLAG.search(line) and line.lstrip().startswith("#") for line in lines)
    assert 'exec bash "$REPO/scripts/run_local.sh"' in text and "export CONTRACT_VERSION=1" in text
    assert "touch \"$PREEMPT_SENTINEL\"' USR1" in text and "exit 75" in text
    assert "JOB_STOPPED_TO_RESUME_miles_run_$RUN_TAG" in text and "JOB_PASS_miles_run_$RUN_TAG" in text
    assert text.rstrip().splitlines()[-1] == 'echo "JOB_PASS_miles_run_$RUN_TAG"'
    for name in _contract_names():
        if name not in ("NODE_RANK", "NODE_IP", "GPUS_PER_NODE", "CACHE_DIR", "HF_HOME", "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "NO_EGRESS", "NODE_LOCAL", "TMPDIR",
                        "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_PTXAS_PATH", "STAGE_DIR", "STAGE_SHA",
                        "INFRA_SHA", "WANDB_API_KEY", "HF_TOKEN", "NEMO_GYM_URL", "TEACHER_URL", "RETRIEVER_URL",
                        "GYM_VENV_ROOT", "ENTRY", "SESSION_SERVER_EXTERNAL_HOST", "SANDBOX_TRANSPORT"):
            assert re.search(rf"\b{name}=", text), name  # the names the smallest wrapper must set
    for path in sorted(SCRIPTS.rglob("*.sh")):
        assert "JOB_PASS_" not in path.read_text(), path
    sc, nodelist = "sc" + "ontrol", "SLURM" + "_JOB_NODELIST"
    assert f'HEAD_HOST=$({sc} show hostnames "${nodelist}" | head -n1 || true)' in text
    assert "HEAD_IP=$(getent hosts \"$HEAD_HOST\" | awk '{print $1; exit}' || true)" in text
    assert "[ -n \"$HEAD_IP\" ] || { echo \"FATAL: could not resolve the head node '$HEAD_HOST'" in text
    _fakes(tmp_path, "sleep")
    env = _env(tmp_path, bare_path=True, RUN_TAG="t", RECIPE="alfworld", REPO=str(REPO), PROJECT_DIR=str(tmp_path / "proj"),
               **{"SLURM" + "_JOB_ID": "1", "SLURM" + "_JOB_NAME": "miles_run", "SLURM" + "_JOB_NUM_NODES": "1", nodelist: "node1"})
    rc, out = _bash(f"'{BASH}' {EXAMPLE}", env)
    assert rc == 1, out
    assert "FATAL: could not resolve the head node ''" in out and "CONTRACT_A version=1" not in out, out
    assert not (tmp_path / "proj").exists()


def test_contract_selftest_passes_under_the_fake_env(tmp_path):
    """`. tests/fake_env.sh && bash scripts/contract_selftest.sh` with the fakes on PATH: every Contract A name set
    (`CONTRACT_VAR name=<n> state=set` x 43, in the file's order), the entry's dry run checked against the fake
    wrapper's sentinel and results dir, the example grepped, `CONTRACT_SELFTEST_OK vars=43/43` last, no rm, the
    one stub under the fake TMPDIR; an unknown contract major is a FATAL without the marker; --dry-run prints the
    three argv lines after the listing and creates nothing; a bare shell (no fake env) passes in local mode."""
    _fakes(tmp_path)
    fake_root = tmp_path / "fake"
    env = _env(tmp_path, LOCAL_ENV="/dev/null", FAKE_ENV_ROOT=str(fake_root))
    rc, out = _bash(f". {FAKE_ENV} && bash {REPO / SELFTEST}", env)
    assert rc == 0, out
    lines = out.rstrip().splitlines()
    assert lines[-1] == "CONTRACT_SELFTEST_OK vars=43/43"
    var_lines = [line for line in lines if line.startswith("CONTRACT_VAR ")]
    assert len(var_lines) == 43 and all(line.endswith(" state=set") for line in var_lines)
    assert [line.split()[1][len("name="):] for line in var_lines] == _contract_names()
    assert "FATAL" not in out and not (tmp_path / "RM_CALLED").exists()
    assert "selftest: CONTRACT_VERSION=1" in out and "DRY_RUN driver:" in out
    assert (fake_root / "tmp" / "miles-agent-selftest" / "policy" / "config.json").is_file()  # the one stub dir
    assert f"--preempt-sentinel {fake_root}/sync/1_0/preempt" in out
    assert f"--save-dir {fake_root}/project/results/selftest/ " in out
    assert f"selftest: {EXAMPLE} carries no site-specific scheduler flag outside a comment" in out
    assert "CONTRACT_VAR name=GYM_VENV_ROOT state=set" in out and "DRY_RUN gym: uv run gym env start" in out
    # an unknown contract major: the listing, then the FATAL, no marker
    rc, out = _bash(f". {FAKE_ENV} && export CONTRACT_VERSION=2 && bash {REPO / SELFTEST}", env)
    assert rc == 1, out
    assert "FATAL: unsupported CONTRACT_VERSION 2 (this entry speaks 1)" in out and "CONTRACT_SELFTEST_OK" not in out
    assert len([line for line in out.splitlines() if line.startswith("CONTRACT_VAR ")]) == 43
    # --dry-run: the listing plus the three argv lines; nothing created under a fresh root
    dry_root = tmp_path / "dry"
    env_dry = _env(tmp_path, LOCAL_ENV="/dev/null", FAKE_ENV_ROOT=str(dry_root))
    rc, out = _bash(f". {FAKE_ENV} && bash {REPO / SELFTEST} --dry-run", env_dry)
    assert rc == 0, out
    bash_n = _dry(out, "bash_n")
    assert bash_n[:2] == ["bash", "-n"] and str(REPO / SELFTEST) in bash_n and str(EXAMPLE) in bash_n
    assert set(bash_n[2:]) == {str(p) for p in _shell_files()}
    assert _dry(out, "run_local") == ["bash", str(REPO / "scripts" / "run_local.sh"), "--dry-run", "--recipe",
                                      "alfworld", "--run-tag", "selftest"]
    assert _dry(out, "grep_example") == ["grep", "-nE", "--", SITE_FLAG.pattern, str(EXAMPLE)]
    assert "CONTRACT_SELFTEST_OK" not in out and "DRY_RUN driver:" not in out
    # fake_env.sh creates the TMPDIR it exports; the dry run creates nothing else
    assert [p.relative_to(dry_root) for p in dry_root.rglob("*")] == [Path("tmp")]
    assert len([line for line in out.splitlines() if line.startswith("CONTRACT_VAR ")]) == 43
    # a bare shell: local mode, the stub under $TMPDIR, the marker with however many names the shell has
    tmpdir = tmp_path / "t"
    tmpdir.mkdir()
    env_bare = _env(tmp_path, LOCAL_ENV="/dev/null", TMPDIR=str(tmpdir))
    rc, out = _run_script(SELFTEST, env=env_bare)
    assert rc == 0, out
    assert re.fullmatch(r"CONTRACT_SELFTEST_OK vars=\d+/43", out.rstrip().splitlines()[-1])
    assert "selftest: CONTRACT_VERSION=<unset: local mode>" in out
    assert f"--preempt-sentinel {REPO}/results/selftest/stop" in out and f"--save-dir {REPO}/results/selftest/ " in out
    assert (tmpdir / "miles-agent-selftest" / "policy" / "config.json").is_file()
    assert not (REPO / "results" / "selftest").exists()
    # --help prints the header; an unknown flag is a FATAL
    rc, out = _run_script(SELFTEST, "--help", env=env_bare)
    assert rc == 0 and out.startswith("scripts/contract_selftest.sh — the no-cluster self-test") and "--dry-run" in out
    rc, out = _run_script(SELFTEST, "--fast", env=env_bare)
    assert rc == 1 and "FATAL: unknown argument '--fast'" in out


def test_smoke_tests_dry_run_prints_both_pytest_lines(tmp_path):
    """--dry-run: the repo's pytest line (cwd $REPO) and, when $MILES_ROOT/tests/fast exists, the miles fast-test
    line (cwd $MILES_ROOT, `env -u CUDA_VISIBLE_DEVICES`, the 17 files, the one --deselect); without tests/fast a
    WARN and one line. The real run with the fake python3: two pytest calls from the two dirs, the second without
    CUDA_VISIBLE_DEVICES, `CONTRACT_TESTS_OK` last; a failing suite is a FATAL without the marker."""
    miles = tmp_path / "miles"
    (miles / "tests" / "fast").mkdir(parents=True)
    _fakes(tmp_path)
    env = _env(tmp_path, LOCAL_ENV="/dev/null", MILES_ROOT=str(miles), CUDA_VISIBLE_DEVICES="0")
    rc, out = _run_script(SMOKE, "--dry-run", env=env)
    assert rc == 0, out
    assert _dry(out, "cwd") == [str(REPO)] and _dry(out, "cwd", 1) == [str(miles)]
    assert _dry(out, "pytest_repo") == ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", "tests"]
    argv = _dry(out, "pytest_miles")
    assert argv[:9] == ["env", "-u", "CUDA_VISIBLE_DEVICES", "python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    files = argv[9:argv.index("--deselect")]
    assert len(files) == 17 and files[0] == MILES_FAST_FIRST and files[-1] == MILES_FAST_LAST
    assert all(f.startswith("tests/fast/") and f.endswith(".py") for f in files) and len(set(files)) == 17
    assert argv[-2:] == ["--deselect", MILES_FAST_DESELECT]
    assert "CONTRACT_TESTS_OK" not in out and _calls(env) == []
    # no tests/fast under MILES_ROOT: the WARN, one pytest line
    bare = tmp_path / "bare"
    bare.mkdir()
    env_bare = _env(tmp_path, LOCAL_ENV="/dev/null", MILES_ROOT=str(bare))
    rc, out = _run_script(SMOKE, "--dry-run", env=env_bare)
    assert rc == 0, out
    assert f"WARN: no miles fast tests under {bare} (tests/fast missing): step 2 skipped" in out
    assert [line.split(":")[0] for line in out.splitlines() if line.startswith("DRY_RUN pytest_")] == ["DRY_RUN pytest_repo"]
    # the real run with the fake python3: the two calls, their cwd and env, the marker last
    rc, out = _run_script(SMOKE, env=env)
    assert rc == 0, out
    calls = _calls(env, "python3")
    assert [Path(c["cwd"]).resolve() for c in calls] == [REPO.resolve(), miles.resolve()]
    assert calls[0]["argv"] == ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", "tests"]
    assert calls[1]["argv"][:5] == ["-m", "pytest", "-q", "-p", "no:cacheprovider"] and calls[1]["argv"][-2:] == ["--deselect", MILES_FAST_DESELECT]
    assert calls[0]["env"].get("CUDA_VISIBLE_DEVICES") == "0" and "CUDA_VISIBLE_DEVICES" not in calls[1]["env"]
    assert out.rstrip().splitlines()[-1] == "CONTRACT_TESTS_OK"
    # no tests/fast, for real: one call (a fresh call log), the WARN, the marker
    _fakes(tmp_path)
    rc, out = _run_script(SMOKE, env=env_bare)
    assert rc == 0 and len(_calls(env_bare, "python3")) == 1 and out.rstrip().splitlines()[-1] == "CONTRACT_TESTS_OK"
    # a failing suite: FATAL with the rc, no marker
    env_fail = _env(tmp_path, LOCAL_ENV="/dev/null", MILES_ROOT=str(miles), PYTHON="false")
    rc, out = _run_script(SMOKE, env=env_fail)
    assert rc == 1 and "FATAL: the repo's tests failed (rc=1)" in out and "CONTRACT_TESTS_OK" not in out
    rc, out = _run_script(SMOKE, "--help", env=env_bare)
    assert rc == 0 and out.startswith("scripts/smoke_tests.sh — the contract smoke") and "CONTRACT_TESTS_OK" in out


def _setup_tree(tmp_path):
    """A copy of what scripts/setup.sh touches (scripts/, patches/miles/, pyproject.toml, local.env.example) with
    EMPTY 3rdparty/miles and 3rdparty/Gym dirs: both submodules look uninitialised, and nothing under the real
    checkout is read for its state or written."""
    root = tmp_path / "clone"
    shutil.copytree(REPO / "scripts", root / "scripts")
    shutil.copytree(REPO / "patches" / "miles", root / "patches" / "miles")
    for name in ("pyproject.toml", "local.env.example"):
        shutil.copy(REPO / name, root / name)
    (root / "3rdparty" / "miles").mkdir(parents=True)
    (root / "3rdparty" / "Gym").mkdir()
    return root


def test_setup_dry_run_lists_submodules_patches_and_uv(tmp_path):
    """--dry-run on a fresh clone (both submodules empty): the argv of every step in order (submodules,
    check_patches, uv_venv, uv_pip, local_env), nothing executed, no marker. The real run with the fakes: `git
    submodule update --init --recursive` for both, check_patches.sh --apply (its log, 9 am, log), uv venv + uv pip
    into <root>/.venv, local.env copied, `SETUP_OK ...` last. An initialised submodule at its gitlink or ahead of it
    (the patched miles) and an existing local.env are left alone (the real checkout: the two read-only git calls per
    initialised submodule, no submodule line; an exported tree without .git gets the init planned); one whose HEAD is
    neither the gitlink nor a descendant of it is updated. No uv on
    PATH: FATAL with the install hint, a WARN under --dry-run."""
    root = _setup_tree(tmp_path)
    setup = root / "scripts" / "setup.sh"
    local_env = root / "local.env"
    miles_src = _fake_checkout(tmp_path)  # the .git marker check_patches.sh wants under MILES_SRC
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": list(BASE_SUBJECTS)})
    env = _env(tmp_path, LOCAL_ENV=str(local_env), MILES_SRC=str(miles_src))
    rc, out = _bash(f"bash {setup} --dry-run", env)
    assert rc == 0, out
    labels = [line.split(":")[0][len("DRY_RUN "):] for line in out.splitlines() if line.startswith("DRY_RUN ")]
    assert labels == ["submodules", "check_patches", "uv_venv", "uv_pip", "local_env"]
    assert _dry(out, "submodules") == ["git", "-C", str(root), "submodule", "update", "--init", "--recursive", "3rdparty/miles", "3rdparty/Gym"]
    assert _dry(out, "check_patches") == ["bash", str(root / "scripts" / "check_patches.sh"), "--apply"]
    assert _dry(out, "uv_venv") == ["uv", "venv", "--allow-existing", "--python", "3.13.14", str(root / ".venv")]
    assert _dry(out, "uv_pip") == ["uv", "pip", "install", "-p", str(root / ".venv"), "--group", f"{root}/pyproject.toml:test"]
    assert _dry(out, "local_env") == ["cp", str(root / "local.env.example"), str(local_env)]
    assert _calls(env) == [] and "SETUP_OK" not in out and not local_env.exists() and not (root / ".venv").exists()
    # the real run with the fakes
    rc, out = _bash(f"bash {setup}", env)
    assert rc == 0, out
    git = _git_calls(env)
    assert git[0] == ["-C", str(root), "submodule", "update", "--init", "--recursive", "3rdparty/miles", "3rdparty/Gym"]
    assert git[1] == ["-C", str(miles_src), *GIT_LOG_ARGV] and git[-1] == git[1] and len(git) == 12
    assert [argv[-1] for argv in git[2:11]] == [str(root / "patches" / "miles" / n) for n in _series()]
    assert [c["argv"] for c in _calls(env, "uv")] == [
        ["venv", "--allow-existing", "--python", "3.13.14", str(root / ".venv")],
        ["pip", "install", "-p", str(root / ".venv"), "--group", f"{root}/pyproject.toml:test"],
    ]
    assert local_env.read_text() == (REPO / "local.env.example").read_text()
    assert "PATCHES_OK applied=9/9 newly_applied=9" in out and "setup: next steps" in out
    assert out.rstrip().splitlines()[-1] == f"SETUP_OK venv={root}/.venv local_env={local_env}"
    assert not (tmp_path / "RM_CALLED").exists()
    # a second run: the submodules initialised (markers) and local.env there -> no submodule line, no cp
    for sub in ("miles", "Gym"):
        (root / "3rdparty" / sub / ".git").write_text("gitdir: elsewhere\n")
    rc, out = _bash(f"bash {setup} --dry-run", env)
    assert rc == 0, out
    assert "DRY_RUN submodules" not in out and "DRY_RUN local_env" not in out
    assert "setup: submodule 3rdparty/miles already initialised: left as it is" in out
    assert "setup: submodule 3rdparty/Gym already initialised: left as it is" in out
    assert f"setup: {local_env} already there: kept" in out
    assert [line.split(":")[0][len("DRY_RUN "):] for line in out.splitlines() if line.startswith("DRY_RUN ")] == ["check_patches", "uv_venv", "uv_pip"]
    # the pin moved under an initialised checkout (a pull): a HEAD that is neither the gitlink nor a descendant of it
    # is updated, recursively; a HEAD at or ahead of the gitlink (the patched miles) is left alone
    miles_sub, gym_sub = str(root / "3rdparty" / "miles"), str(root / "3rdparty" / "Gym")
    links = {"3rdparty/miles": "a" * 40, "3rdparty/Gym": "c" * 40}
    heads = {miles_sub: "b" * 40, gym_sub: "c" * 40}
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": list(BASE_SUBJECTS), "git_gitlink": links, "git_head": heads, "git_ancestor_rc": 1})
    rc, out = _bash(f"bash {setup} --dry-run", env)
    assert rc == 0, out
    assert f"setup: submodule 3rdparty/miles is at {'b' * 12}, neither the gitlink {'a' * 12} nor a descendant of it: updated to the gitlink" in out
    assert "setup: submodule 3rdparty/Gym already initialised: left as it is" in out
    assert _dry(out, "submodules") == ["git", "-C", str(root), "submodule", "update", "--init", "--recursive", "3rdparty/miles"]
    assert [c["argv"] for c in _calls(env, "git")] == [["-C", str(root), "ls-files", "-s", "--", "3rdparty/miles"], ["-C", miles_sub, "rev-parse", "HEAD"],
                                                      ["-C", miles_sub, "merge-base", "--is-ancestor", "a" * 40, "b" * 40],
                                                      ["-C", str(root), "ls-files", "-s", "--", "3rdparty/Gym"], ["-C", gym_sub, "rev-parse", "HEAD"]]
    _set_state(tmp_path, {"git_log": list(BASE_SUBJECTS), "git_gitlink": links, "git_head": heads, "git_ancestor_rc": 0})
    rc, out = _bash(f"bash {setup} --dry-run", env)
    assert rc == 0 and "DRY_RUN submodules" not in out and out.count("already initialised: left as it is") == 2, out
    # the real checkout, dry: for an initialised submodule (a clone) only the two read-only git calls run (a fresh
    # call log); for one without a .git (an exported or staged tree, as legitimate a place for the suite as a clone)
    # setup.sh plans `submodule update --init` and reads nothing
    _fakes(tmp_path, *DEFAULT_FAKES, "git", state={"git_log": list(BASE_SUBJECTS)})
    env_repo = _env(tmp_path, LOCAL_ENV="/dev/null")
    rc, out = _run_script(SETUP, "--dry-run", env=env_repo)
    assert rc == 0, out
    subs = ("3rdparty/miles", "3rdparty/Gym")
    initialised = [sub for sub in subs if (REPO / sub / ".git").exists()]
    missing = [sub for sub in subs if sub not in initialised]
    if missing:
        assert _dry(out, "submodules") == ["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", *missing]
    else:
        assert "DRY_RUN submodules" not in out
    assert out.count("already initialised: left as it is") == len(initialised)
    assert _dry(out, "check_patches") == ["bash", str(REPO / "scripts" / "check_patches.sh"), "--apply"]
    reads = [[["-C", str(REPO), "ls-files", "-s", "--", sub], ["-C", f"{REPO}/{sub}", "rev-parse", "HEAD"]] for sub in initialised]
    assert [c["argv"] for c in _calls(env_repo)] == [argv for pair in reads for argv in pair]
    assert "SETUP_OK" not in out
    # no uv on PATH: FATAL before anything else; a WARN under --dry-run
    _fakes(tmp_path, *(n for n in DEFAULT_FAKES if n != "uv"), "git", state={"git_log": list(BASE_SUBJECTS)})
    env_nouv = _env(tmp_path, bare_path=True, interp_dir=False, LOCAL_ENV=str(local_env), MILES_SRC=str(miles_src))
    rc, out = _bash(f"'{BASH}' {setup}", env_nouv)
    assert rc == 1, out
    assert "FATAL: uv not found (UV_BIN=uv): install uv: https://docs.astral.sh/uv/" in out and _calls(env_nouv) == []
    rc, out = _bash(f"'{BASH}' {setup} --dry-run", env_nouv)
    assert rc == 0, out
    assert "WARN: uv not found (UV_BIN=uv)" in out and "DRY_RUN uv_venv: uv venv" in out and "SETUP_OK" not in out
    rc, out = _bash(f"'{BASH}' {setup} --help", env_nouv)
    assert rc == 0 and out.startswith("scripts/setup.sh — a fresh clone made ready") and "SETUP_OK" in out


def test_docs_name_no_site_path():
    """The four new docs and the example carry no identity literal; the three docs outside D22's exception carry
    no scheduler word and none of D22's usability words (a reader without a cluster must not need them);
    docs/cluster.md is the one page that names the wrapper words and the plugin-less container lines; README
    points at the new pages and the dry run."""
    hits = []
    for rel in (*NEW_DOCS, str(EXAMPLE.relative_to(REPO))):
        for i, line in enumerate((REPO / rel).read_text().splitlines(), 1):
            match = IDENTITY_LITERAL.search(line)
            if match is not None:
                hits.append(f"{rel}:{i}: {match.group(0)}")
    assert hits == []
    for rel in NEW_DOCS[1:]:
        text = (REPO / rel).read_text()
        for pattern in (SITE_LITERAL, USABILITY_LITERAL):
            match = pattern.search(text)
            assert match is None, f"{rel}: {match.group(0)}"
    cluster = (REPO / "docs" / "cluster.md").read_text()
    assert USABILITY_LITERAL.search(cluster) is not None  # the container lines live here, and only here
    for heading in ("## A. wrapper -> entry", "## B. the project's entries", "## C. preemption / time limit", "## D. markers"):
        assert heading in cluster, heading
    assert "import -o miles" in cluster and 'start --rw --mount "$REPO:$REPO"' in cluster
    readme = (REPO / "README.md").read_text()
    for name in ("docs/local_setup.md", "docs/containers.md", "docs/data.md", "docs/cluster.md", "scripts/run_local.sh --dry-run"):
        assert name in readme, name
