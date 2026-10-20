"""The public R2S scripts run with fake commands on PATH: scripts/lib/sglang.sh, scripts/teacher_node.sh,
scripts/lib/retired_knobs.sh, scripts/r2s_teacher_ft.sh and scripts/r2s_pipeline.sh (their tests moved here from the old
job-script test file with the public half of its helpers), the anchors that tie scripts/run_local.sh and its libs to
the launcher's CLI, and the public helpers' freedom from any scheduler call."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# the public entry and the public libs it is built on
ENTRY = REPO / "scripts" / "run_local.sh"
RAY_SH = REPO / "scripts" / "lib" / "ray.sh"
LADDER_SH = REPO / "scripts" / "lib" / "gym_ladder.sh"
# the public R2S scripts under test
SGLANG_SH = REPO / "scripts" / "lib" / "sglang.sh"
TEACHER_SH = REPO / "scripts" / "teacher_node.sh"
RETIRED_SH = REPO / "scripts" / "lib" / "retired_knobs.sh"
FT_SH = REPO / "scripts" / "r2s_teacher_ft.sh"
PIPELINE_SH = REPO / "scripts" / "r2s_pipeline.sh"

# The knob -> a word of the replacement its FATAL names (design §2.6).
RETIRED_KNOBS = {
    "TEACHER2_MODEL": "TEACHER_MODEL",
    "ONESHOT_PROTOCOL": "PROMPT=r2s_oneshot",
    "ALFWORLD_OPD_PROMPT": "PROMPT=r2s_oneshot",
    "SCIW_OPD_PROMPT": "PROMPT=r2s_oneshot",
    "OPD_TOOL_CALL_FORMAT": "MODEL_FAMILIES",
    "TEACHER_TOOL_PARSER": "MODEL_FAMILIES",
    "WEBSHOP_TOOL_CALL_STYLE": "MODEL_FAMILIES",
    "ALFWORLD_OPD_REFS": "FTB",
    "SCIW_OPD_REFS": "FTB",
    "WEBSHOP_OPD_REFS": "FTB",
    "WEBSHOP_OPD_REF_FILE": "FTB",
    "TOOLS_FILE": "--env",
    "VOTE_KEY": "--env",
    "TAU_SELECT": "R2S_ARGS_C",
}

# Stand-ins for the commands the teacher node and the gate run; every call lands in $ROOT/events. sleep: the 5 s
# and 10 s polls take 20 ms. nvidia-smi -L lists FAKE_GPUS (4) GPUs. python3 -m sglang.launch_server serves (touches
# $ROOT/served after 50 ms) until TERM, or dies at boot with FAKE_SGLANG_DIES=1. curl answers /health_generate once
# $ROOT/served exists or FAKE_HEALTH=ok. The tokenizer probe passes unless FAKE_PROBE_FAIL=1 (a drift verdict) or
# FAKE_PROBE_CRASH=1 (it never compares: a traceback); FAKE_PROBE_NOISE=1 writes a warning without its newline ahead
# of the verdict. torchrun stands in for stage E's launcher: it writes the report (and a weights file) into --out-dir
# and prints FT_PASS (exit 0), or with FAKE_FT=fail the report with passed false after FT_FAIL and exits 1 as the
# real torchrun does whatever the worker's code, or with FAKE_FT=skipgate the report with passed false and the
# weights after FT_GATE_SKIPPED (exit 0: --skip-gate), or with FAKE_FT=crash dies at once with nothing written.
# python3 -m miles_agent.r2s.<stage> stands in for the five pipeline stages: it
# records its flags, skips itself when its summary already carries the marker (unless --force, as common.skip_if_done),
# writes FAKE_ROWS_<letter> (3) rows and the summary closed by the marker, or under FAKE_STAGE_<letter> crashes
# (exit 1), refuses its gate (the FAIL marker, exit 2), stops on a sentinel it raises itself (stop: no marker, exit 0)
# or ends without a marker; python3 -m miles_agent.launcher serve-template / reasoning-parser write a template and
# print neutral parser names; a python3 -c expression runs on the real interpreter with the repo first (FAKE_REPO).
FAKE = r"""
import json, os, signal, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
root = Path(os.environ["ROOT"])
def event(line):
    with open(root / "events", "a") as f:
        f.write(line + "\n")
if name == "sleep":
    time.sleep(0.02); sys.exit(0)
if name == "getent":
    print("10.0.0.7 fakehost"); sys.exit(0)
if name == "nvidia-smi":
    for i in range(int(os.environ.get("FAKE_GPUS", "4"))): print(f"GPU {i}: Fake GPU (UUID: GPU-{i})")
    sys.exit(0)
if name == "setsid":
    try:
        os.setsid()
    except OSError:
        pass
    os.execvp(sys.argv[1], sys.argv[1:])
if name == "curl":
    args = sys.argv[1:]
    url = [a for i, a in enumerate(args) if not a.startswith("-") and (i == 0 or args[i - 1] != "-o")][-1]
    event("curl " + url)
    ok = url.endswith("/health_generate") and ((root / "served").exists() or os.environ.get("FAKE_HEALTH") == "ok")
    sys.exit(0 if ok else 22)
if name == "python3" and sys.argv[1:3] == ["-m", "sglang.launch_server"]:
    event("sglang cuda=" + os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + " " + " ".join(sys.argv[3:]))
    (root / "server.pid").write_text(str(os.getpid()))
    if os.environ.get("FAKE_SGLANG_DIES") == "1":
        print("fake sglang: dying at boot"); sys.exit(1)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    time.sleep(0.05)
    (root / "served").touch()
    while True:
        time.sleep(0.05)
if name == "python3" and sys.argv[1:4] == ["-m", "miles_agent.launcher", "tokenizer-probe"]:
    event("probe " + " ".join(sys.argv[4:]))
    if os.environ.get("FAKE_PROBE_NOISE") == "1":
        sys.stderr.write("warning: slow tokenizer in use")  # no trailing newline: the verdict lands on this line
    if os.environ.get("FAKE_PROBE_FAIL") == "1":
        sys.exit("TOKENIZER_PROBE_FAIL vocabulary sizes differ: student 1, teacher 2")
    if os.environ.get("FAKE_PROBE_CRASH") == "1":
        raise OSError(f"{sys.argv[5]}/tokenizer_config.json: no such file")
    print("TOKENIZER_PROBE_OK"); sys.exit(0)
if name == "torchrun":
    event("torchrun " + " ".join(sys.argv[1:]))
    facts = {k: os.environ.get(k, "") for k in ("PYTHONPATH", "TRITON_CACHE_DIR")}
    event("torchrun-env " + " ".join(f"{k}={v}" for k, v in facts.items()))
    mode = os.environ.get("FAKE_FT", "pass")
    if mode == "crash":
        print("fake torchrun: a worker died before the gate"); sys.exit(1)
    out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    passed = mode == "pass"
    after = 0.9 if passed else 1.1
    report = {"val_hind_ce_before": 1.0, "val_hind_ce_after": after, "passed": passed, "objective": "suffix"}
    (out / "r2s_ft_report.json").write_text(json.dumps(report, indent=2))
    if passed or mode == "skipgate":  # --skip-gate: exported unimproved, the worker exits 0 after FT_GATE_SKIPPED
        (out / "model.safetensors").write_text("weights")
        print("[00:00:00] FT_PASS" if passed else "[00:00:00] FT_GATE_SKIPPED"); sys.exit(0)
    print("[00:00:00] FT_FAIL val hind CE did not improve (1.0000 -> 1.1000)"); sys.exit(1)
STAGES = {
    "divergence": ("A", "DIVERGENCE_DONE"),
    "takeover": ("B", "TAKEOVER_DONE"),
    "judge": ("J", "JUDGE_DONE"),
    "select": ("C", "SELECT_DONE"),
    "label": ("D", "LABEL_DONE"),
}
if name == "python3" and sys.argv[1:2] == ["-m"] and sys.argv[2].startswith("miles_agent.r2s."):
    letter, marker = STAGES[sys.argv[2].rsplit(".", 1)[1]]
    argv = sys.argv[3:]
    event(f"stage {letter} " + " ".join(argv))
    def flag(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default
    out = flag("--out")
    summary = flag("--summary", out + ".summary.json")
    if "--force" not in argv and os.path.exists(summary) and f'"{marker}": true' in Path(summary).read_text():
        event(f"skip {letter}")
        print(f"{marker} already in {summary}; nothing to do (--force reruns)"); sys.exit(0)
    mode = os.environ.get(f"FAKE_STAGE_{letter}", "done")
    if mode == "crash":
        print("fake stage: Traceback (most recent call last)"); sys.exit(1)
    rows = int(os.environ.get(f"FAKE_ROWS_{letter}", "3"))
    with open(out, "a" if letter in "BD" else "w") as f:  # B and D append per row, the others write once
        for i in range(rows):
            f.write(json.dumps({"state_id": f"s{i}", "sample": 0}) + "\n")
    if mode == "stop":  # a preemption during the stage: the stop file seen between rows, no marker, exit 0
        Path(flag("--stop-file")).touch()
        print(f"stop file raised: exiting without {marker}"); sys.exit(0)
    if mode == "nomarker":
        sys.exit(0)
    done = marker if mode != "gate" else marker.replace("DONE", "FAIL")
    Path(summary).write_text(json.dumps({"rows": rows, done: True}, indent=2) + f"\n{done}\n")
    print(done); sys.exit(2 if mode == "gate" else 0)
if name == "python3" and sys.argv[1:4] == ["-m", "miles_agent.launcher", "serve-template"]:
    event("launcher " + " ".join(sys.argv[3:]))
    Path(sys.argv[5]).write_text("{{ fake template }}")
    print("fake_tool_parser"); sys.exit(0)
if name == "python3" and sys.argv[1:4] == ["-m", "miles_agent.launcher", "reasoning-parser"]:
    event("launcher " + " ".join(sys.argv[3:]))
    print("fake_reasoning_parser"); sys.exit(0)
if name == "python3" and sys.argv[1:2] == ["-c"]:
    os.environ["PYTHONPATH"] = os.environ["FAKE_REPO"]
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
raise RuntimeError(sys.argv)
"""
# The wrapper's side of the contract around teacher_node_main (the EXIT-trap accumulator), with a stand-in
# head: once the URL is up, a USR1 (a preemption reaches every task) and then the end of the job.
TEACHER_HARNESS = r"""
set -eu
G_CLEANUP=":"
cleanup() { set +e; eval "$G_CLEANUP"; }
add_cleanup() { G_CLEANUP="$1; $G_CLEANUP"; }
trap cleanup EXIT
. "$SGLANG_SH"
. "$TEACHER_SH"
( until [ -f "$SYNC_DIR/TEACHER_URL" ] || [ -f "$SYNC_DIR/TEACHER_FAILED" ]; do sleep 1; done
  [ -f "$SYNC_DIR/TEACHER_URL" ] || exit 0
  kill -USR1 $$; sleep 1; touch "$SYNC_DIR/ENDED" ) &
teacher_node_main
echo NOT_REACHED
"""
GATE_HARNESS = r"""
set -eu
add_cleanup() { :; }
. "$SGLANG_SH"
. "$TEACHER_SH"
teacher_gate
echo "ARG=$TEACHER_URL_ARG"
"""
# The gate with a preemption arriving during the teacher's boot: the sentinel comes up after five (fake) seconds.
GATE_PREEMPT_HARNESS = r"""
set -eu
add_cleanup() { :; }
. "$SGLANG_SH"
. "$TEACHER_SH"
( for _ in 1 2 3 4 5; do sleep 1; done; touch "$PREEMPT_SENTINEL" ) &
teacher_gate
echo "ARG=$TEACHER_URL_ARG"
"""
# start_sglang under a caller's set -e: guarded first, then bare (the pipeline stages' shape).
SGLANG_HARNESS = r"""
set -eu
add_cleanup() { :; }
. "$SGLANG_SH"
start_sglang teacher "$MODEL_DIR" 13141 "${CUDA_IDS-all}" 1 || echo "GUARDED rc=$?"
start_sglang teacher "$MODEL_DIR" 13141 "${CUDA_IDS-all}" 1
echo NOT_REACHED
"""
# wait_sglang against a server pid that never answers (tail -f: alive, no $ROOT/served for curl): its stop file
# absent, then raised while waiting, then the call without the argument once the file exists.
WAIT_HARNESS = r"""
set -eu
add_cleanup() { :; }
. "$SGLANG_SH"
tail -f /dev/null &
SGLANG_PID_teacher=$!
trap 'kill "$SGLANG_PID_teacher"' EXIT
SGLANG_LOG_teacher=$ROOT/teacher.log
printf 'loading shards\n' > "$SGLANG_LOG_teacher"
rc=0; wait_sglang teacher http://127.0.0.1:13141/health_generate 3 1 "$STOP" || rc=$?
echo "RC_ABSENT=$rc"
( for _ in 1 2 3; do sleep 1; done; touch "$STOP" ) &
rc=0; wait_sglang teacher http://127.0.0.1:13141/health_generate 400 1 "$STOP" || rc=$?
echo "RC_RAISED=$rc"
rc=0; wait_sglang teacher http://127.0.0.1:13141/health_generate 2 1 || rc=$?
echo "RC_NOARG=$rc"
"""
# free_port against a listener on the preferred port (the leftover of an earlier job): the next port is printed, a free
# preferred port is kept, a span of one over the busy port finds nothing.
FREE_PORT_HARNESS = r"""
set -eu
add_cleanup() { :; }
. "$SGLANG_SH"
python3 -c 'import socket, sys, time
s = socket.socket()
s.bind(("0.0.0.0", int(sys.argv[1])))
s.listen()
print("LISTENING", flush=True)
time.sleep(30)' "$BUSY" > "$ROOT/listener.out" &
LPID=$!
trap 'kill "$LPID" 2>/dev/null' EXIT
for _ in $(seq 100); do grep -q LISTENING "$ROOT/listener.out" 2>/dev/null && break; sleep 0.1; done
grep -q LISTENING "$ROOT/listener.out"
echo "SHIFTED=$(free_port "$BUSY" 20)"
echo "KEPT=$(free_port "$((BUSY + 1))" 20)"
rc=0; out=$(free_port "$BUSY" 1) || rc=$?
echo "NONE_RC=$rc NONE_OUT=$out"
"""
# The wrapper's side of the contract around r2s_pipeline_main (the EXIT-trap accumulator) with the servers as a
# recorder: every start / wait / stop lands in events in the order the driver issues them, the stop of each started
# server registered with the trap as the real start_sglang does, and a stop of a server already down a no-op as the
# real stop_sglang's (no pid). The wait records its stop-file argument and returns 2 when that file exists, as the
# real wait_sglang does; FAKE_WAIT_FAIL=<name> makes that server's wait fail (return 1: it never came up).
PIPELINE_HARNESS = r"""
set -eu
G_CLEANUP=":"
cleanup() { set +e; eval "$G_CLEANUP"; }
add_cleanup() { G_CLEANUP="$1; $G_CLEANUP"; }
trap cleanup EXIT
. "$PIPELINE_SH"
start_sglang() {
  echo "start $*" >> "$ROOT/events"; : > "$ROOT/up.$1"; add_cleanup "stop_sglang $1"; echo "SGLANG_START $1"
}
wait_sglang() {
  echo "wait $1 $2${5:+ stop=$5}" >> "$ROOT/events"
  [ -z "${5:-}" ] || [ ! -e "$5" ] || return 2
  [ "${FAKE_WAIT_FAIL:-}" != "$1" ] || return 1
}
stop_sglang() { [ -e "$ROOT/up.$1" ] || return 0; mv "$ROOT/up.$1" "$ROOT/down.$1"; echo "stop $1" >> "$ROOT/events"; }
r2s_pipeline_main
echo HARNESS_END
"""


# stage E's knobs and the generic paths its script reads: never inherited from the developer's shell
FT_KNOBS = {"TAG", "LABELS", "BASE_MODEL", "LR", "EPOCHS", "GRAD_ACCUM", "FT_EXTRA", "SRC_MOUNT", "LOCAL", "JOB_ID"}
# the pipeline's knobs (scripts/r2s_pipeline.sh header), likewise
PIPELINE_KNOBS = {
    "RECIPE",
    "STAGES",
    "DUMP_GLOB",
    "STATES_IN",
    "FOLD_IN",
    "JUDGE_MODEL",
    "JUDGE_HF",
    "TOP_N",
    "MAX_PER_GOAL",
    "SAMPLES",
    "TAKEOVER_TEMPERATURE",
    "K",
    "MIN_STATES",
    "MIN_REPLAY_OK",
    "STOP_AFTER_FOLD",
    "CONCURRENCY",
}


def _case(tmp_path, name, **settings):
    """A fresh root with the fake commands, a sync dir and the job environment (the MILES_AGENT_* dirs and RUNS_DIR
    under $ROOT, RUN_TAG t1, job 123). The job's own CUDA_VISIBLE_DEVICES and scheduler variables never leak in
    (the scripts under test read the first; a suite run inside a job carries both)."""
    root = tmp_path / name
    (root / "sync").mkdir(parents=True)
    fake_bin = root / "bin"
    fake_bin.mkdir()
    for command in ("sleep", "getent", "nvidia-smi", "setsid", "curl", "python3", "torchrun"):
        path = fake_bin / command
        path.write_text("#!" + sys.executable + "\n" + FAKE)
        path.chmod(0o755)
    (root / "events").write_text("")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("TEACHER", "FAKE_", "SYNC_DIR", "RUNS_DIR", "MILES_AGENT_", "POLICY_", "PREEMPT_", "R2S_", "SLU" "RM_")
        )
        and key != "CUDA_VISIBLE_DEVICES"
        and key not in FT_KNOBS
        and key not in PIPELINE_KNOBS
    }
    env.update(
        ROOT=str(root),
        PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
        SGLANG_SH=str(SGLANG_SH),
        TEACHER_SH=str(TEACHER_SH),
        MILES_AGENT_MODELS=str(root / "models"),
        MILES_AGENT_DATA=str(root / "data"),
        MILES_AGENT_RESULTS=str(root / "results"),
        RUNS_DIR=str(root / "runs"),
        RUN_TAG="t1",
        JOB_ID="123",
        LOCAL_ENV="/dev/null",
        SYNC_DIR=str(root / "sync"),
    )
    env.update({key: str(value) for key, value in settings.items()})
    return root, env


def _model_dir(root, name):
    path = root / "models" / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


def _run(root, harness, env):
    (root / "harness.sh").write_text(harness)
    try:
        result = subprocess.run(
            ["bash", str(root / "harness.sh")], env=env, capture_output=True, text=True, timeout=60
        )
    finally:
        server_alive = _reap_server(root)
    return result, (root / "events").read_text().splitlines(), server_alive


def _reap_server(root):
    """True when the fake sglang server outlived the harness (it is then killed here)."""
    pid_file = root / "server.pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.02)
    os.kill(pid, signal.SIGKILL)
    return True


def test_serve_model_sh_calls_write_serve_template():
    """scripts/serve_model.sh takes the served template and its parser from launcher.write_serve_template through
    `python -m miles_agent.launcher serve-template` with the checkout on PYTHONPATH (SRC_MOUNT, else its own REPO:
    the old inner script's absolute mount literal is gone); SERVE_TITO_MODEL and SERVE_THINKING stay the knobs,
    the SERVE_PARSER echo stays, no copy of the TITO code."""
    script = REPO / "scripts" / "serve_model.sh"
    text = script.read_text()
    assert "PYTHONPATH=${SRC_MOUNT:-$REPO}${PYTHONPATH:+:$PYTHONPATH} \"$PYTHON\" -m miles_agent.launcher serve-template" in text
    assert "PYTHONPATH=/opt/miles" "-agent" not in text
    assert '"$SERVE_MODEL_PATH" "$TEMPLATE" --thinking "$THINKING" --tito-model "$TITO"' in text
    assert 'echo "SERVE_PARSER $PARSER"' in text and "TITOTokenizerType" not in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_teacher_node_writes_url_and_exits_on_ended(tmp_path):
    """The teacher node starts the campaign's server line over the node's GPUs, publishes http://<ip>:13141 whole
    (.tmp, then mv) once /health_generate answers, serves through a USR1 and ends on $SYNC_DIR/ENDED with
    TEACHER_ENDED, its server stopped by the EXIT trap; a server that dies at boot or a missing model dir leaves
    TEACHER_FAILED and exit 1 instead, and the TEACHER_* knobs reach the server line."""
    root, env = _case(tmp_path, "ended", TEACHER_MODEL="Qwen3.5-9B")
    hf = _model_dir(root, "Qwen3.5-9B")
    result, events, server_alive = _run(root, TEACHER_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "SGLANG_START teacher" in out and f"TEACHER_UP model={hf} url=http://10.0.0.7:13141 dp=4" in out
    assert "TEACHER_SIGNAL: serving until" in out and out.rstrip().endswith("TEACHER_ENDED")
    assert "NOT_REACHED" not in out
    assert (root / "sync" / "TEACHER_URL").read_text().strip() == "http://10.0.0.7:13141"
    assert not (root / "sync" / "TEACHER_URL.tmp").exists() and not (root / "sync" / "TEACHER_FAILED").exists()
    assert not server_alive
    assert [event for event in events if event.startswith("sglang ")] == [
        f"sglang cuda=unset --model-path {hf} --host 0.0.0.0 --port 13141 --tp 1 --dp-size 4 "
        "--mem-fraction-static 0.75 --chunked-prefill-size 4096 --context-length 33792 --disable-cuda-graph "
        "--log-level warning"
    ]
    assert "curl http://127.0.0.1:13141/health_generate" in events
    assert (root / "runs" / "t1_teacher_123.log").is_file()
    # written whole: the head never reads half a URL
    assert (
        '"$SYNC_DIR/TEACHER_URL.tmp" && mv "$SYNC_DIR/TEACHER_URL.tmp" "$SYNC_DIR/TEACHER_URL"'
        in TEACHER_SH.read_text()
    )

    # a boot failure: TEACHER_FAILED, no URL, exit 1 with the log's tail; the knobs and TEACHER_HF reach the line
    root, env = _case(
        tmp_path,
        "dies",
        FAKE_SGLANG_DIES="1",
        TEACHER_HF=str(tmp_path / "elsewhere" / "t9"),
        TEACHER_DP="2",
        TEACHER_MEM_FRAC="0.6",
        TEACHER_CONTEXT_LEN="8192",
    )
    hf = tmp_path / "elsewhere" / "t9"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text("{}")
    result, events, server_alive = _run(root, TEACHER_HARNESS, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FATAL: teacher {hf} failed to boot" in result.stdout and "fake sglang: dying at boot" in result.stdout
    assert (root / "sync" / "TEACHER_FAILED").exists() and not (root / "sync" / "TEACHER_URL").exists()
    assert "TEACHER_UP" not in result.stdout and "TEACHER_ENDED" not in result.stdout and not server_alive
    assert [event for event in events if event.startswith("sglang ")] == [
        f"sglang cuda=unset --model-path {hf} --host 0.0.0.0 --port 13141 --tp 1 --dp-size 2 "
        "--mem-fraction-static 0.6 --chunked-prefill-size 4096 --context-length 8192 --disable-cuda-graph "
        "--log-level warning"
    ]

    # no model dir: refused before anything starts
    root, env = _case(tmp_path, "nodir", TEACHER_MODEL="absent")
    result, events, _ = _run(root, TEACHER_HARNESS, env)
    assert result.returncode == 1
    assert f"FATAL: start_sglang teacher: no HF model dir {root / 'models' / 'absent'}" in result.stdout
    assert (root / "sync" / "TEACHER_FAILED").exists() and not events


def test_teacher_gate_prefers_teacher_url_override(tmp_path):
    """TEACHER_URL in the job environment wins over the sync dir's TEACHER_URL (an external teacher: health-checked,
    no probe without a teacher dir); without it the gate reads the file the teacher node published, and
    TEACHER_FAILED, a silent teacher or a failing health check are FATAL."""
    root, env = _case(tmp_path, "override", TEACHER_URL="http://1.2.3.4:5", FAKE_HEALTH="ok")
    (root / "sync" / "TEACHER_URL").write_text("http://10.0.0.7:13141\n")
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TEACHER_GATE_OK url=http://1.2.3.4:5" in result.stdout
    assert "ARG=--teacher-url http://1.2.3.4:5" in result.stdout
    assert "tokenizer probe skipped (TEACHER_URL names no TEACHER_MODEL / TEACHER_HF" in result.stdout
    assert events == ["curl http://1.2.3.4:5/health_generate"]

    root, env = _case(tmp_path, "published", FAKE_HEALTH="ok")
    (root / "sync" / "TEACHER_URL").write_text("http://10.0.0.7:13141\n")
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ARG=--teacher-url http://10.0.0.7:13141" in result.stdout
    assert events == ["curl http://10.0.0.7:13141/health_generate"]

    root, env = _case(tmp_path, "failed", FAKE_HEALTH="ok")
    (root / "sync" / "TEACHER_FAILED").touch()
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 1 and "FATAL: the teacher node reported a failure" in result.stdout
    assert "TEACHER_GATE_OK" not in result.stdout and not events

    root, env = _case(tmp_path, "silent", FAKE_HEALTH="ok", TEACHER_WAIT_TRIES="3")
    result, _, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 1 and "FATAL: no teacher URL after 3 x 5 s" in result.stdout

    root, env = _case(tmp_path, "unhealthy", TEACHER_URL="http://1.2.3.4:5")
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert (
        result.returncode == 1 and "FATAL: teacher http://1.2.3.4:5 does not answer /health_generate" in result.stdout
    )
    assert events == ["curl http://1.2.3.4:5/health_generate"]


def test_teacher_gate_runs_tokenizer_probe_unless_disabled(tmp_path):
    """With a teacher dir known (TEACHER_MODEL under MILES_AGENT_MODELS, or TEACHER_HF)
    the gate runs
    `python3 -m miles_agent.launcher tokenizer-probe <POLICY_HF> <teacher>` before TEACHER_GATE_OK; a drift is FATAL;
    TEACHER_TOKENIZER_CHECK=0 skips it."""
    student = tmp_path / "students" / "S"
    student.mkdir(parents=True)
    common = dict(TEACHER_URL="http://1.2.3.4:5", FAKE_HEALTH="ok", POLICY_HF=str(student))

    root, env = _case(tmp_path, "probe", TEACHER_MODEL="Qwen3.5-9B", **common)
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert events == ["curl http://1.2.3.4:5/health_generate", f"probe {student} {root / 'models' / 'Qwen3.5-9B'}"]
    assert result.stdout.index("TOKENIZER_PROBE_OK") < result.stdout.index("TEACHER_GATE_OK url=http://1.2.3.4:5")

    root, env = _case(tmp_path, "drift", TEACHER_MODEL="Qwen3.5-9B", FAKE_PROBE_FAIL="1", **common)
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 1
    assert "TOKENIZER_PROBE_FAIL vocabulary sizes differ" in result.stderr
    assert "FATAL: teacher tokenizer differs from the student's" in result.stdout
    assert "TEACHER_TOKENIZER_CHECK=0 skips the probe" in result.stdout and "TEACHER_GATE_OK" not in result.stdout
    assert (root / "runs" / "t1_tokenizer_probe_123.log").read_text().startswith("TOKENIZER_PROBE_FAIL")

    # the verdict behind stderr without a trailing newline (a warning on the same line): still the drift, never
    # "could not run"
    root, env = _case(
        tmp_path, "noisy_drift", TEACHER_MODEL="Qwen3.5-9B", FAKE_PROBE_FAIL="1", FAKE_PROBE_NOISE="1", **common
    )
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 1 and "FATAL: teacher tokenizer differs from the student's" in result.stdout
    assert "could not run" not in result.stdout and "TEACHER_GATE_OK" not in result.stdout
    log = (root / "runs" / "t1_tokenizer_probe_123.log").read_text()
    assert log.startswith("warning: slow tokenizer in useTOKENIZER_PROBE_FAIL vocabulary sizes differ")

    # a probe that never compared the tokenizers (a missing dir, an import error): its stderr and the knobs that
    # named the dirs, not the drift verdict
    root, env = _case(tmp_path, "crash", TEACHER_MODEL="Qwen3.5-9B", FAKE_PROBE_CRASH="1", **common)
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 1 and "OSError" in result.stderr and "TEACHER_GATE_OK" not in result.stdout
    assert "FATAL: the tokenizer probe could not run" in result.stdout
    assert "teacher tokenizer differs" not in result.stdout
    knobs = f"TEACHER_MODEL=Qwen3.5-9B TEACHER_HF= -> {root / 'models' / 'Qwen3.5-9B'}, POLICY_HF={student}"
    assert knobs in result.stdout
    assert "OSError" in (root / "runs" / "t1_tokenizer_probe_123.log").read_text()

    root, env = _case(tmp_path, "disabled", TEACHER_MODEL="Qwen3.5-9B", TEACHER_TOKENIZER_CHECK="0", **common)
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert events == ["curl http://1.2.3.4:5/health_generate"]
    assert (
        "tokenizer probe skipped (TEACHER_TOKENIZER_CHECK=0)" in result.stdout and "TEACHER_GATE_OK" in result.stdout
    )

    # the dir the probe reads: TEACHER_HF as given, else the name under MILES_AGENT_MODELS
    root, env = _case(tmp_path, "hf", TEACHER_HF="/elsewhere/t9", TEACHER_MODEL="ignored", **common)
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0 and events[-1] == f"probe {student} /elsewhere/t9"
    root, env = _case(tmp_path, "models_dir", TEACHER_MODEL="Qwen3.5-9B", MILES_AGENT_MODELS="/mnt/models", **common)
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0 and events[-1] == f"probe {student} /mnt/models/Qwen3.5-9B"


def test_teacher_gate_exits_on_preempt_sentinel(tmp_path):
    """A preemption while the teacher boots: the wait loop sees PREEMPT_SENTINEL on its next poll, logs
    TEACHER_GATE_PREEMPT and takes the entry's "PREEMPT before launch" path (exit 0, the batch shell's
    preempt_finish requeues on the sentinel) instead of holding the head for the remaining TEACHER_WAIT_TRIES polls:
    no health check, no probe, no TEACHER_GATE_OK. An external TEACHER_URL waits for nothing, so the entry's own
    launch gate, two steps later, is the one that sees its sentinel."""
    sentinel = tmp_path / "preempt" / "sync" / "preempt"
    root, env = _case(tmp_path, "preempt", TEACHER_WAIT_TRIES="400", PREEMPT_SENTINEL=sentinel)
    result, events, _ = _run(root, GATE_PREEMPT_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TEACHER_GATE_PREEMPT" in result.stdout
    assert "PREEMPT before launch: not starting the driver" in result.stdout
    assert "TEACHER_GATE_OK" not in result.stdout and "ARG=" not in result.stdout and "FATAL" not in result.stdout
    assert events == [] and sentinel.exists()
    # the entry's launch gate (scripts/run_local.sh, the body the old inner script execs): the same line, the same exit status
    assert 'echo "PREEMPT before launch: not starting the driver"; exit 0' in ENTRY.read_text()

    # a sentinel already up when the gate starts: the first poll takes it
    sentinel = tmp_path / "early" / "sync" / "preempt"
    root, env = _case(tmp_path, "early", PREEMPT_SENTINEL=sentinel)
    sentinel.touch()
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0 and "TEACHER_GATE_PREEMPT" in result.stdout and events == []

    sentinel = tmp_path / "external" / "sync" / "preempt"
    root, env = _case(
        tmp_path, "external", TEACHER_URL="http://1.2.3.4:5", FAKE_HEALTH="ok", PREEMPT_SENTINEL=sentinel
    )
    sentinel.touch()
    result, events, _ = _run(root, GATE_HARNESS, env)
    assert result.returncode == 0 and "TEACHER_GATE_OK url=http://1.2.3.4:5" in result.stdout
    assert "TEACHER_GATE_PREEMPT" not in result.stdout


def test_start_sglang_returns_nonzero_without_gpus(tmp_path):
    """Where nvidia-smi lists no GPU (or the CUDA id list is empty) start_sglang prints its one-line FATAL and
    returns 1 before any server starts, under a caller's set -e too: a bare call then ends the caller WITH the
    reason (the bare `grep -c` it had exited 1 on a zero count and took the caller down without a word), a guarded
    call goes on."""
    root, env = _case(tmp_path, "nogpu", FAKE_GPUS="0")
    env["MODEL_DIR"] = str(_model_dir(root, "Qwen3.5-9B"))
    result, events, server_alive = _run(root, SGLANG_HARNESS, env)
    assert result.returncode == 1, result.stdout + result.stderr
    fatal = "FATAL: start_sglang teacher: --dp-size 1 over 0 GPU(s) (all)"
    assert result.stdout.count(fatal) == 2 and "GUARDED rc=1" in result.stdout
    assert "NOT_REACHED" not in result.stdout and "SGLANG_START" not in result.stdout
    assert events == [] and not server_alive

    root, env = _case(tmp_path, "emptyids", CUDA_IDS="")
    env["MODEL_DIR"] = str(_model_dir(root, "Qwen3.5-9B"))
    result, events, server_alive = _run(root, SGLANG_HARNESS, env)
    assert result.returncode == 1
    assert result.stdout.count("FATAL: start_sglang teacher: --dp-size 1 over 0 GPU(s) ()") == 2
    assert "NOT_REACHED" not in result.stdout and events == [] and not server_alive

    # the lib has no other bare count: every grep -c carries its `|| true`
    code = "\n".join(line for line in SGLANG_SH.read_text().splitlines() if not line.lstrip().startswith("#"))
    assert code.count("grep -c") == 2 == code.count("|| true)")


def test_wait_sglang_returns_2_on_stop_file(tmp_path):
    """wait_sglang's optional fifth argument is a stop file (the pipeline's preempt sentinel), checked first at every
    poll: raised while the server still boots (its pid alive, /health_generate not answering) the wait ends at that
    poll with `wait_sglang: <name>: stop file <path> raised while waiting` and return 2, no log tail; absent, or
    without the argument, the wait runs its tries and fails with the tail and return 1 as before."""
    root, env = _case(tmp_path, "stop")
    stop = root / "stop"
    env["STOP"] = str(stop)
    result, events, _ = _run(root, WAIT_HARNESS, env)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "RC_ABSENT=1" in out and "RC_RAISED=2" in out and "RC_NOARG=1" in out and "SGLANG_UP" not in out
    assert f"wait_sglang: teacher: stop file {stop} raised while waiting" in out
    assert out.count("not ready after 3 x 1s") == 1 and out.count("not ready after 2 x 1s") == 1
    assert out.count("last 100 lines of") == 2 == out.count("loading shards")  # the tail twice, never on the stop
    assert events.count("curl http://127.0.0.1:13141/health_generate") >= 5 and stop.exists()


def test_retired_knobs_are_fatal():
    """Every campaign-branch knob main does not read is refused with its replacement when set (an empty value, as the
    campaign's TEACHER2_MODEL=, is not a setting); a clean environment passes silently."""
    base = {
        key: value for key, value in os.environ.items() if key not in RETIRED_KNOBS and not key.endswith("_OPD_REFS")
    }

    def source(**knobs):
        return subprocess.run(
            ["bash", "-c", 'set -eu; . "$1"', "_", str(RETIRED_SH)],
            env={**base, **knobs},
            capture_output=True,
            text=True,
            timeout=30,
        )

    clean = source()
    assert clean.returncode == 0 and clean.stdout == "" and clean.stderr == ""
    for knob, replacement in RETIRED_KNOBS.items():
        result = source(**{knob: "1"})
        assert result.returncode == 1, knob
        assert f"FATAL: {knob} is a retired knob: " in result.stdout and replacement in result.stdout, knob
    assert source(TEACHER2_MODEL="").returncode == 0
    # the pattern, not a list: any *_OPD_REFS
    assert source(SEARCH_R1_OPD_REFS="1").returncode == 1


def test_public_helper_anchors():
    """The public R2S helpers and the split's public scripts parse and make no scheduler call (a plain machine with two
    servers sources them; the scheduler's words are spelled in pieces here, the site-literal gate scans this file), and
    the helpers go through the launcher's CLI: sglang.sh's serve-template, teacher_node.sh's tokenizer-probe;
    r2s_pipeline.sh sources sglang.sh from the checkout (SRC_MOUNT, default: this repo)."""
    for script in (SGLANG_SH, TEACHER_SH, RETIRED_SH, PIPELINE_SH, ENTRY, RAY_SH, LADDER_SH):
        subprocess.run(["bash", "-n", str(script)], check=True)
    scheduler = re.compile(r"\b(sr" "un|sb" "atch|scon" r"trol|squeue|scancel)\b" "|SLU" "RM_")
    for script in (SGLANG_SH, TEACHER_SH, RETIRED_SH, PIPELINE_SH):
        code = "\n".join(line for line in script.read_text().splitlines() if not line.lstrip().startswith("#"))
        assert not scheduler.search(code), script
    # the split's public scripts: no scheduler word at all (ray.sh's two `unset` lines scrub the scheduler's family)
    scheduler_any = re.compile(scheduler.pattern + "|SLU" "RM")
    for script in (ENTRY, REPO / "scripts" / "ray_cluster.sh", REPO / "scripts" / "lib" / "common.sh", RAY_SH, LADDER_SH):
        code = "\n".join(line for line in script.read_text().splitlines()
                         if not line.lstrip().startswith("#") and not (script == RAY_SH and line.lstrip().startswith("unset ")))
        assert not scheduler_any.search(code), script
    assert 'python3 -m miles_agent.launcher serve-template "$1" "$2"' in SGLANG_SH.read_text()
    assert 'python3 -m miles_agent.launcher tokenizer-probe "${POLICY_HF:?}" "$hf"' in TEACHER_SH.read_text()
    assert '. "${SRC_MOUNT:-$REPO}/scripts/lib/sglang.sh"' in PIPELINE_SH.read_text()


def test_entry_script_anchors():
    """scripts/run_local.sh carries the teacher plumbing where the design puts it (the anchors the old inner script
    held before I2a): the public helpers sourced from the checkout, the teacher node before the ray-worker branch,
    the gate between the gym start (gym_start_and_wait prints GYM_UP) and PREEMPT_ARGS with its --teacher-url on the
    driver line, the retired knobs next to the STUDENT_MODEL refusal, ray's GPU expectation over the ray nodes only
    (the count on the entry's line, the one call shape in ray.sh); MODE=r2s_pipeline sources and runs the public
    stage driver right after the gym (before the teacher gate: the pipeline's teacher is its own server), exits 0
    with RUN_INNER_PREEMPT_STOP on a sentinel stop and with RUN_INNER_PASS otherwise."""
    entry = ENTRY.read_text()
    assert (
        entry.index('. "$LIB/sglang.sh"')
        < entry.index('. "$REPO/scripts/teacher_node.sh"')
        < entry.index("then teacher_node_main; fi")
        < entry.index("# ---- ray worker branch")
    )
    assert 'async_4n_teacher|async_2n_teacher) [ -n "${TEACHER_URL:-}" ] || TEACHER_RANK=$(( NNODES - 1 )) ;;' in entry
    gym_at = entry.index("\n  gym_start_and_wait\n")
    assert gym_at < entry.index("then teacher_gate; fi") < entry.index('PREEMPT_ARGS=""')
    assert "  $PREEMPT_ARGS \\\n  $TEACHER_URL_ARG \\\n  ${EXTRA_ARGS:-} &" in entry
    assert '--num-nodes "$RAY_NODES" --num-gpus-per-node "$GPUS_PER_NODE"' in entry
    branch_at = entry.index('if [ "$MODE" = r2s_pipeline ]; then')
    assert gym_at < branch_at < entry.index("then teacher_gate; fi")
    branch = entry[branch_at : entry.index("\nfi\n", branch_at)]
    assert '  . "$REPO/scripts/r2s_pipeline.sh"\n  r2s_pipeline_main\n' in branch
    assert 'echo "RUN_INNER_PREEMPT_STOP rc=0 durable=' in branch and 'echo "RUN_INNER_PASS"; exit 0' in branch
    lines = entry.splitlines()
    student = next(
        i for i, line in enumerate(lines) if "FATAL: STUDENT_MODEL / STUDENT_HF are now POLICY_MODEL" in line
    )
    retired = next(i for i, line in enumerate(lines) if line == '. "$LIB/retired_knobs.sh"')
    assert 0 < retired - student <= 2
    assert "RAY_NODES=$(( NNODES - 1 ))" in entry and "MILES_EXPECT_GPUS=$(( RAY_NODES * GPUS_PER_NODE ))" in entry
    assert "MILES_EXPECT_GPUS=$(( NNODES * GPUS_PER_NODE ))" not in entry
    assert "MILES_EXPECT_GPUS=$1 MILES_GCS=$2 \"$PYTHON\" - <<'PYEOF'" in RAY_SH.read_text()
    assert 'echo "GYM_UP agent=http://127.0.0.1:$NEMO_GYM_PORT"' in LADDER_SH.read_text()
    assert 'echo "PREEMPT before launch: not starting the driver"; exit 0' in entry
    assert lines[-1] == 'echo "RUN_INNER_PASS"' and 'echo "RUN_INNER_PREEMPT_STOP rc=$RUN_RC durable=' in entry
    assert "MILES_PIN_ACTIVE path=" in entry and "miles pin active:" not in entry
    # the exec is the wrapper's; the entry IS the body (its header may say so in a comment)
    assert [l for l in entry.splitlines() if not l.lstrip().startswith("#") and "exec bash" in l] == []


def test_r2s_teacher_ft_sh_anchors():
    """Stage E's entry (scripts/r2s_teacher_ft.sh): it parses, makes no scheduler call and deletes nothing (an earlier
    .tmp is set aside), its header names the output convention, it sources scripts/lib/common.sh for the roots and the
    job id (the models dir is MILES_AGENT_MODELS, a wrapper's MODELS_DIR, else the repo's; the scratch dir LOCAL, a
    wrapper's NODE_LOCAL, else /tmp), takes --dry-run, and runs torchrun into <TAG>.tmp before the mv on the pass."""
    subprocess.run(["bash", "-n", str(FT_SH)], check=True)
    sh = FT_SH.read_text()
    code = "\n".join(line for line in sh.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"(^|[\s;&|(`])rm\s", code)
    assert not re.search(r"\b(sr" "un|sb" "atch|scon" r"trol|squeue|scancel)\b", code)
    assert "r2s-<env>-<teacher>-teacher-for-<student>" in sh.split("set -e")[0]
    assert "MODELS=$MILES_AGENT_MODELS" in sh and "OUT_DIR=$MODELS/$TAG" in sh and "SRC=${SRC_MOUNT:-" in sh
    assert 'SCRATCH=${LOCAL:-${NODE_LOCAL:-/tmp}}' in sh and '. "$SRC/scripts/lib/common.sh"' in sh
    assert 'DRY_RUN=${DRY_RUN:-}' in sh
    assert 'torchrun --standalone --nproc-per-node "$NGPU" -m miles_agent.r2s.teacher_ft' in sh
    assert sh.index('--out-dir "$OUT_DIR.tmp"') < sh.index('mv "$OUT_DIR.tmp" "$OUT_DIR"')
    assert "aside=$OUT_DIR.failed-$JOB n=0" in sh and 'mv "$OUT_DIR.tmp" "$aside"' in sh


def _ft_case(tmp_path, name, **settings):
    """A stage E root: the base teacher under $ROOT/models, a labels file, the knobs; LOCAL, SRC_MOUNT and the job id
    set as a job wrapper (or a plain machine) would."""
    root, env = _case(tmp_path, name)
    _model_dir(root, "Qwen3.5-9B")
    (root / "labels.jsonl").write_text('{"state_id": "s", "objective": "suffix", "ids": [1], "positions": []}\n')
    env.update(
        {
            "TAG": "r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b",
            "LABELS": str(root / "labels.jsonl"),
            "LOCAL": str(root / "local"),
            "SRC_MOUNT": str(REPO),
            "JOB_ID": "123",
            "MILES_AGENT_MODELS": str(root / "models"),
            **{key: str(value) for key, value in settings.items()},
        }
    )
    result = subprocess.run(["bash", str(FT_SH)], env=env, capture_output=True, text=True, timeout=60)
    return root, result, (root / "events").read_text().splitlines()


def test_r2s_teacher_ft_sh_promotes_on_pass_only(tmp_path):
    """scripts/r2s_teacher_ft.sh with a fake torchrun: the campaign's command line over the machine's GPUs into
    <models>/<TAG>.tmp, moved to <models>/<TAG> and its report printed on an exit 0 (R2S_FT_PASS); a gate refusal
    (torchrun exits 1 whatever the worker's code; the report says passed false) keeps the .tmp with its report,
    promotes nothing and exits 2 (R2S_FT_FAIL); a crash before the verdict exits with torchrun's code; an existing
    <TAG> is refused before anything runs; an earlier .tmp is set aside as .failed-<job>, or .failed-<job>.<n> with
    the first free n when that name is taken (never nested inside it); --skip-gate's unimproved export (torchrun
    exits 0, the report says passed false) is promoted with gate=skipped on the marker; LR / EPOCHS / GRAD_ACCUM /
    FT_EXTRA and MILES_AGENT_MODELS reach the command line; no GPU or no labels file is FATAL."""
    tag = "r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b"
    root, result, events = _ft_case(tmp_path, "pass")
    assert result.returncode == 0, result.stdout + result.stderr
    models = root / "models"
    assert (models / tag / "r2s_ft_report.json").is_file() and (models / tag / "model.safetensors").is_file()
    assert not (models / f"{tag}.tmp").exists()
    start = f"R2S_FT_START tag={tag} labels={root / 'labels.jsonl'} base={models / 'Qwen3.5-9B'}"
    assert f"{start} out={models / tag} ngpu=4" in result.stdout
    assert '"passed": true' in result.stdout and result.stdout.rstrip().endswith(f"R2S_FT_PASS out={models / tag}")
    assert events[0] == (
        f"torchrun --standalone --nproc-per-node 4 -m miles_agent.r2s.teacher_ft --data {root / 'labels.jsonl'} "
        f"--model-path {models / 'Qwen3.5-9B'} --out-dir {models / tag}.tmp --lr 2e-6 --epochs 1 --grad-accum 8"
    )
    assert events[1].startswith(f"torchrun-env PYTHONPATH={REPO}")
    assert f"TRITON_CACHE_DIR={root / 'local'}/jit/triton" in events[1]

    root, result, events = _ft_case(tmp_path, "fail", FAKE_FT="fail")
    assert result.returncode == 2, result.stdout + result.stderr
    models = root / "models"
    assert '"passed": false' in (models / f"{tag}.tmp" / "r2s_ft_report.json").read_text()
    assert not (models / tag).exists() and "R2S_FT_PASS" not in result.stdout
    assert f"R2S_FT_FAIL tag={tag}: the gate refused the teacher" in result.stdout
    assert "FT_FAIL val hind CE did not improve" in result.stdout and len(events) == 2

    root, result, events = _ft_case(tmp_path, "crash", FAKE_FT="crash")
    assert result.returncode == 1 and "FATAL: torchrun exited 1 before the gate verdict" in result.stdout
    assert not (root / "models" / f"{tag}.tmp").exists() and not (root / "models" / tag).exists()

    # an existing output: refused before torchrun; an earlier .tmp (a failed run's report): set aside, then a pass
    (tmp_path / "existing" / "models" / tag).mkdir(parents=True)
    root, result, events = _ft_case(tmp_path, "existing")
    assert result.returncode == 1 and f"FATAL: {root / 'models' / tag} exists; pick a new TAG" in result.stdout
    assert events == []
    stale = tmp_path / "stale" / "models" / f"{tag}.tmp"
    stale.mkdir(parents=True)
    (stale / "r2s_ft_report.json").write_text('{"passed": false}')
    root, result, events = _ft_case(tmp_path, "stale")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "models" / f"{tag}.failed-123" / "r2s_ft_report.json").read_text() == '{"passed": false}'
    assert '"passed": true' in (root / "models" / tag / "r2s_ft_report.json").read_text()

    # the same job id again (a requeue, a third run): .failed-<job> taken -> .failed-<job>.1, then .2; never nested
    for name, taken in (("aside1", ("",)), ("aside2", ("", ".1"))):
        models = tmp_path / name / "models"
        stale = models / f"{tag}.tmp"
        stale.mkdir(parents=True)
        (stale / "r2s_ft_report.json").write_text('{"passed": false, "run": "stale"}')
        for suffix in taken:
            (models / f"{tag}.failed-123{suffix}").mkdir()
            (models / f"{tag}.failed-123{suffix}" / "r2s_ft_report.json").write_text('{"run": "earlier"}')
        root, result, events = _ft_case(tmp_path, name)
        assert result.returncode == 0, result.stdout + result.stderr
        free = models / f"{tag}.failed-123.{len(taken)}"
        assert (free / "r2s_ft_report.json").read_text() == '{"passed": false, "run": "stale"}'
        for suffix in taken:
            assert (models / f"{tag}.failed-123{suffix}" / "r2s_ft_report.json").read_text() == '{"run": "earlier"}'
            assert not (models / f"{tag}.failed-123{suffix}" / f"{tag}.tmp").exists()
        assert '"passed": true' in (models / tag / "r2s_ft_report.json").read_text()

    # --skip-gate (FT_EXTRA): torchrun exits 0 with a report saying passed false; promoted, and the marker says so
    root, result, events = _ft_case(tmp_path, "skipgate", FAKE_FT="skipgate", FT_EXTRA="--skip-gate")
    assert result.returncode == 0, result.stdout + result.stderr
    models = root / "models"
    assert '"passed": false' in (models / tag / "r2s_ft_report.json").read_text()
    assert (models / tag / "model.safetensors").is_file() and not (models / f"{tag}.tmp").exists()
    assert result.stdout.rstrip().endswith(f"R2S_FT_PASS gate=skipped out={models / tag}")
    assert "R2S_FT_FAIL" not in result.stdout and events[0].endswith("--grad-accum 8 --skip-gate")

    # the knobs and the models dir
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "Qwen3-30B-A3B-Instruct-2507").mkdir(parents=True)
    (elsewhere / "Qwen3-30B-A3B-Instruct-2507" / "config.json").write_text("{}")
    root, result, events = _ft_case(
        tmp_path,
        "knobs",
        MILES_AGENT_MODELS=str(elsewhere),
        BASE_MODEL="Qwen3-30B-A3B-Instruct-2507",
        LR="1e-6",
        EPOCHS="2",
        GRAD_ACCUM="4",
        FT_EXTRA="--skip-gate --warmup-steps 3",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert events[0].endswith(
        f"--data {root / 'labels.jsonl'} --model-path {elsewhere / 'Qwen3-30B-A3B-Instruct-2507'} "
        f"--out-dir {elsewhere / tag}.tmp --lr 1e-6 --epochs 2 --grad-accum 4 --skip-gate --warmup-steps 3"
    )
    assert (elsewhere / tag / "r2s_ft_report.json").is_file() and not (root / "models" / tag).exists()

    root, result, events = _ft_case(tmp_path, "nogpu", FAKE_GPUS="0")
    assert result.returncode == 1 and "FATAL: nvidia-smi lists no GPU" in result.stdout and events == []
    root, result, events = _ft_case(tmp_path, "nolabels", LABELS=str(tmp_path / "absent.jsonl"))
    assert result.returncode == 1 and f"FATAL: no labels file {tmp_path / 'absent.jsonl'}" in result.stdout
    assert events == []


def test_r2s_teacher_ft_sh_dry_run_under_the_wrapper_names(tmp_path):
    """The FT entry under a job wrapper's Contract A names alone (MODELS_DIR, NODE_LOCAL, JOB_ID; no LOCAL, no
    MILES_AGENT_* name): --dry-run prints the torchrun argv it would run, the models dir taken from MODELS_DIR through
    scripts/lib/common.sh, and exits 0 before any gate or start (no FATAL, no R2S_FT_START, no torchrun call)."""
    root, env = _case(tmp_path, "wrapper")
    _model_dir(root, "Qwen3.5-9B")
    (root / "labels.jsonl").write_text('{"state_id": "s", "objective": "suffix", "ids": [1], "positions": []}\n')
    for key in list(env):
        if key == "LOCAL" or key.startswith("MILES_AGENT_"):
            del env[key]
    tag = "r2s-alfworld-qwen3.5-9b-teacher-for-qwen3.5-4b"
    env.update(MODELS_DIR=str(root / "models"), NODE_LOCAL=str(root / "local"), JOB_ID="123", LOCAL_ENV="/dev/null",
               SRC_MOUNT=str(REPO), TAG=tag, LABELS=str(root / "labels.jsonl"), BASE_MODEL="Qwen3.5-9B")
    result = subprocess.run(["bash", str(FT_SH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout.splitlines()
    assert len(out) == 1 and out[0].startswith("DRY_RUN torchrun: torchrun --standalone --nproc-per-node "), result.stdout
    assert (f" -m miles_agent.r2s.teacher_ft --data {root / 'labels.jsonl'} --model-path {root / 'models' / 'Qwen3.5-9B'} "
            f"--out-dir {root / 'models' / tag}.tmp ") in out[0]
    assert "FATAL" not in result.stdout + result.stderr and "R2S_FT_START" not in result.stdout
    assert not [event for event in (root / "events").read_text().splitlines() if event.startswith("torchrun")]


def _pipeline_case(tmp_path, name, **settings):
    """A pipeline root: the base teacher, the student and the judge named under $ROOT/models, an ALFWorld job (RECIPE,
    TAG, DUMP_GLOB, POLICY_HF as scripts/run_local.sh sets it) with MIN_STATES 2 for the fake stages' three rows; the
    harness runs from the root, where the driver's one `python3 -c` (the env's check) imports the repo's select."""
    knobs = dict(TEACHER_MODEL="Qwen3.5-9B", RECIPE="alfworld", TAG="p1", MIN_STATES="2")
    knobs.update(settings)
    root, env = _case(tmp_path, name, **knobs)
    for model in {"Qwen3.5-9B", "Qwen3.5-4B", knobs.get("JUDGE_MODEL", "Qwen3.5-9B")}:
        _model_dir(root, model)
    env.setdefault("POLICY_HF", str(root / "models" / "Qwen3.5-4B"))
    env.setdefault("DUMP_GLOB", str(root / "dump_*.jsonl"))
    env.update(PIPELINE_SH=str(PIPELINE_SH), SRC_MOUNT=str(REPO), FAKE_REPO=str(REPO))
    (root / "harness.sh").write_text(PIPELINE_HARNESS)
    result = subprocess.run(
        ["bash", str(root / "harness.sh")], env=env, cwd=root, capture_output=True, text=True, timeout=120
    )
    return root, result, (root / "events").read_text().splitlines()


def _kinds(events, *prefixes):
    """The second word of every event of the given kinds (start / stop / stage / skip), in order."""
    return [event.split()[1] for event in events if event.startswith(prefixes)]


def test_r2s_stages_markers_and_gates(tmp_path):
    """scripts/r2s_pipeline.sh with fake stage CLIs and the servers as a recorder: the campaign's server lines (the
    teacher on GPUs 0,1 DP2 with the served template and both parsers, alive to the end; the student on GPU 2 for
    stage A only), every stage's flags in order, the markers as gates (MIN_STATES with its default of 100, a gate
    refusal, a crash, a missing marker), STOP_AFTER_FOLD, STAGES subsets with the inputs they need, J before C for a
    judge env on the teacher's own server or, when JUDGE_MODEL is another dir, on :13143 for stage J alone while the
    teacher is never stopped before LABEL_DONE and stage D prefills on :13141 (F1), R2S_ARGS_<S> verbatim and last,
    the stages' DONE skip and --force, and a sentinel raised inside stage B: <sentinel>.done, exit 0, nothing after."""
    root, result, events = _pipeline_case(tmp_path, "alf")
    assert result.returncode == 0, result.stdout + result.stderr
    out = root / "data" / "r2s" / "p1"
    teacher, student = root / "models" / "Qwen3.5-9B", root / "models" / "Qwen3.5-4B"
    server = "--mem-fraction-static 0.8 --chunked-prefill-size 4096 --context-length 32768"
    served = "--tool-call-parser fake_tool_parser --reasoning-parser fake_reasoning_parser --served-model-name"
    stage_a = (
        f"stage A --env alfworld --dump-glob {root}/dump_*.jsonl --student-url http://127.0.0.1:13142 "
        f"--teacher-url http://127.0.0.1:13141 --tokenizer-path {student} --top-n 2400 --max-per-goal 3 --workers 32 "
        f"--out {out}/states.jsonl"
    )
    stage_b = (
        f"stage B --env alfworld --states {out}/states.jsonl --teacher-url http://127.0.0.1:13141 "
        f"--teacher-model teacher --samples 8 --temperature 1.0 --concurrency 32 --min-replay-ok 0.7 "
        f"--out {out}/fold.jsonl"
    )
    stage_d = (
        f"stage D --env alfworld --states {out}/states.jsonl --fold {out}/fold.jsonl --tau {out}/tau.jsonl "
        f"--tokenizer-path {student} --teacher-url http://127.0.0.1:13141 --k 16 --out {out}/labels.jsonl"
    )
    assert events == [
        f"launcher serve-template {teacher} {out}/teacher_template.jinja",
        f"launcher reasoning-parser {teacher}",
        f"start teacher {teacher} 13141 0,1 2 {server} --chat-template {out}/teacher_template.jinja {served} teacher "
        "--log-level warning",
        f"start student {student} 13142 2 1 {server} --disable-cuda-graph --log-level warning",
        "wait teacher http://127.0.0.1:13141/health_generate",
        "wait student http://127.0.0.1:13142/health_generate",
        stage_a,
        "stop student",
        stage_b,
        f"stage C --env alfworld --fold {out}/fold.jsonl --out {out}/tau.jsonl",
        stage_d,
        "stop teacher",
    ]
    lines = result.stdout.splitlines()
    for marker in (
        "R2S_PIPELINE_START tag=p1 env=alfworld check=vote stages=A,B,C,D ",
        "R2S_DIVERGENCE_PASS states=3",
        f"R2S_TAKEOVER_PASS fold={out}/fold.jsonl",
        f"R2S_SELECT_PASS tau={out}/tau.jsonl",
        f"R2S_LABEL_PASS labels={out}/labels.jsonl",
        "R2S_PIPELINE_PASS",
    ):
        assert any(line.startswith(marker) for line in lines), marker
    assert lines[-1] == "HARNESS_END" and "fold_only" not in result.stdout
    assert (out / "teacher_template.jinja").is_file()
    assert (out / "labels.jsonl.summary.json").read_text().endswith("}\nLABEL_DONE\n")

    # the gate after A: the campaign's 100 states by default, counted off the states file
    root, result, events = _pipeline_case(tmp_path, "thin", MIN_STATES="", FAKE_ROWS_A="99")
    assert result.returncode == 1 and "FATAL: 99 states under MIN_STATES=100" in result.stdout
    assert _kinds(events, "stage ") == ["A"] and "R2S_DIVERGENCE_PASS" not in result.stdout
    root, result, events = _pipeline_case(tmp_path, "enough", MIN_STATES="", FAKE_ROWS_A="100")
    assert result.returncode == 0 and "R2S_DIVERGENCE_PASS states=100" in result.stdout

    # a stage's gate refusal (exit 2, the FAIL marker), a crash and a missing marker end the job: nothing runs after
    root, result, events = _pipeline_case(tmp_path, "gate", FAKE_STAGE_B="gate")
    assert result.returncode == 1 and "FATAL: stage B (miles_agent.r2s.takeover) exited 2" in result.stdout
    assert (root / "data" / "r2s" / "p1" / "fold.jsonl.summary.json").read_text().endswith("TAKEOVER_FAIL\n")
    assert _kinds(events, "stage ") == ["A", "B"] and "HARNESS_END" not in result.stdout
    root, result, events = _pipeline_case(tmp_path, "crash", FAKE_STAGE_D="crash")
    assert result.returncode == 1 and "FATAL: stage D (miles_agent.r2s.label) exited 1" in result.stdout
    root, result, events = _pipeline_case(tmp_path, "nomarker", FAKE_STAGE_C="nomarker")
    assert result.returncode == 1 and "FATAL: stage C ended without SELECT_DONE" in result.stdout
    assert _kinds(events, "stage ") == ["A", "B", "C"]

    # STOP_AFTER_FOLD=1: the fold alone
    root, result, events = _pipeline_case(tmp_path, "fold", STOP_AFTER_FOLD="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "R2S_PIPELINE_PASS fold_only" in result.stdout and "R2S_SELECT_PASS" not in result.stdout
    assert _kinds(events, "stage ") == ["A", "B"] and result.stdout.rstrip().endswith("HARNESS_END")

    # STAGES subsets with their inputs: no student server without A, no server at all for a selection alone
    states_in = tmp_path / "earlier_states.jsonl"
    states_in.write_text('{"state_id": "s"}\n' * 5)
    fold_in = tmp_path / "earlier_fold.jsonl"
    fold_in.write_text('{"state_id": "s", "sample": 0}\n')
    root, result, events = _pipeline_case(tmp_path, "from_states", STAGES="B,C,D", STATES_IN=str(states_in))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stages=B,C,D " in result.stdout and "R2S_DIVERGENCE_PASS states=5" in result.stdout
    assert _kinds(events, "start ", "stage ") == ["teacher", "B", "C", "D"]
    assert any(event.startswith(f"stage B --env alfworld --states {states_in} ") for event in events)
    root, result, events = _pipeline_case(tmp_path, "from_fold", STATES_IN=str(states_in), FOLD_IN=str(fold_in))
    assert result.returncode == 0 and "stages=C,D " in result.stdout
    assert _kinds(events, "start ", "stage ") == ["teacher", "C", "D"]
    assert any(event.startswith(f"stage D --env alfworld --states {states_in} --fold {fold_in} ") for event in events)
    root, result, events = _pipeline_case(tmp_path, "select_only", STAGES="C", FOLD_IN=str(fold_in))
    assert result.returncode == 0 and "R2S_PIPELINE_PASS" in result.stdout
    assert _kinds(events, "launcher ", "start ", "stage ") == ["C"]
    root, result, events = _pipeline_case(
        tmp_path, "label_only", STAGES="D", STATES_IN=str(states_in), FOLD_IN=str(fold_in)
    )
    tau = root / "data" / "r2s" / "p1" / "tau.jsonl"
    assert result.returncode == 1 and f"FATAL: no selection {tau}: stage C in STAGES" in result.stdout
    assert events == []
    root, result, events = _pipeline_case(tmp_path, "unknown_env", RECIPE="search_r1")
    assert result.returncode == 1 and "FATAL: RECIPE=search_r1 is not an R2S environment" in result.stdout
    assert events == []
    root, result, events = _pipeline_case(tmp_path, "no_gpu_left", TEACHER_DP="4")
    assert result.returncode == 1 and events == []
    assert "FATAL: TEACHER_DP=4 takes all 4 GPUs: none left for the student server (stage A)" in result.stdout
    # STOP_AFTER_FOLD=1 without B in STAGES (STATES_IN + FOLD_IN default to C,D): nothing to stop after, FATAL before
    # any server boots, no fold_only pass
    root, result, events = _pipeline_case(
        tmp_path, "fold_no_b", STOP_AFTER_FOLD="1", STATES_IN=str(states_in), FOLD_IN=str(fold_in)
    )
    assert result.returncode == 1 and "FATAL: STOP_AFTER_FOLD=1 needs stage B in STAGES" in result.stdout
    assert events == [] and "fold_only" not in result.stdout
    # a generic dir unset: the expansion's one FATAL line, never set -u's "unbound variable"
    for given, missing in (({"MILES_AGENT_MODELS": "/m"}, "MILES_AGENT_DATA"), ({}, "MILES_AGENT_MODELS")):
        run = subprocess.run(
            ["bash", "-c", 'set -eu; add_cleanup() { :; }; . "$SRC_MOUNT/scripts/r2s_pipeline.sh"; r2s_pipeline_main'],
            env={
                "PATH": os.environ["PATH"],
                "SRC_MOUNT": str(REPO),
                "TAG": "p1",
                "RUN_TAG": "t1",
                "RECIPE": "alfworld",
                "TEACHER_MODEL": "Qwen3.5-9B",
                **given,
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert run.returncode == 1 and f"{missing}: {missing} must be set" in run.stderr, run.stderr
        assert "unbound variable" not in run.stderr

    # a judge env (WebShop): J before C, on the teacher's own server when JUDGE_MODEL (Qwen3.5-9B) is the teacher's dir
    root, result, events = _pipeline_case(tmp_path, "ws_shared", RECIPE="webshop")
    assert result.returncode == 0, result.stdout + result.stderr
    out = root / "data" / "r2s" / "p1"
    assert "check=judge stages=A,B,J,C,D " in result.stdout
    assert f"R2S_JUDGE_PASS judge={out}/judge.jsonl" in result.stdout
    assert _kinds(events, "start ", "stage ") == ["teacher", "student", "A", "B", "J", "C", "D"]
    assert (
        f"stage J --env webshop --fold {out}/fold.jsonl --judge-url http://127.0.0.1:13141 --judge-model teacher "
        f"--out {out}/judge.jsonl"
    ) in events
    assert (
        f"stage C --env webshop --fold {out}/fold.jsonl --judge-file {out}/judge.jsonl --out {out}/tau.jsonl"
    ) in events
    # F1: another judge dir gets its own server on the GPUs after the teacher's, for stage J alone; stage D prefills
    # on the teacher (:13141), which is not stopped before LABEL_DONE
    root, result, events = _pipeline_case(tmp_path, "ws_judge", RECIPE="webshop", JUDGE_MODEL="Qwen3.5-27B")
    assert result.returncode == 0, result.stdout + result.stderr
    out = root / "data" / "r2s" / "p1"
    judge = root / "models" / "Qwen3.5-27B"
    order = ["teacher", "student", "A", "student", "B", "judge", "J", "judge", "C", "D", "teacher"]
    assert _kinds(events, "start ", "stop ", "stage ") == order
    assert f"launcher serve-template {judge} {out}/judge_template.jinja" in events
    assert f"launcher reasoning-parser {judge}" in events
    assert (
        f"start judge {judge} 13143 2,3 2 {server} --chat-template {out}/judge_template.jinja {served} judge "
        "--log-level warning"
    ) in events
    assert "wait judge http://127.0.0.1:13143/health_generate" in events
    assert (
        f"stage J --env webshop --fold {out}/fold.jsonl --judge-url http://127.0.0.1:13143 --judge-model judge "
        f"--out {out}/judge.jsonl"
    ) in events
    label = next(event for event in events if event.startswith("stage D "))
    assert " --teacher-url http://127.0.0.1:13141 " in label
    assert "stop teacher" not in events[: events.index(label)] and events.index("stop teacher") > events.index(label)

    # R2S_ARGS_<S> verbatim and last, after the driver's flags and --stop-file (PREEMPT_SENTINEL set)
    sentinel = tmp_path / "args" / "sync" / "preempt"
    root, result, events = _pipeline_case(
        tmp_path, "args", PREEMPT_SENTINEL=sentinel, R2S_ARGS_B="--samples 2 --limit 5", R2S_ARGS_C="--m 1"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = root / "data" / "r2s" / "p1"
    stages = [event for event in events if event.startswith("stage ")]
    assert stages[0].endswith(f"--out {out}/states.jsonl --stop-file {sentinel}")
    assert stages[1].endswith(f"--out {out}/fold.jsonl --stop-file {sentinel} --samples 2 --limit 5")
    assert stages[2].endswith(f"--out {out}/tau.jsonl --stop-file {sentinel} --m 1")
    assert stages[3].endswith(f"--out {out}/labels.jsonl --stop-file {sentinel}")
    assert not sentinel.exists() and not sentinel.with_name("preempt.done").exists()

    # a finished stage says so and does nothing (the stages' own resume after a requeue); R2S_ARGS_<S>=--force
    # reruns it
    done = '{\n  "DIVERGENCE_DONE": true,\n  "rows": 4\n}\nDIVERGENCE_DONE\n'
    for name in ("resume", "force"):
        earlier = tmp_path / name / "data" / "r2s" / "p1"
        earlier.mkdir(parents=True)
        (earlier / "states.jsonl").write_text('{"state_id": "s"}\n' * 4)
        (earlier / "states.jsonl.summary.json").write_text(done)
    root, result, events = _pipeline_case(tmp_path, "resume")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _kinds(events, "stage ", "skip ", "stop ") == ["A", "A", "student", "B", "C", "D", "teacher"]
    assert "DIVERGENCE_DONE already in" in result.stdout and "R2S_DIVERGENCE_PASS states=4" in result.stdout
    root, result, events = _pipeline_case(tmp_path, "force", R2S_ARGS_A="--force")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _kinds(events, "skip ") == []
    assert any(event.startswith("stage A ") and event.endswith(" --force") for event in events)
    assert "R2S_DIVERGENCE_PASS states=3" in result.stdout

    # a sentinel raised inside stage B (the stage stops between rows, no marker, exit 0): <sentinel>.done names the
    # stage and its rows, the driver prints R2S_PIPELINE_PREEMPT_STOP and returns 0 (the batch shell's
    # preempt_finish requeues on the sentinel), C and D never start, the servers stop with the job
    sentinel = tmp_path / "preempt" / "sync" / "preempt"
    root, result, events = _pipeline_case(tmp_path, "preempt", PREEMPT_SENTINEL=sentinel, FAKE_STAGE_B="stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.exists()
    assert json.loads(sentinel.with_name("preempt.done").read_text()) == {"stage": "B", "rows": 3}
    assert "R2S_PIPELINE_PREEMPT_STOP stage=B rc=0 rows=3" in result.stdout
    assert _kinds(events, "stage ") == ["A", "B"] and events[-1] == "stop teacher"
    assert "R2S_TAKEOVER_PASS" not in result.stdout and "R2S_PIPELINE_PASS" not in result.stdout
    assert result.stdout.rstrip().endswith("HARNESS_END")


def test_r2s_pipeline_stops_on_sentinel_during_boot(tmp_path):
    """A sentinel raised while a server boots: the driver hands PREEMPT_SENTINEL to every wait_sglang as its stop file
    (the fake returns 2 when the file exists, as the real one does), writes <sentinel>.done with stage "boot" and no
    rows, prints R2S_PIPELINE_PREEMPT_STOP stage=boot and returns 0 before any stage starts (no R2S_STAGE_START, no
    FATAL); the servers it started stop with the job. A wait that fails for any other reason stays FATAL."""
    sentinel = tmp_path / "boot_sentinel" / "preempt"
    sentinel.parent.mkdir(parents=True)
    sentinel.touch()
    root, result, events = _pipeline_case(tmp_path, "boot", PREEMPT_SENTINEL=sentinel)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(sentinel.with_name("preempt.done").read_text()) == {"stage": "boot", "rows": 0}
    assert "R2S_PIPELINE_PREEMPT_STOP stage=boot rc=2 rows=0" in result.stdout
    assert "R2S_STAGE_START" not in result.stdout and "FATAL" not in result.stdout
    assert "R2S_PIPELINE_PASS" not in result.stdout and result.stdout.rstrip().endswith("HARNESS_END")
    kinds = ["teacher", "student", "teacher", "student", "teacher"]  # start, start, the teacher's wait, the EXIT trap
    assert _kinds(events, "start ", "wait ", "stage ", "stop ") == kinds
    assert f"wait teacher http://127.0.0.1:13141/health_generate stop={sentinel}" in events
    assert not (root / "data" / "r2s" / "p1" / "states.jsonl").exists()

    root, result, events = _pipeline_case(tmp_path, "boot_fail", FAKE_WAIT_FAIL="student")
    assert result.returncode == 1 and "FATAL: the student server did not come up" in result.stdout
    assert _kinds(events, "wait ", "stage ") == ["teacher", "student"]
    assert "R2S_PIPELINE_PREEMPT_STOP" not in result.stdout and "HARNESS_END" not in result.stdout


def test_free_port_moves_off_a_busy_port(tmp_path):
    """free_port <preferred> [span] prints the first port from <preferred> that binds (a probe socket, closed at once):
    the port a leftover server holds moves the pipeline's server up instead of an `address already in use` at boot;
    a free preferred port is kept; a span that holds no free port returns 1 and prints nothing. The listener is a real
    socket (the fake python3 runs -c on the real interpreter), on a high port so the test never meets a real server."""
    root, env = _case(tmp_path, "freeport")
    busy = 43141 + os.getpid() % 400
    env["BUSY"] = str(busy)
    env["FAKE_REPO"] = str(REPO)  # the fake python3 runs a -c expression on the real interpreter with this on the path
    (root / "harness.sh").write_text(FREE_PORT_HARNESS)
    result = subprocess.run(["bash", str(root / "harness.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"SHIFTED={busy + 1}" in result.stdout
    assert f"KEPT={busy + 1}" in result.stdout
    assert "NONE_RC=1 NONE_OUT=" in result.stdout
