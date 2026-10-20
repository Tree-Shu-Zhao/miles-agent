#!/bin/bash
# scripts/lib/ray.sh — the external ray cluster of a run (sourced after scripts/lib/common.sh; no scheduler call).
# Always external: miles' non-external preamble runs `ray stop --force; pkill -9 ray`, which would kill the gym
# servers' private ray cluster, and the head must start BEFORE gym: whoever starts first wins the default dashboard
# port 8265, and miles' execute_train hardcodes `ray job submit --address http://127.0.0.1:8265` — if gym's private
# ray won that race, training would silently be submitted to a cluster without the GPUs of a multi-node topology.
# The PMI / PMIX / MPI / OMPI families and the scheduler's own variables are unset around every `ray start` (they
# confuse ray; every value the run needs is already expanded by then; OpenMPI / PMIx variables exist on bare
# machines too).
#   ray_head <ip> <gpus>            `ray start --head` on <ip>:6379 with <gpus> GPUs, dashboard on all interfaces
#                                   (it registers <ip>:8265 with the GCS, which miles' worker manager — an actor on
#                                   any node — reads for list_nodes(); loopback still serves `ray job submit`);
#                                   exports MASTER_ADDR=<ip> and MILES_SCRIPT_EXTERNAL_RAY=1; hard gate on
#                                   http://127.0.0.1:8265/api/version (12 x 5 s, FATAL: our dashboard must own
#                                   8265); registers `ray stop --force` with the cleanup trap as soon as the start
#                                   returns 0, before that gate (a head that fails the gate is stopped too; a plain
#                                   machine must not keep a headless ray; harmless in a container that ends with
#                                   the step)
#   ray_running                     `ray status` answers: a ray cluster is already up on this machine
#   ray_worker_join <head_ip> <own_ip> <gpus> <node_rank>
#                                   join <head_ip>:6379 (60 tries x 10 s; FATAL when $SYNC_DIR/ENDED is up before
#                                   the join), RAY_WORKER_JOINED node_rank= ip=, idle until $SYNC_DIR/ENDED or the
#                                   raylet is gone, `ray stop`, FATAL when ENDED is still missing (the worker died
#                                   before the driver finished), RAY_WORKER_DONE node_rank=; the caller exits 0
#   ray_wait_gpus <expected> <gcs>  wait up to 900 s until ray.cluster_resources()["GPU"] >= <expected> through
#                                   $PYTHON (the ray package), FATAL after 15 min; the caller computes <expected>
#                                   (nodes x GPUs per node, the teacher node of a *_teacher topology excluded)

ray_head() {  # <ip> <gpus>
  local ip=$1 gpus=$2 dash_ok i
  export MASTER_ADDR=$ip
  (
    unset $(env | grep -E '^(PMI|PMIX|MPI|OMPI|SLURM)' | cut -d= -f1)
    # All interfaces: the dashboard registers <node ip>:8265 with the GCS, which miles' worker
    # manager (an actor on any node) reads for list_nodes(); bound to loopback it registers
    # 127.0.0.1:8265, unreachable from the other nodes. Loopback still serves `ray job submit`.
    "$RAY_BIN" start --head --node-ip-address="$ip" --port=6379 \
      --num-gpus "$gpus" --disable-usage-stats --dashboard-host=0.0.0.0 \
      --min-worker-port=2000 --max-worker-port=2999
  ) || { echo "FATAL: ray start --head failed on $ip"; exit 1; }
  # registered before the dashboard gate: a head that started but fails the gate is stopped by the EXIT trap too
  add_cleanup '"$RAY_BIN" stop --force >/dev/null 2>&1 || true'
  export MILES_SCRIPT_EXTERNAL_RAY=1
  # Hard gate: our dashboard must own 8265 or job submission goes astray again.
  dash_ok=0
  for i in $(seq 12); do
    "$CURL" -sf -o /dev/null "http://127.0.0.1:8265/api/version" && { dash_ok=1; break; }
    sleep 5
  done
  [ "$dash_ok" -eq 1 ] || { echo "FATAL: our ray dashboard is not serving on 8265"; exit 1; }
}

ray_running() { "$RAY_BIN" status >/dev/null 2>&1; }

ray_worker_join() {  # <head_ip> <own_ip> <gpus> <node_rank>
  local head_ip=$1 own_ip=$2 gpus=$3 node_rank=$4 joined
  # scheduler env confuses ray workers; safe here — every value this
  # branch needs is already expanded
  unset $(env | grep -E '^(PMI|PMIX|MPI|OMPI|SLURM)' | cut -d= -f1)
  joined=0
  for _ in $(seq 60); do
    [ -f "$SYNC_DIR/ENDED" ] && { echo "FATAL: head exited before ray came up"; exit 1; }
    if "$RAY_BIN" start --address "$head_ip:6379" --node-ip-address "$own_ip" \
         --num-gpus "$gpus" --disable-usage-stats \
         --min-worker-port=2000 --max-worker-port=2999; then
      joined=1; break
    fi
    sleep 10
  done
  [ "$joined" -eq 1 ] || { echo "FATAL: could not join ray head at $head_ip:6379"; exit 1; }
  echo "RAY_WORKER_JOINED node_rank=$node_rank ip=$own_ip"
  until [ -f "$SYNC_DIR/ENDED" ]; do
    if command -v pgrep >/dev/null && ! pgrep -f raylet >/dev/null; then
      break
    fi
    sleep 10
  done
  "$RAY_BIN" stop || true
  [ -f "$SYNC_DIR/ENDED" ] || { echo "FATAL: ray worker died before the driver finished"; exit 1; }
  echo "RAY_WORKER_DONE node_rank=$node_rank"
}

ray_wait_gpus() {  # <expected gpus> <gcs address>
  # Explicit GCS address: address="auto" connects to the "latest" cluster,
  # which after the head-before-gym order would be gym's private one.
  MILES_EXPECT_GPUS=$1 MILES_GCS=$2 "$PYTHON" - <<'PYEOF'
import os, sys, time
import ray
ray.init(address=os.environ["MILES_GCS"])
need = float(os.environ["MILES_EXPECT_GPUS"])
deadline = time.time() + 900
while time.time() < deadline:
    have = ray.cluster_resources().get("GPU", 0.0)
    if have >= need:
        print(f"ray cluster ready: {have}/{need} GPUs"); sys.exit(0)
    time.sleep(5)
print(f"FATAL: ray cluster stuck below {need} GPUs after 15min"); sys.exit(1)
PYEOF
}
