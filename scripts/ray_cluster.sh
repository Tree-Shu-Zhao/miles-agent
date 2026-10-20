#!/bin/bash
# scripts/ray_cluster.sh — a ray cluster across plain machines, without a scheduler: the same `ray start` lines a
# run uses (scripts/lib/ray.sh), driven by hand or over ssh. Then, on the head, `scripts/run_local.sh --topology <t>
# --num-nodes N` reuses the running head (ray_running) and waits for the workers' GPUs.
#
#   scripts/ray_cluster.sh [--dry-run] head
#       the ray head on this machine (<ip>:6379, dashboard :8265; GPUs = GPUS_PER_NODE, else the CUDA_VISIBLE_DEVICES
#       count, else `nvidia-smi -L`); prints RAY_HEAD=<ip>:6379; the head outlives this script (`stop` ends it)
#   scripts/ray_cluster.sh [--dry-run] join <head_ip>
#       join <head_ip>:6379 from this machine (no idle loop: the shell owns the node); prints RAY_JOINED head= ip= gpus=
#   scripts/ray_cluster.sh [--dry-run] ssh-join <head_ip> <host>...
#       `$SSH -o BatchMode=yes <host> "$REMOTE_PREFIX bash <this script> join <head_ip>"` per host; REMOTE_PREFIX is
#       e.g. `docker exec miles` or `cd /same/path &&` (the checkout must sit at this script's own path on every host)
#   scripts/ray_cluster.sh status
#       `ray status`
#   scripts/ray_cluster.sh [--dry-run] stop [<host>...]
#       `ray stop --force` here, then the same over ssh on each host; prints RAY_STOPPED hosts=<n>
# Requirements: the same checkout, model and data paths on every node; ports 6379, 8265 and 2000-2999 open between
# the nodes; the gym on the head only (scripts/run_local.sh starts it there). Tools: RAY_BIN, SSH, NVIDIA_SMI
# (scripts/lib/common.sh). --dry-run prints every command, one line per host, and runs none of them.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$HERE/ray_cluster.sh

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

args=()
for a in "$@"; do
  if [ "$a" = --dry-run ]; then DRY_RUN=1; else args+=("$a"); fi
done
set -- ${args[@]+"${args[@]}"}

. "$HERE/lib/common.sh"
. "$LIB/ray.sh"
[ -z "${DRY_RUN:-}" ] || export DRY_RUN

COMMANDS="head | join <head_ip> | ssh-join <head_ip> <host>... | status | stop [<host>...]"

own_ip() {  # this node's IPv4 (common.sh node_ip); under --dry-run 127.0.0.1 stands in when nothing knows it
  local ip
  if ip=$(node_ip 2>/dev/null); then printf '%s\n' "$ip"; return 0; fi
  if ! is_dry_run; then
    echo "FATAL: could not determine this node's IP (getent hosts, hostname -I and NODE_IP are all empty)" >&2
    return 1
  fi
  echo "WARN: node IP unknown: 127.0.0.1 stands in under --dry-run" >&2
  printf '127.0.0.1\n'
}

remote() {  # <label> <host> <command...>: the command on <host> through $SSH, under REMOTE_PREFIX
  local label=$1 host=$2
  shift 2
  if is_dry_run; then dry_run_print "$label" "$SSH" -o BatchMode=yes "$host" "${REMOTE_PREFIX:+$REMOTE_PREFIX }$*"; return 0; fi
  "$SSH" -o BatchMode=yes "$host" "${REMOTE_PREFIX:+$REMOTE_PREFIX }$*"
}

cmd=${1:-}
case "$cmd" in
  -h|--help) usage; exit 0 ;;
  '') usage; fatal "a command is required: $COMMANDS" ;;
esac
shift
case "$cmd" in
  head)
    ip=$(own_ip)
    gpus=$(measure_gpus)
    if is_dry_run; then
      dry_run_print ray_head "$RAY_BIN" start --head --node-ip-address="$ip" --port=6379 --num-gpus "$gpus" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --min-worker-port=2000 --max-worker-port=2999
      exit 0
    fi
    [ "$gpus" -gt 0 ] || echo "WARN: no GPU measured on this head (GPUS_PER_NODE, CUDA_VISIBLE_DEVICES, nvidia-smi -L all empty)"
    ray_head "$ip" "$gpus"
    # ray_head registers `ray stop --force` for a run's exit; this head must outlive the script (run_local.sh reuses
    # it, `stop` ends it), so the registration is dropped again
    G_CLEANUP=":"
    echo "RAY_HEAD=$ip:6379"
    ;;
  join)
    head_ip=${1:-}
    [ -n "$head_ip" ] || fatal "join needs <head_ip>"
    ip=$(own_ip)
    gpus=$(measure_gpus)
    JOIN=("$RAY_BIN" start --address "$head_ip:6379" --node-ip-address "$ip" --num-gpus "$gpus" --disable-usage-stats
          --min-worker-port=2000 --max-worker-port=2999)
    if is_dry_run; then dry_run_print ray_join "${JOIN[@]}"; exit 0; fi
    "${JOIN[@]}" || fatal "could not join the ray head at $head_ip:6379 from $ip"
    echo "RAY_JOINED head=$head_ip:6379 ip=$ip gpus=$gpus"
    ;;
  ssh-join)
    head_ip=${1:-}
    [ -n "$head_ip" ] || fatal "ssh-join needs <head_ip> <host>..."
    shift
    [ $# -gt 0 ] || fatal "ssh-join needs at least one <host> after <head_ip>"
    for h in "$@"; do
      echo "ray_cluster: ssh-join $h -> $head_ip:6379"
      remote ssh_join "$h" bash "$SELF" join "$head_ip" || fatal "ssh-join failed on $h"
    done
    echo "ray_cluster: ssh-join done hosts=$#"
    ;;
  status)
    if is_dry_run; then dry_run_print ray_status "$RAY_BIN" status; exit 0; fi
    "$RAY_BIN" status
    ;;
  stop)
    if is_dry_run; then dry_run_print ray_stop "$RAY_BIN" stop --force; else "$RAY_BIN" stop --force || true; fi
    for h in "$@"; do
      remote ray_stop "$h" bash "$SELF" stop || echo "WARN: stop failed on $h"
    done
    echo "RAY_STOPPED hosts=$#"
    ;;
  *) fatal "unknown command '$cmd': $COMMANDS" ;;
esac
