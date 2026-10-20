# Containers

The training side of this repo (miles, Megatron, sglang, ray) runs in the miles image: the tag `3rdparty/miles`
pins, `radixark/miles:v0.1.2` today (`git -C 3rdparty/miles describe --tags` prints `v0.1.2-<n>-g...`: the pin
plus the `n` patches of `patches/miles/`). The image carries miles' dependencies; the miles code it runs is YOUR
checkout, mounted at `/root/miles` (`scripts/lib/common.sh`: `MILES_ROOT` is `/root/miles` whenever that dir
exists), so the patches are in effect inside the container exactly as `scripts/check_patches.sh` left them in
the submodule, and `scripts/run_local.sh` checks it (`MILES_PIN_ACTIVE path=/root/miles/miles`: the package's own
path under the mounted checkout).

## docker

```
docker run --rm -it --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -v "$REPO:$REPO" -v "$REPO/3rdparty/miles:/root/miles" -w "$REPO" \
  radixark/miles:v0.1.2
```

- `--network=host`: ray (6379, 8265, 2000-2999), the gym servers (8100, 8200, 8300-8999), the session servers of
  the rollout engines (7100-7131) and their router (7200) talk over loopback ports and hand each other
  `http://127.0.0.1:<port>` URLs; a bridged network would hide them from one another. The fixed ports are listed
  in docs/local_setup.md.
- `-v "$REPO:$REPO"`, same path inside and outside: every path a script prints (the `DRY_RUN` lines, the log
  paths, `results/<tag>/stop`) is valid in both places, and the gym venvs built inside stay valid (next section).
- `-v "$REPO/3rdparty/miles:/root/miles"`: the patched miles checkout at the path the image's python imports
  `miles` from.
- the three roots: when `MILES_AGENT_MODELS`, `MILES_AGENT_DATA`, `MILES_AGENT_RESULTS` (`local.env`) point
  outside the checkout, bind them at the same paths too (`-v /data/models:/data/models`, ...). By default they
  hang under the repo and ride along with it.
- `--ipc=host`, `--shm-size`, the ulimits: what sglang, NCCL and the dataloaders need (miles' own quick-start line).

Everything of the quickstart runs inside: `scripts/setup.sh`, `scripts/build_gym_venvs.sh`,
`scripts/fetch_data.sh`, `scripts/run_local.sh`, `scripts/eval_ckpt.sh`, `scripts/export_hf.sh`,
`scripts/serve_model.sh`, `scripts/smoke_tests.sh`. A model or data download (`hf download`, `scripts/fetch_data.sh`)
can run outside as well: the paths are the same.

## The gym venvs are built inside the same container

`scripts/build_gym_venvs.sh [<root>]` installs a CPython 3.13.14 and one venv per gym server under `GYM_VENV_ROOT`
(default `$REPO/.gym_venvs`). uv venvs are not relocatable (shebangs, `activate`, `pyvenv.cfg` and the editable
`.pth` files bake absolute paths): build them at the path the runs will use, from inside the container that will
run them (same mounts), and keep `GYM_VENV_ROOT` equal to that build root. An empty `GYM_VENV_ROOT` makes a run
start the servers through `uv run` from `3rdparty/Gym` instead (network at the first start, no prebuilt venvs).

## Several containers, several machines

A multi-node run needs the same container (same image, same mounts, same paths) on every machine, and the ports
6379, 8265 and 2000-2999 open between them; `scripts/ray_cluster.sh` drives the `ray start` lines on each
(`REMOTE_PREFIX`, e.g. `docker exec miles`, runs the join inside a named container). docs/local_setup.md,
"Multi-node without a scheduler", has the requirements; docs/cluster.md the form under a job scheduler.

## Without docker

A plugin-less container runtime of a cluster does the same with its own import / create / start commands;
docs/cluster.md, "Containers without a scheduler plugin", shows the three lines for one such runtime, with the
same two mounts.
