#!/usr/bin/env python3
"""torch_dist -> HF export for this pin's checkpoint layout (no common.pt); or, with --input-hf-dir, the finishing of an
HF export the run wrote itself (miles --save-hf, for the Megatron-Bridge models such as Nemotron-H that the torch_dist
converters do not know): its shards and index are copied as they are, then finished and checked as below.

tools/convert_torch_dist_to_hf.py reads the megatron args namespace from the
checkpoint's common.pt, but this pin's saver writes pure DCP shards
(__N_0.distcp + .metadata + metadata.json{sharded_backend,...}) with no
common.pt anywhere. The conversion itself only consults static model-shape
fields (num_layers, hidden_size, num_attention_heads, num_query_groups,
kv_channels, vocab_size, num_experts, untie flag), all of which are encoded
in scripts/models/<model>.py model_args(). Synthesize the namespace from that
script and reuse the tool's own load/convert/save machinery verbatim.

--fill-from-origin copies the tensors the origin checkpoint has and the export
lacks, which the trained Megatron model does not hold (Qwen3.5's vision tower
and MTP layer), into one more shard, so that the export loads as the origin's
architecture does. --origin-dtypes casts each exported tensor to the origin's
dtype for it (Qwen3.5 stores its GDN norms in fp32; Megatron trains them in
bf16, so the cast loses nothing). --check-origin then requires the export's
names to be the origin's, each tensor of the origin's shape and dtype, and
reports how far the trained tensors moved from the origin's (relative L2;
--max-rel-diff fails a wrongly mapped tensor).
"""

import argparse
import importlib.util
import json
import os
import shlex
import time
from collections import defaultdict

LINEAR_ATTN_LAYOUT = 1  # miles_plugins.models.linear_attn.WEIGHT_LAYOUT_VERSION at this pin


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MegatronArgs:
    """Attribute bag; unset attrs resolve to None like an argparse namespace."""

    def __getattr__(self, name):
        return None


def parse_model_args(flag_string):
    ns = MegatronArgs()
    tokens = shlex.split(flag_string)
    i = 0
    while i < len(tokens):
        assert tokens[i].startswith("--"), f"expected flag, got {tokens[i]!r}"
        key = tokens[i][2:].replace("-", "_")
        values = []
        i += 1
        while i < len(tokens) and not tokens[i].startswith("--"):
            values.append(number_or_text(tokens[i]))
            i += 1
        # a flag alone is True, one value is that value, several (--spec <module> <function>) a list
        setattr(ns, key, True if not values else values[0] if len(values) == 1 else values)
    return ns


def number_or_text(raw):
    for kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            pass
    return raw


def weight_map(hf_dir):
    with open(os.path.join(hf_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def copy_hf_export(input_dir, output_dir):
    """The weights of an HF export the run wrote itself, shards and index as they are (the rest comes from the origin)."""
    import shutil

    names = sorted(name for name in os.listdir(input_dir) if name.endswith(".safetensors"))
    assert names and os.path.isfile(os.path.join(input_dir, "model.safetensors.index.json")), (
        f"{input_dir}: no sharded safetensors export"
    )
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    for name in names + ["model.safetensors.index.json"]:
        shutil.copyfile(os.path.join(input_dir, name), os.path.join(output_dir, name))
    print(f"copied the {len(names)} shards of {input_dir} in {time.time() - start:.1f}s")


def convert_torch_dist(tool, args):
    """The torch_dist checkpoint in --input-dir as HF shards in --output-dir (the tool's own machinery)."""
    margs = parse_model_args(load_module("margs_src", args.model_args_script).model_args())
    assert margs.num_layers and margs.hidden_size and margs.vocab_size, (
        f"model_args() parse looks wrong: layers={margs.num_layers} "
        f"hidden={margs.hidden_size} vocab={margs.vocab_size}"
    )
    # the converters that read the HF config (Qwen3.5's GDN heads) find it here, as in the tool's own main
    margs.hf_checkpoint = args.origin_hf_dir

    state_dict = {}
    start = time.time()
    tool.dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=tool.WrappedStorageReader(args.input_dir),
        planner=tool.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"checkpoint loaded in {time.time() - start:.1f}s, {len(state_dict)} entries")
    # The linear-attention layers' layout marker is a buffer, not a weight (miles_plugins.models.linear_attn): it says
    # the GDN q/k/v and conv rows are group-major (version 1 at this pin), which the converters undo. No HF tensor.
    markers = [name for name in state_dict if name.endswith("weight_layout_version")]
    versions = {int(state_dict.pop(name).flatten()[0]) for name in markers}
    if markers:
        assert versions == {LINEAR_ATTN_LAYOUT}, (
            f"linear-attention layout {versions}, the converters know {LINEAR_ATTN_LAYOUT}"
        )
        print(f"dropped {len(markers)} linear-attention layout markers (version {LINEAR_ATTN_LAYOUT})")

    tool.save_tensors(margs, args.model_name, state_dict, args.output_dir, args.chunk_size)


def fill_from_origin(origin_dir, output_dir):
    """Copies the origin's tensors the export lacks into one more shard of the export; returns their names."""
    import safetensors.torch
    from safetensors import safe_open

    origin, export = weight_map(origin_dir), weight_map(output_dir)
    missing = sorted(set(origin) - set(export))
    if not missing:
        return []
    by_file = defaultdict(list)
    for name in missing:
        by_file[origin[name]].append(name)
    tensors = {}
    for filename, names in by_file.items():
        with safe_open(os.path.join(origin_dir, filename), framework="pt") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    shard = "model-origin-filled.safetensors"
    safetensors.torch.save_file(tensors, os.path.join(output_dir, shard), metadata={"format": "pt"})
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    for name, tensor in tensors.items():
        index["weight_map"][name] = shard
        index["metadata"]["total_size"] += tensor.numel() * tensor.element_size()
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return missing


def cast_to_origin_dtypes(origin_dir, output_dir):
    """Casts each exported tensor to the origin's dtype for it, rewriting the shards that hold one of another dtype
    (Qwen3.5 keeps its GDN norms in fp32, which Megatron trains in bf16); returns {(from, to): count}."""
    import safetensors.torch
    import torch
    from safetensors import safe_open

    dtypes = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}

    def dtypes_of(hf_dir, names_by_file):
        out = {}
        for filename, names in names_by_file.items():
            with safe_open(os.path.join(hf_dir, filename), framework="pt") as f:
                for name in names:
                    out[name] = f.get_slice(name).get_dtype()
        return out

    origin, export = weight_map(origin_dir), weight_map(output_dir)
    common = set(export) & set(origin)
    by_origin_file, by_export_file = defaultdict(list), defaultdict(list)
    for name in common:
        by_origin_file[origin[name]].append(name)
        by_export_file[export[name]].append(name)
    want, have = dtypes_of(origin_dir, by_origin_file), dtypes_of(output_dir, by_export_file)
    differ = sorted(name for name in common if want[name] != have[name])
    if not differ:
        return {}
    for name in differ:
        assert want[name] in dtypes and have[name] in dtypes, f"{name}: {have[name]} -> {want[name]}"
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    counts = defaultdict(int)
    for filename in sorted({export[name] for name in differ}):
        path = os.path.join(output_dir, filename)
        tensors = safetensors.torch.load_file(path)
        for name in differ:
            if export[name] == filename:
                before = tensors[name].numel() * tensors[name].element_size()
                tensors[name] = tensors[name].to(dtypes[want[name]])
                index["metadata"]["total_size"] += tensors[name].numel() * tensors[name].element_size() - before
                counts[(have[name], want[name])] += 1
        safetensors.torch.save_file(tensors, path, metadata={"format": "pt"})
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return dict(counts)


def check_origin(origin_dir, output_dir, max_rel_diff):
    """The export's names are the origin's, each tensor of the origin's shape and dtype, none moved more than
    max_rel_diff (relative L2) from the origin's; raises otherwise, prints how far the tensors moved."""
    from safetensors import safe_open

    origin, export = weight_map(origin_dir), weight_map(output_dir)
    only_export, only_origin = sorted(set(export) - set(origin)), sorted(set(origin) - set(export))
    problems = [f"not in the origin: {only_export[:5]} ({len(only_export)})"] if only_export else []
    if only_origin:
        problems.append(f"not in the export: {only_origin[:5]} ({len(only_origin)})")
    moved, identical = [], 0
    by_file = defaultdict(list)
    for name in sorted(set(export) & set(origin)):
        by_file[export[name]].append(name)
    for filename, names in by_file.items():
        with safe_open(os.path.join(output_dir, filename), framework="pt") as fe:
            for name in names:
                a = fe.get_tensor(name)
                with safe_open(os.path.join(origin_dir, origin[name]), framework="pt") as fo:
                    b = fo.get_tensor(name)
                if a.shape != b.shape or a.dtype != b.dtype:
                    problems.append(f"{name}: {tuple(a.shape)} {a.dtype} vs the origin's {tuple(b.shape)} {b.dtype}")
                    continue
                diff = (a.float() - b.float()).norm().item()
                if diff == 0:
                    identical += 1
                    continue
                rel = diff / max(b.float().norm().item(), 1e-12)
                moved.append((rel, name))
                if rel > max_rel_diff:
                    problems.append(f"{name}: moved {rel:.3g} from the origin (over {max_rel_diff})")
    moved.sort()
    print(f"CHECK_ORIGIN {len(export)} tensors: {identical} identical to the origin's, {len(moved)} moved; "
          f"relative L2 median {moved[len(moved) // 2][0] if moved else 0:.3g}, max {moved[-1][0] if moved else 0:.3g}"
          f" ({moved[-1][1] if moved else '-'})")
    if problems:
        raise SystemExit("CHECK_ORIGIN_FAILED " + "; ".join(problems[:20]))
    print("CHECK_ORIGIN_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--miles-root", default="/root/miles")
    parser.add_argument("--model-args-script", help="scripts/models/<model>.py providing model_args() (--input-dir)")
    parser.add_argument("--model-name", default="qwen2")
    parser.add_argument("--input-dir", help="iter_NNNNNNN dir with .distcp shards")
    parser.add_argument("--input-hf-dir", help="instead of --input-dir: an HF export the run wrote itself (--save-hf)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=5 * 1024 * 1024 * 1024)
    parser.add_argument("--fill-from-origin", action="store_true",
                        help="copy the origin's tensors the export lacks (a vision tower, an MTP layer)")
    parser.add_argument("--origin-dtypes", action="store_true",
                        help="cast each exported tensor to the origin's dtype for it (Qwen3.5's fp32 GDN norms)")
    parser.add_argument("--check-origin", action="store_true",
                        help="require the origin's names, shapes and dtypes; report how far the tensors moved")
    parser.add_argument("--max-rel-diff", type=float, default=0.05)
    args = parser.parse_args()

    assert (args.input_dir is None) != (args.input_hf_dir is None), "give one of --input-dir, --input-hf-dir"
    assert args.input_hf_dir or args.model_args_script, "--input-dir needs --model-args-script"

    tool = load_module("cvt_tool", f"{args.miles_root}/tools/convert_torch_dist_to_hf.py")
    if args.input_hf_dir:
        copy_hf_export(args.input_hf_dir, args.output_dir)
    else:
        convert_torch_dist(tool, args)
    tool.copy_assets(args.origin_hf_dir, args.output_dir)
    if args.fill_from_origin:
        filled = fill_from_origin(args.origin_hf_dir, args.output_dir)
        prefixes = sorted({".".join(name.split(".")[:2]) for name in filled})
        print(f"FILLED_FROM_ORIGIN {len(filled)} tensors: {prefixes}")
    if args.origin_dtypes:
        cast = cast_to_origin_dtypes(args.origin_hf_dir, args.output_dir)
        print("CAST_TO_ORIGIN_DTYPES " + (", ".join(f"{n} {a}->{b}" for (a, b), n in cast.items()) or "none"))
    if args.check_origin:
        check_origin(args.origin_hf_dir, args.output_dir, args.max_rel_diff)
    print(f"HF export complete: {args.output_dir}")


if __name__ == "__main__":
    main()
