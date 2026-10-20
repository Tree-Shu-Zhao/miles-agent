"""Stage E of the R2S pipeline: full-parameter FT of the base teacher on stage D's labels (torchrun + FSDP).

Objective (label.py, "suffix"): the privileged top-k target (the frozen teacher with the selected takeover in
context) on every assistant token of the pilot episode from the mined turn to the end, thinking, tool call and
turns without a tool call alike; the input is the plain episode. Loss per labeled position p (the model's
distribution FOR token p sits at logits[p-1]):
    CE = -sum_k softmax(tgt_lps)_k * log_softmax(logits[p-1])[tgt_ids_k]
row loss = mean over the row's positions; the step loss is the mean over the batch rows. Label files with any
other objective or a non-privileged position are refused: the teacher trains label.py's objective and nothing
else, and the report records it. Optimizer: AdamW(lr, betas 0.9/0.95, wd 0), linear warmup --warmup-steps then
cosine to --min-lr-ratio, grad clip --clip, global batch --grad-accum rows, --epochs epochs. Split: goal-disjoint
(md5 of goal mod 100 < --val-pct -> val). Gate: the held-out CE on the labeled positions must drop start -> end
(FT_PASS), else FT_FAIL, exit 2, the report written and nothing exported; --skip-gate exports anyway
(FT_GATE_SKIPPED). The defaults are the campaign's (select.FT_*); the logic is REF's (the campaign branch's
miles_agent/r2s/teacher_ft.py) with torch and transformers imported where they are used, so the label checks,
the split, the schedule and the export sidecars run (and are tested) without them.

Qwen3.5 checkpoints are VLM wrappers (Qwen3_5ForConditionalGeneration, decoder under text_config, vision tower
sibling): the wrapper class is loaded so the vision tower is kept (frozen) in the export, the base config.json is
served verbatim (patch_saved_config) and the processor sidecars are copied, so sglang serves the result like the
base model.

    torchrun --standalone --nproc-per-node 4 -m miles_agent.r2s.teacher_ft \\
        --data labels.jsonl --model-path $MILES_AGENT_MODELS/Qwen3.5-9B --out-dir $MILES_AGENT_MODELS/r2s-alfworld-qwen3.5-4b-teacher

scripts/r2s_teacher_ft.sh runs it into <out>.tmp and promotes the dir on FT_PASS (a scheduler wrapper may run that
script on a cluster node: docs/cluster.md). Output name convention: r2s-<env>-<student>-teacher (the launcher's family rule resolves it).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import time
from typing import TYPE_CHECKING, Any

from miles_agent.r2s import common, label, select

if TYPE_CHECKING:
    import torch

AUX_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
)
_VISUAL_PREFIXES = ("model.visual.", "visual.", "model.vision_tower.", "vision_tower.")
OBJECTIVE = label.OBJECTIVE  # the one objective this stage trains; recorded in r2s_ft_report.json
REPORT = "r2s_ft_report.json"


def log(msg: str) -> None:
    """Rank 0 only (torchrun sets RANK; a single process has none)."""
    if os.environ.get("RANK", "0") == "0":
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_rows(rows: list[dict[str, Any]], val_pct: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Goal-disjoint split: md5(goal) mod 100 < val_pct -> val, so every state of a goal lands on one side
    (REF teacher_ft.py:59-65; a row without a goal hashes its state_id)."""
    train, val = [], []
    for r in rows:
        key = str(r.get("goal") or r["state_id"])
        h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
        (val if h < val_pct else train).append(r)
    return train, val


def renorm_weights(lps: list[float]) -> torch.Tensor:
    """The top-k log-probs as a distribution over the k targets (softmax renormalises what top-k cut)."""
    import torch

    return torch.softmax(torch.tensor(lps, dtype=torch.float32), dim=-1)


def position_loss(logits_sel: torch.Tensor, positions: list[dict[str, Any]], device) -> tuple[torch.Tensor, int]:
    """logits_sel [P, V] aligned with positions -> (CE sum over the positions, count)."""
    import torch

    logp = torch.log_softmax(logits_sel.float(), dim=-1)
    ce_sum = logp.new_zeros(())
    for j, x in enumerate(positions):
        ids = torch.tensor(x["tgt_ids"], dtype=torch.long, device=device)
        w = renorm_weights(x["tgt_lps"]).to(device)
        ce_sum = ce_sum - (w * logp[j, ids]).sum()
    return ce_sum, len(positions)


def check_labels(rows: list[dict[str, Any]]) -> None:
    """Every row must carry label.py's objective and only privileged ("hind") positions: the pre-suffix label
    files (turns without a tool call skipped, plain "anchor" action targets) train a different teacher."""
    other = sum(r.get("objective") != OBJECTIVE for r in rows)
    plain = sum(x["kind"] != "hind" for r in rows for x in r["positions"])
    if other or plain:
        raise SystemExit(
            f"FATAL: {other} label rows lack objective {OBJECTIVE!r} and {plain} positions are not privileged. "
            "teacher_ft trains only the R2S objective of label.py (privileged targets on every assistant token from "
            "the mined turn to the episode end): relabel older files with it. The campaign's pre-suffix teachers are "
            "reproduced from their branch (docs/r2s.md section 7), not from main"
        )


def row_loss(model, row: dict[str, Any], device) -> tuple[torch.Tensor, tuple[torch.Tensor, int]]:
    """One row's mean CE over its labeled positions, plus (CE sum, count) for the global eval."""
    import torch

    input_ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
    positions = row["positions"]
    idx = torch.tensor([x["p"] - 1 for x in positions], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, logits_to_keep=idx, use_cache=False)
    logits_sel = out.logits[0]
    if logits_sel.size(0) != idx.numel():  # the model ignored logits_to_keep: full logits
        logits_sel = logits_sel.index_select(0, idx)
    ce_sum, n = position_loss(logits_sel, positions, device)
    return ce_sum / max(1, n), (ce_sum.detach(), n)


def load_model(model_path: str, dtype):
    """The checkpoint's own architecture class (a VLM wrapper keeps its vision tower, frozen), sdpa attention."""
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if cls is None:
        cls = transformers.AutoModelForCausalLM
    model = cls.from_pretrained(model_path, trust_remote_code=True, attn_implementation="sdpa", dtype=dtype)
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith(_VISUAL_PREFIXES):
            p.requires_grad = False
            n_frozen += 1
    return model, cfg, n_frozen


def decoder_layer_cls(model):
    """The decoder block class FSDP wraps per layer."""
    import torch

    for mod in model.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 1 and mod.__class__.__name__ == "ModuleList":
            cls = type(mod[0])
            if "DecoderLayer" in cls.__name__ or "Block" in cls.__name__:
                return cls
    raise RuntimeError("could not find the decoder layer class for FSDP auto-wrap")


def patch_saved_config(out_dir: str, base_model_dir: str | None = None) -> None:
    """FT never changes the architecture, but the runtime transformers re-serializes config.json in a schema
    older serving stacks read wrong (dtype: float32 from training, Qwen3-MoE num_experts -> num_local_experts,
    rope_theta nested). Serve with the base config verbatim when we have it; otherwise at least stamp bfloat16."""
    path = os.path.join(out_dir, "config.json")
    base = os.path.join(base_model_dir, "config.json") if base_model_dir else None
    if base and os.path.isfile(base):
        shutil.copy2(base, path)
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    def stamp(d):
        for key in [k for k in ("dtype", "torch_dtype") if k in d] or ["torch_dtype"]:
            d[key] = "bfloat16"

    stamp(cfg)
    if isinstance(cfg.get("text_config"), dict):
        stamp(cfg["text_config"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


def copy_aux_files(model_dir: str, out_dir: str) -> list[str]:
    """The tokenizer / processor / generation sidecars the export needs to serve like the base model."""
    copied = []
    for name in AUX_FILES:
        src, dst = os.path.join(model_dir, name), os.path.join(out_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def eval_schedule(n_rows: int, world: int, rank: int) -> list[tuple[int, bool]]:
    """(row index, counts) per forward pass on `rank`: every rank runs ceil(n/world) forwards (FSDP all-gathers
    are collective, so an idle rank deadlocks the others); the padding passes re-use a real row, uncounted."""
    n_rows = max(1, n_rows)
    per_rank = math.ceil(n_rows / world)
    return [((rank + j * world) % n_rows, rank + j * world < n_rows) for j in range(per_rank)]


def eval_hind_ce(model, rows: list[dict[str, Any]], device, world: int, rank: int) -> float:
    """Global held-out CE on the labeled (privileged) positions (sum over all rows / count), identical on
    every rank."""
    import torch
    import torch.distributed as dist

    model.eval()
    with torch.no_grad():
        total, count = torch.zeros((), device=device), torch.zeros((), device=device)
        for i, counts in eval_schedule(len(rows), world, rank):
            _, (ce_sum, n) = row_loss(model, rows[i], device)
            if counts:
                total += ce_sum
                count += n
        if world > 1:
            dist.all_reduce(total)
            dist.all_reduce(count)
    model.train()
    return (total / count.clamp_min(1)).item()


def ft_report(
    ce0: float, ce1: float, passed: bool, counts: tuple[int, int, int], steps: int, args: argparse.Namespace, copied
) -> dict[str, Any]:
    """r2s_ft_report.json (REF's keys): the gate's two numbers and verdict, the objective, the split, the knobs."""
    rows, train, val = counts
    return {
        "val_hind_ce_before": ce0,
        "val_hind_ce_after": ce1,
        "passed": passed,
        "objective": OBJECTIVE,
        "rows": rows,
        "train": train,
        "val": val,
        "steps": steps,
        "args": vars(args),
        "copied_aux": list(copied),
    }


def write_report(out_dir: str, report: dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, REPORT), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="R2S stage E: full-parameter teacher FT on stage D's labels (torchrun)")
    ap.add_argument("--data", required=True, help="stage D's labels.jsonl (objective suffix)")
    ap.add_argument("--model-path", required=True, help="the base teacher's HF dir")
    ap.add_argument("--out-dir", required=True, help="the bf16 HF export (and the report, FT_FAIL included)")
    ap.add_argument("--epochs", type=int, default=select.FT_EPOCHS)
    ap.add_argument("--lr", type=float, default=select.FT_LR)
    ap.add_argument("--grad-accum", type=int, default=select.FT_GRAD_ACCUM, help="rows per optimizer step")
    ap.add_argument("--warmup-steps", type=int, default=select.FT_WARMUP_STEPS)
    ap.add_argument("--min-lr-ratio", type=float, default=select.FT_MIN_LR_RATIO, help="the cosine floor")
    ap.add_argument("--clip", type=float, default=select.FT_CLIP)
    ap.add_argument("--val-pct", type=int, default=select.FT_VAL_PCT, help="held-out goals, percent")
    ap.add_argument("--seed", type=int, default=select.SEED)
    ap.add_argument("--max-ctx", type=int, default=select.MAX_CTX, help="rows with more ids are left out")
    ap.add_argument("--skip-gate", action="store_true", help="export even if held-out hind CE did not improve")
    return ap


def main(argv: list[str] | None = None) -> None:
    import torch
    import torch.distributed as dist

    args = build_parser().parse_args(argv)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    rows = common.read_jsonl(args.data)
    check_labels(rows)
    rows = [r for r in rows if r["positions"] and len(r["ids"]) <= args.max_ctx]
    train, val = split_rows(rows, args.val_pct)
    log(f"rows={len(rows)} train={len(train)} val={len(val)} goals={len({r['goal'] for r in rows})}")
    if not train or not val:
        raise SystemExit("FATAL: empty train or val split")
    counts = (len(rows), len(train), len(val))

    model, cfg, n_frozen = load_model(args.model_path, torch.float32)
    log(f"loaded {type(model).__name__}; frozen vision params: {n_frozen}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fsdp_model = None
    if world > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={decoder_layer_cls(model)})
        model = FSDP(
            model,
            auto_wrap_policy=wrap,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            device_id=local_rank,
            sync_module_states=True,
            use_orig_params=True,
        )
        fsdp_model = model
    else:
        model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    accum = max(1, args.grad_accum // world)
    rows_per_rank = len(train) // world
    steps_per_epoch = math.ceil(rows_per_rank / accum)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

    ce0 = eval_hind_ce(model, val, device, world, rank)
    log(f"val hind CE before: {ce0:.4f}")
    step = 0
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        order = torch.randperm(len(train), generator=g).tolist()
        mine = [train[i] for i in order][rank::world][:rows_per_rank]
        for b in range(0, len(mine), accum):
            batch = mine[b : b + accum]
            if len(batch) < accum and world > 1:
                break  # lockstep: every rank does the same number of forward passes
            for lr_group in opt.param_groups:
                lr_group["lr"] = lr_at(step)
            loss_acc = 0.0
            for row in batch:
                loss, _ = row_loss(model, row, device)
                (loss / len(batch)).backward()
                loss_acc += loss.item() / len(batch)
            if fsdp_model is not None:
                fsdp_model.clip_grad_norm_(args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or step == total_steps:
                log(f"epoch {epoch} step {step}/{total_steps} loss {loss_acc:.4f} lr {lr_at(step - 1):.2e}")
    ce1 = eval_hind_ce(model, val, device, world, rank)
    log(f"val hind CE after: {ce1:.4f} (before {ce0:.4f})")
    passed = ce1 < ce0
    if not passed and not args.skip_gate:
        # the verdict is kept with the (unpromoted) out dir: the job reads it, the weights are not exported
        log(f"FT_FAIL val hind CE did not improve ({ce0:.4f} -> {ce1:.4f})")
        if rank == 0:
            write_report(args.out_dir, ft_report(ce0, ce1, passed, counts, step, args, []))
        if world > 1:
            dist.barrier()
        raise SystemExit(2)
    log("FT_PASS" if passed else "FT_GATE_SKIPPED")

    # export bf16 HF
    if fsdp_model is not None:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            fsdp_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            sd = fsdp_model.state_dict()
        if rank == 0:
            sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
            os.makedirs(args.out_dir, exist_ok=True)
            fsdp_model.module.save_pretrained(args.out_dir, state_dict=sd, safe_serialization=True)
    elif rank == 0:
        sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        os.makedirs(args.out_dir, exist_ok=True)
        model.save_pretrained(args.out_dir, state_dict=sd, safe_serialization=True)
    if rank == 0:
        copied = copy_aux_files(args.model_path, args.out_dir)
        patch_saved_config(args.out_dir, args.model_path)
        write_report(args.out_dir, ft_report(ce0, ce1, passed, counts, step, args, copied))
        log(f"exported {args.out_dir} (aux: {copied})")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
