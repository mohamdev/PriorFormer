"""Train the research reference model (M0, docs/plan-v2.md): Backbone, then the Versatile model.

    python -m priorformer.train_v2 --model backbone --data <processed train clips> --out runs/reference_backbone
    python -m priorformer.train_v2 --model versatile --data <processed train clips> \
        --init runs/reference_backbone/best.pt --out runs/reference_versatile

Training items follow protocol.REFERENCE: upright cameras (`--cameras`, wild by default), 2D input
in normalized camera coordinates with detector-like noise, camera-frame targets, Sliding window
warmup. The Backbone always gets all Priors; the Versatile model draws a Masking mode
(MASKING_MODES_V2) per batch and, unlike v1, fine-tunes every parameter, since the flag inputs are
new. Validation holds out whole training Clips; the Test clips are never touched.

Resumable: if --out holds last.pt, training continues after its epoch (same command again after a
timeout or cancel), and a finished run exits at once. Every epoch is seeded, so a resumed run
matches an uninterrupted one. --epochs-per-run stops early on purpose, to fit a walltime.
"""
import argparse
import json
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from priorformer.data import WINDOW, ProtocolWindowDataset, load_clips_v2
from priorformer.losses import reference_loss
from priorformer.model import MASKING_MODES_V2, ReferenceLifter, load_reference, model_inputs
from priorformer.protocol import CAMERA_SETTINGS, REFERENCE
from priorformer.splits import split_clips

VAL_STRIDE = 4


def train_masking_modes(model_name):
    return list(MASKING_MODES_V2) if model_name == "versatile" else ["full_priors"]


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, modes, device):
    """Mean MPJPE (m) over the real frames of the validation Windows, per Masking mode."""
    model.eval()
    out = {}
    for mode in modes:
        total, count = 0.0, 0.0
        for batch in loader:
            batch = to_device(batch, device)
            err = (model(*model_inputs(batch, mode)) - batch["y3d"]).norm(dim=-1).mean(dim=-1)
            total += (err * batch["frame_mask"]).sum().item()
            count += batch["frame_mask"].sum().item()
        out[mode] = total / max(count, 1)
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["backbone", "versatile"])
    p.add_argument("--data", required=True, help="folder of processed training clips (.npz)")
    p.add_argument("--init", help="checkpoint to start from (required for versatile)")
    p.add_argument("--epochs", type=int, help="default: 50 for the backbone, 20 for the versatile model")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, help="default: 1e-3 for the backbone, 1e-4 for fine-tuning")
    p.add_argument("--cameras", choices=list(CAMERA_SETTINGS), default="wild")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--warmup", type=float, default=REFERENCE.warmup, help="probability of a warmup Window")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-clips", type=int, help="use only the first N training clips (local checks)")
    p.add_argument("--max-steps", type=int, help="stop each epoch after N batches (local checks)")
    p.add_argument("--epochs-per-run", type=int, help="stop after N epochs in this run (resume later)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    finetune = args.model != "backbone"
    if finetune and not args.init:
        p.error("--init is required for the versatile model")
    args.lr = args.lr or (1e-4 if finetune else 1e-3)
    args.epochs = args.epochs or (20 if finetune else 50)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = replace(REFERENCE, cameras=CAMERA_SETTINGS[args.cameras], warmup=args.warmup,
                       noise=None if args.no_noise else REFERENCE.noise)

    names = sorted(f.name for f in Path(args.data).glob("*.npz"))[:args.max_clips]
    train_names, val_names = split_clips(names, args.val_fraction, args.seed)
    (out / "val_clips.txt").write_text("\n".join(val_names) + "\n")
    train_ds = ProtocolWindowDataset(load_clips_v2(args.data, train_names), protocol, seed=args.seed)
    val_ds = ProtocolWindowDataset(load_clips_v2(args.data, val_names), replace(protocol, warmup=0.0),
                                   stride=VAL_STRIDE, seed=args.seed + 1)
    print(f"{len(train_names)} train clips ({len(train_ds)} windows), {len(val_names)} val clips ({len(val_ds)} windows)")
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=args.workers, pin_memory=True)

    model = ReferenceLifter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    modes = train_masking_modes(args.model)
    config = {**vars(args), "kind": "reference", "window": WINDOW, "protocol": repr(protocol),
              "train_clips": len(train_names), "val_clips": len(val_names), "commit": git_commit()}

    start, best = 1, float("inf")
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"resuming after epoch {ckpt['epoch']}")
    elif args.init:
        model.load_state_dict(load_reference(args.init, device).state_dict())
    if start > args.epochs:
        print("already finished")
        return

    last = args.epochs if not args.epochs_per_run else min(args.epochs, start + args.epochs_per_run - 1)
    for epoch in range(start, last + 1):
        random.seed(args.seed * 1000 + epoch)
        torch.manual_seed(args.seed * 1000 + epoch)
        model.train()
        train_ds.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=True,
                            generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        t0, step_log = time.time(), []
        pbar = tqdm(loader, desc=f"epoch {epoch}", mininterval=30, total=args.max_steps or len(loader))
        for step, batch in enumerate(pbar):
            if args.max_steps and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            mode = random.choice(modes)
            with torch.autocast(device.type, enabled=args.amp):
                loss, metrics = reference_loss(model(*model_inputs(batch, mode)), batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step_log.append({"step": step, "mode": mode, "loss": loss.item(), **metrics})
            pbar.set_postfix(mpjpe=f"{metrics['mpjpe']:.4f}", refresh=False)
        sched.step()

        val = validate(model, val_loader, modes, device)
        log = {"epoch": epoch, "train_mpjpe": float(np.mean([s["mpjpe"] for s in step_log])),
               "val_mpjpe": val, "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps(log), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
        with open(out / "steps.jsonl", "a") as f:
            f.writelines(json.dumps({"epoch": epoch, **s}) + "\n" for s in step_log)

        best_now = val["full_priors"] < best
        best = min(best, val["full_priors"])
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_mpjpe": val, "config": config,
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "best": best}
        torch.save(ckpt, out / "last.tmp")
        Path(out / "last.tmp").replace(out / "last.pt")   # atomic: a kill mid-save keeps the old last.pt
        if best_now:
            torch.save(ckpt, out / "best.pt")


if __name__ == "__main__":
    main()
