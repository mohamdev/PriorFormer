"""Train the Backbone, or fine-tune it into the Expert or Versatile model.

    python -m priorformer.train --mode expert --data <processed train clips> \
        --init checkpoints/best_lifter_128_ddp_epoch50.pt --epochs 10 --out runs/expert

Validation holds out whole training clips (never windows of clips that are trained on), and
the test clips in splits/test_clips.tsv are never touched. Fine-tuning (expert, versatile)
updates only the Transformer encoder and the regressor.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from priorformer.data import WINDOW, WindowDataset, load_clips
from priorformer.losses import total_loss
from priorformer.model import MASKING_MODES, TransformerLifter, mask_priors
from priorformer.splits import split_clips

VAL_STRIDE = 4  # validate on every 4th window of the held-out clips


def train_modes(mode):
    """Masking modes drawn uniformly per batch during training."""
    return list(MASKING_MODES) if mode == "versatile" else ["full_priors"]


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, modes, device):
    model.eval()
    out = {}
    for mode in modes:
        errs = []
        for batch in loader:
            batch = to_device(batch, device)
            k, seg = mask_priors(batch["k"], batch["seg"], mode)
            errs.append((model(batch["x2d"], k, seg) - batch["y3d"]).norm(dim=-1).mean(dim=(1, 2)))
        out[mode] = torch.cat(errs).mean().item()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["backbone", "expert", "versatile"])
    p.add_argument("--data", required=True, help="folder of processed training clips (.npz)")
    p.add_argument("--init", help="checkpoint to start from (required for expert/versatile)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, help="default: 1e-3 for the backbone, 1e-4 for fine-tuning")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.mode != "backbone" and not args.init:
        p.error("--init is required for fine-tuning")
    args.lr = args.lr or (1e-3 if args.mode == "backbone" else 1e-4)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = sorted(f.name for f in Path(args.data).glob("*.npz"))
    train_names, val_names = split_clips(names, args.val_fraction, args.seed)
    (out / "val_clips.txt").write_text("\n".join(val_names) + "\n")
    train_ds = WindowDataset(load_clips(args.data, train_names), seed=args.seed)
    val_ds = WindowDataset(load_clips(args.data, val_names), stride=VAL_STRIDE, seed=args.seed + 1)
    print(f"{len(train_names)} train clips ({len(train_ds)} windows), {len(val_names)} val clips ({len(val_ds)} windows)")
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=args.workers, pin_memory=True)

    model = TransformerLifter().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model_state_dict"])
    if args.mode != "backbone":
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith(("transformer.", "regressor."))
    opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    modes = train_modes(args.mode)
    config = {**vars(args), "window": WINDOW, "train_clips": len(train_names), "val_clips": len(val_names)}

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=True)
        t0 = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch}", mininterval=30)
        for batch in pbar:
            batch = to_device(batch, device)
            k, seg = mask_priors(batch["k"], batch["seg"], random.choice(modes))
            with torch.autocast(device.type, enabled=args.amp):
                loss, metrics = total_loss(model(batch["x2d"], k, seg), batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(mpjpe=f"{metrics['mpjpe']:.4f}", refresh=False)
        sched.step()

        val = validate(model, val_loader, modes, device)
        log = {"epoch": epoch, "val_mpjpe": val, "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps(log), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_mpjpe": val, "config": config}
        torch.save(ckpt, out / "last.pt")
        if val["full_priors"] < best:
            best = val["full_priors"]
            torch.save(ckpt, out / "best.pt")


if __name__ == "__main__":
    main()
