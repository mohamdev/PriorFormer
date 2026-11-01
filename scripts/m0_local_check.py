"""Local small-scale training check of the research reference model against the v1 training (ticket #9).

    python scripts/m0_local_check.py --data <processed train clips> --out results/m0_local_check

Two checks, on the same chosen Clips for every variant:
1. Overfit: one fixed batch of Windows (fixed cameras and noise), many steps. The loss must fall
   close to zero for v1 and for the research reference model.
2. Short training: a few epochs over a small set of training Clips with new cameras each epoch,
   MPJPE on held-out Clips, for v1 and for the reference model with each M0 option alone (including
   the camera estimated Masking mode) and all together, so that a broken option shows up as a curve that doesn't fall like the others.
v1 errors are in the Clip pelvis frame and reference errors in the camera frame (both root-relative).
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from priorformer.data import Clip, ProtocolWindowDataset, load_clip_v2
from priorformer.losses import reference_loss, total_loss
from priorformer.model import ReferenceLifter, TransformerLifter, model_inputs
from priorformer.protocol import CENTERED, REFERENCE, V1, WILD, NoiseSetting, Protocol
from priorformer.splits import split_clips

BASE = Protocol(cameras=CENTERED, input="normalized", target="camera")
# name -> (protocol, Masking mode of the reference model)
VARIANTS = {
    "v1": (V1, None),
    "reference: centered, no noise, no warmup": (BASE, "full_priors"),
    "+ camera estimated": (BASE, "camera_estimated"),
    "+ wild cameras": (replace(BASE, cameras=WILD), "full_priors"),
    "+ 2D noise": (replace(BASE, noise=NoiseSetting()), "full_priors"),
    "+ warmup": (replace(BASE, warmup=REFERENCE.warmup), "full_priors"),
    "reference: all M0 options": (REFERENCE, "full_priors"),
}


def step(model, batch, protocol, mode):
    if protocol.is_v1:
        return total_loss(model(batch["x2d"], batch["k"], batch["seg"]), batch)
    return reference_loss(model(*model_inputs(batch, mode)), batch)


@torch.no_grad()
def val_mpjpe(model, loader, protocol, mode, device):
    model.eval()
    total, count = 0.0, 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if protocol.is_v1:
            err = (model(batch["x2d"], batch["k"], batch["seg"]) - batch["y3d"]).norm(dim=-1).mean(dim=-1)
            mask = torch.ones_like(err)
        else:
            err = (model(*model_inputs(batch, mode)) - batch["y3d"]).norm(dim=-1).mean(dim=-1)
            mask = batch["frame_mask"]
        total += (err * mask).sum().item()
        count += mask.sum().item()
    model.train()
    return 1000 * total / count


def new_model(protocol, device):
    torch.manual_seed(0)
    return (TransformerLifter() if protocol.is_v1 else ReferenceLifter()).to(device)


def overfit(clips, protocol, mode, device, steps, batch_size):
    ds = ProtocolWindowDataset(clips, protocol, stride=7, seed=0)
    batch = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False)))
    batch = {k: v.to(device) for k, v in batch.items()}
    model = new_model(protocol, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    curve = []
    for i in range(steps):
        loss, metrics = step(model, batch, protocol, mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append({"step": i, "loss": loss.item(), "mpjpe_mm": 1000 * metrics["mpjpe"]})
    model.eval()  # final error without dropout
    with torch.no_grad():
        _, metrics = step(model, batch, protocol, mode)
    return curve, 1000 * metrics["mpjpe"]


def short_training(train, val, protocol, mode, device, epochs, steps, batch_size, workers):
    train_ds = ProtocolWindowDataset(train, protocol, seed=0)
    val_ds = ProtocolWindowDataset(val, replace(protocol, warmup=0.0), stride=4, seed=1)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=workers)
    model = new_model(protocol, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    curve = [{"epoch": 0, "train_mpjpe_mm": None, "val_mpjpe_mm": val_mpjpe(model, val_loader, protocol, mode, device)}]
    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                            generator=torch.Generator().manual_seed(epoch))
        errs = []
        for i, batch in enumerate(loader):
            if i >= steps:
                break
            loss, metrics = step(model, {k: v.to(device) for k, v in batch.items()}, protocol, mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
            errs.append(1000 * metrics["mpjpe"])
        curve.append({"epoch": epoch, "train_mpjpe_mm": float(np.mean(errs)),
                      "val_mpjpe_mm": val_mpjpe(model, val_loader, protocol, mode, device)})
    return curve


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, r in results.items():
        c = r["overfit_curve"]
        axes[0].plot([p["step"] for p in c], [p["mpjpe_mm"] for p in c], label=name)
        c = r["training_curve"]
        axes[1].plot([p["epoch"] for p in c], [p["val_mpjpe_mm"] for p in c], marker="o", label=name)
    axes[0].set(title="Overfit one fixed batch", xlabel="step", ylabel="train MPJPE (mm)", yscale="log")
    axes[1].set(title="Short training, held-out Clips", xlabel="epoch", ylabel="val MPJPE (mm)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=60, help="first N training clips (sorted names)")
    p.add_argument("--overfit-steps", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps", type=int, default=300, help="batches per epoch")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = sorted(f.name for f in Path(args.data).glob("*.npz"))[:args.clips]
    train_names, val_names = split_clips(names, 0.2, 0)
    clips = {n: load_clip_v2(Path(args.data) / n) for n in names}
    train, val = [clips[n] for n in train_names], [clips[n] for n in val_names]
    print(f"{len(train)} train clips, {len(val)} held-out clips, device {device}")

    results = {}
    for name, (protocol, mode) in VARIANTS.items():
        t0 = time.time()
        curve, final = overfit(train[:8], protocol, mode, device, args.overfit_steps, args.batch)
        training = short_training(train, val, protocol, mode, device, args.epochs, args.steps, args.batch, args.workers)
        results[name] = {"overfit_curve": curve, "overfit_first_mm": curve[0]["mpjpe_mm"], "overfit_final_mm": final,
                         "training_curve": training, "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{name}: overfit {curve[0]['mpjpe_mm']:.0f} -> {final:.1f} mm; "
              f"val {training[0]['val_mpjpe_mm']:.0f} -> {training[-1]['val_mpjpe_mm']:.1f} mm ({results[name]['minutes']} min)")

    (out / "results.json").write_text(json.dumps({"args": vars(args), "train_clips": train_names,
                                                  "val_clips": val_names, "results": results}, indent=1))
    plot(results, out / "curves.png")
    rows = ["| Variant | Overfit: first → final MPJPE (mm) | Val MPJPE epoch 0 → 1 → last (mm) |", "|---|---|---|"]
    for name, r in results.items():
        c = r["training_curve"]
        rows.append(f"| {name} | {r['overfit_first_mm']:.0f} → {r['overfit_final_mm']:.1f} | "
                    f"{c[0]['val_mpjpe_mm']:.0f} → {c[1]['val_mpjpe_mm']:.0f} → {c[-1]['val_mpjpe_mm']:.1f} |")
    (out / "table.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
