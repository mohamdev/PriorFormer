"""Train the research reference model (M0, docs/plan-v2.md): Backbone, then the Versatile model.

    python -m priorformer.train_v2 --model backbone --data <processed train clips> --out runs/reference_backbone
    python -m priorformer.train_v2 --model versatile --data <processed train clips> \
        --init runs/reference_backbone/best.pt --out runs/reference_versatile
    python -m priorformer.train_v2 --model versatile --data <processed train clips> --out runs/reference_versatile_scratch
    python -m priorformer.train_v2 --model versatile --data <train clips> <train_extra clips> --out runs/reference_full_amass

Training items follow protocol.REFERENCE: upright cameras (`--cameras`, wild by default), 2D input
in normalized camera coordinates with detector-like noise, camera-frame targets, Sliding window
warmup. The Backbone always gets all Priors; the Versatile model draws a Masking mode
(MASKING_MODES_V2) per batch and, unlike v1, fine-tunes every parameter, since the flag inputs are
new. Without --init, the Versatile model is trained from scratch with the Backbone's schedule.

Generation 2 options: --schedule onecycle (linear warmup to --lr over the first 5% of steps, then
cosine to ~0, stepped every batch), --ema (validation, best.pt and evaluation use the EMA weights;
last.pt keeps both), --seg-jitter (with probability 1/2, a training Window's Segment prior is scaled
by one global factor and one factor per segment, so the model learns not to trust a wrong prior;
the 3D target stays exact). Validation holds out whole training Clips; the Test clips are never touched.

Resumable: if --out holds last.pt, training continues where it stopped (same command again after a
timeout or cancel), and a finished run writes DONE and exits at once. Every epoch is seeded and a
mid-epoch save keeps the batch position and RNG states, so a resumed run matches an uninterrupted
one. --epochs-per-run and --time-limit stop early on purpose, to fit a walltime; --save-every saves
mid-epoch in case the job is killed.
"""
import argparse
import copy
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

from priorformer.data import WINDOW, ProtocolWindowDataset, load_clip_v2
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
    """Mean MPJPE (m) over the real frames of every Window of the Validation clips, per Masking mode."""
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


SEG_JITTER_P = 0.5


def jitter_segments(seg, global_max, segment_max):
    """Segment priors (B,S) scaled, for half the Windows, by (1 + U(-g, g)) * (1 + U(-s, s)) per segment."""
    hit = (torch.rand(len(seg), 1, device=seg.device) < SEG_JITTER_P).float()
    g = 1 + global_max * (2 * torch.rand(len(seg), 1, device=seg.device) - 1)
    s = 1 + segment_max * (2 * torch.rand(seg.shape, device=seg.device) - 1)
    return seg * (1 + hit * (g * s - 1))


class EMA:
    """Exponential moving average of a model's floating-point state."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(src[k])


def save_last(out, ckpt):
    torch.save(ckpt, out / "last.tmp")
    Path(out / "last.tmp").replace(out / "last.pt")   # atomic: a kill mid-save keeps the old last.pt


def rng_states():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_states(states):
    random.setstate(states["python"])
    torch.set_rng_state(states["torch"].cpu())   # the checkpoint may have been loaded onto the GPU
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:  # no git (cluster compute nodes): read the checked-out ref
        git = Path(__file__).resolve().parents[1] / ".git"
        try:
            ref = (git / "HEAD").read_text().strip().removeprefix("ref: ")
            if (git / ref).exists():
                return (git / ref).read_text().strip()[:7]
            return next(line[:7] for line in (git / "packed-refs").read_text().splitlines() if line.endswith(" " + ref))
        except (OSError, StopIteration):
            return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["backbone", "versatile"])
    p.add_argument("--data", required=True, nargs="+",
                   help="folder(s) of processed training clips (.npz); names must be unique across folders")
    p.add_argument("--init", help="checkpoint to fine-tune (the versatile model starts from the backbone's)")
    p.add_argument("--epochs", type=int, help="default: 20 when fine-tuning (--init), else 50")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, help="peak learning rate; default: 1e-3, 1e-4 when fine-tuning")
    p.add_argument("--schedule", choices=["cosine", "onecycle"], default="cosine",
                   help="cosine: per epoch from --lr, no warmup; onecycle: per step, 5%% warmup then cosine")
    p.add_argument("--ema", type=float, help="EMA decay per step (e.g. 0.9999); evaluate the EMA weights")
    p.add_argument("--seg-jitter", type=float, nargs=2, metavar=("GLOBAL", "SEGMENT"),
                   help="max relative Segment prior error in training, e.g. 0.2 0.05")
    p.add_argument("--cameras", choices=list(CAMERA_SETTINGS), default="wild")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--warmup", type=float, default=REFERENCE.warmup, help="probability of a warmup Window")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-clips", type=int, help="use only the first N training clips (local checks)")
    p.add_argument("--max-steps", type=int, help="stop each epoch after N batches (local checks)")
    p.add_argument("--epochs-per-run", type=int, help="stop after N epochs in this run (resume later)")
    p.add_argument("--time-limit", type=float, help="minutes: save and stop mid-epoch after this (resume later)")
    p.add_argument("--save-every", type=float, help="minutes between mid-epoch saves of last.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    finetune = args.init is not None
    args.lr = args.lr or (1e-4 if finetune else 1e-3)
    args.epochs = args.epochs or (20 if finetune else 50)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = replace(REFERENCE, cameras=CAMERA_SETTINGS[args.cameras], warmup=args.warmup,
                       noise=None if args.no_noise else REFERENCE.noise)

    folder_of = {f.name: f.parent for d in args.data for f in Path(d).glob("*.npz")}
    names = sorted(folder_of)[:args.max_clips]
    train_names, val_names = split_clips(names, args.val_fraction, args.seed)
    (out / "val_clips.txt").write_text("\n".join(val_names) + "\n")
    train_ds = ProtocolWindowDataset([load_clip_v2(folder_of[n] / n) for n in train_names], protocol, seed=args.seed)
    val_ds = ProtocolWindowDataset([load_clip_v2(folder_of[n] / n) for n in val_names], replace(protocol, warmup=0.0),
                                   stride=VAL_STRIDE, seed=args.seed + 1)
    print(f"{len(train_names)} train clips ({len(train_ds)} windows), {len(val_names)} val clips ({len(val_ds)} windows)")
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=args.workers, pin_memory=True)

    model = ReferenceLifter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_ds) // args.batch
    if args.max_steps:
        steps_per_epoch = min(args.max_steps, steps_per_epoch)
    if args.schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                    pct_start=0.05, cycle_momentum=False, div_factor=25,
                                                    final_div_factor=1e4)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = EMA(model, args.ema) if args.ema else None
    scaler = torch.amp.GradScaler(enabled=args.amp)
    modes = train_masking_modes(args.model)
    config = {**vars(args), "kind": "reference", "window": WINDOW, "protocol": repr(protocol),
              "train_clips": len(train_names), "val_clips": len(val_names), "commit": git_commit()}

    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))   # readable without torch
    start, best, partial = 1, float("inf"), None
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema:
            ema.model.load_state_dict(ckpt["ema_state_dict"])
        start, best, partial = ckpt["epoch"] + 1, ckpt["best"], ckpt.get("partial")
        print(f"resuming after epoch {ckpt['epoch']}" + (f", step {partial['step']}" if partial else ""))
    elif args.init:
        model.load_state_dict(load_reference(args.init, device).state_dict())
        if ema:
            ema.model.load_state_dict(model.state_dict())
    if start > args.epochs:
        print("already finished")
        (out / "DONE").touch()
        return

    run_start = last_save = time.time()
    last = args.epochs if not args.epochs_per_run else min(args.epochs, start + args.epochs_per_run - 1)
    for epoch in range(start, last + 1):
        if args.time_limit is not None and epoch > start and time.time() - run_start > 60 * args.time_limit:
            print(f"time limit: stopping before epoch {epoch}")
            return
        random.seed(args.seed * 1000 + epoch)
        torch.manual_seed(args.seed * 1000 + epoch)
        skip, step_log = 0, []
        if partial and partial["epoch"] == epoch:
            skip, step_log = partial["step"], partial["step_log"]
            set_rng_states(partial["rng"])
        model.train()
        train_ds.set_epoch(epoch)
        gen = torch.Generator().manual_seed(args.seed * 1000 + epoch)   # also seeds the workers, not the global RNG
        order = torch.randperm(len(train_ds), generator=gen)
        loader = DataLoader(train_ds, batch_size=args.batch, sampler=order[skip * args.batch:].tolist(),
                            num_workers=args.workers, pin_memory=True, drop_last=True, generator=gen)
        t0 = time.time()
        steps = steps_per_epoch
        pbar = tqdm(loader, desc=f"epoch {epoch}", mininterval=30, initial=skip, total=steps)
        for step, batch in enumerate(pbar, start=skip):
            if step >= steps:
                break
            batch = to_device(batch, device)
            if args.seg_jitter:
                batch["seg"] = jitter_segments(batch["seg"], *args.seg_jitter)
            mode = random.choice(modes)
            with torch.autocast(device.type, enabled=args.amp):
                loss, metrics = reference_loss(model(*model_inputs(batch, mode)), batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if args.schedule == "onecycle":
                sched.step()
            if ema:
                ema.update(model)
            step_log.append({"step": step, "mode": mode, "loss": loss.item(), **metrics})
            pbar.set_postfix(mpjpe=f"{metrics['mpjpe']:.4f}", refresh=False)

            stop = args.time_limit is not None and time.time() - run_start > 60 * args.time_limit
            if stop or (args.save_every and time.time() - last_save > 60 * args.save_every):
                save_last(out, {"epoch": epoch - 1, "model_state_dict": model.state_dict(), "config": config,
                                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                                "scaler": scaler.state_dict(), "best": best,
                                "ema_state_dict": ema.model.state_dict() if ema else None,
                                "partial": {"epoch": epoch, "step": step + 1, "step_log": step_log,
                                            "rng": rng_states()}})
                last_save = time.time()
                if stop:
                    print(f"time limit: saved epoch {epoch} at step {step + 1}")
                    return
        if args.schedule == "cosine":
            sched.step()

        val = validate(ema.model if ema else model, val_loader, modes, device)
        log = {"epoch": epoch, "train_loss": float(np.mean([s["loss"] for s in step_log])),
               "train_mpjpe": float(np.mean([s["mpjpe"] for s in step_log])),
               "val_mpjpe": val, "minutes": round((time.time() - t0) / 60, 1)}
        if ema:  # val_mpjpe is the EMA model's; the raw weights for comparison
            log["val_mpjpe_raw"] = validate(model, val_loader, ["full_priors"], device)
        print(json.dumps(log), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
        with open(out / "steps.jsonl", "a") as f:
            f.writelines(json.dumps({"epoch": epoch, **s}) + "\n" for s in step_log)

        best_now = val["full_priors"] < best
        best = min(best, val["full_priors"])
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_mpjpe": val, "config": config,
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "best": best, "ema_state_dict": ema.model.state_dict() if ema else None}
        save_last(out, ckpt)
        last_save = time.time()
        if best_now:  # the weights that were validated: EMA ones if any
            torch.save({**ckpt, "model_state_dict": (ema.model if ema else model).state_dict()}, out / "best.pt")
    if last == args.epochs:
        (out / "DONE").touch()


if __name__ == "__main__":
    main()
