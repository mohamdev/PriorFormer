"""Evaluation protocol: MPJPE of a checkpoint on the test clips, per Task and Masking mode.

    python -m priorformer.evaluate --checkpoint checkpoints/best_lifter_versatile_128_ddp_epoch12.pt \
        --clips <processed test clips> --out results/versatile_epoch12.json

Each test clip is seen through `--cameras` random cameras. Camera j of a clip is drawn from an
RNG seeded with (seed, clip index in splits/test_clips.tsv, j), so every clip always gets the
same cameras. Every Window of the clip is lifted (segment prior measured on the Window's first
frame) and overlapping predictions are averaged; MPJPE is averaged over frames and joints, then
over cameras, then over clips. Like the paper, the Task "average" is the mean of Task means.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from priorformer.camera import normalize_intrinsics, project, sample_camera
from priorformer.data import WINDOW, load_clip
from priorformer.model import MASKING_MODES, load_lifter, mask_priors
from priorformer.skeleton import pose_segment_priors
from priorformer.splits import read_split, task_clips

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def predict_clip(model, clip, K, R, t, mode, window=WINDOW, batch=2048):
    """Lift every Window of a clip (F,J,3) seen through camera (K,R,t); returns (F,J,3)."""
    device = next(model.parameters()).device
    starts = np.arange(len(clip) - window + 1)
    x2d = project(clip[starts[:, None] + np.arange(window)], K, R, t)  # (W,T,J,2)
    seg = pose_segment_priors(clip[starts])                            # (W,6), each Window's first frame
    k = np.repeat(normalize_intrinsics(K)[None], len(starts), axis=0)

    preds = []
    for i in range(0, len(starts), batch):
        kb, sb = mask_priors(torch.from_numpy(k[i:i + batch]).to(device),
                             torch.from_numpy(seg[i:i + batch]).to(device), mode)
        preds.append(model(torch.from_numpy(x2d[i:i + batch]).to(device), kb, sb).cpu().numpy())
    preds = np.concatenate(preds)                                       # (W,T,J,3)

    acc, count = np.zeros_like(clip), np.zeros((len(clip), 1, 1), np.float32)
    for offset in range(window):
        acc[offset:offset + len(starts)] += preds[:, offset]
        count[offset:offset + len(starts)] += 1
    return acc / count


def evaluate_clips(model, clips, n_cameras, seed, modes, clip_index=None):
    """MPJPE (m) per mode, clip and camera: {mode: {name: array(n_cameras)}}.

    `clips` maps names to (F,J,3) arrays; `clip_index` gives each clip's camera seed index
    (default: its position among the sorted names). Clips shorter than a Window are skipped.
    """
    clip_index = clip_index or {name: i for i, name in enumerate(sorted(clips))}
    errors = {mode: {} for mode in modes}
    for name, clip in tqdm(clips.items(), desc="clips", mininterval=10):
        if len(clip) < WINDOW:
            continue
        per_cam = {mode: [] for mode in modes}
        for cam in range(n_cameras):
            K, R, t = sample_camera(clip, np.random.default_rng([seed, clip_index[name], cam]))
            for mode in modes:
                pred = predict_clip(model, clip, K, R, t, mode)
                per_cam[mode].append(np.linalg.norm(pred - clip, axis=-1).mean())
        for mode in modes:
            errors[mode][name] = np.array(per_cam[mode])
    return errors


def _stats(values_m):
    v = np.asarray(values_m) * 1000
    return {"mean_mm": float(v.mean()), "std_mm": float(v.std())}


def summarize(errors, tasks):
    """Per mode: all-clip and per-Task mean ± std over clip means, and the mean ± std of Task means."""
    out = {}
    for mode, per_clip in errors.items():
        clip_mean = {name: e.mean() for name, e in per_clip.items()}
        task_stats = {}
        for task, names in tasks.items():
            vals = [clip_mean[n] for n in names if n in clip_mean]
            if vals:
                task_stats[task] = {"n": len(vals), **_stats(vals)}
        out[mode] = {
            "all_clips": {"n": len(clip_mean), **_stats(list(clip_mean.values()))},
            "tasks": task_stats,
            "task_average": _stats([s["mean_mm"] / 1000 for s in task_stats.values()]),
        }
    return out


def markdown_table(summary):
    modes = list(summary)
    head = "| Task | n | " + " | ".join(m.replace("_", " ") for m in modes) + " |"
    rows = [head, "|---|---|" + "---|" * len(modes)]
    fmt = lambda s: f"{s['mean_mm']:.1f} ± {s['std_mm']:.1f}"  # noqa: E731
    for task, s in summary[modes[0]]["tasks"].items():
        rows.append(f"| {task} | {s['n']} | " + " | ".join(fmt(summary[m]["tasks"][task]) for m in modes) + " |")
    rows.append("| **Task average** | | " + " | ".join(fmt(summary[m]["task_average"]) for m in modes) + " |")
    n = summary[modes[0]]["all_clips"]["n"]
    rows.append(f"| **All test clips** | {n} | " + " | ".join(fmt(summary[m]["all_clips"]) for m in modes) + " |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clips", required=True, help="folder of processed test clips (.npz)")
    p.add_argument("--split", default=REPO / "splits" / "test_clips.tsv")
    p.add_argument("--cameras", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", nargs="+", default=list(MASKING_MODES), choices=list(MASKING_MODES))
    p.add_argument("--out", required=True, help="results .json (a .md table is written next to it)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lifter(args.checkpoint, device)
    names = read_split(args.split)
    clips = {n: load_clip(Path(args.clips) / n) for n in names}
    errors = evaluate_clips(model, clips, args.cameras, args.seed, args.modes,
                            clip_index={n: i for i, n in enumerate(names)})
    summary = summarize(errors, task_clips(names))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    protocol = {k: str(v) for k, v in vars(args).items()}
    out.write_text(json.dumps({"protocol": protocol, "summary": summary,
                               "per_clip_mm": {m: {n: (e * 1000).round(2).tolist() for n, e in pc.items()}
                                               for m, pc in errors.items()}}, indent=1))
    table = markdown_table(summary)
    out.with_suffix(".md").write_text(f"`{Path(args.checkpoint).name}`, {args.cameras} cameras/clip, seed {args.seed}\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    main()
