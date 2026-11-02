"""Prior perturbation sweep of a v1 checkpoint under the v1 protocol, for comparison with the research
reference model (evaluate_v2 --sweep-clips) and with outside reproductions of v1.

    python scripts/prior_sweep_v1.py --checkpoint checkpoints/best_lifter_versatile_128_ddp_epoch12.pt \
        --clips <v1 processed test clips> --out results/v1_prior_sweep.json

On N evenly spaced Test clips, each seen through the first `--cameras` v1 cameras (seeded as in
priorformer.evaluate), with full priors: the Segment prior is scaled by (1 + eps), or the focal length
of the Camera prior is (1 + eps) times the true one. Predictions are lifted and averaged over
overlapping Windows as in the v1 evaluation. Reports MPJPE and bone ratio per eps and their slopes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from priorformer.camera import normalize_intrinsics, project, sample_camera
from priorformer.data import WINDOW, load_clip
from priorformer.evaluate_v2 import SWEEP_EPS, bone_ratio, sweep_report, sweep_summary
from priorformer.model import load_lifter
from priorformer.skeleton import pose_segment_priors
from priorformer.splits import read_split

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def predict_clip_scaled(model, clip, K, R, t, seg_scale=1.0, focal_scale=1.0, window=WINDOW, batch=2048):
    """v1 evaluation's averaged prediction (F,J,3), full priors, with scaled Segment and focal priors."""
    device = next(model.parameters()).device
    starts = np.arange(len(clip) - window + 1)
    x2d = torch.from_numpy(project(clip[starts[:, None] + np.arange(window)], K, R, t)).to(device)
    seg = torch.from_numpy(pose_segment_priors(clip[starts]) * seg_scale).to(device)
    k = normalize_intrinsics(K).copy()
    k[0] *= focal_scale
    k = torch.from_numpy(np.repeat(k[None], len(starts), axis=0)).to(device)
    preds = np.concatenate([model(x2d[i:i + batch], k[i:i + batch], seg[i:i + batch]).cpu().numpy()
                            for i in range(0, len(starts), batch)])
    acc, count = np.zeros_like(clip), np.zeros((len(clip), 1, 1), np.float32)
    for offset in range(window):
        acc[offset:offset + len(starts)] += preds[:, offset]
        count[offset:offset + len(starts)] += 1
    return acc / count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clips", required=True, help="folder of v1 processed test clips (.npz)")
    p.add_argument("--split", default=REPO / "splits" / "test_clips.tsv")
    p.add_argument("--sweep-clips", type=int, default=200)
    p.add_argument("--cameras", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lifter(args.checkpoint, device)
    names = read_split(args.split)
    chosen = names[::max(1, len(names) // args.sweep_clips)][:args.sweep_clips]
    errs = {"segments": [[] for _ in SWEEP_EPS], "focal": [[] for _ in SWEEP_EPS]}
    ratios = {"segments": [[] for _ in SWEEP_EPS], "focal": [[] for _ in SWEEP_EPS]}
    for name in tqdm(chosen, desc="sweep", mininterval=10):
        clip = load_clip(Path(args.clips) / name)
        if len(clip) < WINDOW:
            continue
        for cam in range(args.cameras):
            K, R, t = sample_camera(clip, np.random.default_rng([args.seed, names.index(name), cam]))
            for i, e in enumerate(SWEEP_EPS):
                for channel, kw in [("segments", {"seg_scale": 1 + e}), ("focal", {"focal_scale": 1 + e})]:
                    pred = predict_clip_scaled(model, clip, K, R, t, **kw)
                    errs[channel][i].append(np.linalg.norm(pred - clip, axis=-1).mean())
                    ratios[channel][i].append(bone_ratio(pred, clip))

    sweep = sweep_summary(SWEEP_EPS, errs, ratios, len(chosen), args.cameras)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": {k: str(v) for k, v in vars(args).items()}, "prior_sweep": {"v1": sweep}}, indent=1))
    report = sweep_report({"v1": sweep}).replace("Newest-frame MPJPE", "All-frame MPJPE")
    out.with_suffix(".md").write_text(f"`{Path(args.checkpoint).name}`, v1 protocol (all-frame averaged, "
                                      f"v1 cameras, no 2D noise)\n\n{report}")
    print(report)


if __name__ == "__main__":
    main()
