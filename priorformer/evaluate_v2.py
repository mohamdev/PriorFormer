"""Evaluation protocol v2 of a research reference model on the Test clips (docs/plan-v2.md).

    python -m priorformer.evaluate_v2 --checkpoint runs/reference_versatile/best.pt \
        --clips <processed test clips> --out results/reference_versatile.json

Each Clip is seen through `--cameras` cameras per camera setting ("centered": v1 ranges, upright;
"wild": wide focal range, off-center subject, random image size, roll). Camera j of a Clip is drawn
from an RNG seeded with (seed, setting index, Clip index in splits/test_clips.tsv, j); the 2D noise
of that camera comes from the same RNG, so it is the same for every Masking mode.

Every frame is lifted as the newest frame of its Window, as in a live session: the first 12 frames
are in Sliding window warmup. Metrics (root-relative, camera frame): newest-frame MPJPE (headline),
N-MPJPE, PA-MPJPE and bone-length error on frames with a full Window; MPJPE per joint; MPJPE by
frames of history (warmup); and all-frame MPJPE, overlapping full Windows averaged as in v1.
Errors are averaged over frames, then cameras, then Clips; the Task average is the mean of Task means.

--sweep-clips N adds the prior perturbation sweep on N evenly spaced Test clips: with full priors,
the Segment prior is scaled by (1 + eps), or the focal length used to normalize the 2D input is
(1 + eps) times the true one, on the same cameras and noise for every eps. It reports newest-frame
MPJPE per eps and, on each side, the least-squares slope (mm per unit eps) of the increase over eps = 0
(± its standard error over Clips and cameras); and the bone ratio, predicted / true length of the
Segment prior pairs: its slope (follow) is 1 when the prediction takes the scaled lengths, 0 when it
ignores them.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from priorformer.camera import sample_camera_v2
from priorformer.data import WINDOW, load_clip_v2, observe
from priorformer.evaluate import summarize
from priorformer.metrics import bone_length_error, mpjpe, n_mpjpe, pa_mpjpe, per_joint_error
from priorformer.model import MASKING_MODES_V2, load_reference, model_inputs
from priorformer.protocol import CAMERA_SETTINGS, REFERENCE
from priorformer.skeleton import pose_segment_priors, stretch_to_segments
from priorformer.splits import read_split, task_clips

REPO = Path(__file__).resolve().parents[1]
SCALAR_METRICS = ["mpjpe", "n_mpjpe", "pa_mpjpe", "bone_error", "mpjpe_all_frames"]
SWEEP_EPS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def newest_frame_windows(n_frames, window=WINDOW):
    """Frame indices (F,T) of the Window ending at each frame, and its frame mask (F,T)."""
    idx = np.arange(n_frames)[:, None] - (window - 1) + np.arange(window)[None]
    return np.clip(idx, 0, None), (idx >= 0).astype(np.float32)


@torch.no_grad()
def predict_windows(model, obs, joints, mode, window=WINDOW, batch=2048, seg_scale=1.0, focal_scale=1.0):
    """Predictions (F,T,J,3) of the Window ending at each frame of an observed Clip.

    seg_scale scales the Segment prior; focal_scale normalizes the 2D input with that multiple of the
    true focal length (perturbation sweep)."""
    device = next(model.parameters()).device
    idx, mask = newest_frame_windows(len(joints), window)
    seg = pose_segment_priors(joints[idx[:, 0]]) * seg_scale
    preds = []
    for i in range(0, len(idx), batch):
        sl = slice(i, i + batch)
        m = torch.from_numpy(mask[sl]).to(device)
        b = {k: torch.from_numpy(obs[k][idx[sl]]).to(device) for k in ["x2d", "x2d_est", "vis"]}
        b["x2d"] = b["x2d"] * m[:, :, None, None] / focal_scale
        b["x2d_est"] = b["x2d_est"] * m[:, :, None, None]
        b.update(frame_mask=m, seg=torch.from_numpy(seg[sl]).to(device))
        preds.append(model(*model_inputs(b, mode)).cpu().numpy())
    return np.concatenate(preds)


def clip_metrics(preds, gt, window=WINDOW):
    """Metrics (m) of one Clip and camera from predictions (F,T,J,3) of newest-frame Windows."""
    newest = preds[:, -1]
    full = slice(window - 1, None)
    out = {"mpjpe": mpjpe(newest[full], gt[full]).mean(),
           "n_mpjpe": n_mpjpe(newest[full], gt[full]).mean(),
           "pa_mpjpe": pa_mpjpe(newest[full], gt[full]).mean(),
           "bone_error": bone_length_error(newest[full], gt[full]).mean(),
           "per_joint": per_joint_error(newest[full], gt[full]).mean(axis=0),
           "warmup": mpjpe(newest[:window - 1], gt[:window - 1])}   # history 1 .. window-1 frames
    full_windows = preds[window - 1:]                                 # starting at frames 0 .. F-window
    acc, count = np.zeros_like(gt), np.zeros((len(gt), 1, 1), np.float32)
    for offset in range(window):
        acc[offset:offset + len(full_windows)] += full_windows[:, offset]
        count[offset:offset + len(full_windows)] += 1
    out["mpjpe_all_frames"] = mpjpe(acc / count, gt).mean()
    return out


def evaluate_clips_v2(model, clips, protocol, n_cameras, seed, modes, clip_index=None, setting_index=0,
                      stretch=False):
    """Metrics per mode, metric and Clip: {mode: {metric: {name: array(n_cameras, ...)}}}.

    stretch adds "full_priors_stretched" (bone-stretching test): the full_priors predictions with each
    Segment prior bone stretched to the prior's length."""
    clip_index = clip_index or {name: i for i, name in enumerate(sorted(clips))}
    modes = list(modes)
    out = {mode: {} for mode in modes + (["full_priors_stretched"] if stretch else [])}
    for name, clip in tqdm(clips.items(), desc="clips", mininterval=10):
        if len(clip.joints) < WINDOW:
            print(f"[warn] skipping {name}: {len(clip.joints)} frames < one Window")
            continue
        per_cam = {mode: [] for mode in out}
        for cam_i in range(n_cameras):
            rng = np.random.default_rng([seed, setting_index, clip_index[name], cam_i])
            cam = sample_camera_v2(clip.joints, rng, protocol.cameras, clip.up)
            obs = observe(clip.joints, cam, protocol, rng)
            for mode in modes:
                preds = predict_windows(model, obs, clip.joints, mode)
                per_cam[mode].append(clip_metrics(preds, obs["y3d"]))
                if stretch and mode == "full_priors":
                    seg = pose_segment_priors(clip.joints[newest_frame_windows(len(clip.joints))[0][:, 0]])
                    per_cam["full_priors_stretched"].append(clip_metrics(stretch_to_segments(preds, seg[:, None]), obs["y3d"]))
        for mode in out:
            for metric in per_cam[mode][0]:
                out[mode].setdefault(metric, {})[name] = np.array([m[metric] for m in per_cam[mode]])
    return out


def side_slopes(eps, err_mm):
    """Least-squares slopes (mm per unit eps) of err(eps) - err(0) for eps > 0 and eps < 0.

    err_mm is (n_eps,) or (n_eps, n): then one slope per column."""
    eps, err = list(eps), np.asarray(err_mm, dtype=float)
    d = err - err[eps.index(0.0)]
    e = np.asarray(eps).reshape(-1, *[1] * (d.ndim - 1))
    pos, neg = e.ravel() > 0, e.ravel() < 0
    slope = lambda m: (e[m] * d[m]).sum(axis=0) / (e[m] ** 2).sum()
    if d.ndim == 1:
        return {"positive": float(slope(pos)), "negative": float(slope(neg))}
    return {"positive": slope(pos), "negative": slope(neg)}


def bone_ratio(pred, gt):
    """Mean predicted / true length of the Segment prior pairs over frames (pred, gt: (F,J,3))."""
    return float((pose_segment_priors(pred) / np.maximum(pose_segment_priors(gt), 1e-6)).mean())


def prior_sweep(model, clips, protocol, n_cameras, seed, clip_index, setting_index, eps=SWEEP_EPS):
    """Newest-frame MPJPE (mm, full priors) and bone ratio per eps, for a scaled Segment prior and a
    scaled focal length; slopes of both on each side of eps = 0."""
    errs = {"segments": [[] for _ in eps], "focal": [[] for _ in eps]}
    ratios = {"segments": [[] for _ in eps], "focal": [[] for _ in eps]}
    for name, clip in tqdm(clips.items(), desc="sweep", mininterval=10):
        if len(clip.joints) < WINDOW:
            continue
        for cam_i in range(n_cameras):
            rng = np.random.default_rng([seed, setting_index, clip_index[name], cam_i])
            obs = observe(clip.joints, sample_camera_v2(clip.joints, rng, protocol.cameras, clip.up), protocol, rng)
            gt = obs["y3d"][WINDOW - 1:]
            for i, e in enumerate(eps):
                for channel, kw in [("segments", {"seg_scale": 1 + e}), ("focal", {"focal_scale": 1 + e})]:
                    pred = predict_windows(model, obs, clip.joints, "full_priors", **kw)[WINDOW - 1:, -1]
                    errs[channel][i].append(mpjpe(pred, gt).mean())
                    ratios[channel][i].append(bone_ratio(pred, gt))
    return sweep_summary(eps, errs, ratios, len(clips), n_cameras)


def sweep_summary(eps, errs, ratios, n_clips, n_cameras):
    """Summary of per-eps lists of per-(Clip, camera) errors (m) and bone ratios, per channel."""
    out = {"eps": list(eps), "clips": n_clips, "cameras": n_cameras}
    for channel, per_eps in errs.items():
        mm = 1000 * np.array(per_eps)                                    # (n_eps, n)
        each = side_slopes(eps, mm)
        ratio = np.array(ratios[channel])
        out[channel] = {"mpjpe_mm": [round(float(v), 2) for v in mm.mean(axis=1)],
                        "slope_mm_per_eps": side_slopes(eps, mm.mean(axis=1)),
                        "slope_sem": {k: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0 for k, v in each.items()},
                        "bone_ratio": [round(float(v), 4) for v in ratio.mean(axis=1)],
                        "follow": side_slopes(eps, ratio.mean(axis=1))}
    return out


def sweep_report(sweeps):
    lines = []
    for setting, sw in sweeps.items():
        lines += [f"## Prior perturbation sweep, {setting} cameras ({sw['clips']} Clips x {sw['cameras']} cameras, full priors)",
                  "", "Newest-frame MPJPE (mm) with the prior scaled by (1 + eps):", "",
                  "| eps | " + " | ".join(f"{e:+.2f}" for e in sw["eps"]) + " |", "|---|" + "---|" * len(sw["eps"])]
        for channel in ["segments", "focal"]:
            lines.append(f"| {channel} | " + " | ".join(f"{v:.1f}" for v in sw[channel]["mpjpe_mm"]) + " |")
        lines.append("")
        lines += ["Bone ratio (predicted / true length of the Segment prior pairs):", "",
                  "| eps | " + " | ".join(f"{e:+.2f}" for e in sw["eps"]) + " |", "|---|" + "---|" * len(sw["eps"])]
        for channel in ["segments", "focal"]:
            lines.append(f"| {channel} | " + " | ".join(f"{v:.3f}" for v in sw[channel]["bone_ratio"]) + " |")
        lines += ["", "| | Slope eps > 0 (mm per unit eps) | Slope eps < 0 | Follow eps > 0 | Follow eps < 0 |",
                  "|---|---|---|---|---|"]
        for channel in ["segments", "focal"]:
            sl, se, fo = sw[channel]["slope_mm_per_eps"], sw[channel]["slope_sem"], sw[channel]["follow"]
            lines.append(f"| {channel} | {sl['positive']:+.1f} ± {se['positive']:.1f} | {sl['negative']:+.1f} ± {se['negative']:.1f} "
                         f"| {fo['positive']:.2f} | {fo['negative']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def summarize_v2(errors, tasks):
    """Scalar metrics summarized like v1 (mm); per-joint and warmup curves averaged over Clips (mm)."""
    out = {}
    for mode, per_metric in errors.items():
        out[mode] = {m: summarize({mode: per_metric[m]}, tasks)[mode] for m in SCALAR_METRICS}
        for curve in ["per_joint", "warmup"]:
            out[mode][curve] = (1000 * np.mean([e.mean(axis=0) for e in per_metric[curve].values()], axis=0)).round(2).tolist()
    return out


def markdown_report(results):
    from priorformer.evaluate import markdown_table
    from priorformer.skeleton import JOINT_NAMES

    lines = []
    fmt = lambda s: f"{s['mean_mm']:.1f} ± {s['std_mm']:.1f}"  # noqa: E731
    for setting, summary in results.items():
        modes = list(summary)
        lines += [f"## {setting} cameras", "", "Newest-frame MPJPE (mm) per Task:", "",
                  markdown_table({m: summary[m]["mpjpe"] for m in modes}), "",
                  "All metrics, all Test clips (mean ± std over Clips, mm):", "",
                  "| Metric | " + " | ".join(m.replace("_", " ") for m in modes) + " |",
                  "|---|" + "---|" * len(modes)]
        for metric in SCALAR_METRICS:
            lines.append(f"| {metric} | " + " | ".join(fmt(summary[m][metric]["all_clips"]) for m in modes) + " |")
        lines += ["", "Newest-frame MPJPE per joint (mm):", "",
                  "| Joint | " + " | ".join(m.replace("_", " ") for m in modes) + " |", "|---|" + "---|" * len(modes)]
        for j, name in enumerate(JOINT_NAMES):
            lines.append(f"| {name} | " + " | ".join(f"{summary[m]['per_joint'][j]:.1f}" for m in modes) + " |")
        lines += ["", "Sliding window warmup: newest-frame MPJPE (mm) by frames of history:", "",
                  "| History | " + " | ".join(m.replace("_", " ") for m in modes) + " |", "|---|" + "---|" * len(modes)]
        for h in range(WINDOW - 1):
            lines.append(f"| {h + 1} | " + " | ".join(f"{summary[m]['warmup'][h]:.1f}" for m in modes) + " |")
        lines.append(f"| {WINDOW} (full) | " + " | ".join(fmt(summary[m]["mpjpe"]["all_clips"]) for m in modes) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clips", required=True, help="folder of processed test clips (.npz)")
    p.add_argument("--split", default=REPO / "splits" / "test_clips.tsv")
    p.add_argument("--cameras", type=int, default=10)
    p.add_argument("--settings", nargs="+", default=list(CAMERA_SETTINGS), choices=list(CAMERA_SETTINGS))
    p.add_argument("--noise", choices=["on", "off"], default="on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modes", nargs="+", default=list(MASKING_MODES_V2), choices=list(MASKING_MODES_V2))
    p.add_argument("--max-clips", type=int, help="first N Test clips only (quick checks)")
    p.add_argument("--stretch", action="store_true", help="add the bone-stretching test (full_priors_stretched)")
    p.add_argument("--sweep-clips", type=int, default=0, help="prior perturbation sweep on N evenly spaced Test clips")
    p.add_argument("--sweep-cameras", type=int, default=3)
    p.add_argument("--out", required=True, help="results .json (a .md report is written next to it)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_reference(args.checkpoint, device)
    names = read_split(args.split)[:args.max_clips]
    clips = {n: load_clip_v2(Path(args.clips) / n) for n in names}
    index = {n: i for i, n in enumerate(names)}
    base = REFERENCE if args.noise == "on" else replace(REFERENCE, noise=None)

    results, per_clip, sweeps = {}, {}, {}
    for s_i, setting in enumerate(CAMERA_SETTINGS):
        if setting not in args.settings:
            continue
        errors = evaluate_clips_v2(model, clips, replace(base, cameras=CAMERA_SETTINGS[setting]), args.cameras,
                                   args.seed, args.modes, clip_index=index, setting_index=s_i, stretch=args.stretch)
        results[setting] = summarize_v2(errors, task_clips(names))
        per_clip[setting] = {m: {n: (e * 1000).round(2).tolist() for n, e in errors[m]["mpjpe"].items()} for m in errors}
        if args.sweep_clips:
            chosen = names[::max(1, len(names) // args.sweep_clips)][:args.sweep_clips]
            sweeps[setting] = prior_sweep(model, {n: clips[n] for n in chosen}, replace(base, cameras=CAMERA_SETTINGS[setting]),
                                          args.sweep_cameras, args.seed, index, s_i)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(prm.numel() for prm in model.parameters())
    protocol = {k: str(v) for k, v in vars(args).items()}
    out.write_text(json.dumps({"protocol": protocol, "parameters": n_params, "summary": results,
                               "per_clip_newest_mpjpe_mm": per_clip, "prior_sweep": sweeps}, indent=1))
    report = markdown_report(results) + ("\n" + sweep_report(sweeps) if sweeps else "")
    out.with_suffix(".md").write_text(f"`{Path(args.checkpoint).name}`, {args.cameras} cameras/clip and setting, "
                                      f"seed {args.seed}, noise {args.noise}, {n_params:,} parameters\n\n{report}")
    print(report)


if __name__ == "__main__":
    main()
