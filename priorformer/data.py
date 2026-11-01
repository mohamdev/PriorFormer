"""Processed clips and the on-the-fly projected Window dataset."""
import bisect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from priorformer.camera import (default_intrinsics, normalize_intrinsics, normalize_keypoints, pixel_projection,
                               project, sample_camera, sample_camera_v2)
from priorformer.noise import add_noise
from priorformer.skeleton import JOINT_NAMES, segment_prior

R_HIP, L_HIP = JOINT_NAMES.index("r_hip"), JOINT_NAMES.index("l_hip")

WINDOW = 13


def load_clip(path):
    """3D joints (F,12,3) of a processed clip, in its pelvis frame."""
    return np.load(path)["joints_3d"].astype(np.float32)


def load_clips(folder, names):
    return [load_clip(Path(folder) / n) for n in names]


class WindowDataset(Dataset):
    """Every `stride`-th Window of each clip, projected through a random camera per item.

    Cameras are drawn from an RNG seeded by (seed, epoch, index): call set_epoch() to get new
    cameras each epoch; leave the epoch at 0 for a fixed validation set.
    """

    def __init__(self, clips, window=WINDOW, stride=1, seed=0):
        self.clips, self.window, self.stride, self.seed, self.epoch = clips, window, stride, seed, 0
        counts = [max(0, (len(c) - window) // stride + 1) for c in clips]
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        if self.offsets[-1] == 0:
            raise ValueError(f"no clip has >= {window} frames")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        c = bisect.bisect_right(self.offsets, idx) - 1
        start = (idx - self.offsets[c]) * self.stride
        clip = self.clips[c]
        y3d = clip[start:start + self.window]

        rng = np.random.default_rng([self.seed, self.epoch, idx])
        K, R, t = sample_camera(clip, rng)
        return v1_item(y3d, K, R, t)


def v1_item(y3d, K, R, t):
    return {
        "x2d": torch.from_numpy(project(y3d, K, R, t)),
        "y3d": torch.from_numpy(y3d),
        "k": torch.from_numpy(normalize_intrinsics(K)),
        "seg": torch.from_numpy(segment_prior(y3d)),
        "K": torch.from_numpy(K),
        "R": torch.from_numpy(R),
        "t": torch.from_numpy(t),
    }


@dataclass
class Clip:
    """3D joints (F,12,3) in the Clip's pelvis frame, and the gravity up direction in that frame."""
    joints: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], np.float32))


def load_clip_v2(path):
    """A processed Clip with its gravity up (AMASS world z, from the stored pelvis-frame rotation R)."""
    d = np.load(path)
    return Clip(d["joints_3d"].astype(np.float32), (d["R"] @ np.array([0.0, 0.0, 1.0])).astype(np.float32))


def load_clips_v2(folder, names):
    return [load_clip_v2(Path(folder) / n) for n in names]


class ProtocolWindowDataset(Dataset):
    """WindowDataset for any Protocol; with V1 it returns exactly WindowDataset's items.

    Same Window indexing and per-item camera seeding (seed, epoch, index) as WindowDataset.
    """

    def __init__(self, clips, protocol, window=WINDOW, stride=1, seed=0):
        self.protocol = protocol
        self.ups = [c.up for c in clips]
        self.base = WindowDataset([c.joints for c in clips], window, stride, seed)
        self.radii = [np.linalg.norm(c.joints.reshape(-1, 3), axis=1).max() for c in clips]

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.protocol.is_v1:
            return self.base[idx]
        b = self.base
        c = bisect.bisect_right(b.offsets, idx) - 1
        start = (idx - b.offsets[c]) * b.stride
        clip = b.clips[c]
        y3d = clip[start:start + b.window]
        rng = np.random.default_rng([b.seed, b.epoch, idx])
        return protocol_item(clip, y3d, self.ups[c], self.radii[c], self.protocol, rng)


def protocol_item(clip, y3d, up, r_body, protocol, rng):
    """Training item of a Window y3d (T,J,3) of `clip` under a non-V1 protocol."""
    cam = sample_camera_v2(y3d, rng, protocol.cameras, up, r_body)
    if protocol.input == "pixels":
        if protocol.cameras.random_size or protocol.target != "pelvis":
            raise ValueError("pixel input (pixels / 1920) needs fixed-size images and pelvis-frame targets")
        return v1_item(y3d, cam.K, cam.R, cam.t)
    item = observe(y3d, cam, protocol, rng)
    if protocol.warmup and rng.random() < protocol.warmup:
        hidden = len(y3d) - int(rng.integers(1, len(y3d)))   # keep 1 .. T-1 newest frames
        item = hide_oldest_frames(item, hidden)
    return {k: torch.from_numpy(v) for k, v in item.items()}


def hide_oldest_frames(item, n):
    """Sliding window warmup: the `n` oldest frames of a Window are not real frames yet."""
    item = dict(item)
    item["frame_mask"] = item["frame_mask"].copy()
    item["frame_mask"][:n] = 0
    for key in ["x2d", "x2d_est", "vis"]:
        item[key] = item[key].copy()
        item[key][:n] = 0
    return item


def observe(joints, cam, protocol, rng):
    """Inputs and targets (numpy) of joints (T,J,3) seen by `cam` under a normalized-input protocol.

    x2d: observed 2D in normalized camera coordinates with the true intrinsics; x2d_est: the same
    with default intrinsics (camera estimated); x2d_clean: noise-free x2d; vis (T,J): 1 if observed;
    frame_mask (T,): 1 for real frames; y3d: root-relative (mid-hip) targets in the camera frame;
    root (T,3): mid-hip in the camera frame; seg: Segment prior.
    """
    if protocol.target != "camera":
        raise ValueError("normalized input needs camera-frame targets")
    cam_pts = joints @ cam.R.T + cam.t
    root = cam_pts[:, [R_HIP, L_HIP]].mean(axis=1)
    uv = pixel_projection(joints, cam.K, cam.R, cam.t)
    vis = np.ones(uv.shape[:-1], np.float32)
    uv_obs = uv
    if protocol.noise is not None:
        uv_obs, vis = add_noise(uv, protocol.noise, rng)
    return {
        "x2d": normalize_keypoints(uv_obs, cam.K) * vis[..., None],
        "x2d_est": normalize_keypoints(uv_obs, default_intrinsics(cam.W, cam.H)) * vis[..., None],
        "x2d_clean": normalize_keypoints(uv, cam.K),
        "vis": vis,
        "frame_mask": np.ones(len(joints), np.float32),
        "y3d": (cam_pts - root[:, None]).astype(np.float32),
        "root": root.astype(np.float32),
        "seg": segment_prior(joints),
    }
