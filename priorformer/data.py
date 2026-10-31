"""Processed clips and the on-the-fly projected Window dataset."""
import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from priorformer.camera import normalize_intrinsics, project, sample_camera
from priorformer.skeleton import segment_prior

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
        return {
            "x2d": torch.from_numpy(project(y3d, K, R, t)),
            "y3d": torch.from_numpy(y3d),
            "k": torch.from_numpy(normalize_intrinsics(K)),
            "seg": torch.from_numpy(segment_prior(y3d)),
            "K": torch.from_numpy(K),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),
        }
