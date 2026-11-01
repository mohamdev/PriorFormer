"""Detector-like 2D noise (protocol.NoiseSetting) on pixel keypoints."""
import numpy as np

from priorformer.skeleton import JOINT_NAMES

# (right, left) joint pairs, exchanged by a left/right swap.
LEFT_RIGHT = [(JOINT_NAMES.index(f"r_{j}"), JOINT_NAMES.index(f"l_{j}"))
              for j in ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]]


def add_noise(uv, setting, rng):
    """Noisy copy of pixel keypoints (T,J,2) and their visibility (T,J).

    One jitter std per call, drawn from U(0, sigma) times the 2D body size (bounding-box diagonal);
    then outliers, left/right swaps of whole frames, and missing joints (zeroed, visibility 0).
    """
    T, J, _ = uv.shape
    size = np.linalg.norm(uv.reshape(-1, 2).max(axis=0) - uv.reshape(-1, 2).min(axis=0))
    out = uv.astype(np.float64)
    if setting.sigma > 0:
        out = out + rng.normal(0, rng.uniform(0, setting.sigma) * size, out.shape)
    if setting.p_outlier > 0:
        hit = rng.random((T, J)) < setting.p_outlier
        out[hit] += rng.normal(0, setting.outlier_sigma * size, (hit.sum(), 2))
    if setting.p_swap > 0:
        swap = rng.random(T) < setting.p_swap
        right, left = (list(x) for x in zip(*LEFT_RIGHT))
        out[np.ix_(swap, right + left)] = out[np.ix_(swap, left + right)]
    vis = np.ones((T, J), np.float32)
    if setting.p_missing > 0:
        vis[rng.random((T, J)) < setting.p_missing] = 0
    return (out * vis[..., None]).astype(np.float32), vis
