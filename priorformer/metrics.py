"""Evaluation protocol v2 metrics, per pose: predictions and targets (...,J,3), root-relative, metres."""
import numpy as np

from priorformer.skeleton import SKELETON_EDGES


def per_joint_error(pred, gt):
    return np.linalg.norm(pred - gt, axis=-1)


def mpjpe(pred, gt):
    return per_joint_error(pred, gt).mean(axis=-1)


def n_mpjpe(pred, gt):
    """MPJPE after the least-squares scale of each predicted pose: isolates scale errors."""
    scale = (pred * gt).sum(axis=(-1, -2), keepdims=True) / ((pred * pred).sum(axis=(-1, -2), keepdims=True) + 1e-12)
    return mpjpe(scale * pred, gt)


def pa_mpjpe(pred, gt):
    """MPJPE after the best similarity transform (rotation, scale, translation) of each pose."""
    mu_p, mu_g = pred.mean(axis=-2, keepdims=True), gt.mean(axis=-2, keepdims=True)
    p, g = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(np.swapaxes(p, -1, -2) @ g)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones_like(S)
    D[..., -1] = d
    R = (U * D[..., None, :]) @ Vt                      # p @ R best matches g
    scale = (S * D).sum(axis=-1)[..., None, None] / ((p * p).sum(axis=(-1, -2), keepdims=True) + 1e-12)
    return mpjpe(scale * p @ R + mu_g, gt)


def bone_lengths(x, edges=SKELETON_EDGES):
    return np.stack([np.linalg.norm(x[..., i, :] - x[..., j, :], axis=-1) for i, j in edges], axis=-1)


def bone_length_error(pred, gt):
    """Mean absolute bone-length error over the 12 skeleton edges."""
    return np.abs(bone_lengths(pred) - bone_lengths(gt)).mean(axis=-1)
