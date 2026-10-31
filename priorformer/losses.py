"""Training losses: MPJPE, bone-length consistency, and pixel reprojection of the first frame."""
import torch

from priorformer.camera import IMG_W
from priorformer.skeleton import SKELETON_EDGES

REPROJ_WEIGHT = 1e-2


def mpjpe(pred, target):
    return (pred - target).norm(dim=-1).mean()


def bone_lengths(x, edges=SKELETON_EDGES):
    return torch.stack([(x[..., i, :] - x[..., j, :]).norm(dim=-1) for i, j in edges], dim=-1)


def bone_loss(pred, target):
    return (bone_lengths(pred) - bone_lengths(target)).abs().mean()


def reprojection_loss(pred, x2d, K, R, t):
    """Mean pixel error between the first predicted frame projected through (K,R,t) and its 2D input."""
    cam = torch.einsum("bij,bkj->bki", R, pred[:, 0]) + t[:, None]
    uvw = torch.einsum("bij,bkj->bki", K, cam)
    uv = uvw[..., :2] / (uvw[..., 2:] + 1e-8)
    return (uv - x2d[:, 0] * IMG_W).abs().mean()


def total_loss(pred, batch):
    """Returns (loss, metrics) with loss = MPJPE + bone + 0.01 * reprojection / image width."""
    pose_err = mpjpe(pred, batch["y3d"])
    bone_err = bone_loss(pred, batch["y3d"])
    reproj_px = reprojection_loss(pred, batch["x2d"], batch["K"], batch["R"], batch["t"])
    loss = pose_err + bone_err + REPROJ_WEIGHT * reproj_px / IMG_W
    return loss, {"mpjpe": pose_err.item(), "bone": bone_err.item(), "reproj_px": reproj_px.item()}
