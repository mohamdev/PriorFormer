"""Synthetic monocular cameras looking at the pelvis origin, and the 2D/intrinsics normalization."""
import numpy as np

IMG_W, IMG_H = 1920, 1080
F_MIN, F_MAX = 800.0, 1400.0
DIST_MIN, DIST_MAX = 1.5, 2.5   # camera distance, in units of the clip's body radius
PP_JITTER = 0.03                # principal point offset, fraction of image size


def look_at_R(cam_pos, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])):
    z = target - cam_pos
    z /= np.linalg.norm(z) + 1e-8
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def sample_camera(clip, rng):
    """Random camera for a clip (F,J,3): returns K (3,3), R_w2c (3,3), t_w2c (3,)."""
    coords = clip.reshape(-1, 3)
    r_body = np.linalg.norm(coords, axis=1).max()
    min_z = coords[:, 2].min()
    min_xy = r_body * DIST_MIN / 2
    while True:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v) + 1e-8
        cam_pos = v * r_body * rng.uniform(DIST_MIN, DIST_MAX)
        if cam_pos[2] >= min_z and np.linalg.norm(cam_pos[:2]) >= min_xy:
            break
    R = look_at_R(cam_pos)
    t = -R @ cam_pos

    f = rng.uniform(F_MIN, F_MAX)
    cx = IMG_W / 2 + rng.uniform(-PP_JITTER, PP_JITTER) * IMG_W
    cy = IMG_H / 2 + rng.uniform(-PP_JITTER, PP_JITTER) * IMG_H
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
    return K, R.astype(np.float32), t.astype(np.float32)


def normalize_intrinsics(K):
    """Camera prior: [f/2000, (cx-W/2)/W, (cy-H/2)/H]."""
    return np.array([K[0, 0] / 2000.0, (K[0, 2] - IMG_W / 2) / IMG_W, (K[1, 2] - IMG_H / 2) / IMG_H], np.float32)


def project(points, K, R, t):
    """Project (...,3) world points to model input: pixel coordinates divided by IMG_W."""
    cam = points @ R.T + t
    uvw = cam @ K.T
    return (uvw[..., :2] / (uvw[..., 2:] + 1e-8) / IMG_W).astype(np.float32)
