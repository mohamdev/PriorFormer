"""Synthetic monocular cameras looking at the pelvis origin, and the 2D/intrinsics normalization."""
from typing import NamedTuple

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


# ---- v2 cameras (protocol.CameraSetting): upright, optionally wild ----

class Camera(NamedTuple):
    K: np.ndarray       # (3,3) pixels
    R: np.ndarray       # (3,3) Clip frame -> camera
    t: np.ndarray       # (3,)
    W: int              # image size, pixels
    H: int


def _image_size(rng, setting):
    if not setting.random_size:
        return IMG_W, IMG_H
    long_side = int(rng.uniform(640, 3840))
    short_side = int(round(long_side / (4 / 3 if rng.random() < 0.5 else 16 / 9)))
    return (long_side, short_side) if rng.random() < 0.5 else (short_side, long_side)


def _roll(R, deg):
    a = np.deg2rad(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ R


def in_image(points, K, R, t, W, H, margin=0.0):
    """True if every point (...,3) is in front of the camera and projects inside the image."""
    cam = points.reshape(-1, 3) @ R.T + t
    if (cam[:, 2] <= 1e-3).any():
        return False
    uv = (cam @ K.T)[:, :2] / cam[:, 2:]
    m = margin * min(W, H)
    return bool((uv[:, 0] >= m).all() and (uv[:, 0] <= W - m).all() and (uv[:, 1] >= m).all() and (uv[:, 1] <= H - m).all())


def sample_camera_v2(points, rng, setting, up, r_body=None, tries=100):
    """Random upright camera for root-relative joints `points` (...,J,3) in the Clip frame.

    Image up is gravity `up` (up to a random roll); the camera is never below the lowest joint.
    The camera sits around the mid-hip (origin) and looks at it, or with `target_offset` at a
    random point near it, so that the mid-hip projects off-center; the points are then kept inside
    the image (after `tries` failed draws, the camera looks at the mid-hip). Without an offset,
    like v1, the body is not forced inside the image.
    """
    pts = points.reshape(-1, 3)
    r_body = r_body or np.linalg.norm(pts, axis=1).max()
    up = up / np.linalg.norm(up)
    min_height = (pts @ up).min()
    W, H = _image_size(rng, setting)
    for attempt in range(tries + 1):
        if setting.f_px is not None:
            f = rng.uniform(*setting.f_px)
        else:
            f = rng.uniform(*setting.f_rel) * np.hypot(W, H)
        if setting.fill is not None:
            dist = max(r_body * f / (rng.uniform(*setting.fill) * min(W, H) / 2), 1.2 * r_body)
        else:
            dist = r_body * rng.uniform(*setting.distance)
        for _ in range(10000):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v) + 1e-8
            cam_pos = v * dist
            height = cam_pos @ up
            if height >= min_height and np.linalg.norm(cam_pos - height * up) >= r_body * DIST_MIN / 2:
                break
        else:
            raise RuntimeError("no camera position above the lowest joint: check the gravity direction")
        offset = setting.target_offset if attempt < tries else 0.0
        target = np.zeros(3)
        if offset > 0:  # shift the look-at target so that the mid-hip lands up to `offset` half-images off-center
            R0 = look_at_R(cam_pos, target, up=-up)
            o = rng.uniform(-offset, offset, 2) * np.array([W, H]) / 2 * dist / f
            target = o[0] * R0[0] + o[1] * R0[1]
        R = look_at_R(cam_pos, target, up=-up)  # camera y points down the image, i.e. along -up
        if setting.roll_deg:
            R = _roll(R, rng.uniform(-setting.roll_deg, setting.roll_deg))
        t = -R @ cam_pos
        cx = W / 2 + rng.uniform(-setting.pp_jitter, setting.pp_jitter) * W
        cy = H / 2 + rng.uniform(-setting.pp_jitter, setting.pp_jitter) * H
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        if offset == 0 or in_image(pts, K, R, t, W, H):
            return Camera(K, R.astype(np.float32), t.astype(np.float32), W, H)


def pixel_projection(points, K, R, t):
    """Pixel coordinates (...,2) of points (...,3) in the Clip frame."""
    cam = points @ R.T + t
    uvw = cam @ K.T
    return (uvw[..., :2] / uvw[..., 2:]).astype(np.float32)


def normalize_pixels(uv, K):
    """Normalized camera coordinates ((u - cx)/fx, (v - cy)/fy) of pixel positions (...,2): X/Z, Y/Z."""
    return ((uv - K[:2, 2]) / np.array([K[0, 0], K[1, 1]], np.float32)).astype(np.float32)


def default_intrinsics(W, H):
    """Intrinsics assumed when the Camera prior is unknown: f = image diagonal, principal point = center."""
    f = np.hypot(W, H)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
