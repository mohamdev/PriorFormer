import numpy as np
import pytest

from priorformer.camera import sample_camera_v2
from priorformer.protocol import CENTERED, WILD, CameraSetting

from test_priors_camera import t_pose


def pixels(points, cam):
    c = points @ cam.R.T + cam.t
    uvw = c @ cam.K.T
    return uvw[..., :2] / uvw[..., 2:]


def tilted(up_deg):
    a = np.deg2rad(up_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)], np.float32)


@pytest.mark.parametrize("up", [np.array([0, 0, 1], np.float32), tilted(30)])
def test_gravity_projects_straight_down_through_the_mid_hip(up):
    clip = np.stack([t_pose()] * 3)
    for seed in range(30):
        cam = sample_camera_v2(clip, np.random.default_rng(seed), CENTERED, up)
        above, below = pixels(np.stack([0.5 * up, -0.5 * up]), cam)
        hip = pixels(np.zeros((1, 3), np.float32), cam)[0]
        # the vertical through the looked-at mid-hip projects onto the image column of the hip
        np.testing.assert_allclose([above[0], below[0]], [hip[0], hip[0]], atol=1e-2)
        assert above[1] < hip[1] < below[1]  # image v grows downward


def test_roll_tilts_the_projected_vertical_within_the_roll_range():
    clip = np.stack([t_pose()] * 3)
    up = np.array([0, 0, 1], np.float32)
    angles = []
    for seed in range(50):
        cam = sample_camera_v2(clip, np.random.default_rng(seed), CameraSetting(roll_deg=10.0), up)
        d = pixels(-0.5 * up[None], cam)[0] - pixels(np.zeros((1, 3), np.float32), cam)[0]
        angles.append(np.degrees(np.arctan2(d[0], d[1])))
    assert max(np.abs(angles)) <= 10.0 + 1e-3
    assert max(np.abs(angles)) > 5.0


def test_cameras_are_seeded_and_never_below_the_lowest_joint():
    clip = np.stack([t_pose()] * 3)
    up = tilted(20)
    for seed in range(20):
        a = sample_camera_v2(clip, np.random.default_rng(seed), WILD, up)
        b = sample_camera_v2(clip, np.random.default_rng(seed), WILD, up)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        cam_pos = -a.R.T @ a.t
        assert cam_pos @ up >= (clip.reshape(-1, 3) @ up).min() - 1e-6


def test_normalized_keypoints_are_x_over_z_and_y_over_z():
    from priorformer.camera import normalize_keypoints

    cam_points = np.array([[0.3, -0.2, 2.0], [-0.5, 0.4, 4.0]], np.float32)
    K = np.array([[900, 0, 700], [0, 900, 500], [0, 0, 1]], np.float32)
    uvw = cam_points @ K.T
    uv = uvw[:, :2] / uvw[:, 2:]
    np.testing.assert_allclose(normalize_keypoints(uv, K), cam_points[:, :2] / cam_points[:, 2:], atol=1e-6)


def test_a_focal_length_twice_too_long_halves_the_normalized_coordinates():
    from priorformer.camera import default_intrinsics, normalize_keypoints

    K = default_intrinsics(1920, 1080)
    np.testing.assert_allclose(K[0, 0], np.hypot(1920, 1080))
    np.testing.assert_allclose(K[:2, 2], [960, 540])
    uv = np.array([[1200.0, 300.0], [100.0, 900.0]], np.float32)
    K_true = K.copy()
    K_true[0, 0] = K_true[1, 1] = K[0, 0] / 2
    np.testing.assert_allclose(normalize_keypoints(uv, K), 0.5 * normalize_keypoints(uv, K_true), atol=1e-6)


def test_wild_cameras_move_the_subject_off_center_but_keep_it_in_the_image():
    from priorformer.camera import in_image

    clip = np.stack([t_pose()] * 3)
    up = np.array([0, 0, 1], np.float32)
    offsets, sizes, focal = [], set(), []
    for seed in range(200):
        cam = sample_camera_v2(clip, np.random.default_rng(seed), WILD, up)
        assert in_image(clip, cam.K, cam.R, cam.t, cam.W, cam.H)
        hip = pixels(np.zeros((1, 3), np.float32), cam)[0]
        offsets.append(np.abs(hip - cam.K[:2, 2]) / np.array([cam.W, cam.H]))
        sizes.add((cam.W, cam.H))
        focal.append(cam.K[0, 0] / np.hypot(cam.W, cam.H))
    offsets = np.array(offsets)
    assert np.median(offsets.max(axis=1)) > 0.05          # the root is usually well away from the principal point
    assert len(sizes) > 100 and any(w < h for w, h in sizes) and any(w > h for w, h in sizes)  # portrait and landscape
    assert 0.35 <= min(focal) and max(focal) <= 1.5 and max(focal) - min(focal) > 0.8
