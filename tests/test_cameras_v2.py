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
