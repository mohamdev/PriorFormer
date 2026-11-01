import numpy as np

from priorformer.noise import LEFT_RIGHT, add_noise
from priorformer.protocol import NoiseSetting

QUIET = NoiseSetting(sigma=0.0, p_outlier=0.0, p_swap=0.0, p_missing=0.0)


def keypoints(n_frames=50, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(400, 800, (n_frames, 12, 2))).astype(np.float32)


def test_noise_is_seeded():
    uv = keypoints()
    a = add_noise(uv, NoiseSetting(), np.random.default_rng(1))
    b = add_noise(uv, NoiseSetting(), np.random.default_rng(1))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_zero_noise_returns_the_exact_keypoints_all_visible():
    uv = keypoints()
    out, vis = add_noise(uv, QUIET, np.random.default_rng(0))
    np.testing.assert_array_equal(out, uv)
    assert vis.shape == (50, 12) and (vis == 1).all()


def test_a_swap_exchanges_left_and_right_joints():
    uv = keypoints()
    out, _ = add_noise(uv, NoiseSetting(sigma=0.0, p_outlier=0.0, p_swap=1.0, p_missing=0.0), np.random.default_rng(0))
    right, left = zip(*LEFT_RIGHT)
    np.testing.assert_array_equal(out[:, list(right)], uv[:, list(left)])
    np.testing.assert_array_equal(out[:, list(left)], uv[:, list(right)])


def test_missing_joints_are_flagged_and_carry_no_position():
    uv = keypoints(400)
    out, vis = add_noise(uv, NoiseSetting(sigma=0.0, p_outlier=0.0, p_swap=0.0, p_missing=0.2), np.random.default_rng(0))
    assert 0.15 < 1 - vis.mean() < 0.25
    assert (out[vis == 0] == 0).all()
    np.testing.assert_array_equal(out[vis == 1], uv[vis == 1])


def test_jitter_scales_with_the_2d_body_size():
    uv = keypoints(400)
    size = np.linalg.norm(uv.reshape(-1, 2).max(0) - uv.reshape(-1, 2).min(0))
    out, _ = add_noise(uv, NoiseSetting(sigma=0.02, p_outlier=0.0, p_swap=0.0, p_missing=0.0), np.random.default_rng(0))
    err = np.abs(out - uv)
    assert 0 < err.std() <= 0.02 * size
