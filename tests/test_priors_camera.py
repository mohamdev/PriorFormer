import numpy as np

from priorformer.camera import IMG_W, normalize_intrinsics, project, sample_camera
from priorformer.skeleton import segment_prior


def t_pose():
    """12 joints in metres: arms out along y, legs down along -z (pelvis frame)."""
    return np.array([
        [0, -0.20, 0.50], [0, 0.20, 0.50],    # shoulders
        [0, -0.48, 0.50], [0, 0.48, 0.50],    # elbows
        [0, -0.74, 0.50], [0, 0.74, 0.50],    # wrists
        [0, -0.10, 0.00], [0, 0.10, 0.00],    # hips
        [0, -0.10, -0.44], [0, 0.10, -0.44],  # knees
        [0, -0.10, -0.86], [0, 0.10, -0.86],  # ankles
    ], dtype=np.float32)


def test_segment_prior_is_six_joint_pair_lengths_over_two():
    seg = segment_prior(t_pose())
    # R/L forearm 0.26, shoulder width 0.40, R shoulder-hip sqrt(0.1^2+0.5^2), R/L thigh 0.44
    expected = np.array([0.26, 0.26, 0.40, np.hypot(0.1, 0.5), 0.44, 0.44]) / 2.0
    np.testing.assert_allclose(seg, expected, atol=1e-6)


def test_segment_prior_of_a_window_uses_its_first_frame():
    window = np.stack([t_pose(), t_pose() * 2.0])
    np.testing.assert_allclose(segment_prior(window), segment_prior(t_pose()), atol=1e-6)


def test_intrinsics_are_normalized_by_2000_and_image_size():
    K = np.array([[1000, 0, 960 + 19.2], [0, 1000, 540 - 10.8], [0, 0, 1]], np.float32)
    np.testing.assert_allclose(normalize_intrinsics(K), [0.5, 0.01, -0.01], atol=1e-6)


def test_camera_sampling_is_seeded_and_respects_constraints():
    clip = np.stack([t_pose()] * 5)
    r_body = np.linalg.norm(clip.reshape(-1, 3), axis=1).max()
    for seed in range(20):
        K, R, t = sample_camera(clip, np.random.default_rng(seed))
        K2, R2, t2 = sample_camera(clip, np.random.default_rng(seed))
        np.testing.assert_array_equal(K, K2); np.testing.assert_array_equal(R, R2); np.testing.assert_array_equal(t, t2)

        cam_pos = -R.T @ t
        assert 1.5 * r_body - 1e-5 <= np.linalg.norm(cam_pos) <= 2.5 * r_body + 1e-5
        assert cam_pos[2] >= clip[..., 2].min()
        assert 800 <= K[0, 0] <= 1400
        # the camera looks at the pelvis origin: it projects onto the principal point
        uv = project(np.zeros((1, 1, 3), np.float32), K, R, t)[0, 0] * IMG_W
        np.testing.assert_allclose(uv, K[:2, 2], atol=1e-3)


def test_projection_divides_pixels_by_image_width():
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], np.float32)
    R, t = np.eye(3, dtype=np.float32), np.array([0, 0, 2], np.float32)
    uv = project(np.array([[[0.2, 0.1, 0.0]]], np.float32), K, R, t)
    np.testing.assert_allclose(uv[0, 0], [(960 + 100) / IMG_W, (540 + 50) / IMG_W], atol=1e-6)


def test_limb_length_spread_is_the_std_of_per_clip_median_lengths():
    from priorformer.skeleton import limb_length_spread

    same = [np.stack([t_pose()] * 3), np.stack([t_pose()] * 4)]
    assert limb_length_spread(same, (6, 8)) == 0.0
    varied = [np.stack([t_pose()] * 3), np.stack([t_pose() * 1.1] * 4)]
    # R thigh 0.44 m and 0.484 m: std 0.022 m
    np.testing.assert_allclose(limb_length_spread(varied, (6, 8)), 0.022, atol=1e-6)


def test_stretching_gives_the_given_lengths_and_keeps_directions_and_hips():
    from priorformer.skeleton import SEG_NORM, SEGMENT_PAIRS, stretch_to_segments

    rng = np.random.default_rng(0)
    pose = t_pose() + rng.normal(0, 0.03, (12, 3))
    lengths = np.array([0.30, 0.22, 0.41, 0.0, 0.47, 0.36])     # (0, 6) is not a bone: left as is
    out = stretch_to_segments(pose, lengths / SEG_NORM)

    for k, (a, b) in enumerate(SEGMENT_PAIRS[:3] + SEGMENT_PAIRS[4:]):
        given = np.delete(lengths, 3)[k]
        np.testing.assert_allclose(np.linalg.norm(out[b] - out[a]), given, atol=1e-6)
        before, after = pose[b] - pose[a], out[b] - out[a]
        np.testing.assert_allclose(after / np.linalg.norm(after), before / np.linalg.norm(before), atol=1e-6)
    np.testing.assert_array_equal(out[[6, 7]], pose[[6, 7]])
    np.testing.assert_allclose(out[[0, 1]].mean(axis=0), pose[[0, 1]].mean(axis=0), atol=1e-6)


def test_stretching_a_pose_to_its_own_lengths_changes_nothing():
    from priorformer.skeleton import pose_segment_priors, stretch_to_segments

    poses = t_pose()[None] + np.random.default_rng(1).normal(0, 0.03, (5, 12, 3))
    np.testing.assert_allclose(stretch_to_segments(poses, pose_segment_priors(poses)), poses, atol=1e-6)
