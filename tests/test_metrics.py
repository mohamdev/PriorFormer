import numpy as np

from priorformer.metrics import bone_length_error, mpjpe, n_mpjpe, pa_mpjpe, per_joint_error

from test_priors_camera import t_pose


def rotation_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)


def test_mpjpe_of_a_constant_offset_is_its_length():
    gt = np.stack([t_pose()] * 2)
    pred = gt + np.array([0.003, 0.004, 0.0], np.float32)
    np.testing.assert_allclose(mpjpe(pred, gt), [0.005, 0.005], atol=1e-7)
    np.testing.assert_allclose(per_joint_error(pred, gt), np.full((2, 12), 0.005), atol=1e-7)


def test_n_mpjpe_ignores_scale_but_not_rotation():
    gt = t_pose()[None]
    assert mpjpe(gt * 1.2, gt)[0] > 0.05
    np.testing.assert_allclose(n_mpjpe(gt * 1.2, gt), [0.0], atol=1e-6)
    assert n_mpjpe(gt @ rotation_z(30).T, gt)[0] > 0.05


def test_pa_mpjpe_ignores_rotation_scale_and_translation():
    gt = t_pose()[None]
    pred = 0.8 * gt @ rotation_z(40).T + np.array([0.1, 0.0, 0.2], np.float32)
    np.testing.assert_allclose(pa_mpjpe(pred, gt), [0.0], atol=1e-6)


def test_bone_length_error_compares_the_12_edges():
    gt = t_pose()[None]
    # every bone 10% longer: mean of 0.1 * the 12 t_pose edge lengths
    edges = [0.28, 0.26, 0.28, 0.26, 0.44, 0.42, 0.44, 0.42, 0.40, 0.20, np.hypot(0.1, 0.5), np.hypot(0.1, 0.5)]
    np.testing.assert_allclose(bone_length_error(gt * 1.1, gt), [0.1 * np.mean(edges)], atol=1e-6)
