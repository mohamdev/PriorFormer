import numpy as np
import pytest
import torch

from priorformer.data import Clip
from priorformer.evaluate_v2 import (clip_metrics, evaluate_clips_v2, newest_frame_windows, predict_windows,
                                     summarize_v2)
from priorformer.model import ReferenceLifter
from priorformer.protocol import REFERENCE

from test_priors_camera import t_pose
from test_protocol import moving_clip


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ReferenceLifter().eval()


def test_each_frame_is_the_newest_frame_of_its_window():
    idx, mask = newest_frame_windows(4, window=3)
    np.testing.assert_array_equal(idx, [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(mask, [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]])


def test_perfect_predictions_score_zero_everywhere():
    gt = moving_clip(20)
    idx, _ = newest_frame_windows(len(gt))
    m = clip_metrics(gt[idx], gt)
    for key in ["mpjpe", "n_mpjpe", "pa_mpjpe", "bone_error", "mpjpe_all_frames"]:
        assert m[key] == pytest.approx(0.0, abs=1e-6)
    assert m["per_joint"].shape == (12,) and m["warmup"].shape == (12,)


def test_cameras_and_noise_are_fixed_per_clip_and_camera_and_shared_by_modes(model):
    clips = {"a.npz": Clip(moving_clip(20, 0)), "b.npz": Clip(moving_clip(16, 1))}
    modes = ["full_priors", "camera_estimated"]
    two = evaluate_clips_v2(model, clips, REFERENCE, n_cameras=2, seed=0, modes=modes)
    again = evaluate_clips_v2(model, clips, REFERENCE, n_cameras=2, seed=0, modes=modes)
    only_b = evaluate_clips_v2(model, {"b.npz": clips["b.npz"]}, REFERENCE, n_cameras=1, seed=0, modes=modes,
                               clip_index={"b.npz": 1})
    assert two["full_priors"]["mpjpe"]["a.npz"].shape == (2,)
    np.testing.assert_array_equal(two["full_priors"]["mpjpe"]["a.npz"], again["full_priors"]["mpjpe"]["a.npz"])
    np.testing.assert_array_equal(two["camera_estimated"]["mpjpe"]["b.npz"][:1], only_b["camera_estimated"]["mpjpe"]["b.npz"])

    s = summarize_v2(two, {"walk": ["a.npz", "b.npz"]})
    assert set(s["full_priors"]) >= {"mpjpe", "n_mpjpe", "pa_mpjpe", "bone_error", "mpjpe_all_frames", "per_joint", "warmup"}
    assert len(s["full_priors"]["warmup"]) == 12


def test_newest_frame_prediction_matches_a_single_window(model):
    from priorformer.camera import sample_camera_v2
    from priorformer.data import observe
    from priorformer.model import model_inputs
    from priorformer.skeleton import segment_prior

    joints = moving_clip(18)
    rng = np.random.default_rng(0)
    obs = observe(joints, sample_camera_v2(joints, rng, REFERENCE.cameras, np.array([0, 0, 1.0])), REFERENCE, rng)
    preds = predict_windows(model, obs, joints, "full_priors", batch=4)
    f = 15
    single = {k: torch.from_numpy(obs[k][f - 12:f + 1])[None] for k in ["x2d", "x2d_est", "vis", "frame_mask"]}
    single["seg"] = torch.from_numpy(segment_prior(joints[f - 12:f + 1]))[None]
    with torch.no_grad():
        expected = model(*model_inputs(single, "full_priors"))[0, -1].numpy()
    np.testing.assert_allclose(preds[f, -1], expected, atol=1e-5)


def test_side_slopes_are_fitted_separately_on_each_side_of_zero():
    from priorformer.evaluate_v2 import side_slopes
    eps = [-0.2, -0.1, 0.0, 0.1, 0.2]
    err = [41.0, 40.5, 40.0, 50.0, 60.0]                      # +100 mm/eps above, -5 mm/eps below
    assert side_slopes(eps, err) == pytest.approx({"positive": 100.0, "negative": -5.0})


def test_side_slopes_of_a_matrix_are_the_slopes_of_each_column():
    from priorformer.evaluate_v2 import side_slopes
    eps = [-0.2, -0.1, 0.0, 0.1, 0.2]
    a, b = [41.0, 40.5, 40.0, 50.0, 60.0], [30.0, 30.0, 30.0, 32.0, 34.0]
    both = side_slopes(eps, np.array([a, b]).T)
    np.testing.assert_allclose(both["positive"], [100.0, 20.0])
    np.testing.assert_allclose(both["negative"], [-5.0, 0.0])


def test_follow_is_one_when_the_prediction_takes_the_scaled_lengths_and_zero_when_it_ignores_them():
    from priorformer.evaluate_v2 import bone_ratio, side_slopes, sweep_summary
    gt = t_pose()[None]
    eps = [-0.2, 0.0, 0.2]
    follows = [[bone_ratio(gt * (1 + e), gt)] for e in eps]       # pose about the origin: lengths scale by 1 + eps
    ignores = [[bone_ratio(gt, gt)] for e in eps]
    assert side_slopes(eps, np.array(follows)[:, 0]) == pytest.approx({"positive": 1.0, "negative": 1.0})
    errs = [[0.04], [0.04], [0.04]]
    out = sweep_summary(eps, {"segments": errs, "focal": errs}, {"segments": follows, "focal": ignores}, 1, 1)
    assert out["segments"]["follow"] == pytest.approx({"positive": 1.0, "negative": 1.0})
    assert out["focal"]["follow"] == pytest.approx({"positive": 0.0, "negative": 0.0})
    assert out["segments"]["bone_ratio"] == pytest.approx([0.8, 1.0, 1.2])


def test_the_unperturbed_sweep_point_equals_the_full_priors_evaluation(model):
    from priorformer.evaluate_v2 import prior_sweep
    clips = {"a.npz": Clip(moving_clip(20, 0)), "b.npz": Clip(moving_clip(16, 1))}
    index = {"a.npz": 0, "b.npz": 1}
    sweep = prior_sweep(model, clips, REFERENCE, 2, 0, index, 1, eps=[-0.1, 0.0, 0.1])
    full = evaluate_clips_v2(model, clips, REFERENCE, 2, 0, ["full_priors"], clip_index=index, setting_index=1)
    expected = 1000 * np.mean([e.mean() for e in full["full_priors"]["mpjpe"].values()])
    for channel in ["segments", "focal"]:
        assert sweep[channel]["mpjpe_mm"][1] == pytest.approx(expected, abs=0.01)
        assert sweep[channel]["mpjpe_mm"][0] != sweep[channel]["mpjpe_mm"][2]   # the perturbation reaches the model
