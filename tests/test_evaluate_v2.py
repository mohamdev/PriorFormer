import numpy as np
import pytest
import torch

from priorformer.data import Clip
from priorformer.evaluate_v2 import (clip_metrics, evaluate_clips_v2, newest_frame_windows, predict_windows,
                                     summarize_v2)
from priorformer.model import ReferenceLifter
from priorformer.protocol import REFERENCE

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
