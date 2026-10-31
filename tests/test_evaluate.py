import numpy as np
import pytest
import torch

from priorformer.camera import normalize_intrinsics, project, sample_camera
from priorformer.evaluate import evaluate_clips, predict_clip, summarize
from priorformer.model import TransformerLifter
from priorformer.skeleton import segment_prior


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TransformerLifter().eval()


def walking_clip(n_frames, phase=0.0):
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.3, (12, 3)).astype(np.float32)
    swing = np.sin(np.linspace(0, 3, n_frames) + phase)[:, None, None].astype(np.float32)
    return base[None] + 0.05 * swing * np.ones((1, 12, 3), np.float32)


def naive_prediction(model, clip, K, R, t):
    """One window at a time, segment prior from each window's first frame, overlaps averaged."""
    acc, count = np.zeros_like(clip), np.zeros((len(clip), 1, 1), np.float32)
    k = torch.from_numpy(normalize_intrinsics(K))[None]
    with torch.no_grad():
        for s in range(len(clip) - 12):
            win = clip[s:s + 13]
            x2d = torch.from_numpy(project(win, K, R, t))[None]
            seg = torch.from_numpy(segment_prior(win))[None]
            acc[s:s + 13] += model(x2d, k, seg)[0].numpy()
            count[s:s + 13] += 1
    return acc / count


def test_batched_clip_prediction_matches_one_window_at_a_time(model):
    clip = walking_clip(30)
    K, R, t = sample_camera(clip, np.random.default_rng(3))
    batched = predict_clip(model, clip, K, R, t, "full_priors", batch=7)
    np.testing.assert_allclose(batched, naive_prediction(model, clip, K, R, t), atol=1e-5)


def test_masked_mode_zeroes_the_masked_prior(model):
    clip = walking_clip(13)
    K, R, t = sample_camera(clip, np.random.default_rng(3))
    x2d = torch.from_numpy(project(clip, K, R, t))[None]
    with torch.no_grad():
        expected = model(x2d, torch.zeros(1, 3), torch.from_numpy(segment_prior(clip))[None])[0].numpy()
    np.testing.assert_allclose(predict_clip(model, clip, K, R, t, "camera_masked"), expected, atol=1e-5)


def test_cameras_are_fixed_per_clip_and_camera_index(model):
    clips = {"a.npz": walking_clip(20), "b.npz": walking_clip(16, phase=1.0)}
    four = evaluate_clips(model, clips, n_cameras=4, seed=0, modes=["full_priors"])
    again = evaluate_clips(model, clips, n_cameras=4, seed=0, modes=["full_priors"])
    fewer = evaluate_clips(model, {"b.npz": clips["b.npz"]}, n_cameras=2, seed=0, modes=["full_priors"],
                           clip_index={"b.npz": 1})

    assert four["full_priors"]["a.npz"].shape == (4,)
    np.testing.assert_array_equal(four["full_priors"]["a.npz"], again["full_priors"]["a.npz"])
    np.testing.assert_array_equal(four["full_priors"]["b.npz"][:2], fewer["full_priors"]["b.npz"])
    assert len(set(four["full_priors"]["a.npz"])) == 4  # different cameras give different errors


def test_summary_averages_cameras_then_clips_and_tasks_paper_style():
    errors = {"full_priors": {"c1": np.array([0.010, 0.030]),   # clip mean 20 mm
                              "c2": np.array([0.040, 0.040]),   # clip mean 40 mm
                              "c3": np.array([0.050, 0.070]),   # clip mean 60 mm
                              "c4": np.array([0.100, 0.100])}}  # in no task
    s = summarize(errors, {"walk": ["c1", "c2"], "kick": ["c3"]})["full_priors"]

    assert s["tasks"]["walk"] == pytest.approx({"n": 2, "mean_mm": 30.0, "std_mm": 10.0})
    assert s["tasks"]["kick"] == pytest.approx({"n": 1, "mean_mm": 60.0, "std_mm": 0.0})
    # paper-style average: mean and std of the task means (30, 60)
    assert s["task_average"] == pytest.approx({"mean_mm": 45.0, "std_mm": 15.0})
    # all clips: mean of clip means (20, 40, 60, 100)
    assert s["all_clips"] == pytest.approx({"n": 4, "mean_mm": 55.0, "std_mm": np.std([20, 40, 60, 100])})
