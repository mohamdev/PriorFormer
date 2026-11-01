import numpy as np
import pytest
import torch

from priorformer.model import MASKING_MODES_V2, ReferenceLifter, model_inputs

B, T, J = 3, 13, 12


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x2d": torch.randn(B, T, J, 2, generator=g) * 0.2,
        "x2d_est": torch.randn(B, T, J, 2, generator=g) * 0.1,
        "vis": torch.ones(B, T, J),
        "frame_mask": torch.ones(B, T),
        "seg": torch.rand(B, 6, generator=g) * 0.2 + 0.1,
    }


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ReferenceLifter().eval()


def test_reference_model_is_small_and_lifts_every_masking_mode(model):
    assert sum(p.numel() for p in model.parameters()) < 600_000
    for mode in MASKING_MODES_V2:
        with torch.no_grad():
            out = model(*model_inputs(batch(), mode))
        assert out.shape == (B, T, J, 3)
        assert torch.isfinite(out).all()


def test_camera_estimated_uses_default_intrinsics_and_raises_the_flag():
    b = batch()
    x2d, vis, frame_mask, seg, flags = model_inputs(b, "camera_estimated")
    torch.testing.assert_close(x2d, b["x2d_est"])
    torch.testing.assert_close(flags, torch.tensor([[1.0, 0.0]] * B))
    x2d, vis, frame_mask, seg, flags = model_inputs(b, "no_priors")
    torch.testing.assert_close(seg, torch.zeros(B, 6))
    torch.testing.assert_close(flags, torch.tensor([[1.0, 1.0]] * B))


def test_onnx_export_round_trips(model, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from priorformer.export_onnx import export_reference

    path = tmp_path / "reference.onnx"
    export_reference(model, path)
    inputs = model_inputs(batch(1), "full_priors")
    with torch.no_grad():
        ours = model(*inputs).numpy()
    sess = ort.InferenceSession(str(path))
    names = [i.name for i in sess.get_inputs()]
    ref = sess.run(None, {n: x.numpy() for n, x in zip(names, inputs)})[0]
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_reference_loss_counts_only_real_frames():
    from priorformer.losses import reference_loss

    b = batch()
    b["y3d"] = torch.randn(B, T, J, 3)
    b["root"] = torch.tensor([0.0, 0.0, 3.0]).expand(B, T, 3)
    b["x2d_clean"] = b["x2d"]
    b["frame_mask"][:, :5] = 0
    pred = b["y3d"].clone()
    pred[:, :5] += 1.0  # errors on masked frames only
    loss, metrics = reference_loss(pred, b)
    assert metrics["mpjpe"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["bone"] == pytest.approx(0.0, abs=1e-6)


def test_frames_outside_the_window_have_no_effect(model):
    b = batch()
    b["frame_mask"][:, :6] = 0
    other = {k: v.clone() for k, v in b.items()}
    other["x2d"][:, :6] = 5.0
    other["vis"][:, :6] = 0.3
    with torch.no_grad():
        a = model(*model_inputs(b, "full_priors"))
        c = model(*model_inputs(other, "full_priors"))
    torch.testing.assert_close(a, c, atol=1e-6, rtol=0)
    with torch.no_grad():  # ... but they do change the output when real
        full = model(*model_inputs(batch(), "full_priors"))
    assert (full[:, -1] - a[:, -1]).abs().max() > 1e-4


def test_onnx_matches_pytorch_during_warmup(model, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from priorformer.export_onnx import export_reference

    path = tmp_path / "reference.onnx"
    export_reference(model, path)
    sess = ort.InferenceSession(str(path))
    names = [i.name for i in sess.get_inputs()]
    for hidden in [12, 7, 1]:
        b = batch(2)
        b["frame_mask"][:, :hidden] = 0
        inputs = model_inputs(b, "camera_estimated")
        with torch.no_grad():
            ours = model(*inputs).numpy()
        ref = sess.run(None, {n: x.numpy() for n, x in zip(names, inputs)})[0]
        np.testing.assert_allclose(ours, ref, atol=1e-4)
