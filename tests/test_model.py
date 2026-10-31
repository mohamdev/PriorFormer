import numpy as np
import torch

from priorformer.model import load_lifter


def test_released_checkpoint_matches_onnx_export(versatile_ckpt, onnx_model):
    import onnxruntime as ort

    model = load_lifter(versatile_ckpt)
    rng = np.random.default_rng(0)
    x2d = rng.uniform(0.2, 0.8, (4, 13, 12, 2)).astype(np.float32)
    k = np.array([[0.55, 0.01, -0.02]], np.float32)
    seg = np.array([[0.13, 0.13, 0.17, 0.25, 0.21, 0.21]], np.float32)

    with torch.no_grad():
        ours = model(torch.from_numpy(x2d), torch.from_numpy(k).expand(4, 3), torch.from_numpy(seg).expand(4, 6)).numpy()
    sess = ort.InferenceSession(str(onnx_model))
    ref = np.concatenate([sess.run(None, {"x2d": x2d[i:i + 1], "k": k, "seg": seg})[0] for i in range(4)])

    assert ours.shape == (4, 13, 12, 3)
    np.testing.assert_allclose(ours, ref, atol=1e-4)
