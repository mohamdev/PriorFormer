import numpy as np
import torch

from priorformer.data import Clip, ProtocolWindowDataset, WindowDataset
from priorformer.protocol import V1


def moving_clip(n_frames, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.3, (12, 3)).astype(np.float32)
    base[[6, 7]] = [[0, -0.1, 0], [0, 0.1, 0]]  # hips symmetric about the origin, like processed Clips
    swing = np.sin(np.linspace(0, 3, n_frames))[:, None, None].astype(np.float32)
    clip = base[None] + 0.05 * swing * np.ones((1, 12, 3), np.float32)
    clip[:, [6, 7]] = base[[6, 7]]
    return clip


def test_v1_protocol_reproduces_the_v1_window_dataset_exactly():
    clips = [moving_clip(20, 0), moving_clip(15, 1)]
    v1 = WindowDataset(clips, seed=3)
    ours = ProtocolWindowDataset([Clip(c) for c in clips], V1, seed=3)
    assert len(ours) == len(v1)
    for epoch in [0, 2]:
        v1.set_epoch(epoch)
        ours.set_epoch(epoch)
        for idx in range(len(v1)):
            a, b = v1[idx], ours[idx]
            assert a.keys() == b.keys()
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
