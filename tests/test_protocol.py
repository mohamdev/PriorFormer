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


def test_upright_v1_format_items_train_the_v1_model():
    from torch.utils.data import DataLoader

    from priorformer.losses import total_loss
    from priorformer.model import TransformerLifter
    from priorformer.protocol import CENTERED, Protocol

    ds = ProtocolWindowDataset([Clip(moving_clip(40, s)) for s in range(3)], Protocol(cameras=CENTERED))
    item = ds[0]
    assert item["x2d"].shape == (13, 12, 2) and item["y3d"].shape == (13, 12, 3)
    torch.manual_seed(0)
    model = TransformerLifter()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = next(iter(DataLoader(ds, batch_size=16, shuffle=False)))
    losses = []
    for _ in range(100):
        loss, _ = total_loss(model(batch["x2d"], batch["k"], batch["seg"]), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_camera_frame_targets_plus_the_root_project_onto_the_2d_input():
    from priorformer.protocol import CENTERED, Protocol

    ds = ProtocolWindowDataset([Clip(moving_clip(30, 2))], Protocol(cameras=CENTERED, input="normalized", target="camera"))
    for idx in [0, 5, 17]:
        item = ds[idx]
        X = (item["y3d"] + item["root"][:, None]).numpy()
        np.testing.assert_allclose(X[..., :2] / X[..., 2:], item["x2d"].numpy(), atol=1e-5)
        hips = item["y3d"][:, [6, 7]].mean(dim=1)
        torch.testing.assert_close(hips, torch.zeros_like(hips), atol=1e-6, rtol=0)
        # camera-frame targets are a rotation of the Clip-frame Window: same bone lengths
        np.testing.assert_allclose(np.linalg.norm(item["y3d"][:, 0] - item["y3d"][:, 2], axis=-1),
                                   np.linalg.norm(moving_clip(30, 2)[idx:idx + 13, 0] - moving_clip(30, 2)[idx:idx + 13, 2], axis=-1),
                                   atol=1e-5)


def test_without_noise_the_2d_input_is_the_exact_projection():
    from priorformer.protocol import WILD, Protocol

    ds = ProtocolWindowDataset([Clip(moving_clip(30, 2))], Protocol(cameras=WILD, input="normalized", target="camera"))
    item = ds[3]
    torch.testing.assert_close(item["x2d"], item["x2d_clean"])
    assert (item["vis"] == 1).all() and (item["frame_mask"] == 1).all()


def test_warmup_hides_only_the_oldest_frames_of_some_windows():
    from priorformer.protocol import REFERENCE

    ds = ProtocolWindowDataset([Clip(moving_clip(200, 3))], REFERENCE)
    masks = torch.stack([ds[i]["frame_mask"] for i in range(len(ds))])
    hidden = masks == 0
    assert 0.15 < hidden.any(dim=1).float().mean() < 0.45   # warmup probability 0.3
    assert (masks[:, -1] == 1).all()                          # the newest frame is always real
    # hidden frames are a prefix: once a frame is real, every newer one is too
    assert (masks[:, 1:] >= masks[:, :-1]).all()
    item = next(ds[i] for i in range(len(ds)) if ds[i]["frame_mask"][0] == 0)
    assert (item["x2d"][item["frame_mask"] == 0] == 0).all() and (item["vis"][item["frame_mask"] == 0] == 0).all()
