"""PriorFormer: Transformer lifter from 2D joint windows + optional priors to 3D joints.

Parameter names match the released checkpoints, so they load with strict=True.
"""
import math

import torch
import torch.nn as nn

from priorformer.skeleton import N_SEGMENTS

D_MODEL = 128
N_HEAD = 4
N_LAYERS = 4
D_FF = 256
D_IN = 2 + 3 + N_SEGMENTS  # 2D joint + camera prior + segment prior


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerLifter(nn.Module):
    """x2d (B,T,J,2), k (B,3), seg (B,6) -> 3D joints (B,T,J,3) in the clip's pelvis frame.

    A masked prior is passed as zeros.
    """

    def __init__(self):
        super().__init__()
        self.token_embed = nn.Linear(D_IN, D_MODEL)
        self.k_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.seg_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.k_embed = nn.Linear(3, D_MODEL)
        self.seg_embed = nn.Linear(N_SEGMENTS, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.regressor = nn.Linear(D_MODEL, 3)

    def forward(self, x2d, k, seg):
        B, T, J, _ = x2d.shape
        k_exp = k[:, None, None, :].expand(B, T, J, 3)
        seg_exp = seg[:, None, None, :].expand(B, T, J, N_SEGMENTS)
        tokens = torch.cat([x2d, k_exp, seg_exp], dim=-1).reshape(B, T * J, D_IN)
        x = self.token_embed(tokens)

        k_tok = self.k_token.expand(B, -1, -1) + self.k_embed(k).unsqueeze(1)
        seg_tok = self.seg_token.expand(B, -1, -1) + self.seg_embed(seg).unsqueeze(1)

        x = self.pos_enc(torch.cat([k_tok, seg_tok, x], dim=1))
        out = self.transformer(x)
        return self.regressor(out[:, 2:, :].reshape(B, T, J, D_MODEL))


def load_lifter(path, device="cpu"):
    """Load a checkpoint (dict with 'model_state_dict', or a bare state dict) in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = TransformerLifter().to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    return model.eval()


# Masking modes: name -> (camera masked, segments masked).
MASKING_MODES = {
    "full_priors": (False, False),
    "camera_masked": (True, False),
    "segments_masked": (False, True),
    "no_priors": (True, True),
}


def mask_priors(k, seg, mode):
    """Zero the priors that `mode` masks."""
    mask_cam, mask_seg = MASKING_MODES[mode]
    return (torch.zeros_like(k) if mask_cam else k), (torch.zeros_like(seg) if mask_seg else seg)
