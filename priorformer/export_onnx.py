"""Export a checkpoint to ONNX (inputs x2d (B,13,12,2), k (B,3), seg (B,6); output out (B,13,12,3)).

    python -m priorformer.export_onnx --checkpoint checkpoints/best_lifter_versatile_128_ddp_epoch12.pt --out model.onnx
"""
import argparse

import torch

from priorformer.data import WINDOW
from priorformer.model import load_lifter
from priorformer.skeleton import N_JOINTS, N_SEGMENTS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = load_lifter(args.checkpoint)
    inputs = (torch.zeros(1, WINDOW, N_JOINTS, 2), torch.zeros(1, 3), torch.zeros(1, N_SEGMENTS))
    batch = {0: "batch"}
    torch.onnx.export(model, inputs, args.out, input_names=["x2d", "k", "seg"], output_names=["out"],
                      dynamic_axes={"x2d": batch, "k": batch, "seg": batch, "out": batch}, opset_version=17)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
