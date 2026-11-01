"""Export a checkpoint to ONNX (inputs x2d (B,13,12,2), k (B,3), seg (B,6); output out (B,13,12,3)).

    python -m priorformer.export_onnx --checkpoint checkpoints/best_lifter_versatile_128_ddp_epoch12.pt --out model.onnx

With --reference, a research reference model checkpoint: inputs x2d (B,13,12,2), vis (B,13,12),
frame_mask (B,13), seg (B,6), flags (B,2); output out (B,13,12,3). The input shapes stay fixed
during Sliding window warmup: unfilled slots have frame_mask 0.
"""
import argparse

import torch

from priorformer.data import WINDOW
from priorformer.model import N_FLAGS, load_lifter, load_reference
from priorformer.skeleton import N_JOINTS, N_SEGMENTS


REFERENCE_INPUTS = ["x2d", "vis", "frame_mask", "seg", "flags"]


def export_reference(model, path):
    inputs = (torch.zeros(1, WINDOW, N_JOINTS, 2), torch.ones(1, WINDOW, N_JOINTS), torch.ones(1, WINDOW),
              torch.zeros(1, N_SEGMENTS), torch.zeros(1, N_FLAGS))
    batch = {0: "batch"}
    torch.onnx.export(model.eval(), inputs, str(path), input_names=REFERENCE_INPUTS, output_names=["out"],
                      dynamic_axes={**{n: batch for n in REFERENCE_INPUTS}, "out": batch}, opset_version=17)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", action="store_true", help="research reference model checkpoint")
    args = p.parse_args()

    if args.reference:
        export_reference(load_reference(args.checkpoint), args.out)
        print(f"wrote {args.out}")
        return
    model = load_lifter(args.checkpoint)
    inputs = (torch.zeros(1, WINDOW, N_JOINTS, 2), torch.zeros(1, 3), torch.zeros(1, N_SEGMENTS))
    batch = {0: "batch"}
    torch.onnx.export(model, inputs, args.out, input_names=["x2d", "k", "seg"], output_names=["out"],
                      dynamic_axes={"x2d": batch, "k": batch, "seg": batch, "out": batch}, opset_version=17)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
