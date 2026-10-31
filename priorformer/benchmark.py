"""Latency of one forward pass (one Window, batch 1), on CPU and, if available, GPU.

    python -m priorformer.benchmark --checkpoint checkpoints/best_lifter_versatile_128_ddp_epoch12.pt
"""
import argparse
import time

import numpy as np
import torch

from priorformer.data import WINDOW
from priorformer.model import load_lifter
from priorformer.skeleton import N_JOINTS, N_SEGMENTS


@torch.no_grad()
def latency_us(model, device, warmup, runs):
    model = model.to(device)
    inputs = [torch.rand(1, WINDOW, N_JOINTS, 2, device=device), torch.rand(1, 3, device=device),
              torch.rand(1, N_SEGMENTS, device=device)]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        model(*inputs)
    times = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter_ns()
        model(*inputs)
        sync()
        times.append((time.perf_counter_ns() - t0) / 1000)
    return np.median(times), np.std(times)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args()

    model = load_lifter(args.checkpoint)
    print(f"parameters: {sum(prm.numel() for prm in model.parameters()):,}")
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for device in devices:
        med, std = latency_us(model, device, args.warmup, args.runs)
        print(f"{device.type}: median {med:.0f} µs (std {std:.0f} µs)")


if __name__ == "__main__":
    main()
