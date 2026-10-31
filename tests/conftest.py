import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path(os.environ.get("PRIORFORMER_CHECKPOINTS", REPO / "checkpoints"))


def require(path):
    if path is None or not Path(path).exists():
        pytest.skip(f"missing {path}")
    return Path(path)


@pytest.fixture
def versatile_ckpt():
    return require(CHECKPOINTS / "best_lifter_versatile_128_ddp_epoch12.pt")


@pytest.fixture
def onnx_model():
    return require(CHECKPOINTS / "model.onnx")
