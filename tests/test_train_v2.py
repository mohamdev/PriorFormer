import json

import numpy as np
import pytest
import torch

from priorformer import evaluate_v2, train_v2

from test_protocol import moving_clip


@pytest.fixture
def clips_dir(tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    for i in range(6):
        np.savez(folder / f"clip_{i}.npz", joints_3d=moving_clip(30 + 3 * i, i), R=np.eye(3), t=np.zeros(3))
    return folder


def run(clips_dir, out, model="backbone", epochs=2, extra=()):
    train_v2.main(["--model", model, "--data", str(clips_dir), "--out", str(out), "--epochs", str(epochs),
                   "--batch", "8", "--max-steps", "3", "--workers", "0", "--val-fraction", "0.34", *extra])


def test_backbone_then_versatile_train_and_evaluate_end_to_end(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "backbone")
    logs = [json.loads(line) for line in (tmp_path / "backbone" / "log.jsonl").read_text().splitlines()]
    assert [log["epoch"] for log in logs] == [1, 2]
    assert (tmp_path / "backbone" / "best.pt").exists()

    run(clips_dir, tmp_path / "versatile", "versatile", 1, ["--init", str(tmp_path / "backbone" / "best.pt")])
    val = json.loads((tmp_path / "versatile" / "log.jsonl").read_text().splitlines()[-1])["val_mpjpe"]
    assert set(val) == {"full_priors", "camera_estimated", "segments_masked", "no_priors"}

    split = tmp_path / "test.tsv"
    split.write_text("processed_file\tamass_source\n" + "".join(f"clip_{i}.npz\tx\n" for i in range(2)))
    out = tmp_path / "results" / "reference.json"
    import sys
    argv, sys.argv = sys.argv, ["evaluate_v2", "--checkpoint", str(tmp_path / "versatile" / "best.pt"),
                                "--clips", str(clips_dir), "--split", str(split), "--cameras", "2", "--out", str(out)]
    try:
        evaluate_v2.main()
    finally:
        sys.argv = argv
    res = json.loads(out.read_text())
    assert set(res["summary"]) == {"centered", "wild"}
    assert res["parameters"] < 600_000
    assert "Sliding window warmup" in out.with_suffix(".md").read_text()


def test_an_interrupted_run_resumes_to_the_same_result(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "straight", epochs=3)
    for _ in range(4):  # one epoch per run, then a no-op once finished
        run(clips_dir, tmp_path / "chunks", epochs=3, extra=["--epochs-per-run", "1"])
    epochs = [json.loads(line)["epoch"] for line in (tmp_path / "chunks" / "log.jsonl").read_text().splitlines()]
    assert epochs == [1, 2, 3]
    a = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)["model_state_dict"]
    b = torch.load(tmp_path / "chunks" / "last.pt", weights_only=False)["model_state_dict"]
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=1e-4, rtol=1e-4)  # GPU kernels are not bit-deterministic


def test_a_run_cut_mid_epoch_by_its_time_limit_resumes_to_the_same_result(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "straight", epochs=2)
    assert (tmp_path / "straight" / "DONE").exists()
    for _ in range(12):  # --time-limit 0 saves and stops after every step; 6 steps, 2 epoch ends, then no-ops
        run(clips_dir, tmp_path / "cut", epochs=2, extra=["--time-limit", "0"])
    assert (tmp_path / "cut" / "DONE").exists()
    logs = [json.loads(line) for line in (tmp_path / "cut" / "log.jsonl").read_text().splitlines()]
    straight = [json.loads(line) for line in (tmp_path / "straight" / "log.jsonl").read_text().splitlines()]
    assert [log["epoch"] for log in logs] == [1, 2]
    for a, b in zip(straight, logs):
        assert abs(a["train_mpjpe"] - b["train_mpjpe"]) < 1e-4
    a = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)["model_state_dict"]
    b = torch.load(tmp_path / "cut" / "last.pt", weights_only=False)["model_state_dict"]
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=1e-4, rtol=1e-4)
