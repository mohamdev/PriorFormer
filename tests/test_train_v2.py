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


def test_the_versatile_model_trains_from_scratch_without_a_backbone(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "scratch", "versatile", 1)
    ckpt = torch.load(tmp_path / "scratch" / "last.pt", weights_only=False)
    assert ckpt["config"]["init"] is None and ckpt["config"]["lr"] == 1e-3
    assert set(ckpt["val_mpjpe"]) == {"full_priors", "camera_estimated", "segments_masked", "no_priors"}
    assert (tmp_path / "scratch" / "DONE").exists()


GEN2 = ["--lr", "3e-4", "--schedule", "onecycle", "--ema", "0.9", "--seg-jitter", "0.2", "0.05"]


def test_a_generation_2_run_cut_mid_epoch_resumes_to_the_same_weights_and_ema(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "straight", "versatile", 2, GEN2)
    for _ in range(12):
        run(clips_dir, tmp_path / "cut", "versatile", 2, [*GEN2, "--time-limit", "0"])
    assert (tmp_path / "cut" / "DONE").exists()
    a = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)
    b = torch.load(tmp_path / "cut" / "last.pt", weights_only=False)
    for key in ["model_state_dict", "ema_state_dict"]:
        for k in a[key]:
            torch.testing.assert_close(a[key][k], b[key][k], atol=1e-4, rtol=1e-4)
    best = torch.load(tmp_path / "straight" / "best.pt", weights_only=False)
    ema_of_best_epoch = torch.load(tmp_path / "straight" / "best.pt", weights_only=False)["ema_state_dict"]
    for k in best["model_state_dict"]:
        torch.testing.assert_close(best["model_state_dict"][k], ema_of_best_epoch[k])
    log = json.loads((tmp_path / "straight" / "log.jsonl").read_text().splitlines()[-1])
    assert "val_mpjpe_raw" in log


def test_the_onecycle_schedule_warms_up_to_the_peak_then_decays(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "lr", "versatile", 20, ["--lr", "3e-4", "--schedule", "onecycle"])
    ckpt = torch.load(tmp_path / "lr" / "last.pt", weights_only=False)
    group = ckpt["optimizer"]["param_groups"][0]
    assert ckpt["scheduler"]["total_steps"] == 20 * 3 and group["max_lr"] == 3e-4
    assert group["initial_lr"] == 3e-4 / 25          # warmup starts 25x below the peak
    assert ckpt["scheduler"]["_last_lr"][0] < 3e-4 / 1000   # annealed to ~0 at the end


def test_segment_prior_jitter_leaves_half_the_windows_exact_and_stays_in_bounds():
    torch.manual_seed(0)
    seg = torch.ones(4000, 6)
    out = train_v2.jitter_segments(seg, 0.2, 0.05)
    exact = (out == 1).all(dim=1)
    assert 0.45 < exact.float().mean() < 0.55
    ratio = out[~exact]
    assert ratio.min() >= 0.8 * 0.95 - 1e-6 and ratio.max() <= 1.2 * 1.05 + 1e-6
    assert ratio.max() > 1.15 and ratio.min() < 0.85          # the range is used on both sides


def test_training_reads_the_clips_of_several_folders(clips_dir, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    for i in range(3):
        np.savez(extra / f"extra_{i}.npz", joints_3d=moving_clip(30, 10 + i), R=np.eye(3), t=np.zeros(3))
    train_v2.main(["--model", "backbone", "--data", str(clips_dir), str(extra), "--out", str(tmp_path / "both"),
                   "--epochs", "1", "--batch", "8", "--max-steps", "1", "--workers", "0", "--val-fraction", "0.34"])
    config = json.loads((tmp_path / "both" / "config.json").read_text())
    assert config["train_clips"] + config["val_clips"] == 9


def test_a_larger_size_trains_and_reloads_with_its_own_architecture(clips_dir, tmp_path):
    from priorformer.model import SIZES, load_reference

    run(clips_dir, tmp_path / "medium", "versatile", 1, ["--size", "medium"])
    model = load_reference(tmp_path / "medium" / "best.pt")
    small = sum(p.numel() for p in load_reference(_train_base(clips_dir, tmp_path)).parameters())
    assert model.d_model == SIZES["medium"]["d_model"]
    assert sum(p.numel() for p in model.parameters()) > 3 * small


def _train_base(clips_dir, tmp_path):
    run(clips_dir, tmp_path / "base", "versatile", 1)
    return tmp_path / "base" / "best.pt"
