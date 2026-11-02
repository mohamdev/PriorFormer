# PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors

Official implementation of the paper
**PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors**,
*Mohamed Adjel, Vincent Bonnet*, IEEE-RAS International Conference on Humanoid Robots (Humanoids 2025).

PriorFormer is a small Transformer (0.53M parameters) that lifts a 13-frame window of 12 2D joint positions from one camera to 3D. It optionally takes two **priors**, the camera intrinsics and six segment lengths, and was trained with random prior masking so that it keeps working when either prior is missing.

> **Please read [Corrections](#corrections-october-2026)** before comparing with the paper: the headline accuracy and the average effect of the priors reproduce, but Table II's per-task numbers are noisy and several details are misdescribed.

## Install

```bash
git clone https://github.com/mohamdev/PriorFormer.git && cd PriorFormer
pip install -e ".[dev]"       # torch, numpy, tqdm (+ onnx, onnxruntime, pytest)
```

Download the checkpoints from the [`v1.0-checkpoints` release](https://github.com/mohamdev/PriorFormer/releases/tag/v1.0-checkpoints) into `checkpoints/`.

| File | What it is | SHA-256 |
|---|---|---|
| `best_lifter_versatile_128_ddp_epoch12.pt` | **Versatile model**, the released model (same weights as `model.onnx`) | `7693508775243d3dcbdd37657102fa913e9f83bcd3303c7c99f40de2f0fcb514` |
| `best_lifter_versatile_128_ddp_epoch8.pt` | Versatile model, epoch 8 of the same run; very likely the one behind the paper's Table II | `f6a497f23a07f1243b46a72d79533f200b3b8a6b4c55556460e95e44bb0bf294` |
| `best_lifter_128_ddp_epoch50.pt` | **Backbone**: 50 epochs with all priors; starting point of the fine-tuned models | `58c3c295c935a0b00c6c61abea9889ad56fc9631e2a280fd62380e685649e851` |
| `model.onnx` | ONNX export of the versatile model; inputs `x2d` (1,13,12,2), `k` (1,3), `seg` (1,6) | `f50519eecd13e6a2dc750fa64eb5ecd9fab02ceda4b1141ac47fe2f33ab7a8a2` |

The original Expert model (Table I) was lost; a reproduction is being trained (see [Results](#results)).

## Use

```python
import torch
from priorformer.model import load_lifter

model = load_lifter("checkpoints/best_lifter_versatile_128_ddp_epoch12.pt")
x2d = torch.rand(1, 13, 12, 2)   # 2D joints in pixels / image width (see Conventions)
k   = torch.zeros(1, 3)          # camera prior, or zeros if unknown
seg = torch.zeros(1, 6)          # segment prior, or zeros if unknown
pose3d = model(x2d, k, seg)      # (1, 13, 12, 3), metres, pelvis frame
```

`python -m priorformer.export_onnx` exports to ONNX (dynamic batch); `python -m priorformer.benchmark` measures latency.

## Conventions

The model only gives sensible results if its inputs follow these conventions exactly (all in [`priorformer/`](priorformer)).

- **Joint set** (in this order): 0 R shoulder, 1 L shoulder, 2 R elbow, 3 L elbow, 4 R wrist, 5 L wrist, 6 R hip, 7 L hip, 8 R knee, 9 L knee, 10 R ankle, 11 L ankle. Each joint center is the mean of **SMPL-X** mesh vertices from [`smpl/amass/vertices_keypoints_corr.csv`](smpl/amass/vertices_keypoints_corr.csv): shoulder = `rshoulder`/`lshoulder`; elbow = `lelbow`+`melbow`; wrist = `lwrist`+`mwrist`; hip = `ASIS`+`PSIS`; knee = `knee`+`mknee`; ankle = `ankle`+`mankle`.
- **3D output:** metres, in the clip's pelvis frame: origin at the mid-hip of each frame, axes fixed by the hips (y, left→right) and shoulders (z) at the clip's first frame. Not camera coordinates.
- **2D input `x2d`:** pixel coordinates divided by the image width (W=1920, H=1080), not centered.
- **Camera prior `k`:** `[f/2000, (cx−W/2)/W, (cy−H/2)/H]`.
- **Segment prior `seg`:** distances of six joint pairs on the first frame of the window, divided by 2.0: R forearm (2,4), L forearm (3,5), shoulder width (0,1), R shoulder–R hip (0,6), R thigh (6,8), L thigh (7,9).
- **Masking:** a missing prior is passed as zeros.
- **Synthetic cameras** (training and evaluation): look at the pelvis origin from a random direction above the lowest joint, at distance `R_body × U(1.5, 2.5)` (`R_body`: the clip's largest joint distance from the origin), f ~ U(800, 1400) px, principal point within ±3% of the image center.

## Data

Training and evaluation use AMASS ACCAD, BMLmovi and KIT (SMPL-H `*_poses.npz`, from https://amass.is.tue.mpg.de; it cannot be redistributed) and the SMPL-X body models.

```bash
pip install -e ".[preprocess]"   # human_body_prior
python scripts/preprocess_amass.py --amass <AMASS root> --smplx <SMPL-X models> --out data
```

This writes `data/train/` and `data/test/` using the split in [`splits/`](splits): **5452 training and 800 test clips**, split randomly by clip (subjects and all three subsets appear on both sides; ACCAD is not held out). The `.tsv` files map each processed clip to its AMASS file; for 371 clips the motion name and length match several AMASS files (often duplicate recordings) and all candidates are listed. By default, preprocessing reproduces the paper's data, which ignored body shape (see Corrections 7); add `--with-shape` for new models.

**Full AMASS (more training data).** The other AMASS subsets are split by [`scripts/make_amass_full_split.py`](scripts/make_amass_full_split.py), and the v1 split and the 800 Test clips stay unchanged:

| Split file | Clips | Frames | Content |
|---|---|---|---|
| `train_clips.tsv` (v1) | 5,452 | 4.1M | ACCAD, BMLmovi, KIT |
| `train_extra_clips.tsv` | 7,998 | 14.0M (33.8 h) | BMLrub, BMLhandball, CMU, DanceDB, GRAB, HDM05, HUMAN4D, PosePrior, MoSh, SFU, SOMA, TCD hands, 63 KIT clips missing from v1 |
| `unseen_clips.tsv` | 205 | 0.3M (1.1 h) | Transitions, SSM (AMASS's own test subsets), TotalCapture, HumanEva (with real video, for a later benchmark). Never trained on. |

None of the new clips is within 30 mm of a same-length Test clip or Unseen clip (criterion of Correction 8). Clip frame rates are kept as recorded: 120 Hz for most, 100 Hz for KIT, 60 Hz for 495 CMU clips. Each processed clip stores its `fps`.

```bash
python scripts/preprocess_amass.py --amass <AMASS root> --smplx <SMPL-X models> --out data --with-shape --splits train_extra unseen
python -m priorformer.train_v2 --model versatile --data data/train data/train_extra --out runs/reference_full_amass
python -m priorformer.evaluate_v2 --checkpoint runs/reference_full_amass/best.pt --clips data/unseen --split splits/unseen_clips.tsv --out results/unseen.json
```

## Train

```bash
python -m priorformer.train --model backbone  --data data/train --amp --out runs/backbone
python -m priorformer.train --model expert    --data data/train --init checkpoints/best_lifter_128_ddp_epoch50.pt --amp --out runs/expert
python -m priorformer.train --model versatile --data data/train --init checkpoints/best_lifter_128_ddp_epoch50.pt --amp --out runs/versatile
```

Each step projects random windows through a new random camera. The loss is MPJPE + bone-length error on 12 skeleton edges + 0.01 × first-frame reprojection error (pixels / W). Fine-tuning only updates the Transformer encoder and the regressor; the versatile model draws one of the four masking modes per batch; defaults are 50 epochs at lr 1e-3 for the backbone and 10 epochs at lr 1e-4 for fine-tuning, as recorded in the released checkpoints. Checkpoints are selected on **whole validation clips** held out from the training clips (5%, listed in `runs/<name>/val_clips.txt`); the test clips are never used.

## Evaluate

```bash
python -m priorformer.evaluate --checkpoint checkpoints/best_lifter_versatile_128_ddp_epoch12.pt \
    --clips data/test --out results/versatile_epoch12.json
```

Protocol: every test clip is seen through 10 random cameras, camera *j* of clip *i* (index in `splits/test_clips.tsv`) drawn from an RNG seeded with `(0, i, j)`. Every window is lifted and overlapping predictions averaged; MPJPE is averaged over frames and joints, then cameras, then clips. Tasks are selected by filename (rules in [`priorformer/splits.py`](priorformer/splits.py), lists in [`splits/test_tasks/`](splits/test_tasks)); as in the paper, the task average is the mean ± std of the 13 task means. Run `pytest` to check the protocol and conventions.

## Results

MPJPE (mm) on the **test clips**, 10 cameras per clip ([`results/`](results) has per-task tables and per-clip errors).

**Versatile model** (`best_lifter_versatile_128_ddp_epoch12.pt`, seed 0):

| Task | n | full priors | camera masked | segments masked | no priors |
|---|---|---|---|---|---|
| bend | 25 | 35.6 ± 6.9 | 38.1 ± 7.9 | 39.8 ± 7.7 | 42.4 ± 8.8 |
| displace_tilt | 17 | 22.4 ± 4.8 | 22.8 ± 4.5 | 23.9 ± 4.8 | 24.3 ± 4.4 |
| guitar | 2 | 19.6 ± 0.5 | 21.7 ± 0.1 | 21.5 ± 0.3 | 23.2 ± 0.5 |
| kick | 9 | 45.7 ± 9.6 | 46.7 ± 9.5 | 48.8 ± 9.3 | 49.8 ± 9.3 |
| squat | 2 | 37.0 ± 1.8 | 39.5 ± 2.7 | 48.6 ± 3.2 | 48.5 ± 3.5 |
| punch | 3 | 54.6 ± 18.8 | 56.5 ± 18.9 | 58.7 ± 18.4 | 61.0 ± 18.5 |
| pour | 7 | 31.4 ± 2.3 | 33.5 ± 3.2 | 36.1 ± 1.8 | 38.2 ± 1.5 |
| stairs_climbing | 15 | 33.8 ± 8.4 | 34.8 ± 8.7 | 37.3 ± 8.8 | 38.5 ± 9.2 |
| push_recovery | 53 | 26.2 ± 7.4 | 27.6 ± 7.3 | 29.2 ± 7.7 | 30.8 ± 7.3 |
| tennis | 7 | 43.4 ± 11.1 | 45.0 ± 11.2 | 46.1 ± 10.1 | 47.9 ± 10.9 |
| turn | 54 | 39.0 ± 22.9 | 40.3 ± 23.1 | 42.7 ± 24.4 | 44.2 ± 24.6 |
| walk | 96 | 34.3 ± 15.9 | 35.2 ± 16.0 | 37.9 ± 16.0 | 39.0 ± 16.4 |
| wipe | 17 | 27.1 ± 9.2 | 28.1 ± 9.0 | 30.3 ± 10.5 | 31.5 ± 10.4 |
| **Task average** | | 34.6 ± 9.4 | 36.1 ± 9.5 | 38.5 ± 10.2 | 40.0 ± 10.3 |
| **All test clips** | 800 | 39.3 ± 29.9 | 40.4 ± 30.1 | 43.4 ± 31.0 | 44.6 ± 31.2 |

The paper's Table II averages were 36.4 / 38.7 / 39.2 / 41.2 mm. Paired over the 800 test clips, masking the camera prior adds 1.1 mm (95% bootstrap CI 1.0–1.2), masking the segment prior 4.1 mm (3.8–4.5) and masking both 5.3 mm (5.0–5.8); full priors beat no priors on 97% of clips. Excluding the 44 test clips that duplicate or nearly duplicate a training clip (Corrections 8) changes the task average by at most 0.1 mm. The priors help consistently but modestly, and the model stays accurate without them.

**Expert model** (Table I): the reproduction is being retrained with whole-clip validation; results will be added here.

## v1.1: research reference model

The [`v1.1-checkpoints` release](https://github.com/mohamdev/PriorFormer/releases/tag/v1.1-checkpoints) has one Versatile model (PyTorch and ONNX), for research use only. It keeps the v1 architecture and fixes the data and the protocol ([plan](docs/plan-v2.md), milestone M0):
- real AMASS body shapes (Corrections 7)
- upright cameras with roll
- 2D input in normalized camera coordinates, with default intrinsics and a flag when the camera is unknown
- root-relative 3D output in the camera frame
- wide-range cameras, noisy 2D keypoints and Sliding window warmup in training
- Segment prior noise in training (±20% global × ±5% per segment, on half the Windows)
- OneCycle learning rate with EMA weights

Load it with `load_reference` and evaluate it with `python -m priorformer.evaluate_v2`. Training is in `priorformer/train_v2.py`.

The evaluation (protocol v2) differs from the v1 evaluation above:
- every Test clip is seen through random cameras in two Camera settings, **centered** (v1 ranges) and **wild** (focal length 0.35–1.5 × the image diagonal, off-center subject, any image size, roll)
- 2D keypoints are noisy
- each frame is predicted as the newest frame of its Window, as in a live session

The numbers are therefore not comparable with the v1 table.

Newest-frame MPJPE (mm), 800 Test clips, 3 cameras per clip and setting ([report](results/m0_reference_local/report.md)):

| Camera setting | full priors | camera estimated | segments masked | no priors |
|---|---|---|---|---|
| centered | 32.8 | 37.5 | 37.4 | 41.9 |
| wild | 29.3 | 35.1 | 34.7 | 40.3 |

The priors matter more than in v1: masking both costs 9–11 mm. A Segment prior that is 30% too long now adds 8 mm, against 220 mm for v1. Within the training noise range the model treats the prior as a hint, not a rule, but errors of 50% still hurt. 533k parameters; one Window takes 2.2 ms on CPU (i7-11800H) and 1.1 ms on GPU (RTX 3050 Ti laptop), against 1.7 / 0.8 ms for v1.

## Corrections (October 2026)

An audit of the code against the paper found the following. The released model, the paper's average accuracy and the average effect of the priors reproduce; the per-task numbers and several descriptions do not.

1. **Table II test set not archived.** The clips behind Table II were lost. Results above use the documented protocol on the test clips instead.
2. **Per-task results are noisy.** Table II used one random camera per clip, and a clip's error varies by about 5 mm (median standard deviation) from one camera to another, as much as the effect of the priors. Per-task inversions in Table II (e.g. masking the segment prior improving *bend*, *turn* or *walk*) do not reproduce with 10 cameras. The averages and their ordering do (see [Results](#results)): segment priors help more than camera priors, and both together help most.
3. **Checkpoint mismatch.** The old evaluation scripts picked the "latest" checkpoint by alphabetical order, so Table II was very likely computed with epoch 8 while epoch 12 was exported to ONNX. The two agree within 0.1 mm.
4. **In-training validation leak.** The original training scripts validated on a random 5% of the windows of the *training* clips, so the `best_val` stored in the checkpoints and the "36.5 mm / 39.4 mm" in the paper's Discussion are training-set errors. The new training code validates on whole held-out clips.
5. **Description mismatches:** the split is 80/20, not 90/10; joints come from SMPL-X vertices; camera distance scales with body size and is not 1.5–3.5 m; the segments are the joint pairs above, not forearm/upper arm/bi-acromial/spine/thigh/shank; the bone-length loss uses 12 edges on all frames, not six pairs at the central frame; the versatile model's cosine schedule was set for 20 epochs and the best checkpoint (epoch 12) kept, not 10 epochs.
6. **Two segment priors are not rigid.** The shoulder joint is a single surface vertex that moves with the shoulder girdle, so *shoulder width* and *R shoulder–R hip* vary within a clip (median range 22 and 33 mm, 90th percentile 92 and 81 mm, over 300 test clips), against 2–16 mm for the limb segments.
7. **All bodies have a template shape.** Preprocessing did not apply the AMASS shape coefficients (betas), so every subject has the SMPL-X template body of their gender: limb lengths take essentially two values (e.g. R forearm 242 or 267 mm). The data therefore has many subjects' motions but only two body shapes, and the segment prior carries little more than the template's gender. The segment prior's benefit may therefore come largely from telling the two templates apart, and generalization to other body proportions is untested.
8. **Duplicates across the split.** AMASS contains duplicate recordings, and the random split put some of them on both sides: 8 test clips are exact copies of a training clip under another name, and 36 more are within 30 mm (mean joint distance) of a same-length training clip, mostly with the same motion name (likely duplicate trials). They are listed in [`splits/test_clips_near_train.tsv`](splits/test_clips_near_train.tsv); excluding them changes the results by less than 1 mm.

A corrected arXiv version will follow.

## Repository

```
priorformer/   model, skeleton & priors, camera, data, losses, splits, train, evaluate, export_onnx, benchmark
scripts/       preprocess_amass.py
splits/        train/test clip lists and test task lists
smpl/amass/    vertices_keypoints_corr.csv
results/       evaluation outputs
tests/         protocol and convention tests (pytest)
GLOSSARY.md    project vocabulary
```

The original research scripts are in the git history (tag `v1.0-checkpoints`).

## Citation

```bibtex
@inproceedings{adjel2025priorformer,
  title={PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors},
  author={Adjel, Mohamed and Bonnet, Vincent},
  booktitle={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year={2025}
}
```
