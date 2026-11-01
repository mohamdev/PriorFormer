# PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors

Official implementation of the paper
**PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors**,
*Mohamed Adjel, Vincent Bonnet*, IEEE-RAS International Conference on Humanoid Robots (Humanoids 2025).

PriorFormer is a small Transformer (0.53M parameters) that lifts a 13-frame window of 12 2D joint positions from one camera to 3D. It optionally takes two **priors**, the camera intrinsics and six segment lengths, and was trained with random prior masking so that it keeps working when either prior is missing.

> **Please read [Corrections](#corrections-october-2026)** before comparing with the paper: the headline accuracy and the average effect of the priors reproduce, but Table II's per-task numbers are noisy, the versatile model does not beat the expert, and several details are misdescribed.

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
| `best_lifter_expert_retrained_epoch1.pt` | **Expert model**, retrained in 2026 (the original was lost): backbone fine-tuned with all priors, best epoch on whole validation clips | `4e178888442403a0064afd2d8bcf2b08d25ce11949a910e513e11f3c13f08625` |
| `model.onnx` | ONNX export of the versatile model; inputs `x2d` (1,13,12,2), `k` (1,3), `seg` (1,6) | `f50519eecd13e6a2dc750fa64eb5ecd9fab02ceda4b1141ac47fe2f33ab7a8a2` |

The original Expert model (Table I) was lost; the retrained one reproduces it (see [Results](#results)).

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

**Expert model and backbone** (Table I), average of task means (full tables in [`results/`](results)):

| Model | full priors | camera masked | segments masked | no priors |
|---|---|---|---|---|
| Paper, Table I (original expert) | 39.6 | 209.5 | 420.1 | 419.0 |
| Expert, retrained (`best_lifter_expert_retrained_epoch1.pt`) | 33.0 ± 9.5 | 208.9 ± 21.4 | 429.5 ± 20.4 | 425.4 ± 21.0 |
| Backbone (`best_lifter_128_ddp_epoch50.pt`) | 32.4 ± 9.4 | 210.1 ± 21.4 | 420.6 ± 23.1 | 419.5 ± 22.5 |
| Versatile (`best_lifter_versatile_128_ddp_epoch12.pt`) | 34.6 ± 9.4 | 36.1 ± 9.5 | 38.5 ± 10.2 | 40.0 ± 10.3 |

Models never trained with masking collapse when a prior is zeroed, as in the paper. With all priors, they are slightly *more* accurate than the versatile model: paired over the 800 test clips, the versatile model is 1.5 mm worse than the expert (95% CI 1.2–1.7, worse on 84% of clips) and 2.2 mm worse than the backbone (2.0–2.5). Fine-tuning the backbone into the expert brings nothing: validation error was lowest after the first epoch, so the expert is the backbone after one more epoch at lr 1e-4.

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
9. **The versatile model does not outperform the expert.** The paper reports that the versatile model is slightly better than the expert with all priors (36.4 vs 39.6 mm) and that the expert overfits. With 10 cameras, the expert (retrained) is 1.5 mm better than the versatile model and the backbone 2.2 mm better: robustness to missing priors costs about 2 mm when all priors are given. The original expert's 39.6 mm cannot be re-checked because its checkpoint was lost.

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
