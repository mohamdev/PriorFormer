# PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors

Official implementation of the paper:  
**PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors**  
*Mohamed Adjel, Vincent Bonnet*  
Accepted at **IEEE-RAS International Conference on Humanoid Robots (Humanoids 2025)**

---

## Overview
This repository provides code and pretrained models for **PriorFormer**, a lightweight Transformer-based lifter that maps short sequences of **2D human joint positions** to **3D poses** using a single monocular camera.  

Unlike previous approaches, PriorFormer:
- Supports **optional geometric priors** (camera intrinsics and bone segment lengths),
- Operates in both **calibrated and uncalibrated settings**,
- Uses a **masking mechanism** to remain robust when priors are missing,
- Runs in **real time** on CPU and GPU, making it suitable for **edge devices**.

Key features:
- Fast: 380 µs on GPU / 1.8 ms on CPU  
- Lightweight: <0.6M parameters  
- Accurate: ~36 mm MPJPE on the 13 AMASS benchmark tasks (43 mm over all held-out test clips; see Corrections)
- Versatile: Works with or without geometric priors  

---

## Pretrained models

Checkpoints are attached to the [`v1.0-checkpoints` release](https://github.com/mohamdev/PriorFormer/releases/tag/v1.0-checkpoints).

| File | What it is | SHA-256 |
|---|---|---|
| `best_lifter_versatile_128_ddp_epoch12.pt` | Versatile model (epoch 12 of fine-tuning). **Same weights as `model.onnx`.** | `7693508775243d3dcbdd37657102fa913e9f83bcd3303c7c99f40de2f0fcb514` |
| `best_lifter_versatile_128_ddp_epoch8.pt` | Versatile model, epoch 8 of the same run. Very likely the checkpoint the paper's Table II was computed with (see Corrections). | `f6a497f23a07f1243b46a72d79533f200b3b8a6b4c55556460e95e44bb0bf294` |
| `best_lifter_128_ddp_epoch50.pt` | Backbone, trained 50 epochs with all priors. Starting point of the fine-tuned models. | `58c3c295c935a0b00c6c61abea9889ad56fc9631e2a280fd62380e685649e851` |
| `model.onnx` | ONNX export of the epoch-12 versatile model. Inputs: `x2d` (B,13,12,2), `k` (1,3), `seg` (1,6). | `f50519eecd13e6a2dc750fa64eb5ecd9fab02ceda4b1141ac47fe2f33ab7a8a2` |

The `.pt` files are dicts with `model_state_dict`, `epoch` and `best_val`. Note that `best_val` was computed on windows of the *training* clips (see Corrections), so it is not a held-out error.

The expert model of Table I is not available; a reproduction will be added.

## Data and conventions

- **Source data:** AMASS ACCAD, BMLmovi and KIT (SMPL-H `*_poses.npz`). AMASS cannot be redistributed; download it from https://amass.is.tue.mpg.de.
- **Joint set (12, in this order):** 0 R shoulder, 1 L shoulder, 2 R elbow, 3 L elbow, 4 R wrist, 5 L wrist, 6 R hip, 7 L hip, 8 R knee, 9 L knee, 10 R ankle, 11 L ankle.
  Each joint center is the mean of **SMPL-X** mesh vertices listed in [`smpl/amass/vertices_keypoints_corr.csv`](smpl/amass/vertices_keypoints_corr.csv): shoulder = `rshoulder`/`lshoulder`; elbow = `lelbow`+`melbow`; wrist = `lwrist`+`mwrist`; hip = `ASIS`+`PSIS`; knee = `knee`+`mknee`; ankle = `ankle`+`mankle`. Generation: `smpl/smpl_data_parsing/raw_amass_to_3Dposes_npz.py`.
- **3D targets:** pelvis frame centered on the mid-hip, oriented by the hips and shoulders at the clip's first frame (not camera coordinates). Meters.
- **2D input:** pixel coordinates divided by the image width (W=1920, H=1080), not centered.
- **Camera prior `k`:** `[f/2000, (cx-W/2)/W, (cy-H/2)/H]`, with f ~ U(800, 1400) px and principal point within ±3% of the image center.
- **Segment prior `seg`:** six joint-pair distances measured on the first frame of each 13-frame window, divided by 2.0: R forearm (2,4), L forearm (3,5), shoulder width (0,1), R shoulder to R hip (0,6), R thigh (6,8), L thigh (7,9). The six anatomical lengths written by the CSV pipeline (`raw_amass_to_3Dposes_csv.py`) are **not** used by the model.
- **Masking:** a masked prior is zeroed.
- **Cameras:** look-at the mid-hip from a random direction, at distance `R_body * U(1.5, 2.5)` where `R_body` is the clip's largest joint distance from the hip origin.

## Train/test split

The split is random **by clip** over the three subsets (subjects appear on both sides; ACCAD is not held out): **5452 training / 800 test clips**.

- [`splits/train_clips.tsv`](splits/train_clips.tsv), [`splits/test_clips.tsv`](splits/test_clips.tsv): processed file name → AMASS source file. For 371 clips the motion name and length match more than one AMASS file (often identical duplicate recordings); all candidates are listed, separated by `|`.
- [`splits/test_tasks/`](splits/test_tasks): the 13 paper tasks selected from the **test clips** with the filename rules in [`splits/make_task_lists.py`](splits/make_task_lists.py).

The 800 test clips were never used for training or checkpoint selection.

## Corrections (October 2026)

An audit of the code against the paper found the following. The released model and its headline accuracy are unaffected; some descriptions and one conclusion are.

1. **Table II test set not archived.** The clip list behind Table II was lost. Re-evaluating the released model with the original per-task script (`evaluate_models_per_task.py`, one camera per clip, seed 0) on the 13 tasks drawn from the held-out test clips (`splits/test_tasks/`) gives, averaged over task means:

   | | Full priors | Bones masked | Camera masked | No priors |
   |---|---|---|---|---|
   | Re-evaluation (epoch 12, test clips) | 35.4 ± 9.1 | 41.8 ± 15.0 | 40.2 ± 11.7 | 39.6 ± 11.4 |
   | Paper, Table II | 36.4 ± 12.8 | 39.2 ± 12.8 | 38.7 ± 12.0 | 41.2 ± 14.5 |

   On all 800 test clips (all activities, not only the 13 tasks) the full-priors MPJPE is 42.8 mm. A multi-camera protocol with fixed seeds will replace this table.
2. **Effect of the priors is overstated.** The masked modes differ by 1–3 mm, well inside the per-clip spread, and their ordering is not stable. The paper's statements that bone priors help more than camera priors, and that combining both gives the best result, are not supported. Robustness to missing priors does hold.
3. **Checkpoint mismatch.** The evaluation scripts pick the "latest" checkpoint by alphabetical order, so Table II was very likely computed with epoch 8, while epoch 12 was exported to ONNX. Both checkpoints give the same results within 0.1 mm.
4. **In-training validation leak.** The training scripts validate on a random 5% of the 13-frame windows of the *training* clips, so `best_val` and the "36.5 mm / 39.4 mm" quoted in the paper's Discussion are training-set errors.
5. **Description mismatches in the paper:** the split is 80/20, not 90/10; the six segments are the joint pairs above, not forearm/upper arm/bi-acromial/spine/thigh/shank; joints come from SMPL-X vertices; camera distance is relative to body size, not 1.5–3.5 m; the released versatile model was fine-tuned for 12 epochs, not 10.
6. **Two of the six segment priors are not rigid.** The shoulder joint is a single surface vertex that moves with the shoulder girdle, so *shoulder width* and *R shoulder to R hip* vary within a clip (median range 22 mm and 33 mm, 90th percentile 92 mm and 81 mm, over 300 test clips). Forearms, thighs, shanks and hip width vary by 2–16 mm. Because the prior is measured on a single frame, these two values are noisy, which may partly explain the small effect of the segment prior.

A corrected arXiv version will follow.

---

## Paper
If you use this repository, please cite our paper:

```bibtex
@inproceedings{adjel2025priorformer,
  title={PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors},
  author={Adjel, Mohamed and Bonnet, Vincent},
  booktitle={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year={2025}
}
```
