# PriorFormer v2 plan

End goal: live monocular 3D human pose on phones. A COCO-17 2D detector feeds PriorFormer v2, which outputs SOMA joint rotations, and the phone poses a SOMA body with them. Terms follow [GLOSSARY.md](../GLOSSARY.md). Decisions: [ADR 0001](adr/0001-research-data-now-commercial-retrain-later.md) (research data now, commercial retrain later) and [ADR 0002](adr/0002-soma-rotation-output.md) (SOMA rotation output).

## Rules

- The v1 path stays working: `TransformerLifter`, `load_lifter`, the ONNX test and the v1 evaluation. New behavior goes in new classes or flags; v1 code is not edited in place.
- Every change has unit tests in `tests/` that need no dataset, body-model file or checkpoint (`t_pose()` style).
- Never evaluate on generated motion. Never train on Test clips or Test actors.
- Weights trained on research-only data are never shipped (ADR 0001).

## What the code does today (verified)

| | Fact | Where |
|---|---|---|
| F1 | Clips are processed without betas by default: two template Body shapes (Correction 7). | `scripts/preprocess_amass.py` |
| F2 | Camera up = the first frame's left→right hip axis, so the projected hip line is always vertical: every image is rotated about 90° compared with an upright camera. The Clip's world rotation `R` is saved, so gravity is recoverable. | `camera.py: look_at_R`, `preprocess_amass.py` |
| F3 | 3D targets are in the pelvis frame of the Clip's first frame, which isn't visible from a Window mid-Clip. | `preprocess_amass.py: pelvis_frame_joints` |
| F4 | 2D input is pixels / 1920 and the camera always looks at the mid-hip: the subject is always centered, with one image size. | `camera.py` |
| F5 | Training 2D is noise-free ground-truth projection, with no missing joints. | `data.py: WindowDataset` |
| F6 | Evaluation averages every frame of overlapping Windows, so the middle frames see the future. | `evaluate.py` |
| F7 | Prior effects: camera masked +1.1 mm, segments masked +4.1 mm, both masked +5.3 mm. | README Results |

## Design

**Input.** The Keypoint set (COCO-17), in normalized camera coordinates: x_n = (u − cx)/f, y_n = (v − cy)/f. When f is unknown, use f = sqrt(W² + H²) and c = image center, and set an "intrinsics estimated" flag. This is the keypoint equivalent of CLIFF's box conditioning; it is exact and doesn't depend on resolution. In training with the flag on, normalize with the default f while projecting with the true f.

**Window and Sliding window warmup.** 13 slots, a frame-valid mask (attention mask; empty slots are zeros), and the output is the newest frame. Training hides a random number of the oldest frames. The ONNX input shape stays fixed.

**Output (v2).** 6D rotations for about 22 SOMA body joints: the pelvis relative to the camera, every other joint relative to its parent. Fingers, toes and face stay at rest. Twists that COCO-17 can't show (forearm, wrist, foot) are reported separately.

**Body size (v2).** The user's measurements (or height) are turned into a SOMA Body shape on the phone, outside the network. Its skeleton is the Segment prior given to the model; when it's masked, an average Body shape is used. SOMA forward kinematics then gives exactly the given proportions.

**Losses (v2).** Geodesic rotation loss, plus MPJPE after SOMA forward kinematics in PyTorch, plus optional 2D reprojection in normalized coordinates. If SOMA's forward kinematics isn't differentiable in PyTorch, write a small one for the ~22 joints.

**Cameras.** Up = gravity (AMASS world z, or the BONES-SEED vertical), random roll ±10°, f ~ U(500, 2500) px, principal point jitter ±3%, look-at target offset from the body, random image size (640×480 to 3840×2160).

**2D noise.** Per-keypoint Gaussian jitter, occasional large errors (left/right swaps, jumps), random missing keypoints with a visibility input. Later, tune the noise to the chosen detector's real errors.

## Data

| Use | Source | Body | Licence |
|---|---|---|---|
| M0 reference model, train and test | AMASS, current splits, with real betas | SMPL-X Joint set | research |
| v2 train / validation / test | BONES-SEED, split by actor, each mirrored motion on the same side as its original, ~10% Test actors | SOMA (see open question) | research / qualifying startups |
| v2 cross-dataset test | AMASS Test clips converted to SOMA | SOMA | research |
| Real-image test | EMDB or 3DPW with the COCO detector | — | research |
| Product retrain | Rule-generated key poses + MotionBricks in-betweening (optionally Kimodo-SOMA), with SOMA / MHR / Anny Body shapes | SOMA | NVIDIA Open Model License |

Synthetic COCO-17: fixed SOMA mesh vertices for each keypoint, plus 2D noise.

## Evaluation protocol v2

Per Clip, at least 10 cameras, seeded by (seed, Clip index, camera index), in two settings: **centered** (v1 ranges, upright) and **wild** (the wide ranges above). Headline metric: newest-frame MPJPE, root-relative, camera frame. Also report all-frame MPJPE (for v1 continuity), N-MPJPE, PA-MPJPE, bone-length error, per-joint MPJPE, warmup error for 1–12 frames of history, and rotation error for v2. Masking modes: full priors, camera estimated, segments masked, no priors, segments partial. Report mean ± std over Clips, the mean of Task means, the parameter count and `benchmark.py` latency.

## Milestones (one branch / PR each, with tests and a results .json + .md)

- **M0 — Fixes and protocol.** Gravity-up cameras with roll, wild camera ranges, normalized input with default intrinsics, 2D noise, Sliding window warmup, evaluation protocol v2. Retrain the v1 architecture on AMASS with real betas → **research reference model**. Also run the free test: stretch its output bones to the Segment prior lengths, with no retraining.
- **M1 — SOMA data.** BONES-SEED loader and actor split, SOMA Body shapes, COCO-17 vertex mapping, AMASS Test clips converted to SOMA (with a conversion-error report).
- **M2 — Rotation model.** `LifterV2`, SOMA forward-kinematics losses, the Body shape skeleton as Prior, ONNX export and parity test, latency within ±20% of v1. Data-size curve on 25/50/100% of the training actors.
- **M3 — Real images.** EMDB / 3DPW with the COCO detector, comparison against GEM-X, latency on a phone.
- **Product phase.** Rule-generated key poses (joint limits, no self-collision, balance) + MotionBricks, then retrain the validated architecture on commercial-clean data only.

## Open

- BONES-SEED Body shapes: the docs describe BVH skeletons and rotations only. If per-actor SOMA shape coefficients aren't included, fit a SOMA Body shape to each actor's proportional skeleton, and add random Body shapes. Check this when the data is downloaded.
- Gravity direction as an extra output (later).
- Model size and Window length, after the M2 data-size curve.

## Hypotheses

- H1: The M0 fixes worsen the synthetic score but improve real-image error.
- H2: Normalized input mostly helps in the wild camera setting.
- H3: With varied Body shapes, the Segment prior closes the MPJPE − N-MPJPE gap; full priors vs segments masked differs clearly.
- H4: Rotation output plus a Body shape given from measurements beats joint regression in MPJPE on unseen actors.
