`best.pt`, 3 cameras/clip and setting, seed 0, noise on, 533,379 parameters

## centered cameras

Newest-frame MPJPE (mm) per Task:

| Task | n | full priors |
|---|---|---|
| bend | 25 | 24.5 ± 3.0 |
| displace_tilt | 17 | 25.4 ± 3.4 |
| guitar | 2 | 24.7 ± 0.8 |
| kick | 9 | 37.7 ± 5.1 |
| squat | 2 | 43.3 ± 10.2 |
| punch | 3 | 39.5 ± 10.2 |
| pour | 7 | 33.3 ± 6.9 |
| stairs_climbing | 15 | 31.6 ± 6.2 |
| push_recovery | 53 | 24.0 ± 7.5 |
| tennis | 7 | 36.3 ± 5.8 |
| turn | 54 | 26.2 ± 4.7 |
| walk | 96 | 28.1 ± 7.5 |
| wipe | 17 | 30.9 ± 6.5 |
| **Task average** | | 31.2 ± 6.2 |
| **All test clips** | 800 | 31.9 ± 10.8 |

All metrics, all Test clips (mean ± std over Clips, mm):

| Metric | full priors |
|---|---|
| mpjpe | 31.9 ± 10.8 |
| n_mpjpe | 27.6 ± 9.7 |
| pa_mpjpe | 22.6 ± 8.4 |
| bone_error | 12.4 ± 4.9 |
| mpjpe_all_frames | 28.9 ± 9.7 |

Newest-frame MPJPE per joint (mm):

| Joint | full priors |
|---|---|
| r_shoulder | 25.1 |
| l_shoulder | 26.2 |
| r_elbow | 32.5 |
| l_elbow | 33.0 |
| r_wrist | 46.8 |
| l_wrist | 47.0 |
| r_hip | 9.4 |
| l_hip | 9.3 |
| r_knee | 29.9 |
| l_knee | 29.6 |
| r_ankle | 47.3 |
| l_ankle | 46.3 |

Sliding window warmup: newest-frame MPJPE (mm) by frames of history:

| History | full priors |
|---|---|
| 1 | 51.7 |
| 2 | 36.2 |
| 3 | 30.4 |
| 4 | 28.3 |
| 5 | 27.0 |
| 6 | 26.6 |
| 7 | 26.0 |
| 8 | 25.6 |
| 9 | 25.4 |
| 10 | 25.4 |
| 11 | 25.3 |
| 12 | 25.1 |
| 13 (full) | 31.9 ± 10.8 |

## wild cameras

Newest-frame MPJPE (mm) per Task:

| Task | n | full priors |
|---|---|---|
| bend | 25 | 22.7 ± 3.0 |
| displace_tilt | 17 | 21.9 ± 2.8 |
| guitar | 2 | 27.0 ± 2.6 |
| kick | 9 | 33.9 ± 4.5 |
| squat | 2 | 37.3 ± 2.6 |
| punch | 3 | 38.4 ± 7.0 |
| pour | 7 | 28.5 ± 8.9 |
| stairs_climbing | 15 | 27.9 ± 5.4 |
| push_recovery | 53 | 22.1 ± 5.9 |
| tennis | 7 | 29.9 ± 2.5 |
| turn | 54 | 23.3 ± 5.1 |
| walk | 96 | 25.9 ± 7.3 |
| wipe | 17 | 27.6 ± 4.7 |
| **Task average** | | 28.2 ± 5.3 |
| **All test clips** | 800 | 29.0 ± 10.2 |

All metrics, all Test clips (mean ± std over Clips, mm):

| Metric | full priors |
|---|---|
| mpjpe | 29.0 ± 10.2 |
| n_mpjpe | 25.3 ± 9.0 |
| pa_mpjpe | 20.3 ± 7.8 |
| bone_error | 11.2 ± 4.7 |
| mpjpe_all_frames | 26.2 ± 9.3 |

Newest-frame MPJPE per joint (mm):

| Joint | full priors |
|---|---|
| r_shoulder | 22.1 |
| l_shoulder | 22.8 |
| r_elbow | 28.6 |
| l_elbow | 28.4 |
| r_wrist | 41.9 |
| l_wrist | 41.8 |
| r_hip | 8.8 |
| l_hip | 8.8 |
| r_knee | 28.1 |
| l_knee | 27.9 |
| r_ankle | 44.7 |
| l_ankle | 44.0 |

Sliding window warmup: newest-frame MPJPE (mm) by frames of history:

| History | full priors |
|---|---|
| 1 | 48.1 |
| 2 | 33.8 |
| 3 | 27.6 |
| 4 | 25.5 |
| 5 | 24.5 |
| 6 | 23.8 |
| 7 | 23.4 |
| 8 | 23.0 |
| 9 | 22.9 |
| 10 | 22.8 |
| 11 | 22.5 |
| 12 | 22.5 |
| 13 (full) | 29.0 ± 10.2 |

## Prior perturbation sweep, centered cameras (200 Clips x 3 cameras, full priors)

Newest-frame MPJPE (mm) with the prior scaled by (1 + eps):

| eps | -0.50 | -0.30 | -0.20 | -0.10 | -0.05 | +0.00 | +0.05 | +0.10 | +0.20 | +0.30 | +0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| segments | 100.2 | 43.9 | 30.0 | 33.8 | 30.9 | 31.4 | 35.7 | 36.4 | 37.4 | 43.6 | 52.4 |
| focal | 59.8 | 38.6 | 33.8 | 31.4 | 31.2 | 31.4 | 32.1 | 33.0 | 35.0 | 37.2 | 41.2 |

Bone ratio (predicted / true length of the Segment prior pairs):

| eps | -0.50 | -0.30 | -0.20 | -0.10 | -0.05 | +0.00 | +0.05 | +0.10 | +0.20 | +0.30 | +0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| segments | 0.789 | 0.938 | 1.014 | 1.023 | 1.020 | 1.036 | 1.050 | 1.050 | 1.057 | 1.077 | 1.093 |
| focal | 1.033 | 1.037 | 1.036 | 1.036 | 1.036 | 1.036 | 1.035 | 1.035 | 1.035 | 1.034 | 1.034 |

| | Slope eps > 0 (mm per unit eps) | Slope eps < 0 | Follow eps > 0 | Follow eps < 0 |
|---|---|---|---|---|
| segments | +40.7 ± 1.3 | -96.9 ± 1.5 | 0.12 | 0.40 |
| focal | +19.1 ± 0.5 | -42.7 ± 1.1 | -0.00 | 0.00 |

## Prior perturbation sweep, wild cameras (200 Clips x 3 cameras, full priors)

Newest-frame MPJPE (mm) with the prior scaled by (1 + eps):

| eps | -0.50 | -0.30 | -0.20 | -0.10 | -0.05 | +0.00 | +0.05 | +0.10 | +0.20 | +0.30 | +0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| segments | 117.4 | 44.1 | 27.8 | 31.5 | 28.7 | 28.8 | 33.2 | 33.6 | 34.6 | 41.6 | 50.9 |
| focal | 60.7 | 38.5 | 32.6 | 29.4 | 28.8 | 28.8 | 29.2 | 30.0 | 32.0 | 34.2 | 38.6 |

Bone ratio (predicted / true length of the Segment prior pairs):

| eps | -0.50 | -0.30 | -0.20 | -0.10 | -0.05 | +0.00 | +0.05 | +0.10 | +0.20 | +0.30 | +0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| segments | 0.747 | 0.929 | 1.011 | 1.020 | 1.016 | 1.031 | 1.045 | 1.044 | 1.052 | 1.073 | 1.073 |
| focal | 1.031 | 1.031 | 1.031 | 1.031 | 1.031 | 1.031 | 1.031 | 1.030 | 1.030 | 1.030 | 1.030 |

| | Slope eps > 0 (mm per unit eps) | Slope eps < 0 | Follow eps > 0 | Follow eps < 0 |
|---|---|---|---|---|
| segments | +42.8 ± 1.2 | -124.9 ± 1.4 | 0.10 | 0.45 |
| focal | +18.7 ± 0.7 | -50.2 ± 1.6 | -0.00 | -0.00 |
