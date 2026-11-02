`best.pt`, 3 cameras/clip and setting, seed 0, noise on, 533,379 parameters

## centered cameras

Newest-frame MPJPE (mm) per Task:

| Task | n | full priors | full priors stretched |
|---|---|---|---|
| bend | 25 | 25.1 ± 3.3 | 22.9 ± 3.1 |
| displace_tilt | 17 | 24.5 ± 3.3 | 22.4 ± 3.3 |
| guitar | 2 | 29.1 ± 0.3 | 25.5 ± 0.7 |
| kick | 9 | 39.0 ± 5.6 | 36.5 ± 5.0 |
| squat | 2 | 46.4 ± 10.6 | 44.0 ± 12.1 |
| punch | 3 | 43.1 ± 8.7 | 40.7 ± 9.0 |
| pour | 7 | 33.3 ± 7.7 | 30.7 ± 8.4 |
| stairs_climbing | 15 | 31.6 ± 6.5 | 28.5 ± 5.9 |
| push_recovery | 53 | 25.0 ± 8.1 | 22.9 ± 6.8 |
| tennis | 7 | 36.5 ± 2.4 | 34.4 ± 3.0 |
| turn | 54 | 27.2 ± 5.0 | 24.7 ± 4.7 |
| walk | 96 | 29.0 ± 7.5 | 26.3 ± 7.5 |
| wipe | 17 | 31.8 ± 7.2 | 28.9 ± 6.9 |
| **Task average** | | 32.4 ± 6.7 | 29.9 ± 6.8 |
| **All test clips** | 800 | 32.8 ± 11.0 | 29.8 ± 10.3 |

All metrics, all Test clips (mean ± std over Clips, mm):

| Metric | full priors | full priors stretched |
|---|---|---|
| mpjpe | 32.8 ± 11.0 | 29.8 ± 10.3 |
| n_mpjpe | 28.4 ± 9.9 | 28.5 ± 9.9 |
| pa_mpjpe | 23.1 ± 8.4 | 22.9 ± 8.4 |
| bone_error | 12.9 ± 5.4 | 7.8 ± 3.4 |
| mpjpe_all_frames | 29.6 ± 10.1 | 26.5 ± 9.2 |

Newest-frame MPJPE per joint (mm):

| Joint | full priors | full priors stretched |
|---|---|---|
| r_shoulder | 27.1 | 25.5 |
| l_shoulder | 27.7 | 26.2 |
| r_elbow | 33.7 | 31.7 |
| l_elbow | 33.6 | 31.6 |
| r_wrist | 48.3 | 44.3 |
| l_wrist | 47.0 | 43.7 |
| r_hip | 10.1 | 10.1 |
| l_hip | 10.0 | 10.0 |
| r_knee | 29.8 | 25.0 |
| l_knee | 29.3 | 24.7 |
| r_ankle | 48.6 | 42.8 |
| l_ankle | 48.1 | 42.2 |

Sliding window warmup: newest-frame MPJPE (mm) by frames of history:

| History | full priors | full priors stretched |
|---|---|---|
| 1 | 63.4 | 60.0 |
| 2 | 38.6 | 35.2 |
| 3 | 31.2 | 27.6 |
| 4 | 28.7 | 25.2 |
| 5 | 27.7 | 24.2 |
| 6 | 26.6 | 23.3 |
| 7 | 26.2 | 22.8 |
| 8 | 25.7 | 22.5 |
| 9 | 25.6 | 22.4 |
| 10 | 25.8 | 22.5 |
| 11 | 25.9 | 22.6 |
| 12 | 25.8 | 22.5 |
| 13 (full) | 32.8 ± 11.0 | 29.8 ± 10.3 |

## wild cameras

Newest-frame MPJPE (mm) per Task:

| Task | n | full priors | full priors stretched |
|---|---|---|---|
| bend | 25 | 23.1 ± 4.1 | 21.0 ± 3.9 |
| displace_tilt | 17 | 20.9 ± 2.5 | 19.4 ± 2.5 |
| guitar | 2 | 25.3 ± 2.0 | 22.3 ± 1.8 |
| kick | 9 | 33.8 ± 4.5 | 31.3 ± 4.0 |
| squat | 2 | 38.6 ± 0.5 | 36.4 ± 1.1 |
| punch | 3 | 37.7 ± 7.6 | 35.9 ± 8.2 |
| pour | 7 | 29.1 ± 8.9 | 27.2 ± 9.1 |
| stairs_climbing | 15 | 28.0 ± 5.9 | 25.5 ± 5.4 |
| push_recovery | 53 | 22.0 ± 6.0 | 20.4 ± 5.6 |
| tennis | 7 | 31.3 ± 1.5 | 28.3 ± 2.0 |
| turn | 54 | 23.7 ± 5.3 | 21.5 ± 5.3 |
| walk | 96 | 26.7 ± 7.4 | 24.2 ± 7.4 |
| wipe | 17 | 27.4 ± 4.0 | 24.9 ± 3.5 |
| **Task average** | | 28.3 ± 5.5 | 26.0 ± 5.4 |
| **All test clips** | 800 | 29.3 ± 10.1 | 26.6 ± 9.3 |

All metrics, all Test clips (mean ± std over Clips, mm):

| Metric | full priors | full priors stretched |
|---|---|---|
| mpjpe | 29.3 ± 10.1 | 26.6 ± 9.3 |
| n_mpjpe | 25.4 ± 8.9 | 25.5 ± 8.9 |
| pa_mpjpe | 20.5 ± 7.6 | 20.2 ± 7.6 |
| bone_error | 11.8 ± 5.1 | 7.0 ± 3.3 |
| mpjpe_all_frames | 26.5 ± 9.3 | 23.7 ± 8.3 |

Newest-frame MPJPE per joint (mm):

| Joint | full priors | full priors stretched |
|---|---|---|
| r_shoulder | 24.1 | 22.7 |
| l_shoulder | 24.5 | 23.1 |
| r_elbow | 29.6 | 27.8 |
| l_elbow | 29.0 | 27.2 |
| r_wrist | 42.5 | 38.5 |
| l_wrist | 41.9 | 38.4 |
| r_hip | 9.6 | 9.6 |
| l_hip | 9.3 | 9.3 |
| r_knee | 26.3 | 21.9 |
| l_knee | 26.5 | 22.3 |
| r_ankle | 43.9 | 38.9 |
| l_ankle | 44.1 | 39.2 |

Sliding window warmup: newest-frame MPJPE (mm) by frames of history:

| History | full priors | full priors stretched |
|---|---|---|
| 1 | 51.5 | 48.0 |
| 2 | 35.0 | 31.8 |
| 3 | 28.4 | 25.1 |
| 4 | 26.2 | 22.9 |
| 5 | 25.2 | 22.0 |
| 6 | 24.4 | 21.3 |
| 7 | 24.0 | 20.9 |
| 8 | 23.6 | 20.5 |
| 9 | 23.4 | 20.2 |
| 10 | 23.2 | 20.1 |
| 11 | 23.1 | 20.0 |
| 12 | 22.9 | 19.9 |
| 13 (full) | 29.3 ± 10.1 | 26.6 ± 9.3 |
