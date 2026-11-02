`best_lifter_versatile_128_ddp_epoch12.pt`, v1 protocol (all-frame averaged, v1 cameras, no 2D noise)

## Prior perturbation sweep, v1 cameras (200 Clips x 3 cameras, full priors)

All-frame MPJPE (mm) with the prior scaled by (1 + eps):

| eps | -0.50 | -0.30 | -0.20 | -0.10 | -0.05 | +0.00 | +0.05 | +0.10 | +0.20 | +0.30 | +0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| segments | 44.6 | 44.0 | 43.6 | 43.0 | 41.3 | 39.9 | 43.4 | 55.5 | 123.9 | 259.6 | 414.6 |
| focal | 39.9 | 39.9 | 39.9 | 39.9 | 39.9 | 39.9 | 39.9 | 39.9 | 39.9 | 39.9 | 40.0 |

Bone ratio (predicted / true length of the Segment prior pairs):

| eps | -0.50 | -0.30 | -0.20 | -0.10 | -0.05 | +0.00 | +0.05 | +0.10 | +0.20 | +0.30 | +0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| segments | 1.004 | 1.002 | 1.002 | 1.003 | 1.009 | 1.015 | 1.023 | 1.036 | 0.940 | 0.540 | 0.095 |
| focal | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 | 1.015 |

| | Slope eps > 0 (mm per unit eps) | Slope eps < 0 | Follow eps > 0 | Follow eps < 0 |
|---|---|---|---|---|
| segments | +692.6 ± 4.8 | -12.0 ± 0.7 | -1.57 | 0.03 |
| focal | +0.1 ± 0.1 | -0.1 ± 0.1 | -0.00 | -0.00 |
