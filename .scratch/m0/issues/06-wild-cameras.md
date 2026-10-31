# M0: Wild camera setting
<!-- label: ready-for-agent -->
<!-- blocked-by: 03, 05 -->

## What to build

A "wild" camera setting next to "centered" (v1 ranges, upright): f from 500 to 2500 px, principal point jitter ±3%, look-at target offset from the body so the root is no longer at the image center, random image size (640×480 to 3840×2160) and a small roll. Training can use it, and the evaluation reports both settings.

## Acceptance criteria

- [ ] With wild cameras, the projected mid-hip is spread across the image rather than at the principal point, and the body stays in the image (unit test over seeded cameras).
- [ ] Evaluation outputs separate results for centered and wild cameras, seeded by (seed, Clip index, camera index).
