# M0: Upright cameras
<!-- label: ready-for-agent -->
<!-- blocked-by: 01 -->

## What to build

Synthetic cameras use gravity as up (the AMASS world vertical, recovered from each Clip's stored rotation), with an optional random roll (default 0, e.g. ±10° for training). Today, up is the first frame's hip axis, so every image is rotated about 90° from an upright camera. The v1 cameras stay available through the protocol setting.

## Acceptance criteria

- [ ] For a standing `t_pose()`, gravity projects downward in the image for every seeded camera with roll 0, and within the roll range otherwise.
- [ ] The camera position constraint (no camera below the body) uses the gravity height.
- [ ] A short training run and the evaluation both run with upright cameras.
- [ ] The v1 setting reproduces the v1 cameras exactly.
