# M0: 2D noise and missing joints
<!-- label: ready-for-agent -->
<!-- blocked-by: 05 -->

## What to build

Training 2D input looks like a detector's output: per-joint Gaussian jitter, occasional large errors (left/right swaps, jumps), and random missing joints, with a per-joint visibility input to the research reference model. The evaluation can run with noise on or off.

## Acceptance criteria

- [ ] With noise off, the input equals the exact projection.
- [ ] Noise rates and magnitudes are configurable and seeded; missing joints are flagged as not visible and don't leak their true position (unit test).
- [ ] Training and evaluation run with noise on.
