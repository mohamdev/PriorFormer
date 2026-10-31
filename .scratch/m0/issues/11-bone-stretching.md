# M0: Bone-stretching test
<!-- label: ready-for-agent -->
<!-- blocked-by: 10 -->

## What to build

A zero-training check of whether given lengths help through the skeleton: take the trained research reference model's output, keep each bone's direction, stretch it to the Segment prior length, and rebuild from the root. Add it as an extra row in the results, to decide whether M2's forward-kinematics design pays off.

## Acceptance criteria

- [ ] With exact lengths, the rebuilt pose has exactly those bone lengths (unit test).
- [ ] An extra results row on the Test clips, compared with the raw output in MPJPE, N-MPJPE and per-joint error.
