# M0: Camera-frame targets and evaluation protocol v2 metrics

<!-- label: ready-for-agent -->
<!-- blocked-by: 01 -->

## What to build

3D targets become root-relative (mid-hip origin) in the camera frame of each Window, so they depend only on what the Window shows, not on the Clip's first frame. The evaluation reports the protocol v2 metrics from `docs/plan-v2.md`.

## Acceptance criteria

- [ ] Projecting the camera-frame target plus the root's camera position with K gives back the 2D input (unit test).
- [ ] The evaluation reports newest-frame MPJPE (headline), all-frame MPJPE, N-MPJPE, PA-MPJPE, bone-length error and per-joint MPJPE, as mean ± std over Clips, mean of Task means, and all-Clip mean.
- [ ] N-MPJPE and PA-MPJPE are unit-tested on known cases (a scaled pose gives N-MPJPE 0; a rotated and scaled pose gives PA-MPJPE 0).
- [ ] The v1 evaluation and its numbers are unchanged.
