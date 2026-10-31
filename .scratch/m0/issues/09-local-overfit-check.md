# M0: Local small-scale training check for the research reference model
<!-- label: ready-for-agent -->
<!-- blocked-by: 02, 03, 06, 07, 08 -->

## What to build

Before any cloud run, check locally that the research reference model trains correctly with all M0 options on. It should overfit a few batches, and on the same chosen data samples, its training behaviour should be comparable to the current baseline (v1) training. Report the comparison to the maintainer, who then gives instructions for the cloud runs.

## Acceptance criteria

- [ ] Few-batch overfit: the training loss drops close to zero on a fixed small set of Windows.
- [ ] The v1 training and the research reference model training run on the same chosen samples, with loss and MPJPE curves side by side (each option also turned on alone, so a broken option is visible).
- [ ] A short report (curves, final numbers, anomalies) is written, and the maintainer is told the result. No cloud job is started.
