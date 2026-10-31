# M0: Prefactor: one protocol setting for data and evaluation
<!-- label: ready-for-agent -->
<!-- blocked-by: -->

## What to build

The Window dataset and the evaluation take one protocol setting that describes the cameras, the 2D input format and the target frame. Its default reproduces the v1 protocol exactly. Later M0 tickets add options to it instead of editing v1 code. See `docs/plan-v2.md` (M0) and ADR 0001.

## Acceptance criteria

- [ ] With the default setting, the dataset items and the v1 evaluation on a small fixed fixture are identical to before the change (same seeds, same numbers).
- [ ] `TransformerLifter`, `load_lifter`, the ONNX test and the v1 evaluation still run unchanged.
- [ ] Unit tests need no dataset, body-model file or checkpoint.
