# M0: Sliding window warmup
<!-- label: ready-for-agent -->
<!-- blocked-by: 04, 05 -->

## What to build

The research reference model handles Sliding window warmup: always 13 slots, a frame-valid mask (attention mask, empty slots zero), and the output that counts is the newest frame. Training randomly hides a number of the oldest frames. The evaluation reports error by frames of history (1–12, and full).

## Acceptance criteria

- [ ] Hidden frames have no effect on the output (unit test: changing a masked slot's values doesn't change the prediction).
- [ ] Evaluation reports newest-frame error for each history length.
- [ ] ONNX export keeps a fixed input shape, with the mask as an input; ONNX and PyTorch outputs match (parity test).
