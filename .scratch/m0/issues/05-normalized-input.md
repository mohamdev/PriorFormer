# M0: Camera-normalized input and the research reference model class
<!-- label: ready-for-agent -->
<!-- blocked-by: 01, 04 -->

## What to build

2D input in normalized camera coordinates: (u − cx)/f and (v − cy)/f. When the Camera prior is unknown, default intrinsics are used (f = sqrt(W² + H²), principal point at the image center) and an "intrinsics estimated" flag is set. In training with the flag on, coordinates are normalized with the default f while the projection used the true f. A new model class (the v1 encoder with this input and the flag) trains and evaluates end to end. This is the research reference model; v1 is untouched.

## Acceptance criteria

- [ ] Normalizing the projection of camera-frame points gives X/Z and Y/Z (unit test).
- [ ] With default intrinsics, a 2× error in f scales the normalized coordinates by exactly 0.5× (unit test).
- [ ] The new class runs in every Masking mode, with "camera estimated" replacing zeroed intrinsics.
- [ ] Parameter count stays under ~0.6M; ONNX export of the new class round-trips with random weights.
