# v2 outputs SOMA joint rotations, with body size fixed outside the network

The phone fits a SOMA body, so the v2 model outputs 6D rotations for about 22 SOMA body joints (pelvis relative to the camera, every other joint relative to its parent), not 3D joint positions like v1. Body size enters as a Body shape: the user's measurements are turned into a SOMA Body shape on the phone, outside the network, and its skeleton is given to the model as a Prior. SOMA forward kinematics then rebuilds a body with exactly those measurements. That is what v2 is for: in v1, the Segment prior changed accuracy by only a few millimetres.

## Considered Options

- Joint positions plus on-device IK: rejected, an extra fitting step on the phone, and IK can't recover twist either.
- The network predicts the Body shape from the measurements: rejected, the measurements would no longer be used exactly, and the head would be larger.

## Consequences

- Twists that COCO-17 doesn't show (forearm, wrist, foot) are guesses from the training data, whatever the rotation representation; report them separately.
- Fingers, toes and face joints stay at the rest pose.
