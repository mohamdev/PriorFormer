# PriorFormer

A lightweight Transformer that lifts short sequences of 2D joint positions to 3D poses, optionally guided by geometric priors. Trained and evaluated on synthetic projections of AMASS motion.

## Models

**Backbone**:
The model pre-trained for 50 epochs with all priors always provided; the common starting point of the Expert and Versatile models.
_Avoid_: base model, pretrained lifter

**Expert model**:
The Backbone fine-tuned with all priors always provided (paper Table I).
_Avoid_: full model

**Versatile model**:
The Backbone fine-tuned with random Prior masking; the model released as `model.onnx` (paper Table II).
_Avoid_: masked model

## Inputs

**Joint set**:
The 12 joint centers (shoulders, elbows, wrists, hips, knees, ankles), each the barycenter of fixed SMPL-X mesh vertices.
_Avoid_: keypoints, markers

**Keypoint set**:
The 17 COCO 2D points (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) given to the model as input, as an off-the-shelf 2D detector outputs them. Distinct from the Joint set.
_Avoid_: joints (for the 2D input), COCO joints

**Window**:
The 13 consecutive frames the model lifts at once.

**Sliding window warmup**:
The start of a live session, while the Window holds fewer than 13 real frames; the model then sees only the frames received so far.
_Avoid_: warm-up, cold start, padding phase

**Prior**:
Optional per-clip geometric information given to the model: the Camera prior or the Segment prior.
_Avoid_: side information, conditioning

**Camera prior**:
The 3 normalized camera intrinsics (focal length and principal point).
_Avoid_: calibration, K

**Segment prior**:
Six distances between pairs of joints in the Joint set, measured on the first frame of the Window: right forearm, left forearm, shoulder width, right shoulder–right hip, right thigh, left thigh.
_Avoid_: bone lengths, anatomical lengths, segment lengths (when meaning the CSV-pipeline values)

**Masking mode**:
Which Priors are zeroed at inference or training: full priors, camera masked, segments masked, or no priors.
_Avoid_: ablation setting

## Bodies

**Body parametric model**:
A family of human bodies driven by shape and pose parameters: SOMA, SMPL-X, MHR or Anny.
_Avoid_: body model (alone)

**Body shape**:
One parameter vector of a Body parametric model, fixing one person's proportions and bone lengths.
_Avoid_: body identity, betas (outside SMPL-X), subject

## Data

**Clip**:
One AMASS motion converted to the Joint set in its own pelvis frame; the unit of the train/test split.
_Avoid_: sequence, file, motion

**Test clips**:
The 800 Clips never used for training or checkpoint selection, listed in `splits/test_clips.tsv`.
_Avoid_: validation set, held-out set

**Validation clips**:
Whole Clips held out from the training Clips to select checkpoints during training.
_Avoid_: validation windows

## Evaluation

**Task**:
A named motion category (e.g. walk, kick, wipe) used to group evaluation clips for per-task MPJPE.
_Avoid_: activity, action
