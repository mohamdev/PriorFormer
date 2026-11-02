"""Joint set and segment prior definitions (see GLOSSARY.md)."""
import numpy as np

JOINT_NAMES = [
    "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist",
    "r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle",
]
N_JOINTS = len(JOINT_NAMES)

# Joint pairs whose distances form the segment prior, in this order:
# R forearm, L forearm, shoulder width, R shoulder-R hip, R thigh, L thigh.
SEGMENT_PAIRS = [(2, 4), (3, 5), (0, 1), (0, 6), (6, 8), (7, 9)]
N_SEGMENTS = len(SEGMENT_PAIRS)

SEG_NORM = 2.0

# Edges used by the bone-length loss.
SKELETON_EDGES = [
    (0, 2), (2, 4), (1, 3), (3, 5),
    (6, 8), (8, 10), (7, 9), (9, 11),
    (0, 1), (6, 7), (0, 6), (1, 7),
]


def segment_prior(joints):
    """Segment prior of a pose (J,3), or of a window (T,J,3) measured on its first frame."""
    return pose_segment_priors(joints[0] if joints.ndim == 3 else joints)


def pose_segment_priors(poses):
    """Segment prior of each pose in (...,J,3) -> (...,6)."""
    a, b = zip(*SEGMENT_PAIRS)
    return (np.linalg.norm(poses[..., a, :] - poses[..., b, :], axis=-1) / SEG_NORM).astype(np.float32)


def limb_length_spread(clips, pair):
    """Std across Clips (each (F,J,3)) of the per-Clip median distance between the joints in `pair`."""
    i, j = pair
    lengths = [np.median(np.linalg.norm(c[:, i] - c[:, j], axis=-1)) for c in clips]
    return float(np.std(lengths))


# Segment prior pairs that are bones, as (segment index, parent, child, joints moved with the child).
# Shoulder width is stretched about the mid-shoulder; R shoulder-R hip (segment 3) is not a bone.
STRETCHED_BONES = [(0, 2, 4, [4]), (1, 3, 5, [5]), (4, 6, 8, [8, 10]), (5, 7, 9, [9, 11])]
SHOULDER_WIDTH, R_ARM, L_ARM = 2, [0, 2, 4], [1, 3, 5]


def stretch_to_segments(poses, seg):
    """Bone-stretching test: poses (...,J,3) with each Segment prior bone stretched to its length.

    Each bone keeps its direction and the joints below it move with it; the hips, so the root, stay put.
    seg (...,6) is a Segment prior (lengths / SEG_NORM).
    """
    out = np.array(poses, dtype=np.float64)
    lengths = np.asarray(seg, dtype=np.float64) * SEG_NORM
    for k, parent, child, moved in STRETCHED_BONES:
        bone = out[..., child, :] - out[..., parent, :]
        norm = np.linalg.norm(bone, axis=-1, keepdims=True)
        out[..., moved, :] += (bone * (lengths[..., k, None] / np.maximum(norm, 1e-9) - 1))[..., None, :]
    half = (out[..., 0, :] - out[..., 1, :]) / 2                 # mid-shoulder -> R shoulder
    norm = np.linalg.norm(half, axis=-1, keepdims=True)
    shift = half * (lengths[..., SHOULDER_WIDTH, None] / 2 / np.maximum(norm, 1e-9) - 1)
    out[..., R_ARM, :] += shift[..., None, :]
    out[..., L_ARM, :] -= shift[..., None, :]
    return out.astype(np.asarray(poses).dtype)
