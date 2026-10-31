"""Joint set and segment prior definitions (see GLOSSARY.md)."""
import numpy as np

JOINT_NAMES = [
    "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist",
    "r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle",
]
N_JOINTS = len(JOINT_NAMES)

# Joint pairs whose distances form the segment prior, in this order.
SEGMENT_PAIRS = [(2, 4), (3, 5), (0, 1), (0, 6), (6, 8), (7, 9)]
SEGMENT_NAMES = ["r_forearm", "l_forearm", "shoulder_width", "r_shoulder_hip", "r_thigh", "l_thigh"]
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
