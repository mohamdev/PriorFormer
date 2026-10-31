"""Protocol settings: how a Window is turned into model inputs and 3D targets.

A Protocol fixes the cameras, the 2D input format, the frame of the 3D targets, the 2D noise and
the Sliding window warmup. `V1` is the paper's protocol (WindowDataset and priorformer.evaluate);
`REFERENCE` is the M0 protocol of the research reference model (docs/plan-v2.md).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CameraSetting:
    """Upright cameras (gravity up). Distances are in body radii, focal lengths either in pixels
    (`f_px`, fixed 1920x1080 images) or relative to the image diagonal (`f_rel`, random sizes)."""
    f_px: Optional[tuple] = (800.0, 1400.0)
    f_rel: Optional[tuple] = None
    random_size: bool = False
    distance: tuple = (1.5, 2.5)
    fill: Optional[tuple] = None   # if set, distance makes the body radius fill this fraction of half the short side
    target_offset: float = 0.0     # look-at target offset from the mid-hip, in body radii
    roll_deg: float = 0.0
    pp_jitter: float = 0.03


@dataclass(frozen=True)
class NoiseSetting:
    """Detector-like 2D errors; sizes are fractions of the 2D body size (bounding-box diagonal)."""
    sigma: float = 0.01          # per-Window jitter std drawn from U(0, sigma)
    p_outlier: float = 0.01      # per joint and frame
    outlier_sigma: float = 0.1
    p_swap: float = 0.01         # per frame: left/right swapped
    p_missing: float = 0.05      # per joint and frame


@dataclass(frozen=True)
class Protocol:
    cameras: Optional[CameraSetting] = None   # None: v1 cameras (up = first frame's hip axis)
    input: str = "pixels"                      # "pixels" (v1: pixels / 1920) or "normalized"
    target: str = "pelvis"                     # "pelvis" (v1: Clip pelvis frame) or "camera"
    noise: Optional[NoiseSetting] = None
    warmup: float = 0.0                        # probability that a training Window is in Sliding window warmup

    @property
    def is_v1(self):
        return self.cameras is None and self.input == "pixels" and self.target == "pelvis" \
            and self.noise is None and self.warmup == 0.0


V1 = Protocol()

CENTERED = CameraSetting()
WILD = CameraSetting(f_px=None, f_rel=(0.35, 1.5), random_size=True, fill=(0.3, 0.9),
                     target_offset=0.5, roll_deg=10.0)
CAMERA_SETTINGS = {"centered": CENTERED, "wild": WILD}

REFERENCE = Protocol(cameras=WILD, input="normalized", target="camera", noise=NoiseSetting(), warmup=0.3)
