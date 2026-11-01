"""Check that processed Clips vary in body shape: per-gender std (across Clips) of limb lengths.

    python scripts/check_shape_variety.py --clips <processed clips folder> [--clips <another folder>]

A template-shape set (preprocess_amass.py without --with-shape) gives about 0 mm within each
gender; a shaped set gives clearly more.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from priorformer.skeleton import JOINT_NAMES, limb_length_spread

LIMBS = {"R thigh": ("r_hip", "r_knee"), "R forearm": ("r_elbow", "r_wrist"), "R shank": ("r_knee", "r_ankle")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", action="append", required=True)
    p.add_argument("--max-clips", type=int, default=500, help="first N clips (sorted) per folder")
    args = p.parse_args()
    for folder in args.clips:
        by_gender = defaultdict(list)
        for f in sorted(Path(folder).glob("*.npz"))[:args.max_clips]:
            d = np.load(f)
            gender = str(d["gender"]) if "gender" in d else "unknown"
            by_gender[gender].append(d["joints_3d"])
        print(folder)
        for gender, clips in sorted(by_gender.items()):
            spreads = ", ".join(f"{name} {1000 * limb_length_spread(clips, tuple(JOINT_NAMES.index(j) for j in pair)):.1f} mm"
                                for name, pair in LIMBS.items())
            print(f"  {gender} ({len(clips)} clips): std across clips: {spreads}")


if __name__ == "__main__":
    main()
