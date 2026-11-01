"""Turn raw AMASS clips into processed clips: 12 joint centers in the clip's pelvis frame.

    python scripts/preprocess_amass.py --amass <AMASS root containing ACCAD/, BMLmovi/, KIT/> \
        --smplx <SMPL-X models folder: {female,male,neutral}/model.npz> --out <data folder>

Writes <out>/train/ and <out>/test/ with the file names listed in splits/*_clips.tsv.
Each joint center is the mean of SMPL-X vertices from smpl/amass/vertices_keypoints_corr.csv.

By default the subject's shape coefficients (betas) are NOT applied, exactly like the data the
paper's models were trained on: every body then has the SMPL-X template shape of its gender.
Pass --with-shape to use the real body shapes (recommended for new models), and write them to
their own --out folder. Each processed clip stores its gender and the betas that were applied
(zeros without --with-shape). The AMASS betas fitted for SMPL+H are applied to the SMPL-X model, as
the poses already are: bodies vary like the subjects' but are not their exact SMPL-X shapes.

Requires human_body_prior (https://github.com/nghorbani/human_body_prior).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel

from priorformer.skeleton import JOINT_NAMES

REPO = Path(__file__).resolve().parents[1]
JOINT_VERTICES = {  # joint -> named SMPL-X vertices that are averaged
    "r_shoulder": ["rshoulder"], "l_shoulder": ["lshoulder"],
    "r_elbow": ["r_lelbow", "r_melbow"], "l_elbow": ["l_lelbow", "l_melbow"],
    "r_wrist": ["r_lwrist", "r_mwrist"], "l_wrist": ["l_lwrist", "l_mwrist"],
    "r_hip": ["r_ASIS", "r_PSIS"], "l_hip": ["l_ASIS", "l_PSIS"],
    "r_knee": ["r_knee", "r_mknee"], "l_knee": ["l_knee", "l_mknee"],
    "r_ankle": ["r_ankle", "r_mankle"], "l_ankle": ["l_ankle", "l_mankle"],
}
NUM_BETAS = 16


def named_vertices():
    with open(REPO / "smpl" / "amass" / "vertices_keypoints_corr.csv", encoding="utf-8-sig") as f:
        return {row["Name"]: int(row["Index"]) for row in csv.DictReader(f)}


def pelvis_frame_joints(verts, markers):
    """(F,V,3) mesh vertices -> (F,12,3) joints in the pelvis frame, its rotation, and the first mid-hip."""
    joints = np.stack([verts[:, [markers[m] for m in JOINT_VERTICES[j]]].mean(axis=1) for j in JOINT_NAMES], axis=1)
    r_hip, l_hip, r_sh, l_sh = (JOINT_NAMES.index(j) for j in ["r_hip", "l_hip", "r_shoulder", "l_shoulder"])
    hip_mid = (joints[:, r_hip] + joints[:, l_hip]) / 2
    joints -= hip_mid[:, None]                                           # mid-hip origin, every frame
    y = joints[0, r_hip] - joints[0, l_hip]
    y /= np.linalg.norm(y) + 1e-8                                        # left -> right hip
    z = (joints[0, r_sh] + joints[0, l_sh]) / 2
    z /= np.linalg.norm(z) + 1e-8                                        # towards mid-shoulder
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-8
    z = np.cross(x, y)
    R = np.stack([x, y, z])
    return joints @ R.T, R, hip_mid[0]


CHUNK = 256  # frames per body model call


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amass", required=True)
    p.add_argument("--smplx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--with-shape", action="store_true", help="apply the subject's betas (differs from the paper)")
    p.add_argument("--skip-existing", action="store_true", help="keep clips already in --out (resume a run)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    markers = named_vertices()
    body_models = {}
    for split in ["train", "test"]:
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(REPO / "splits" / f"{split}_clips.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            if args.skip_existing and (out_dir / row["processed_file"]).exists():
                continue
            source = row["amass_source"].split("|")[0]
            if "|" in row["amass_source"]:
                print(f"[warn] {row['processed_file']}: ambiguous source, using {source}")
            data = np.load(Path(args.amass) / source)
            gender = data["gender"].item()
            gender = gender.decode() if isinstance(gender, bytes) else str(gender)
            if gender not in body_models:
                body_models[gender] = BodyModel(bm_fname=str(Path(args.smplx) / gender / "model.npz"),
                                                num_betas=NUM_BETAS).to(device)
            poses = torch.tensor(data["poses"], dtype=torch.float32, device=device)
            params = {"root_orient": poses[:, :3], "pose_body": poses[:, 3:66], "pose_hand": poses[:, 66:],
                      "trans": torch.tensor(data["trans"], dtype=torch.float32, device=device)}
            betas = np.zeros(NUM_BETAS, np.float32)
            if args.with_shape:
                betas = data["betas"][:NUM_BETAS].astype(np.float32)
                params["betas"] = torch.tensor(betas, device=device).expand(len(poses), -1)
            with torch.no_grad():  # in chunks of frames, so that long Clips fit in a small GPU
                verts = np.concatenate([body_models[gender](**{k: v[i:i + CHUNK] for k, v in params.items()})
                                        .v.cpu().numpy() for i in range(0, len(poses), CHUNK)])
            joints, R, hip_mid = pelvis_frame_joints(verts, markers)
            np.savez(out_dir / row["processed_file"], joints_3d=joints.astype(np.float32), R=R, t=hip_mid,
                     betas=betas, gender=gender)
        print(f"{split}: {len(rows)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
