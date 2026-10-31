"""Turn raw AMASS clips into processed clips: 12 joint centers in the clip's pelvis frame.

    python scripts/preprocess_amass.py --amass <AMASS root containing ACCAD/, BMLmovi/, KIT/> \
        --smplx <SMPL-X models folder: {female,male,neutral}/model.npz> --out <data folder>

Writes <out>/train/ and <out>/test/ with the file names listed in splits/*_clips.tsv.
Each joint center is the mean of SMPL-X vertices from smpl/amass/vertices_keypoints_corr.csv.

By default the subject's shape coefficients (betas) are NOT applied, exactly like the data the
paper's models were trained on: every body then has the SMPL-X template shape of its gender.
Pass --with-shape to use the real body shapes (recommended for new models).

Requires human_body_prior (https://github.com/nghorbani/human_body_prior).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel

REPO = Path(__file__).resolve().parents[1]
JOINT_VERTICES = {  # joint -> markers whose vertices are averaged, in model joint order
    "r_shoulder": ["rshoulder"], "l_shoulder": ["lshoulder"],
    "r_elbow": ["r_lelbow", "r_melbow"], "l_elbow": ["l_lelbow", "l_melbow"],
    "r_wrist": ["r_lwrist", "r_mwrist"], "l_wrist": ["l_lwrist", "l_mwrist"],
    "r_hip": ["r_ASIS", "r_PSIS"], "l_hip": ["l_ASIS", "l_PSIS"],
    "r_knee": ["r_knee", "r_mknee"], "l_knee": ["l_knee", "l_mknee"],
    "r_ankle": ["r_ankle", "r_mankle"], "l_ankle": ["l_ankle", "l_mankle"],
}
NUM_BETAS = 16


def marker_vertices():
    with open(REPO / "smpl" / "amass" / "vertices_keypoints_corr.csv", encoding="utf-8-sig") as f:
        return {row["Name"]: int(row["Index"]) for row in csv.DictReader(f)}


def pelvis_frame_joints(verts, markers):
    """(F,V,3) mesh vertices -> (F,12,3) joints in the pelvis frame of the first frame."""
    joints = np.stack([verts[:, [markers[m] for m in ms]].mean(axis=1) for ms in JOINT_VERTICES.values()], axis=1)
    joints -= (joints[:, 6:7] + joints[:, 7:8]) / 2                      # mid-hip origin, every frame
    y = joints[0, 6] - joints[0, 7]
    y /= np.linalg.norm(y) + 1e-8                                        # left -> right hip
    z = (joints[0, 0] + joints[0, 1]) / 2
    z /= np.linalg.norm(z) + 1e-8                                        # towards mid-shoulder
    x = np.cross(y, z)
    x /= np.linalg.norm(x) + 1e-8
    z = np.cross(x, y)
    R = np.stack([x, y, z])
    return joints @ R.T, R


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amass", required=True)
    p.add_argument("--smplx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--with-shape", action="store_true", help="apply the subject's betas (differs from the paper)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    markers = marker_vertices()
    body_models = {}
    for split in ["train", "test"]:
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(REPO / "splits" / f"{split}_clips.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
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
            if args.with_shape:
                betas = torch.tensor(data["betas"][:NUM_BETAS], dtype=torch.float32, device=device)
                params["betas"] = betas.expand(len(poses), -1)
            with torch.no_grad():
                verts = body_models[gender](**params).v.cpu().numpy()
            joints, R = pelvis_frame_joints(verts, markers)
            hip_mid = (verts[0, [markers["r_ASIS"], markers["r_PSIS"], markers["l_ASIS"], markers["l_PSIS"]]]).mean(0)
            np.savez(out_dir / row["processed_file"], joints_3d=joints.astype(np.float32), R=R, t=hip_mid)
        print(f"{split}: {len(rows)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
