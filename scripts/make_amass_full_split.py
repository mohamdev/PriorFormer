"""Split the AMASS clips that the v1 split doesn't use into extra Train clips and Unseen clips.

    python -m scripts.make_amass_full_split --amass <AMASS root with one folder per subset>

The v1 split (splits/train_clips.tsv, splits/test_clips.tsv: ACCAD, BMLmovi, KIT) is left unchanged,
so the 800 Test clips stay comparable with v1 and M0. Every other motion file goes to:
- splits/unseen_clips.tsv if its subset is in UNSEEN_SUBSETS: whole datasets, never trained on, for
  testing on subjects, labs and motions the model has never seen;
- splits/train_extra_clips.tsv otherwise.
Processed names continue the "<index>_<file name>" convention, from index 10000 so that they never
collide with v1 names. Rerunning on the same AMASS folder gives the same files.
"""
import argparse
import csv
from pathlib import Path

from priorformer.splits import normalize_source

REPO = Path(__file__).resolve().parents[1]
# AMASS's own test subsets (Transitions, SSM) and the two subsets with synchronized video (TotalCapture,
# HumanEva), kept unseen for a later benchmark on real images.
UNSEEN_SUBSETS = ("Transitions_mocap", "SSM_synced", "TotalCapture", "HumanEva")
SKIP_SUBSETS = ("AMASS_Filtered",)  # partial copies of ACCAD and KIT clips
FIRST_INDEX = 10000


def assign_new_clips(sources, used, unseen_subsets=UNSEEN_SUBSETS, first_index=FIRST_INDEX):
    """Motion files (paths relative to the AMASS root) not in `used` -> (train_extra, unseen) rows.

    Each row is (processed_file, amass_source). Files are numbered in sorted order of their source.
    """
    used = {normalize_source(s) for s in used}
    train, unseen = [], []
    new = sorted(s for s in sources if s not in used and s.split("/")[0] not in SKIP_SUBSETS)
    for i, source in enumerate(new):
        row = (f"{first_index + i}_{source.split('/')[-1]}", source)
        (unseen if source.split("/")[0] in unseen_subsets else train).append(row)
    return train, unseen


def motion_files(root):
    root = Path(root)
    return sorted(str(p.relative_to(root)) for pattern in ["*_poses.npz", "*_stageii.npz"]
                  for p in root.glob(f"*/**/{pattern}"))


def read_sources(tsv):
    with open(tsv) as f:
        return [s for row in csv.DictReader(f, delimiter="\t") for s in row["amass_source"].split("|")]


def write_rows(tsv, rows):
    with open(tsv, "w") as f:
        f.write("processed_file\tamass_source\n" + "".join(f"{name}\t{src}\n" for name, src in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amass", required=True)
    args = p.parse_args()
    used = read_sources(REPO / "splits" / "train_clips.tsv") + read_sources(REPO / "splits" / "test_clips.tsv")
    train, unseen = assign_new_clips(motion_files(args.amass), used)
    write_rows(REPO / "splits" / "train_extra_clips.tsv", train)
    write_rows(REPO / "splits" / "unseen_clips.tsv", unseen)
    print(f"{len(train)} extra Train clips, {len(unseen)} Unseen clips")


if __name__ == "__main__":
    main()
