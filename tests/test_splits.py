from pathlib import Path

from priorformer.splits import TASKS, read_split, split_clips, task_clips

SPLITS = Path(__file__).resolve().parents[1] / "splits"


def test_clip_split_is_disjoint_complete_and_deterministic():
    clips = [f"{i}_clip.npz" for i in range(200)]
    train, val = split_clips(clips, val_fraction=0.05, seed=0)

    assert len(val) == 10
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == sorted(clips)
    assert split_clips(list(reversed(clips)), val_fraction=0.05, seed=0) == (train, val)
    assert split_clips(clips, val_fraction=0.05, seed=1)[1] != val


def test_task_rules_regenerate_the_published_test_task_lists():
    test_clips = read_split(SPLITS / "test_clips.tsv")
    assert len(test_clips) == 800

    tasks = task_clips(test_clips)

    assert list(tasks) == TASKS and len(TASKS) == 13
    for task, clips in tasks.items():
        published = (SPLITS / "test_tasks" / f"{task}.txt").read_text().split("\n")
        assert clips == [c for c in published if c], task


def test_walk_excludes_stair_clips():
    tasks = task_clips(["1_walking_fast01_poses.npz", "2_upstairs_walk03_poses.npz"])
    assert tasks["walk"] == ["1_walking_fast01_poses.npz"]
    assert tasks["stairs_climbing"] == ["2_upstairs_walk03_poses.npz"]


def test_new_amass_clips_skip_v1_sources_and_keep_whole_subsets_unseen():
    from scripts.make_amass_full_split import assign_new_clips

    sources = ["KIT/3/walk_poses.npz", "KIT/3/run_poses.npz", "CMU/01/01_01_poses.npz",
               "TotalCapture/s1/walking1_poses.npz", "AMASS_Filtered/KIT/3/walk_poses.npz"]
    train, unseen = assign_new_clips(sources, used=["KIT/KIT/3/walk_poses.npz"],
                                      unseen_subsets=("TotalCapture",), first_index=10000)

    assert train == [("10000_01_01_poses.npz", "CMU/01/01_01_poses.npz"),
                     ("10001_run_poses.npz", "KIT/3/run_poses.npz")]
    assert unseen == [("10002_walking1_poses.npz", "TotalCapture/s1/walking1_poses.npz")]


def test_processed_names_are_unique_across_all_split_files():
    names = [n for f in sorted(SPLITS.glob("*_clips.tsv")) if f.name != "test_clips_near_train.tsv"
             for n in read_split(f)]
    v1 = set(read_split(SPLITS / "train_clips.tsv")) | set(read_split(SPLITS / "test_clips.tsv"))
    new = [n for n in names if n not in v1]

    assert len(new) == len(set(new))
    assert not set(new) & v1
