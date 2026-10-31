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
