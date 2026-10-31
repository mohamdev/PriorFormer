"""Train/test clip lists, whole-clip validation split, and Task selection by filename."""
import random
import re

# Paper tasks and the filename rules that select their clips.
TASK_RULES = {
    "bend": r"bend_",
    "displace_tilt": r"displace_tilt",
    "guitar": r"guitar",
    "kick": r"kick",
    "squat": r"squat",
    "punch": r"punch",
    "pour": r"pour",
    "stairs_climbing": r"upstairs|downstairs|stair",
    "push_recovery": r"push_recovery|push_from|RecoveryStepping",
    "tennis": r"tennis",
    "turn": r"(?i)turn",
    "walk": r"(?i)walk",
    "wipe": r"wipe",
}
TASKS = list(TASK_RULES)


def read_split(tsv_path):
    """Processed clip filenames listed in a splits/*_clips.tsv file."""
    lines = open(tsv_path).read().splitlines()[1:]
    return [line.split("\t")[0] for line in lines if line]


def split_clips(clips, val_fraction, seed):
    """Hold out whole clips for validation. Returns (train, val), both sorted."""
    clips = sorted(clips)
    val = set(random.Random(seed).sample(clips, round(val_fraction * len(clips))))
    return [c for c in clips if c not in val], [c for c in clips if c in val]


def task_clips(clips):
    """Map each Task to the sorted clips whose filename matches its rule."""
    stairs = re.compile(TASK_RULES["stairs_climbing"])
    out = {}
    for task, rule in TASK_RULES.items():
        rx = re.compile(rule)
        out[task] = [c for c in sorted(clips)
                     if rx.search(c) and not (task == "walk" and stairs.search(c))]
    return out
