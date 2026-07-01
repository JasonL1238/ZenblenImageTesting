"""Train/val/test split management.

A split is stored as three text files under ``splits/`` — one image identifier
(the file-name stem) per line. The same split files drive every export, so YOLO,
semantic, and COCO always see identical train/val/test partitions.

Splits are *sticky*: once written they are reused verbatim on subsequent runs so
adding a model export never silently reshuffles the data. Pass ``force=True`` to
regenerate (e.g. after adding many new images).
"""
from __future__ import annotations

import random

from .config import SPLITS, DatasetPaths
from .master import Master


def splits_exist(paths: DatasetPaths) -> bool:
    return all(paths.split_file(s).exists() for s in SPLITS)


def read_splits(paths: DatasetPaths) -> dict[str, list[str]]:
    """Load the persisted split files into ``{split: [stem, ...]}``."""
    out: dict[str, list[str]] = {}
    for s in SPLITS:
        f = paths.split_file(s)
        out[s] = f.read_text().splitlines() if f.exists() else []
        out[s] = [line.strip() for line in out[s] if line.strip()]
    return out


def write_splits(paths: DatasetPaths, assignment: dict[str, list[str]]) -> None:
    paths.splits_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        paths.split_file(s).write_text("\n".join(assignment.get(s, [])) + "\n")


def create_splits(
    master: Master,
    paths: DatasetPaths,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    force: bool = False,
) -> dict[str, list[str]]:
    """Assign every master image to train/val/test and persist the split files.

    Ratios are normalized if they don't sum to 1.0. The assignment is a seeded
    shuffle, so it is reproducible but not tied to file order. If split files
    already exist and ``force`` is False, the existing split is returned unchanged.
    """
    if splits_exist(paths) and not force:
        print("  splits already exist — reusing them (pass --force to reshuffle)")
        return read_splits(paths)

    total = train + val + test
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")
    train, val, test = train / total, val / total, test / total

    stems = [master.image_stem(im) for im in master.images]
    if not stems:
        raise ValueError("no images in master — nothing to split")

    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # test takes the remainder so the three always cover every image exactly once.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    assignment = {
        "train": sorted(stems[:n_train]),
        "val": sorted(stems[n_train:n_train + n_val]),
        "test": sorted(stems[n_train + n_val:]),
    }
    write_splits(paths, assignment)
    print(f"  splits written: train={len(assignment['train'])} "
          f"val={len(assignment['val'])} test={len(assignment['test'])}")
    return assignment


def split_of_each_image(paths: DatasetPaths) -> dict[str, str]:
    """Invert the split files into ``{image_stem: split_name}`` for fast lookup."""
    mapping: dict[str, str] = {}
    for split, stems in read_splits(paths).items():
        for stem in stems:
            mapping[stem] = split
    return mapping
