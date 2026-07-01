"""Dataset layout, class definition, and path resolution.

Everything in the framework is anchored to a single dataset root directory
(``--dataset``, default ``smoothie_dataset``). :class:`DatasetPaths` centralises
the on-disk layout so no other module hard-codes a relative path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---- class definition -------------------------------------------------------
# One class only. Two id conventions coexist by ecosystem convention:
#   * YOLO uses 0-indexed class ids  -> smoothie = 0
#   * COCO uses 1-indexed category ids (0 is reserved for background) -> smoothie = 1
SMOOTHIE_NAME = "smoothie"
YOLO_CLASS_ID = 0
COCO_CATEGORY_ID = 1

# Mask pixel values for the binary/semantic masks.
MASK_BG = 0
MASK_FG = 1

SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved absolute paths for one dataset root.

    Construct with :meth:`from_root`; every attribute is an absolute
    :class:`~pathlib.Path`. Nothing here touches the filesystem until
    :meth:`ensure` is called.
    """

    root: Path

    # master (source of truth)
    raw_images: Path
    master_images: Path
    master_annotations: Path
    master_json: Path
    master_masks: Path

    # splits
    splits_dir: Path

    # exports
    yolo_dir: Path
    semantic_dir: Path
    coco_dir: Path

    # debug
    debug_dir: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "DatasetPaths":
        r = Path(root).resolve()
        master = r / "master"
        return cls(
            root=r,
            raw_images=r / "raw_images",
            master_images=master / "images",
            master_annotations=master / "annotations",
            master_json=master / "annotations" / "smoothie_master.json",
            master_masks=master / "masks",
            splits_dir=r / "splits",
            yolo_dir=r / "exports" / "yolo_seg",
            semantic_dir=r / "exports" / "semantic",
            coco_dir=r / "exports" / "coco",
            debug_dir=r / "debug_outputs",
        )

    def ensure(self) -> None:
        """Create the master + splits skeleton (exports are made on demand)."""
        for d in (
            self.raw_images,
            self.master_images,
            self.master_annotations,
            self.master_masks,
            self.splits_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def split_file(self, split: str) -> Path:
        return self.splits_dir / f"{split}.txt"
