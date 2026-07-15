"""The master annotation store — the single source of truth.

The master file (``master/annotations/smoothie_master.json``) is a COCO-style
document:

    {
      "info":        {...},
      "categories":  [{"id": 1, "name": "smoothie"}],
      "images":      [{"id", "file_name", "width", "height"}, ...],
      "annotations": [{"id", "image_id", "category_id", "segmentation",
                       "bbox", "area", "iscrowd"}, ...]
    }

An image with zero annotations is an *intentional negative* (no smoothie visible)
— it is still listed under ``images`` so it flows into every export as an empty
label. Every other module reads the dataset through :class:`Master`, never by
touching the JSON directly.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import geometry
from .config import COCO_CATEGORY_ID, SMOOTHIE_NAME, DatasetPaths, IMAGE_EXTS


@dataclass
class Master:
    """In-memory view of the master COCO document, with convenience accessors."""

    images: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    categories: list[dict] = field(default_factory=lambda: [
        {"id": COCO_CATEGORY_ID, "name": SMOOTHIE_NAME}
    ])
    info: dict = field(default_factory=lambda: {
        "description": "Smoothie segmentation master dataset", "version": "1.0"
    })

    # ---- load / save --------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Master":
        if not path.exists():
            raise FileNotFoundError(
                f"master annotation file not found: {path}\n"
                f"Create it with `import-coco`, `import-labels`, or `init`."
            )
        data = json.loads(path.read_text())
        return cls(
            images=data.get("images", []),
            annotations=data.get("annotations", []),
            categories=data.get("categories") or [{"id": COCO_CATEGORY_ID, "name": SMOOTHIE_NAME}],
            info=data.get("info", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "info": self.info,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }, indent=2))

    # ---- accessors ----------------------------------------------------------
    def annotations_for(self, image_id: int) -> list[dict]:
        return [a for a in self.annotations if a["image_id"] == image_id]

    def image_stem(self, image: dict) -> str:
        """Split-file identifier for an image: its file name without extension."""
        return Path(image["file_name"]).stem

    def next_image_id(self) -> int:
        return (max((im["id"] for im in self.images), default=0)) + 1

    def next_annotation_id(self) -> int:
        return (max((a["id"] for a in self.annotations), default=0)) + 1

    # ---- validation ---------------------------------------------------------
    def validate_categories(self, allow_extra: bool = False) -> list[str]:
        """Return a list of problems with the category table (empty == OK).

        By default the master must contain exactly one category named 'smoothie';
        pass ``allow_extra=True`` to permit additional categories.
        """
        problems: list[str] = []
        names = [c.get("name", "").lower() for c in self.categories]
        if SMOOTHIE_NAME not in names:
            problems.append(f"category '{SMOOTHIE_NAME}' is missing from categories")
        if not allow_extra:
            extra = [n for n in names if n != SMOOTHIE_NAME]
            if extra:
                problems.append(
                    f"unexpected categories {extra}; only '{SMOOTHIE_NAME}' is allowed "
                    f"(pass allow_extra=True to permit others)"
                )
        return problems


# ---------------------------------------------------------------------------
# Importers — build a master from an external source.
# ---------------------------------------------------------------------------
def _read_image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for an image on disk."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def import_from_coco(
    coco_path: Path,
    images_dir: Path,
    paths: DatasetPaths,
    allow_extra_categories: bool = False,
    copy_images: bool = True,
) -> Master:
    """Build the master from an external COCO polygon file.

    Every category is remapped to the single 'smoothie' category (id=1). RLE /
    crowd annotations are skipped with a warning (this pipeline is polygon-first).
    Referenced images are copied into ``master/images`` when ``copy_images``.
    """
    src = json.loads(Path(coco_path).read_text())
    src_cats = {c["id"]: c.get("name", "").lower() for c in src.get("categories", [])}

    if not allow_extra_categories:
        extra = sorted({n for n in src_cats.values() if n and n != SMOOTHIE_NAME})
        if extra:
            raise ValueError(
                f"source COCO contains non-smoothie categories {extra}. "
                f"Re-run with --allow-extra-categories to fold them all into "
                f"'{SMOOTHIE_NAME}', or clean the source first."
            )

    master = Master()
    kept_image_ids: set[int] = set()
    skipped_rle = 0

    for im in src.get("images", []):
        master.images.append({
            "id": int(im["id"]),
            "file_name": Path(im["file_name"]).name,
            "width": int(im["width"]),
            "height": int(im["height"]),
        })
        kept_image_ids.add(int(im["id"]))

    for ann in src.get("annotations", []):
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or (seg and not isinstance(seg[0], list)):
            # dict -> RLE, or list-of-numbers we can't trust: skip.
            skipped_rle += 1
            continue
        if int(ann["image_id"]) not in kept_image_ids:
            continue
        polygons = [list(map(float, p)) for p in seg if len(p) >= 6]
        if not polygons:
            continue
        master.annotations.append(_make_annotation(
            ann_id=int(ann.get("id", master.next_annotation_id())),
            image_id=int(ann["image_id"]),
            polygons=polygons,
            width=_img_wh(master, int(ann["image_id"]))[0],
            height=_img_wh(master, int(ann["image_id"]))[1],
        ))

    if skipped_rle:
        print(f"  warning: skipped {skipped_rle} non-polygon (RLE/crowd) annotations")

    if copy_images:
        _copy_referenced_images(master, images_dir, paths.master_images)

    return master


def import_from_labeling_db(
    db_path: Path,
    source_images_dir: Path,
    paths: DatasetPaths,
    verdicts: tuple[str, ...] = ("good", "corrected"),
    copy_images: bool = True,
) -> Master:
    """Bridge the existing SQLite labeling tool (`labeling/labels.db`) into a master.

    Reads polygons stored on label rows for the given verdicts. Each labeled
    image becomes a master image with one smoothie annotation (an empty/degenerate
    polygon yields a negative image). Prefer the multi-mode exporter
    (``labeling/export_multi.py``) for new standard-mode datasets.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"""
        SELECT l.file_id, l.verdict, l.polygon
        FROM labels l JOIN files f ON f.file_id = l.file_id
        WHERE l.verdict IN ({",".join("?" * len(verdicts))})
        ORDER BY l.file_id
        """,
        verdicts,
    ).fetchall()
    conn.close()

    master = Master()

    for r in rows:
        fid = int(r["file_id"])
        img_path = source_images_dir / f"{fid}.jpg"
        if not img_path.exists():
            print(f"  skip {fid}: image not found at {img_path}")
            continue

        points: list[list[float]] | None = None
        if r["polygon"]:
            points = json.loads(r["polygon"])
        w, h = _read_image_size(img_path)

        image_id = fid
        master.images.append({
            "id": image_id, "file_name": f"{fid}.jpg", "width": w, "height": h,
        })
        # points is [[x, y], ...]; flatten to a single COCO polygon.
        if points and len(points) >= 3:
            flat = [float(v) for xy in points for v in xy]
            master.annotations.append(_make_annotation(
                ann_id=master.next_annotation_id(),
                image_id=image_id, polygons=[flat], width=w, height=h,
            ))
        # else: no usable polygon -> intentional negative (image kept, no ann)

    if copy_images:
        _copy_referenced_images(master, source_images_dir, paths.master_images)

    return master


def scan_raw_images(paths: DatasetPaths, copy_images: bool = True) -> Master:
    """Create a master listing every image in ``raw_images`` with *no* annotations.

    Useful as a starting skeleton before annotating, or to register newly added
    negatives. Existing annotations are not created here.
    """
    master = Master()
    img_paths = sorted(
        p for p in paths.raw_images.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    for i, p in enumerate(img_paths, start=1):
        w, h = _read_image_size(p)
        master.images.append({"id": i, "file_name": p.name, "width": w, "height": h})
        if copy_images:
            dst = paths.master_images / p.name
            if not dst.exists():
                shutil.copyfile(p, dst)
    return master


# ---- shared helpers ---------------------------------------------------------
def _make_annotation(
    ann_id: int, image_id: int, polygons: list[list[float]], width: int, height: int
) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": COCO_CATEGORY_ID,
        "segmentation": polygons,
        "bbox": geometry.polygon_bbox(polygons),
        "area": geometry.polygon_area(polygons, width, height),
        "iscrowd": 0,
    }


def _img_wh(master: Master, image_id: int) -> tuple[int, int]:
    for im in master.images:
        if im["id"] == image_id:
            return int(im["width"]), int(im["height"])
    raise KeyError(f"image_id {image_id} not present in master images")


def _copy_referenced_images(master: Master, src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for im in master.images:
        src = src_dir / im["file_name"]
        if not src.exists():
            missing += 1
            continue
        dst = dst_dir / im["file_name"]
        if not dst.exists():
            shutil.copyfile(src, dst)
    if missing:
        print(f"  warning: {missing} referenced images not found under {src_dir}")
