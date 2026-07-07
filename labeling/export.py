"""Stage 4 — export a YOLO-seg dataset from human-verified labels.

Output layout (under ``labeling/smoothie_dataset/`` by default):
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  data.yaml

Label format (YOLO-seg):
  0 x1 y1 x2 y2 ... xn yn
  class 0 = smoothie; coordinates normalized to [0, 1].
  One row per smoothie region. Empty file = no smoothie visible.

Only "good" and "corrected" verdicts are included (bad/skip are excluded).
  good      -> polygon from data/polygons_sam/<file_id>.json  (SAM, user-accepted)
  corrected -> user-edited polygon stored in SQLite

Run:
  python labeling/export.py
  python labeling/export.py --val 0.15 --test 0.10
  python labeling/export.py --out /path/to/custom/dir
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeling import db

DATASET_DIR = db.ROOT / "smoothie_dataset"


def _polygon_to_yolo(points: list[list[int]], width: int, height: int) -> str:
    """Convert [[x, y], ...] pixel points to a YOLO-seg label line.

    Returns '0 x1 y1 x2 y2 ...' with coords normalized to [0, 1].
    Returns '' if fewer than 3 points (degenerate / empty polygon).
    """
    if len(points) < 3:
        return ""
    coords = []
    for x, y in points:
        coords.append(f"{x / width:.6f}")
        coords.append(f"{y / height:.6f}")
    return "0 " + " ".join(coords)


def _get_polygon(file_id: int, saved_polygon: str | None) -> tuple[list, int, int]:
    """Return (points, width, height).

    Uses corrected polygon from SQLite when present (verdict == 'corrected'),
    otherwise falls back to SAM's original polygon JSON on disk.
    """
    sam = json.loads((db.POLYGONS_DIR / f"{file_id}.json").read_text())
    w, h = sam["width"], sam["height"]
    if saved_polygon:
        return json.loads(saved_polygon), w, h
    return sam["points"], w, h


def _split_ids(
    ids: list[int], val_frac: float, test_frac: float
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic train/val/test split by sorted file_id (no randomness)."""
    n = len(ids)
    n_test = max(0, round(n * test_frac))
    n_val = max(0, round(n * val_frac))
    train = ids[: n - n_val - n_test]
    val = ids[n - n_val - n_test : n - n_test]
    test = ids[n - n_test :]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO-seg dataset from labels")
    parser.add_argument(
        "--out",
        default=str(DATASET_DIR),
        help=f"Output root directory (default: {DATASET_DIR})",
    )
    parser.add_argument("--val", type=float, default=0.10,
                        help="Validation fraction (default: 0.10)")
    parser.add_argument("--test", type=float, default=0.10,
                        help="Test fraction (default: 0.10)")
    args = parser.parse_args()

    print("\n  [DEPRECATED] Old single-class export from the `labels` table.\n"
          "  Standard annotations were migrated into the multi-mode tool; export\n"
          "  the unified pipeline with:  python labeling/export_multi.py --mode standard\n")

    out = Path(args.out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    conn = db.connect()
    rows = conn.execute(
        """
        SELECT l.file_id, l.verdict, l.polygon
        FROM labels l
        JOIN files f ON f.file_id = l.file_id
        WHERE l.verdict IN ('good', 'corrected') AND f.downloaded = 1
        ORDER BY l.file_id
        """
    ).fetchall()

    if not rows:
        print("no good/corrected labels found — nothing to export")
        return

    # Validate that both image and SAM polygon exist on disk.
    items: list[tuple[int, str, str | None]] = []
    for r in rows:
        fid = r["file_id"]
        if not (db.IMAGES_DIR / f"{fid}.jpg").exists():
            print(f"skip {fid}: image missing on disk")
            continue
        if not (db.POLYGONS_DIR / f"{fid}.json").exists():
            print(f"skip {fid}: SAM polygon missing (run run_sam.py first)")
            continue
        items.append((fid, r["verdict"], r["polygon"]))

    if not items:
        print("no exportable items after validation")
        return

    ids = [fid for fid, _, _ in items]
    train_ids, val_ids, test_ids = _split_ids(ids, args.val, args.test)
    split_map: dict[int, str] = {}
    split_map.update({fid: "train" for fid in train_ids})
    split_map.update({fid: "val" for fid in val_ids})
    split_map.update({fid: "test" for fid in test_ids})

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    empty = 0

    for fid, verdict, polygon_json in items:
        split = split_map[fid]

        shutil.copyfile(
            db.IMAGES_DIR / f"{fid}.jpg",
            out / "images" / split / f"{fid}.jpg",
        )

        # good  -> accept SAM polygon as-is (polygon_json ignored)
        # corrected -> use user-edited polygon from SQLite
        pts, w, h = _get_polygon(
            fid, polygon_json if verdict == "corrected" else None
        )
        label_line = _polygon_to_yolo(pts, w, h)

        label_path = out / "labels" / split / f"{fid}.txt"
        label_path.write_text(label_line + "\n" if label_line else "")
        if not label_line:
            empty += 1

        counts[split] += 1

    # data.yaml — absolute path so YOLO can be run from any cwd.
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: smoothie\n"
    )

    total = sum(counts.values())
    print(f"exported {total} images -> {out}")
    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")
    if empty:
        print(f"  {empty} empty label files (no polygon / fewer than 3 points)")
    print("  data.yaml written")


if __name__ == "__main__":
    main()
