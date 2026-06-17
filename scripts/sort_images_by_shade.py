#!/usr/bin/env python3
"""
Sort smoothie images in data/images/ into red_pink/ and yellow/ subfolders.

Uses LAB color inside the detected smoothie ROI (see smoothie_cv.classification.shade).
Writes a JSON manifest listing each file, its assigned shade, and LAB diagnostics.

Usage:
    python scripts/sort_images_by_shade.py
    python scripts/sort_images_by_shade.py --source data/images --dry-run
    python scripts/sort_images_by_shade.py --copy   # copy instead of move
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smoothie_cv.classification.shade import SmoothieShade, classify_smoothie_shade

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(source_dir: Path) -> list[Path]:
    """Return image files in source_dir and any existing shade subfolders."""
    shade_names = {s.value for s in SmoothieShade}
    images: list[Path] = []

    def add_if_image(path: Path) -> None:
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            images.append(path)

    for path in sorted(source_dir.iterdir()):
        if path.is_file():
            add_if_image(path)
        elif path.is_dir() and path.name in shade_names:
            for child in sorted(path.iterdir()):
                add_if_image(child)

    return images


def sort_images(
    source_dir: Path,
    *,
    dry_run: bool = False,
    copy: bool = False,
) -> dict:
    """
    Classify and move (or copy) images into source_dir/red_pink/ and source_dir/yellow/.

    Returns manifest dict suitable for JSON serialization.
    """
    shade_dirs = {
        SmoothieShade.RED_PINK: source_dir / SmoothieShade.RED_PINK.value,
        SmoothieShade.YELLOW: source_dir / SmoothieShade.YELLOW.value,
    }

    if not dry_run:
        for folder in shade_dirs.values():
            folder.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    counts = {SmoothieShade.RED_PINK: 0, SmoothieShade.YELLOW: 0}

    for image_path in collect_images(source_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"SKIP (unreadable): {image_path.name}")
            continue

        result = classify_smoothie_shade(image)
        dest_dir = shade_dirs[result.shade]
        dest_path = dest_dir / image_path.name
        action = "copy" if copy else "move"

        if not dry_run:
            if copy:
                shutil.copy2(image_path, dest_path)
            else:
                shutil.move(str(image_path), str(dest_path))

        counts[result.shade] += 1
        entries.append(
            {
                "filename": image_path.name,
                "shade": result.shade.value,
                "median_a": round(result.median_a, 2),
                "median_b": round(result.median_b, 2),
                "hue_deg": round(result.hue_deg, 2),
                "roi_coverage": round(result.roi_coverage, 4),
                "destination": str(dest_path.relative_to(source_dir)),
            }
        )
        print(
            f"[{result.shade.value:8s}] {image_path.name}  "
            f"a*={result.median_a:+.1f}  b*={result.median_b:+.1f}  "
            f"hue={result.hue_deg:.0f}°  →  {dest_path.relative_to(source_dir)}"
            + ("  (dry-run)" if dry_run else "")
        )

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "action": "dry_run" if dry_run else action,
        "counts": {shade.value: counts[shade] for shade in SmoothieShade},
        "images": entries,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort smoothie images into red_pink/ and yellow/ subfolders by LAB shade."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "images",
        help="Directory containing flat .jpg/.png files (default: data/images)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print assignments without moving files",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into shade folders instead of moving them",
    )
    args = parser.parse_args()

    source_dir = args.source.resolve()
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory.")
        sys.exit(1)

    manifest = sort_images(source_dir, dry_run=args.dry_run, copy=args.copy)

    manifest_path = source_dir / "shade_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nManifest → {manifest_path}")

    print(
        f"\nDone: red_pink={manifest['counts']['red_pink']}  "
        f"yellow={manifest['counts']['yellow']}  "
        f"total={len(manifest['images'])}"
    )


if __name__ == "__main__":
    main()
