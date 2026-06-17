#!/usr/bin/env python3
"""Quick visual test of detect_container (LAB chroma) — saves ROI overlays only."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from smoothie_cv.detection.container import detect_container, draw_container_overlay

IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("outputs/container_detection_test")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"SKIP (unreadable): {img_path.name}")
            continue

        roi_mask, bbox = detect_container(image)
        vis = draw_container_overlay(image, roi_mask)

        out_path = OUTPUT_DIR / f"{img_path.stem}_roi.png"
        cv2.imwrite(str(out_path), vis)

        coverage = (roi_mask > 0).sum() / (image.shape[0] * image.shape[1]) * 100
        print(f"{img_path.name}  bbox={bbox}  coverage={coverage:.1f}%  → {out_path.name}")


if __name__ == "__main__":
    main()
