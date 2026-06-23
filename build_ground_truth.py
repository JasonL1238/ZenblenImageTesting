#!/usr/bin/env python3
"""
Run LAB chroma detection on red/pink smoothie images and save as ground truth.

Red/pink images (data/images/red_pink/) are well-handled by LAB chroma
detection, so we use those ROIs as the baseline to compare future strategies.

Outputs:
  outputs/ground_truth/<stem>_roi.png   — ROI overlay images
  outputs/ground_truth/manifest.json   — per-image stats for later comparison
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from smoothie_cv.detection import detect_container, draw_container_overlay

RED_PINK_DIR = Path("data/images/red_pink")
GT_DIR = Path("outputs/ground_truth")


def roi_stats(image: np.ndarray, roi_mask: np.ndarray) -> dict:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    roi_pixels = lab[roi_mask > 0]
    a_c = roi_pixels[:, 1].astype(np.float32) - 128
    b_c = roi_pixels[:, 2].astype(np.float32) - 128
    return {
        "mean_L":      round(float(roi_pixels[:, 0].mean()), 2),
        "mean_a":      round(float(roi_pixels[:, 1].mean()), 2),
        "mean_b":      round(float(roi_pixels[:, 2].mean()), 2),
        "mean_chroma": round(float(np.hypot(a_c, b_c).mean()), 2),
    }


def main() -> None:
    GT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(RED_PINK_DIR.glob("*.jpg")) + sorted(RED_PINK_DIR.glob("*.png"))
    if not images:
        print(f"No images found in {RED_PINK_DIR}")
        return

    records = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"SKIP (unreadable): {img_path.name}")
            continue

        roi_mask, bbox = detect_container(image)
        stats = roi_stats(image, roi_mask)
        coverage = round(float((roi_mask > 0).sum()) / (image.shape[0] * image.shape[1]), 4)

        # save ROI overlay
        vis = draw_container_overlay(image, roi_mask)
        out_path = GT_DIR / f"{img_path.stem}_roi.png"
        cv2.imwrite(str(out_path), vis)

        record = {
            "image":    str(img_path),
            "stem":     img_path.stem,
            "roi_path": str(out_path),
            "bbox":     bbox,
            "coverage": coverage,
            **stats,
        }
        records.append(record)

        print(
            f"{img_path.name[:55]:<55}  "
            f"coverage={coverage:.1%}  a={stats['mean_a']:.1f}  "
            f"chroma={stats['mean_chroma']:.1f}"
        )

    manifest = {"image_count": len(records), "source_dir": str(RED_PINK_DIR), "images": records}
    manifest_path = GT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nGround truth: {len(records)} red/pink images → {GT_DIR}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
