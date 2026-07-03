"""Cache SAM container ROI masks (legacy/reference detector) for every image.

SAM needs no training data, so its masks are an independent reference when
evaluating a newly trained YOLO model (scripts/compare_yolo_vs_sam.py).

Writes <cache>/<stem>.png (uint8 ROI mask) for each image under data/images/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container

CACHE = Path("outputs/roi_cache_sam")
CACHE.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = Config()
    imgs = sorted(Path("data/images").rglob("*.jpg"))
    print(f"caching {len(imgs)} ROIs → {CACHE}")
    for i, p in enumerate(imgs, 1):
        out = CACHE / f"{p.stem}.png"
        if out.exists():
            print(f"[{i}/{len(imgs)}] skip {p.stem}")
            continue
        img = cv2.imread(str(p))
        # force SAM: the default priority is YOLO, but this cache is the SAM reference
        mask, _bbox, meta = detect_container(img, cfg, prefer=["sam", "classical"],
                                             return_meta=True)
        cv2.imwrite(str(out), mask)
        print(f"[{i}/{len(imgs)}] {p.stem}  det={meta['detector']} fb={meta['fallback']}")
    print("done")


if __name__ == "__main__":
    main()
