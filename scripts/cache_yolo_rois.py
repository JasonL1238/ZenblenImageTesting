"""Cache YOLO-seg container ROI masks so chunk-detector tuning is fast + deterministic.

Writes <out>/<stem>.png (uint8 ROI mask, 0/255) for each image under data/images/.
Re-run after promoting new weights (delete the cache dir first — existing masks
are skipped, not refreshed).

Usage:
  /opt/miniconda3/bin/python scripts/cache_yolo_rois.py
  /opt/miniconda3/bin/python scripts/cache_yolo_rois.py --weights runs/smoothie-seg/nano-v4/weights/best.pt --out outputs/roi_cache_yolo_v4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.detection.yolo import get_yolo_roi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/yolo_smoothie_seg.pt")
    ap.add_argument("--out", default="outputs/roi_cache_yolo")
    args = ap.parse_args()

    cache = Path(args.out)
    cache.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    imgs = sorted(Path("data/images").rglob("*.jpg"))
    print(f"caching {len(imgs)} YOLO ROIs → {cache}")
    for i, p in enumerate(imgs, 1):
        out = cache / f"{p.stem}.png"
        if out.exists():
            continue
        img = cv2.imread(str(p))
        roi = get_yolo_roi(model(img, verbose=False)[0], img.shape[:2])
        cv2.imwrite(str(out), roi)
        print(f"[{i}/{len(imgs)}] {p.stem}  px={int((roi > 0).sum())}")
    print("done")


if __name__ == "__main__":
    main()
