"""Freeze the c5032014 ROI mask and apply it identically to every UserGrab image."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from smoothie_cv.detection.container import detect_container, draw_container_overlay

IMAGES_DIR = Path("data/images")
REF = IMAGES_DIR / "UserGrab_c5032014-b3ad-4988-b35c-3fa78cdfa170_2026_06_16_18_40_13.jpg"
OUT = Path("outputs/frozen_mask_test")
OUT.mkdir(parents=True, exist_ok=True)
FROZEN = Path("outputs/frozen_mask.png")

# --- 1. load frozen mask if present, else compute from the reference image ---
frozen_mask = cv2.imread(str(FROZEN), cv2.IMREAD_GRAYSCALE)
if frozen_mask is not None:
    ref_h, ref_w = frozen_mask.shape[:2]
    print(f"Loaded frozen mask from {FROZEN}  ({ref_w}x{ref_h})\n")
else:
    ref_img = cv2.imread(str(REF))
    frozen_mask, frozen_bbox = detect_container(ref_img)
    ref_h, ref_w = frozen_mask.shape[:2]
    cv2.imwrite(str(FROZEN), frozen_mask)
    print(f"Frozen mask from {REF.name}: bbox={frozen_bbox}, resolution={ref_w}x{ref_h}")
    print(f"Saved -> {FROZEN}\n")

# --- 2. apply that exact mask to every image ---
images = sorted(IMAGES_DIR.glob("UserGrab_*.jpg"))
for img_path in images:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"SKIP (unreadable): {img_path.name}")
        continue

    h, w = image.shape[:2]
    if (h, w) != (ref_h, ref_w):
        print(f"  !! {img_path.name} is {w}x{h}, ref is {ref_w}x{ref_h} — frozen mask won't align")

    overlay = draw_container_overlay(image, frozen_mask)
    cv2.putText(overlay, "FROZEN MASK (c5032014)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out_path = OUT / f"{img_path.stem}_frozen.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"{img_path.name}  ({w}x{h})")

print(f"\nOverlays written to {OUT}/")
