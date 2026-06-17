"""Quick test: run detect_container on every UserGrab image and save ROI overlays."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smoothie_cv.detection.container import detect_container, draw_container_overlay

OUT = ROOT / "outputs/container_test"
OUT.mkdir(parents=True, exist_ok=True)

images = sorted((ROOT / "data/images").glob("UserGrab_*.jpg"))
if not images:
    print("No UserGrab images found.")
    sys.exit(1)

for img_path in images:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"SKIP (unreadable): {img_path.name}")
        continue

    roi_mask, bbox = detect_container(image)

    strategy = "saturation" if bbox is not None else "full_frame"
    overlay = draw_container_overlay(image, roi_mask)

    roi_pixels = int((roi_mask > 0).sum())
    total_pixels = roi_mask.size
    coverage = roi_pixels / total_pixels

    label = f"{strategy}  bbox={bbox}  coverage={coverage:.1%}"
    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = OUT / f"{img_path.stem}_roi.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"{img_path.name}  →  {strategy}  bbox={bbox}  coverage={coverage:.1%}")

print(f"\nOutputs written to {OUT}/")
