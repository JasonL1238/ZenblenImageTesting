"""Stage 2 — run SAM over downloaded images to produce candidate masks + polygons.

For each image in ``data/images/`` this writes:
  data/masks_sam/<file_id>.png       uint8 ROI mask (255 = container)
  data/polygons_sam/<file_id>.json   {"width", "height", "points": [[x, y], ...]}

The polygon is the largest external contour of the mask, simplified with
``cv2.approxPolyDP`` so the labeling UI shows a handful of draggable vertices
instead of thousands of contour points. An empty ``points`` list means SAM found
no plausible container (the UI then lets you draw one or reject).

Resumable: images whose polygon JSON already exists are skipped.

IMPORTANT: SAM2 requires the conda base env (see CLAUDE.md), e.g.
  /opt/miniconda3/bin/python labeling/run_sam.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeling import db
from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.01) -> list[list[int]]:
    """Largest external contour of a binary mask -> simplified [[x, y], ...].

    Returns [] if the mask is empty. ``epsilon_frac`` is the approxPolyDP
    tolerance as a fraction of the contour perimeter (~1% keeps corners/curves
    with a manageable vertex count).
    """
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 0:
        return []
    eps = epsilon_frac * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    return [[int(x), int(y)] for [[x, y]] in approx]


def process_image(path: Path, cfg: Config, epsilon_frac: float, target: str = "container") -> str:
    """Run detection on one image, write mask + polygon. Returns detector name.

    ``target``:
      * ``"container"`` — full cup footprint via ``detect_container`` (SAM→classical).
      * ``"smoothie"``  — raw central liquid mass via ``detect_smoothie`` (no container
        priors). A closer starting point for smoothie-only labels; the top edge still
        needs manual correction in the UI because SAM reads the clear cup + fill as one
        object and cannot reliably cut at the liquid surface.
    """
    fid = path.stem
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    if target == "smoothie":
        from smoothie_cv.detection.sam import detect_smoothie
        mask, _bbox = detect_smoothie(img, cfg)
        detector = "sam-smoothie"
    else:
        mask, _bbox, meta = detect_container(img, cfg, return_meta=True)
        detector = meta["detector"]

    cv2.imwrite(str(db.MASKS_DIR / f"{fid}.png"), mask)

    h, w = img.shape[:2]
    poly = {
        "width": int(w),
        "height": int(h),
        "points": mask_to_polygon(mask, epsilon_frac),
    }
    (db.POLYGONS_DIR / f"{fid}.json").write_text(json.dumps(poly))
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM over downloaded images")
    parser.add_argument("--model", default=None,
                        help="SAM2 model name, e.g. sam2_hiera_tiny (default: Config default). "
                             "Use this instead of --config to avoid inheriting pipeline tuning.")
    parser.add_argument("--epsilon-frac", type=float, default=0.01,
                        help="approxPolyDP tolerance as fraction of perimeter (default 0.01)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (for a quick test batch)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess images even if a polygon JSON already exists")
    parser.add_argument("--target", choices=("container", "smoothie"), default="container",
                        help="What to segment: 'container' (full cup, default) or "
                             "'smoothie' (raw liquid mass, closer to smoothie-only labels)")
    args = parser.parse_args()

    db.ensure_dirs()
    cfg = Config()
    if args.model:
        cfg.sam_model = args.model

    imgs = sorted(db.IMAGES_DIR.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"running SAM on {len(imgs)} images -> {db.MASKS_DIR} / {db.POLYGONS_DIR}")

    done = failed = skipped = 0
    for i, p in enumerate(imgs, 1):
        poly_path = db.POLYGONS_DIR / f"{p.stem}.json"
        if poly_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            det = process_image(p, cfg, args.epsilon_frac, target=args.target)
            done += 1
            print(f"[{i}/{len(imgs)}] {p.stem}  det={det}")
        except Exception as e:  # noqa: BLE001 - keep the batch going, report at end
            failed += 1
            print(f"[{i}/{len(imgs)}] {p.stem}  ERROR {e}")
    print(f"done: {done} processed, {skipped} skipped (cached), {failed} failed")


if __name__ == "__main__":
    main()
