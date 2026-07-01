"""Debug visualizations — overlay image + polygon outline + generated mask.

For each sampled image we render a side-by-side panel:
  left  : original image with polygon outlines drawn
  right : original image with the filled binary mask tinted on top

Panels are written to ``debug_outputs/`` so annotations can be eyeballed without
launching a training run.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry
from .config import DatasetPaths
from .master import Master

_OUTLINE = (0, 255, 0)   # BGR green
_TINT = (122, 192, 57)   # BGR of the semantic-green used in the labeling UI


def _source(paths: DatasetPaths, file_name: str):
    for base in (paths.master_images, paths.raw_images):
        p = base / file_name
        if p.exists():
            return p
    return None


def visualize(master: Master, paths: DatasetPaths, num: int = 20) -> int:
    """Write up to ``num`` overlay panels to ``debug_outputs/``. Returns count."""
    paths.debug_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for im in master.images:
        if written >= num:
            break
        src = _source(paths, im["file_name"])
        if src is None:
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = master.image_stem(im)

        polygons: list[list[float]] = []
        for ann in master.annotations_for(im["id"]):
            polygons.extend(ann["segmentation"])

        # left: outlines
        left = img.copy()
        for poly in polygons:
            if geometry.polygon_point_count(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(left, [pts], isClosed=True, color=_OUTLINE, thickness=2)

        # right: tinted filled mask
        mask = geometry.polygons_to_mask(polygons, w, h, fg=1)
        right = img.copy()
        tint = np.zeros_like(img)
        tint[:] = _TINT
        blended = cv2.addWeighted(right, 0.55, tint, 0.45, 0)
        right = np.where(mask[..., None] > 0, blended, right)

        label = "NEGATIVE (no smoothie)" if not polygons else f"{len(polygons)} polygon(s)"
        panel = np.hstack([left, right])
        cv2.putText(panel, f"{stem}  {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(paths.debug_dir / f"{stem}_overlay.png"), panel)
        written += 1

    return written
