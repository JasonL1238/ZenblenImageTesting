"""
ROI-first helpers: restrict CV analysis to the container interior.

All CV pipelines crop to the ROI bounding box, zero pixels outside the
container contour, run detection on that crop, then paste results back
to full-frame coordinates for scoring and overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoiCrop:
    """A tight bounding-box crop of the container region."""

    image: np.ndarray       # BGR; pixels outside the contour are zeroed
    mask: np.ndarray        # uint8 ROI mask for the crop (255 = inside)
    offset: tuple[int, int]  # (x, y) top-left corner in the full image
    full_shape: tuple[int, int]  # (height, width) of the original image


def crop_to_roi(image: np.ndarray, roi_mask: np.ndarray) -> RoiCrop:
    """
    Extract the tight bbox around roi_mask and black out pixels outside
    the container contour so downstream segmenters only see smoothie area.
    """
    h, w = image.shape[:2]
    roi_bool = roi_mask > 0

    if not roi_bool.any():
        crop = image.copy()
        crop[:] = 0
        return RoiCrop(crop, roi_mask.copy(), (0, 0), (h, w))

    ys, xs = np.where(roi_bool)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    crop_img = image[y1:y2, x1:x2].copy()
    crop_mask = roi_mask[y1:y2, x1:x2].copy()
    crop_img[crop_mask == 0] = 0

    return RoiCrop(crop_img, crop_mask, (x1, y1), (h, w))


def paste_mask(crop_mask: np.ndarray, roi_crop: RoiCrop) -> np.ndarray:
    """Map a crop-space binary mask back to full-frame coordinates."""
    full = np.zeros(roi_crop.full_shape, dtype=np.uint8)
    x, y = roi_crop.offset
    ch, cw = crop_mask.shape
    full[y : y + ch, x : x + cw] = crop_mask
    return full
