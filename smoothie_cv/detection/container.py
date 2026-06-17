"""
Detect the smoothie region in a side-view image and return an ROI mask.

In this rig the cup sits in a stainless-steel blender housing: the smoothie is
the only strongly *saturated* object in frame (vivid color + the printed label),
while the machine is desaturated gray metal. So the primary strategy is color
saturation segmentation, which is robust to the cup being a tapered rectangle
rather than a circle.

Strategy:
  1. Saturation threshold (Otsu) in HSV → largest filled contour (primary).
  2. Full frame (last resort).

Returns a filled binary mask (255 inside the smoothie region, 0 outside) at the
same resolution as the input image, plus a bounding box (x, y, w, h) or None.
"""

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def detect_container(
    image: np.ndarray,
    min_area_frac: float = 0.05,
) -> tuple[np.ndarray, BBox | None]:
    """
    Args:
        image:             BGR image (H x W x 3, uint8)
        min_area_frac:     Reject saturation blobs smaller than this fraction of
                           the frame (filters stray colored specks).

    Returns:
        roi_mask:  H x W uint8 (255 inside detected region, 0 outside)
        bbox:      (x, y, w, h) bounding box of the region, or None on last resort
    """
    h, w = image.shape[:2]
    frame_area = h * w

    # --- primary: saturation segmentation ---
    mask, bbox = _saturation_roi(image, min_area_frac * frame_area)
    if mask is not None:
        return mask, bbox

    # --- last resort: full frame ---
    return np.full((h, w), 255, dtype=np.uint8), None


def _saturation_roi(image: np.ndarray, min_area: float) -> tuple[np.ndarray | None, BBox | None]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # Otsu adapts the saturated-vs-gray split per image (lighting varies).
    _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # close gaps (label text, highlights) then drop specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, None

    return _largest_filled(sat_mask)


def _largest_filled(mask: np.ndarray) -> tuple[np.ndarray | None, BBox | None]:
    """Keep only the largest contour in ``mask``, fill it, and return (filled, bbox)."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(largest)
    return filled, (x, y, bw, bh)


def draw_container_overlay(image: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    """Return a copy of image with the detected ROI boundary drawn in green."""
    vis = image.copy()
    if roi_mask is None or not np.any(roi_mask):
        return vis
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return vis
