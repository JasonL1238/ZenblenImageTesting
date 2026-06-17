"""
Adapt a frozen ROI mask to small per-image changes in a fixed-rig smoothie setup.

The cup is a rigid object held in a stationary blender housing, so the only
legitimate per-image variation is a small position shift and a small apparent
size change. Rather than re-detecting the container from scratch (the saturation
segmentation in ``container.py`` is flaky on pale/foamy smoothies), we take a
known-good frozen mask as a prior and flex it to each frame.

Two techniques, both returning the same contract as ``detect_container``:
  - ``adaptive_roi_transform``: rigid translate + uniform scale of the whole mask.
  - ``adaptive_roi_grabcut``:    snap the boundary to the real cup edge via grabCut.

Both carry a hard "never worse than the static frozen mask" guard: if the adapted
result looks implausible, they return the unmodified frozen mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smoothie_cv.detection.container import BBox, _largest_filled


def load_frozen_mask(path: str | Path) -> np.ndarray | None:
    """Load a saved frozen ROI mask as a single-channel uint8 image, or None."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask


def _bbox_of(mask: np.ndarray) -> BBox | None:
    """Bounding box (x, y, w, h) of the nonzero region, or None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# --------------------------------------------------------------------------- #
# Method 1: constrained similarity transform (rigid translate + uniform scale)
# --------------------------------------------------------------------------- #
def adaptive_roi_transform(
    image: np.ndarray,
    frozen_mask: np.ndarray,
    frozen_bbox: BBox | None = None,
    dx_max: int = 30,
    dy_max: int = 30,
    s_range: tuple[float, float] = (0.92, 1.12),
    conf_min: float = 0.25,
) -> tuple[np.ndarray, BBox | None, dict]:
    """Slide + uniformly scale the frozen mask to match this frame's cup.

    Returns (roi_mask uint8 255-inside, bbox, info). Falls back to the unwarped
    frozen mask (info["accepted"] = False) when evidence is too weak.
    """
    info: dict = {"method": "transform", "dx": 0, "dy": 0, "s": 1.0,
                  "conf": 0.0, "accepted": False}

    if frozen_mask.shape[:2] != image.shape[:2]:
        return frozen_mask, _bbox_of(frozen_mask), info

    h, w = frozen_mask.shape[:2]
    if frozen_bbox is None:
        frozen_bbox = _bbox_of(frozen_mask)
    if frozen_bbox is None:
        return frozen_mask, None, info
    fx, fy, fw, fh = frozen_bbox
    cx, cy = fx + fw / 2.0, fy + fh / 2.0

    # --- evidence map inside a ~30px band around the frozen boundary ---
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dx_max + 1, 2 * dy_max + 1))
    band = cv2.dilate(frozen_mask, band_k)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)

    def _norm_in_band(arr: np.ndarray) -> np.ndarray:
        vals = arr[band > 0]
        if vals.size == 0:
            return np.zeros_like(arr)
        lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
        if hi <= lo:
            return np.zeros_like(arr)
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        out[band == 0] = 0.0
        return out

    evidence = 0.7 * _norm_in_band(edge) + 0.3 * _norm_in_band(sat)

    # --- translation via boundary-template matching within +/- (dx_max, dy_max) ---
    boundary = cv2.morphologyEx(
        frozen_mask, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(np.float32) / 255.0
    boundary = cv2.GaussianBlur(boundary, (0, 0), sigmaX=3)

    templ = boundary[dy_max:h - dy_max, dx_max:w - dx_max]
    dx = dy = 0
    conf = 0.0
    if templ.size and templ.shape[0] > 0 and templ.shape[1] > 0:
        res = cv2.matchTemplate(evidence.astype(np.float32), templ.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        _, conf, _, max_loc = cv2.minMaxLoc(res)
        dx = int(max_loc[0] - dx_max)
        dy = int(max_loc[1] - dy_max)
    info["conf"] = round(float(conf), 3)

    # --- uniform scale from the evidence extent inside the band ---
    col_profile = evidence.sum(axis=0)
    row_profile = evidence.sum(axis=1)

    def _extent(profile: np.ndarray) -> int:
        if profile.max() <= 0:
            return 0
        idx = np.where(profile > 0.15 * profile.max())[0]
        return int(idx.max() - idx.min() + 1) if idx.size else 0

    est_w, est_h = _extent(col_profile), _extent(row_profile)
    scales = [est_w / fw, est_h / fh]
    scales = [s for s in scales if s > 0]
    s = float(np.median(scales)) if scales else 1.0
    s = float(np.clip(s, s_range[0], s_range[1]))

    dx = int(np.clip(dx, -dx_max, dx_max))
    dy = int(np.clip(dy, -dy_max, dy_max))
    info.update({"dx": dx, "dy": dy, "s": round(s, 3)})

    # weak evidence -> keep the frozen mask untouched
    if conf < conf_min:
        return frozen_mask, frozen_bbox, info

    # --- warp: scale about the frozen center, then translate ---
    m = np.array([
        [s, 0.0, cx - s * cx + dx],
        [0.0, s, cy - s * cy + dy],
    ], dtype=np.float32)
    warped = cv2.warpAffine(frozen_mask, m, (w, h), flags=cv2.INTER_NEAREST,
                            borderValue=0)
    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

    filled, bbox = _largest_filled(warped)
    if filled is None:
        return frozen_mask, frozen_bbox, info

    info["accepted"] = True
    return filled, bbox, info


# --------------------------------------------------------------------------- #
# Method 2: prior-seeded grabCut (snap boundary to the real cup edge)
# --------------------------------------------------------------------------- #
def adaptive_roi_grabcut(
    image: np.ndarray,
    frozen_mask: np.ndarray,
    erode_ksize: int = 25,
    dilate_ksize: int = 41,
    iter_count: int = 5,
    max_area_ratio: float = 1.6,
    min_area_ratio: float = 0.6,
    min_iou: float = 0.55,
) -> tuple[np.ndarray, BBox | None, dict]:
    """Snap the frozen prior to this frame's cup boundary via seeded grabCut.

    Returns (roi_mask uint8 255-inside, bbox, info). Falls back to the raw frozen
    mask (info["accepted"] = False) when the result fails the area/IoU guard.
    """
    info: dict = {"method": "grabcut", "area_ratio": 1.0, "iou": 1.0,
                  "accepted": False}

    if frozen_mask.shape[:2] != image.shape[:2]:
        return frozen_mask, _bbox_of(frozen_mask), info

    frozen_bin = (frozen_mask > 0).astype(np.uint8)
    frozen_area = int(frozen_bin.sum())
    if frozen_area == 0:
        return frozen_mask, None, info

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    sure_fg = cv2.erode(frozen_mask, k_erode)
    dil = cv2.dilate(frozen_mask, k_dilate)

    # 4-state seed mask (write sure states last so they win)
    gc = np.full(frozen_mask.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    gc[frozen_mask > 0] = cv2.GC_PR_FGD
    gc[dil == 0] = cv2.GC_BGD
    gc[sure_fg > 0] = cv2.GC_FGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, gc, None, bgd, fgd, iter_count, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return frozen_mask, _bbox_of(frozen_mask), info

    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    filled, bbox = _largest_filled(fg)
    if filled is None:
        return frozen_mask, _bbox_of(frozen_mask), info

    adapted_bin = (filled > 0).astype(np.uint8)
    adapted_area = int(adapted_bin.sum())
    inter = int((adapted_bin & frozen_bin).sum())
    union = int((adapted_bin | frozen_bin).sum())
    area_ratio = adapted_area / frozen_area if frozen_area else 0.0
    iou = inter / union if union else 0.0
    info.update({"area_ratio": round(area_ratio, 3), "iou": round(iou, 3)})

    if (area_ratio > max_area_ratio or area_ratio < min_area_ratio or iou < min_iou):
        return frozen_mask, _bbox_of(frozen_mask), info

    info["accepted"] = True
    return filled, bbox, info
