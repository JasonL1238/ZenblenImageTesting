"""
Shared, detector-agnostic helpers for container detection.

Colour-type classification, the straight-line top-edge prior, top-edge
roughness, and the ROI overlay. Detector-specific logic lives in
``classical.py`` / ``yolo.py``; the public ``detect_container`` dispatcher
lives in ``__init__.py``.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


class SmoothieType(Enum):
    RED_PINK = "red_pink"
    VIVID_YELLOW = "vivid_yellow"
    PALE_YELLOW = "pale_yellow"


def _classify_smoothie(image: np.ndarray) -> SmoothieType:
    """Classify smoothie colour type from center crop median LAB values."""
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    crop = image[cy - h // 6 : cy + h // 6, cx - w // 6 : cx + w // 6]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    med_a = float(np.median(lab[:, :, 1].astype(np.float32))) - 128.0
    med_b = float(np.median(lab[:, :, 2].astype(np.float32))) - 128.0

    if med_a > 8:
        return SmoothieType.RED_PINK
    elif med_b > 20:
        return SmoothieType.VIVID_YELLOW
    else:
        return SmoothieType.PALE_YELLOW


def _robust_line_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    iters: int = 3,
    k: float = 2.0,
) -> np.ndarray:
    """Fit y = m*x + c, iteratively rejecting outliers via MAD reweighting.

    Both upward spikes and downward notches in the top boundary are large
    residuals, so a few reweighting passes pull the line onto the dominant
    straight part of the edge. Returns polynomial coefficients ``[m, c]``.
    """
    coef = np.polyfit(xs, ys, 1)
    for _ in range(iters):
        resid = ys - np.polyval(coef, xs)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-6
        inliers = np.abs(resid - med) < k * 1.4826 * mad
        if inliers.sum() < 2:
            break
        coef = np.polyfit(xs[inliers], ys[inliers], 1)
    return coef


def _top_boundary(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (cols, top, bottom): occupied columns and their first/last fg rows.

    ``top`` and ``bottom`` are full-width arrays indexed by column. Returns None
    if fewer than two columns contain foreground.
    """
    h = mask.shape[0]
    fg = mask > 0
    cols = np.where(fg.any(axis=0))[0]
    if cols.size < 2:
        return None
    top = np.argmax(fg, axis=0)                      # first True from the top
    bottom = h - 1 - np.argmax(fg[::-1, :], axis=0)   # first True from the bottom
    return cols, top, bottom


def top_edge_roughness(mask: np.ndarray, win_frac: float = 0.06) -> float:
    """Quantify how 'squiggly' the top edge of a filled ROI mask is, in pixels.

    Measures *high-frequency* roughness: the per-column top boundary is detrended
    by subtracting a moving average (window ``win_frac`` of the occupied width),
    and the standard deviation of that residual is returned. This isolates true
    column-to-column sawtooth from benign low-frequency shape — a smoothly curved
    cup shoulder or a tilted surface scores near 0, while a jagged tan top scores
    high. (Deviation-from-a-straight-line metrics fail here because the rounded
    cup shoulders inflate them even when the edge is perfectly smooth.)

    A clean top scores ~0.5–2 px; a jagged sawtooth top scores ~3.5–6 px.
    Returns 0.0 for a degenerate mask.
    """
    boundary = _top_boundary(mask)
    if boundary is None:
        return 0.0
    cols, top, _ = boundary
    y = top[cols].astype(np.float64)
    n = len(y)
    win = max(5, int(n * win_frac) | 1)   # odd window
    pad = win // 2
    kernel = np.ones(win) / win
    smooth = np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")[:n]
    return float(np.std(y - smooth))


def flatten_roi_top(
    mask: np.ndarray,
    top_tilt_max_deg: float = 10.0,
) -> tuple[np.ndarray, BBox | None]:
    """Replace the jagged top boundary of a filled ROI mask with a straight line.

    The smoothie surface is physically a near-horizontal straight line, but
    per-image colour thresholding on low-saturation (tan/beige) fills produces a
    sawtooth top edge that oscillates around the true line — spiking up into
    foam/rim and cutting notches down into the smoothie.

    This extracts the topmost foreground row of every occupied column, fits a
    robust line that rejects those spikes/notches as outliers, then rebuilds the
    mask so each column is filled from the fitted line down to its original
    bottom. Spikes above the line are trimmed; notches below it are filled. The
    sides and bottom of the mask are preserved exactly (per-column bottom kept).

    Returns (flattened_mask, bbox). If the mask is empty, returns it unchanged
    with bbox None.
    """
    h = mask.shape[0]
    boundary = _top_boundary(mask)
    if boundary is None:
        return mask, None
    cols, top, bottom = boundary

    xs = cols.astype(np.float64)
    ys = top[cols].astype(np.float64)

    m, c = _robust_line_fit(xs, ys)

    # clamp tilt: |dy across the occupied width| limited by top_tilt_max_deg
    max_slope = np.tan(np.deg2rad(top_tilt_max_deg))
    if abs(m) > max_slope:
        m = max_slope if m > 0 else -max_slope
        # re-anchor c at the median so the clamped line still sits on the edge
        c = float(np.median(ys - m * xs))

    line_y = np.rint(m * xs + c).astype(int)
    line_y = np.clip(line_y, 0, h - 1)

    flat = np.zeros_like(mask)
    for i, x in enumerate(cols):
        yt = line_y[i]
        yb = int(bottom[x])
        if yt > yb:
            continue  # line sits below this column's content; nothing to keep
        flat[yt : yb + 1, x] = 255

    ys2, xs2 = np.where(flat > 0)
    if ys2.size == 0:
        return mask, None
    x0, x1 = int(xs2.min()), int(xs2.max())
    y0, y1 = int(ys2.min()), int(ys2.max())
    return flat, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def draw_container_overlay(image: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    """Return a copy of image with the detected ROI boundary drawn in green."""
    vis = image.copy()
    if roi_mask is None or not np.any(roi_mask):
        return vis
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return vis
