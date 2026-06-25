"""
Shared, detector-agnostic helpers for container detection.

Everything here is used by *both* the classical and the SAM detectors (or by the
runner): colour-type classification, the straight-line top-edge prior, top-edge
roughness, largest-blob filling, and the ROI overlay. Detector-specific logic
lives in ``classical.py`` / ``sam.py``; the public ``detect_container`` dispatcher
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


def refine_cup_sides(mask: np.ndarray, win_frac: float = 0.06) -> np.ndarray:
    """Straighten jagged side walls of a filled cup ROI mask.

    A cup's horizontal cross-section at any height is a single convex span, so the
    per-row left/right boundary should vary smoothly down the cup. But printed logo
    text and low SAM confidence on dark/maroon fills make SAM's side boundary
    scallop inward at each letter and bulge out at others; that ragged edge also
    drags thin dark slivers (cup-wall text, gasket shadow) into the ROI where they
    misfire the chunk detector (e.g. the spurious ~95px "chunk" on 1fd0).

    For each occupied row this replaces the left and right wall x with the *median*
    wall position over a vertical window (``win_frac`` of the occupied height), then
    refills the row between the smoothed walls. The median is robust to inward
    notches *and* outward bulges, and — unlike a min/max envelope — it can never
    push the wall past the surrounding *real* wall positions, so it does not invent
    coverage outside the true cup. A clean wall is left essentially unchanged
    (median of a smooth signal ≈ itself). Top and bottom rows are preserved; the
    straight-line top prior (``flatten_roi_top``) still handles the top edge.
    """
    boundary = _top_boundary(mask)
    if boundary is None:
        return mask
    rows = np.where((mask > 0).any(axis=1))[0]
    if rows.size < 3:
        return mask
    left = np.array([np.argmax(mask[y] > 0) for y in rows], dtype=np.float64)
    right = np.array([mask.shape[1] - 1 - np.argmax(mask[y][::-1] > 0) for y in rows],
                     dtype=np.float64)
    win = max(3, int(rows.size * win_frac) | 1)
    pad = win // 2
    out = np.zeros_like(mask)
    for idx, y in enumerate(rows):
        a, b = max(0, idx - pad), min(rows.size, idx + pad + 1)
        ls = int(np.median(left[a:b]))
        rs = int(np.median(right[a:b]))
        if rs >= ls:
            out[y, ls : rs + 1] = 255
    return out


def extend_roi_to_gasket(
    mask: np.ndarray,
    image: np.ndarray,
    max_frac: float = 0.25,
    dark_drop: float = 0.55,
) -> np.ndarray:
    """Extend a cup ROI mask down to the holder gasket (fixed-rig bottom prior).

    On dark/maroon fills the lower smoothie blends into the shadowed holder, so SAM
    stops mid-cup and a large bottom chunk (often a pale-cream unblended mass) is
    left outside the ROI — the cup then scores falsely clean. In this fixed rig the
    cup always sits in a dark holder gasket, so the *true* cup bottom is the top of
    that gasket: a dark horizontal band spanning the cup's width directly below the
    fill.

    This scans downward from the mask's current bottom (within the cup's central
    column span) for the first row whose median luminance drops below ``dark_drop``
    of the cup's own bottom-band brightness — the gasket. Each column is then filled
    down to that gasket row, stopping early at any dark pixel so dark gaps are not
    filled. The extension is GATED on actually finding the dark gasket within
    ``max_frac`` of the cup height: a cup SAM already segmented to its true bottom
    has only bright metal/reflection below (no dark band found) and is left
    unchanged, so correctly-segmented cups never over-extend onto the holder.
    """
    boundary = _top_boundary(mask)
    if boundary is None:
        return mask
    L = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    y_top, y_bot = int(ys.min()), int(ys.max())
    x_lo, x_hi = int(xs.min()), int(xs.max())
    roi_h = max(y_bot - y_top, 1)
    # cup brightness reference = median L of the cup's bottom 15% band
    band = mask.copy()
    band[: y_bot - int(0.15 * roi_h), :] = 0
    if not (band > 0).any():
        return mask
    ref_L = float(np.median(L[band > 0]))
    dark_thr = dark_drop * ref_L
    # central column span (avoid ragged side edges biasing the row median)
    cx_lo = x_lo + int(0.10 * (x_hi - x_lo))
    cx_hi = x_hi - int(0.10 * (x_hi - x_lo))
    if cx_hi <= cx_lo:
        return mask
    max_extend = int(max_frac * roi_h)
    # find the gasket row: first row below the fill whose central median L goes dark
    gasket_row = None
    for y in range(y_bot + 1, min(y_bot + max_extend + 1, h)):
        if float(np.median(L[y, cx_lo : cx_hi + 1])) < dark_thr:
            gasket_row = y
            break
    if gasket_row is None:        # no gasket below => already at true bottom; leave as-is
        return mask
    out = mask.copy()
    cols = np.where(mask.any(axis=0))[0]
    for x in cols:
        yb = int(np.where(mask[:, x] > 0)[0].max())
        for y in range(yb + 1, gasket_row):
            if L[y, x] < dark_thr:
                break             # hit the gasket / a dark gap in this column
            out[y, x] = 255
    return out


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
