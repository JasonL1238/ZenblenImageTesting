"""
Texture-based blendedness score (chunk-independent).

Idea: a poorly-blended smoothie has spatially CONTIGUOUS colour variation
(streaks, unmixed patches, marbling). Seeds / chia / blueberry skin are small,
isolated speckle — normal recipe texture, not a blend defect. Size is the
separator.

Method:
  1. LAB-convert the liquid ROI.
  2. Build a heavily-blurred local baseline colour (large kernel).
  3. Per-pixel deviation = LAB ΔE between pixel and its local baseline.
  4. Morphologically OPEN the thresholded deviation map with a kernel just
     larger than a seed/fleck — this erases isolated small blobs (seeds/skin)
     while keeping deviation that persists over a larger contiguous area
     (streaks/patches).
  5. Aggregate the surviving deviation into a 0-100% uniformity score.

Self-contained — no imports from active_pipeline. This is the experiment copy;
if validated it gets ported into active_pipeline/smoothie_cv/scoring/texture.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextureParams:
    # Local-baseline blur: large kernel so the baseline is the "should-be"
    # smooth colour, and any real texture stands out against it.
    blur_kernel: int = 121
    # Deviation threshold (LAB ΔE). Pixels below this are treated as blended.
    delta_e_thresh: float = 6.0
    # Morphological opening kernel (px). Set just above a seed/fleck diameter so
    # seeds/chia/skin are erased but larger streaks survive. 0 disables opening.
    open_kernel_px: int = 7
    # ΔE value that maps to score 0 (fully unblended).
    delta_e_max: float = 25.0
    # --- scoring aggregation ---
    #   "area" : penalty = flagged-area fraction of ROI (recommended, strict)
    #   "sev"  : penalty = mean surviving ΔE / delta_e_max (severity-weighted)
    #   "mean" : legacy alias of "sev"
    agg: str = "area"
    # Strictness multiplier applied to the penalty before mapping to a score.
    # Higher = harsher. score = 100 * (1 - min(1, k * penalty)).
    # k=2 with agg="area": ~8% flagged (clean) -> ~84, ~47% flagged -> ~5.
    strictness: float = 2.0


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def deviation_map(
    image: np.ndarray,
    roi_mask: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    params: TextureParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (surviving_delta_e_map, region_bool). The ΔE map is 0 outside the
    surviving flagged regions. Separated from scoring so a tuning harness can try
    many score mappings on the SAME map without re-running the models."""
    p = params or TextureParams()
    h, w = image.shape[:2]

    region = (roi_mask > 0)
    if exclude_mask is not None:
        region &= ~(exclude_mask > 0)

    dev_map = np.zeros((h, w), dtype=np.float32)
    if int(region.sum()) == 0:
        return dev_map, region

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Masked local baseline: blur only over ROI pixels so out-of-ROI colour
    # (steel, background) doesn't bleed into the baseline near the edges.
    k = _odd(p.blur_kernel)
    region_f = region.astype(np.float32)
    weight = cv2.GaussianBlur(region_f, (k, k), 0)
    weight = np.clip(weight, 1e-6, None)
    baseline = np.zeros_like(lab)
    for c in range(3):
        ch = lab[:, :, c] * region_f
        baseline[:, :, c] = cv2.GaussianBlur(ch, (k, k), 0) / weight

    # Per-pixel LAB ΔE (Euclidean in LAB — good enough for a relative score).
    delta_e = np.sqrt(((lab - baseline) ** 2).sum(axis=2))
    delta_e[~region] = 0.0

    # Threshold, then (optionally) morphological OPEN to drop tiny speckle.
    binary = (delta_e >= p.delta_e_thresh).astype(np.uint8)
    if p.open_kernel_px and p.open_kernel_px > 1:
        ksz = _odd(p.open_kernel_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    dev_map = (delta_e * (binary > 0) * region).astype(np.float32)
    return dev_map, region


def score_from_map(
    dev_map: np.ndarray, region: np.ndarray, params: TextureParams | None = None
) -> float:
    """Map a surviving-ΔE map to a 0-100 blendedness score."""
    p = params or TextureParams()
    n_px = int(region.sum())
    if n_px == 0:
        return 100.0
    if p.agg == "area":
        penalty = float((dev_map > 0).sum()) / n_px       # flagged-area fraction
    else:  # "sev" / "mean"
        penalty = float(dev_map[region].mean()) / p.delta_e_max
    return 100.0 * (1.0 - min(1.0, p.strictness * penalty))


def compute_texture_score(
    image: np.ndarray,
    roi_mask: np.ndarray,
    exclude_mask: np.ndarray | None = None,
    params: TextureParams | None = None,
) -> tuple[float, np.ndarray]:
    """(score_0_100, surviving_deviation_map). 100 = fully blended, 0 = worst."""
    dev_map, region = deviation_map(image, roi_mask, exclude_mask, params)
    return score_from_map(dev_map, region, params), dev_map


def overlay_deviation(
    image: np.ndarray,
    dev_map: np.ndarray,
    delta_e_max: float = 25.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Filled heatmap overlay of the surviving deviation."""
    norm = np.clip(dev_map / delta_e_max, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat[dev_map <= 0] = 0
    vis = image.astype(np.float32)
    m = (dev_map > 0)[:, :, None]
    blended = np.where(
        m,
        cv2.addWeighted(image, 1 - alpha, heat, alpha, 0).astype(np.float32),
        vis,
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def outline_deviation(
    image: np.ndarray,
    dev_map: np.ndarray,
    mild_thresh: float = 12.0,
) -> np.ndarray:
    """See-through overlay: OUTLINE the flagged regions instead of filling them,
    so the smoothie underneath stays fully visible.

    Yellow contour = mild unblended patch, red contour = strong. Only the
    boundary is drawn — the interior pixels are untouched.
    """
    vis = image.copy()
    if not (dev_map > 0).any():
        return vis
    flagged = (dev_map > 0).astype(np.uint8)
    severe = (dev_map >= mild_thresh).astype(np.uint8)

    cnts, _ = cv2.findContours(flagged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)      # yellow: all flagged
    cnts_s, _ = cv2.findContours(severe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts_s, -1, (0, 0, 255), 2)      # red: strong deviation
    return vis
