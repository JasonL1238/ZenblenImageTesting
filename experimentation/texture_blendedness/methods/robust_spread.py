"""
Method: robust global color-spread statistic (spatially-blind baseline).

A well-blended smoothie has a TIGHT, unimodal color distribution across the
whole liquid; an unblended one (streaks, clumps, two-tone) has a WIDE spread.
This method measures the global per-channel dispersion of the LAB colors over
the whole scored ROI — no local windows, no spatial structure at all.

To stay robust to seeds / chia / berry-skin flecks (a small fraction of dark
outlier pixels), dispersion is measured with ROBUST statistics:
  * a percentile-TRIMMED standard deviation (trim the top/bottom TRIM_FRAC of
    each channel) — kills the isolated outlier speckle, and
  * the MAD (median absolute deviation, scaled to a std) as a second, even more
    outlier-proof estimate; the two are averaged per channel.

Lightness (L) and chroma (a,b) spreads are combined with separate weights —
chroma streaks are the strongest unblended signal, so a,b carry more weight.
The combined dispersion is mapped linearly (SPREAD_GOOD -> 100, SPREAD_BAD -> 0).

flag01 (overlay only): per-pixel LAB distance from the ROI robust-median color,
normalized — so the overlay still shows where the global spread comes from even
though the score itself is spatially blind.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import scored_region  # noqa: E402

NAME = "robust_spread"

BORDER_ERODE_PX = 16      # drop meniscus band (top) + gasket rim (bottom)
TRIM_FRAC = 0.08          # trim this fraction off EACH tail before the std
W_L = 0.15                # weight on lightness spread (low: lighting gradients
                          #   give good cups real L spread; chroma is the clean cue)
W_AB = 1.0                # weight on chroma (a,b) spread — the strongest cue
# Linear map from combined dispersion to score.
SPREAD_GOOD = 3.5         # dispersion at/below this -> 100 (well blended)
SPREAD_BAD = 10.0         # dispersion at/above this -> 0 (badly unblended)
FLAG_NORM = 22.0          # LAB distance mapped to 1.0 in the overlay


def _trimmed_std(x: np.ndarray, trim: float) -> float:
    """Std of x after discarding the lowest and highest `trim` fraction."""
    if x.size < 8:
        return float(x.std()) if x.size else 0.0
    lo, hi = np.percentile(x, [100.0 * trim, 100.0 * (1.0 - trim)])
    core = x[(x >= lo) & (x <= hi)]
    if core.size < 2:
        return float(x.std())
    return float(core.std())


def _robust_std(x: np.ndarray, trim: float) -> float:
    """Average of a trimmed std and a MAD-based std — both outlier-resistant."""
    tstd = _trimmed_std(x, trim)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad_std = 1.4826 * float(mad)
    return 0.5 * (tstd + mad_std)


def score(image_bgr, roi_mask, logo_mask):
    h, w = image_bgr.shape[:2]
    flag01 = np.zeros((h, w), np.float32)
    region = scored_region(roi_mask, logo_mask, border_erode_px=BORDER_ERODE_PX, image_bgr=image_bgr)
    if int(region.sum()) < 50:
        return 100.0, flag01

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0][region]
    A = lab[:, :, 1][region]
    B = lab[:, :, 2][region]

    sL = _robust_std(L, TRIM_FRAC)
    sA = _robust_std(A, TRIM_FRAC)
    sB = _robust_std(B, TRIM_FRAC)
    chroma_spread = np.sqrt(sA * sA + sB * sB)

    dispersion = np.sqrt(W_L * sL * sL + W_AB * chroma_spread * chroma_spread)

    t = (dispersion - SPREAD_GOOD) / (SPREAD_BAD - SPREAD_GOOD)
    s = 100.0 * (1.0 - float(np.clip(t, 0.0, 1.0)))

    # Overlay: per-pixel LAB distance from the ROI robust-median color.
    med = np.array([np.median(L), np.median(A), np.median(B)], np.float32)
    dist = np.sqrt(((lab - med) ** 2).sum(axis=2))
    fl = np.clip(dist / FLAG_NORM, 0.0, 1.0).astype(np.float32)
    flag01[region] = fl[region]
    return s, flag01
