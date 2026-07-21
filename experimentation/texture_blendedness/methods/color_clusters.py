"""
Method: color clustering / segmentation.

A well-blended smoothie's liquid is essentially ONE dominant colour; an unblended
one carries substantial, spatially-COHERENT off-colour mass (a pale base plus
darker unmixed patches / streaks).

Pipeline:
  1. scored_region = ROI minus logo, border eroded (drops meniscus + gasket rim).
  2. cv2.kmeans on the LAB pixels of that region -> the largest cluster centre is
     the robust "dominant colour" (kmeans, not the mean, so real off-colour
     patches don't drag the reference).
  3. Per-pixel LAB ΔE to that dominant centre -> threshold = candidate off-colour.
  4. Morphological OPEN + connected-component size filter removes small SCATTERED
     components. This is what makes seeds / chia / berry-skin flecks (tiny,
     isolated dark speckle) NOT count — robustness comes from the spatial filter,
     not from clustering alone.
  5. score = 100 * (1 - k * coherent_off_fraction). More spatially-coherent
     off-colour mass -> lower score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import scored_region  # noqa: E402

NAME = "clusters"

BORDER_ERODE_PX = 16      # drop meniscus band + gasket rim
K_CLUSTERS = 5            # kmeans clusters used to find the dominant colour
MAX_SAMPLES = 20000       # cap pixels fed to kmeans (speed)
DELTA_E_THRESH = 24.0     # LAB ΔE from dominant centre -> candidate off-colour
OPEN_KERNEL_PX = 5        # morphological open: kills isolated speckle
MIN_COMPONENT_FRAC = 0.006  # drop off-colour blobs smaller than this frac of region
STRICTNESS = 2.4          # score = 100*(1 - k*coherent_off_fraction)
FLAG_FLOOR = 0.01         # ignore a tiny residual of coherent off-colour


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _dominant_center(samples: np.ndarray) -> np.ndarray:
    """Largest kmeans cluster centre in LAB (float32)."""
    k = min(K_CLUSTERS, len(np.unique(samples, axis=0)))
    if k < 2:
        return samples.mean(axis=0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        samples, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    counts = np.bincount(labels, minlength=k)
    return centers[int(np.argmax(counts))]


def score(image_bgr, roi_mask, logo_mask):
    h, w = image_bgr.shape[:2]
    flag01 = np.zeros((h, w), np.float32)
    region = scored_region(roi_mask, logo_mask, border_erode_px=BORDER_ERODE_PX, image_bgr=image_bgr)
    n_px = int(region.sum())
    if n_px < 200:
        return 100.0, flag01

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[region]                                   # (N,3)

    # kmeans on a subsample -> robust dominant colour
    if len(px) > MAX_SAMPLES:
        idx = np.random.RandomState(0).choice(len(px), MAX_SAMPLES, replace=False)
        samples = px[idx]
    else:
        samples = px
    center = _dominant_center(np.ascontiguousarray(samples, np.float32))

    # ΔE to dominant colour over the whole region
    delta_e = np.sqrt(((lab - center) ** 2).sum(axis=2))
    delta_e[~region] = 0.0

    binary = ((delta_e >= DELTA_E_THRESH) & region).astype(np.uint8)

    # open removes thin speckle; size filter removes scattered small blobs (seeds)
    ksz = _odd(OPEN_KERNEL_PX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    min_area = max(30, int(MIN_COMPONENT_FRAC * n_px))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    coherent = np.zeros((h, w), np.uint8)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            coherent[lbl == i] = 1
    coherent = (coherent > 0) & region

    frac = float(coherent.sum()) / n_px
    penalty = max(0.0, frac - FLAG_FLOOR)
    s = 100.0 * (1.0 - min(1.0, STRICTNESS * penalty))

    flag01[coherent] = np.clip(delta_e[coherent] / 30.0, 0.34, 1.0)
    return float(s), flag01
