"""
Method: intensity-WEIGHTED local-deviation (dev_area's sibling).

dev_area scores by the *count* of off-colour pixels: a pixel just past the
threshold counts the same as a violently unmixed one. This method keeps the same
spatial machinery (blurred local baseline -> ΔE -> threshold -> morph-open to
delete seed/chia/skin speckle) but scores the surviving defect pixels by their
ΔE *magnitude*, so a strong deviation is more powerful than a weak one.

Grounded in the published "uniformity index" for blend quality
(U = 1 - Σ|Ci - C̄| / (2N·C̄), pharmacyinfoline.com) and the coefficient-of-
variation mixing index (std/mean, statiflo.com) — both continuous magnitude
measures, not thresholded counts. Here the magnitude is summed only over pixels
that survive the seed-rejecting morphological open, so recipe speckle still
doesn't contribute.

    penalty = (1/N_region) · Σ_survived  clip(ΔE, 0, ΔE_MAX) / ΔE_MAX
    score   = 100 · (1 - min(1, STRICTNESS · max(0, penalty - FLAG_FLOOR)))
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import scored_region  # noqa: E402

NAME = "dev_weighted"

BLUR_KERNEL = 121
DELTA_E_THRESH = 6.0      # below this = smooth, contributes nothing
DELTA_E_MAX = 25.0        # ΔE at/above which a pixel contributes full weight
OPEN_KERNEL_PX = 7        # delete seed/chia/skin speckle (size separator)
BORDER_ERODE_PX = 18      # drop meniscus band + gasket rim
STRICTNESS = 4.5          # score = 100*(1 - k*intensity_weighted_fraction)
FLAG_FLOOR = 0.01         # ignore a small baseline of residual flecks


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def score(image_bgr, roi_mask, logo_mask):
    h, w = image_bgr.shape[:2]
    flag01 = np.zeros((h, w), np.float32)
    region = scored_region(roi_mask, logo_mask, border_erode_px=BORDER_ERODE_PX, image_bgr=image_bgr)
    n_px = int(region.sum())
    if n_px == 0:
        return 100.0, flag01

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = _odd(BLUR_KERNEL)
    region_f = region.astype(np.float32)
    weight = np.clip(cv2.GaussianBlur(region_f, (k, k), 0), 1e-6, None)
    baseline = np.zeros_like(lab)
    for c in range(3):
        baseline[:, :, c] = cv2.GaussianBlur(lab[:, :, c] * region_f, (k, k), 0) / weight

    delta_e = np.sqrt(((lab - baseline) ** 2).sum(axis=2))
    delta_e[~region] = 0.0

    # spatial seed-rejection: same threshold + open as dev_area, used as a MASK
    binary = (delta_e >= DELTA_E_THRESH).astype(np.uint8)
    ksz = _odd(OPEN_KERNEL_PX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    survived = (binary > 0) & region

    # intensity-weighted: each survivor contributes its ΔE magnitude, not 1
    sev = np.clip(delta_e, 0.0, DELTA_E_MAX) / DELTA_E_MAX
    penalty = float(sev[survived].sum()) / n_px
    penalty = max(0.0, penalty - FLAG_FLOOR)
    s = 100.0 * (1.0 - min(1.0, STRICTNESS * penalty))

    flag01[survived] = np.clip(sev[survived], 0.34, 1.0)
    return s, flag01
