"""
Method: local-deviation flagged-AREA fraction (the current/baseline method).

Per-pixel LAB ΔE from a heavily-blurred local baseline -> threshold ->
morphological open (drops seed/chia speckle) -> score by the FRACTION of the
scored region that stays flagged.

Tuning vs. the first version: the scored region has its BORDER ERODED so the
meniscus reflection band (top) and gasket rim (bottom) — which flag on nicely
blended cups — are excluded. Bad cups have clumps in the CENTRE, so erosion
barely touches them but rescues good cups.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import scored_region  # noqa: E402

NAME = "dev_area"

BLUR_KERNEL = 121
DELTA_E_THRESH = 6.0
OPEN_KERNEL_PX = 7
BORDER_ERODE_PX = 18      # drop meniscus band + gasket rim
STRICTNESS = 2.2         # score = 100*(1 - k*flagged_fraction)
FLAG_FLOOR = 0.03        # ignore a small baseline of residual flecks


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

    binary = (delta_e >= DELTA_E_THRESH).astype(np.uint8)
    ksz = _odd(OPEN_KERNEL_PX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = (binary > 0) & region

    frac = float(binary.sum()) / n_px
    penalty = max(0.0, frac - FLAG_FLOOR)
    s = 100.0 * (1.0 - min(1.0, STRICTNESS * penalty))

    # flag01 for the overlay: normalise ΔE where flagged
    flag01[binary] = np.clip(delta_e[binary] / 25.0, 0.34, 1.0)
    return s, flag01
