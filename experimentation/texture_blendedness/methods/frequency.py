"""
Method: multi-scale frequency / band-pass energy.

A well-blended smoothie is smooth at MID spatial scales: its colour drifts only
gently across the cup. Unblended streaks/patches inject energy at scales LARGER
than a seed but SMALLER than the whole cup. Seeds, chia, and berry-skin flecks
are tiny (a few px) HIGH-frequency speckle — we deliberately smooth them away
below the band so they do not count.

How:
  * Work in LAB (L + a + b) so colour streaks count, not just luminance.
  * Band-pass = difference of two Gaussians, both computed with NORMALIZED
    (masked) convolution — blur(x*mask)/blur(mask) — so the ROI boundary and the
    excluded logo do NOT leak in as spurious edges.
      - SIGMA_LO (~4px) blur removes the finest scale => seeds/flecks are gone
        from the band.
      - SIGMA_HI (~26px) blur is the local baseline (whole-cup colour drift).
      - band = blur_lo - blur_hi  => energy at ~10-55px structures only.
  * Per-pixel mid-scale energy = L2 norm of the 3-channel band, gently robustified
    (sqrt) and confined to the eroded scored region.
  * Score = trimmed-mean band energy over the region, mapped 0-100 (more mid-scale
    energy -> lower score).

The border is eroded (drops meniscus reflection + gasket rim, which are strong
edges on even perfectly blended cups) and the logo is excluded via scored_region.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import scored_region  # noqa: E402

NAME = "frequency"

SIGMA_LO = 4.0           # below this scale (seeds/flecks/chia) is smoothed OUT
SIGMA_HI = 26.0          # above this scale (whole-cup drift) is the baseline
BORDER_ERODE_PX = 16     # drop meniscus band + gasket rim (strong false edges)

# score = 100 * (1 - clip((energy - E_LO) / (E_HI - E_LO)))
E_LO = 0.5               # energy at/below this -> ~100 (well blended)
E_HI = 11.0              # energy at/above this -> ~0 (badly unblended)
TRIM_PCT = 92.0          # aggregate = mean of energies up to this percentile
                         # (ignores the very brightest pixels = residual speckle)


def _norm_blur(img_f, region_f, sigma):
    """Normalized (masked) Gaussian blur: blur(img*mask)/blur(mask).
    Keeps the ROI boundary and logo hole from contaminating the local average."""
    num = cv2.GaussianBlur(img_f * region_f, (0, 0), sigma)
    den = cv2.GaussianBlur(region_f, (0, 0), sigma)
    return num / np.clip(den, 1e-6, None)


def score(image_bgr, roi_mask, logo_mask):
    h, w = image_bgr.shape[:2]
    flag01 = np.zeros((h, w), np.float32)
    region = scored_region(roi_mask, logo_mask, border_erode_px=BORDER_ERODE_PX, image_bgr=image_bgr)
    n_px = int(region.sum())
    if n_px < 50:
        return 100.0, flag01

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    region_f = region.astype(np.float32)

    # Per-channel mid-scale band via normalized DoG.
    energy = np.zeros((h, w), np.float32)
    for c in range(3):
        ch = lab[:, :, c]
        lo = _norm_blur(ch, region_f, SIGMA_LO)
        hi = _norm_blur(ch, region_f, SIGMA_HI)
        band = lo - hi
        energy += band * band
    energy = np.sqrt(energy)          # L2 magnitude of the 3-channel band
    energy[~region] = 0.0

    vals = energy[region]
    # Trimmed mean: ignore the top tail (rare residual speckle / tiny edges).
    cap = np.percentile(vals, TRIM_PCT)
    agg = float(vals[vals <= cap].mean())

    s = 100.0 * (1.0 - np.clip((agg - E_LO) / (E_HI - E_LO), 0.0, 1.0))

    # Overlay: normalized mid-scale energy map inside the region.
    flag01[region] = np.clip(energy[region] / E_HI, 0.0, 1.0)
    return float(s), flag01
