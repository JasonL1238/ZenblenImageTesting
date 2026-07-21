"""
Method: LOCAL COLOR ENTROPY.

A well-blended smoothie is locally one colour: inside any neighbourhood the
colour histogram is a single spike -> near-zero Shannon entropy. An unblended
cup has streaks/patches of a *different* colour, so a neighbourhood straddling a
boundary contains two (or more) colours -> high local entropy.

Design:
  * Work in LAB a*/b* (chroma-hue) only. L* is dropped so that shading, glare and
    the dark speckle of seeds/chia/berry-skin (which are mostly a LUMINANCE
    event, near-neutral in a*/b*) do not drive the histogram.
  * Coarsely quantise a*/b* into a small joint palette (NB x NB bins). Coarse
    bins mean a handful of stray speckle pixels rarely open a *new* colour bin,
    while a real streak of a genuinely different colour does.
  * Compute masked local Shannon entropy over a window (WIN px) that is much
    larger than a seed, using per-bin box-filter counts normalised by the count
    of ROI pixels actually inside the window. A lone seed contributes a tiny
    fraction of one window and so a tiny amount of entropy; a colour boundary
    fills a large share of the window with a second bin -> big entropy.
  * Score by the fraction of the scored region whose local entropy exceeds a
    floor, so isolated speckle (small area) barely counts but broad non-uniform
    regions or long streaks (large area) tank the score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import scored_region  # noqa: E402

NAME = "entropy"

BORDER_ERODE_PX = 16     # drop meniscus band / gasket rim
BIN = 5                  # a*/b* quantisation step in LAB units (~fine palette)
WIN = 49                 # local window (px) -- much larger than a seed
MIN_FILL = 0.5           # window must be >= this fraction ROI to be scored
ENT_FLOOR = 0.9          # entropy (nats) below this = "locally uniform" (weight 0)
ENT_SPAN = 1.2           # entropy above FLOOR by this much -> full weight (1.0)
GAMMA = 2.0              # deviation-intensity emphasis: >1 makes strongly-mixed
                         #   neighbourhoods dominate faint ones; 1.0 = linear
AREA_FLOOR = 0.10        # ignore this baseline weighted fraction of flecks / noise
STRICTNESS = 1.25        # score = 100*(1 - k * intensity-weighted area fraction)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def score(image_bgr, roi_mask, logo_mask):
    h, w = image_bgr.shape[:2]
    flag01 = np.zeros((h, w), np.float32)
    region = scored_region(roi_mask, logo_mask, border_erode_px=BORDER_ERODE_PX, image_bgr=image_bgr)
    n_px = int(region.sum())
    if n_px == 0:
        return 100.0, flag01

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int32)
    b = lab[:, :, 2].astype(np.int32)

    # Fine joint quantisation of a*/b* (step = BIN LAB units). a*/b* only, so the
    # near-neutral dark speckle of seeds does not open a distinct colour bin.
    nb = 256 // BIN + 1
    joint = ((a // BIN) * nb + (b // BIN)).astype(np.int32)

    region_f = region.astype(np.float32)
    win = _odd(WIN)
    # count of ROI pixels inside each window
    total = cv2.boxFilter(region_f, ddepth=cv2.CV_32F, ksize=(win, win),
                          normalize=False)
    total_safe = np.clip(total, 1e-6, None)

    # Masked local Shannon entropy: for every occupied colour bin, box-filter the
    # in-region occurrence map to get its local count, then accumulate -p*log(p).
    entropy = np.zeros((h, w), np.float32)
    present = np.unique(joint[region])
    for bin_id in present:
        occ = ((joint == bin_id) & region).astype(np.float32)
        cnt = cv2.boxFilter(occ, ddepth=cv2.CV_32F, ksize=(win, win),
                            normalize=False)
        p = cnt / total_safe
        nz = p > 1e-6
        entropy[nz] -= p[nz] * np.log(p[nz])

    valid = region & (total >= MIN_FILL * (win * win))
    if not valid.any():
        return 100.0, flag01

    # Per-pixel graded unblendedness: 0 at ENT_FLOOR, ramps to 1 at
    # ENT_FLOOR + ENT_SPAN. This replaces the old hard >=ENT_HI cutoff, so the
    # *magnitude* of local mixing now matters, not just whether it cleared a line.
    unblend = np.clip((entropy - ENT_FLOOR) / ENT_SPAN, 0.0, 1.0)
    unblend[~valid] = 0.0

    # Emphasise INTENSE deviations: GAMMA>1 pushes faint/borderline mixing toward
    # 0 while a strongly-mixed neighbourhood keeps ~full weight, so a small patch
    # of very high entropy can outweigh a large wash of borderline entropy.
    # (--- FOLD IN COLOUR-DISTANCE HERE: multiply `weight` by a per-pixel ΔE-from-
    #  dominant-colour term if you want "how *different*" to matter too, not just
    #  "how mixed". ---)
    weight = unblend ** GAMMA

    # Score: intensity-weighted "unblended area" fraction. Isolated speckle is a
    # tiny weighted area and is absorbed by AREA_FLOOR.
    frac = float(weight[valid].sum()) / max(1, int(valid.sum()))
    penalty = max(0.0, frac - AREA_FLOOR)
    s = 100.0 * (1.0 - min(1.0, STRICTNESS * penalty))

    # Overlay uses the linear (pre-GAMMA) grade so the heatmap stays readable.
    high = valid & (unblend > 0.0)
    flag01[high] = np.clip(unblend[high], 0.34, 1.0)
    return float(s), flag01
