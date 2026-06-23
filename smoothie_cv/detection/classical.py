"""
Classical colour-threshold container detector — the FALLBACK detector.

In this rig the cup sits in a stainless-steel blender housing: the smoothie is
the only strongly *chromatic* object in frame (vivid color), while the machine
body is achromatic (white, black, gray metal).

Strategy — per-colour-type branching:
  1. Sample center crop to classify smoothie colour type.
  2. RED/PINK (a* > 8): warm-biased LAB chroma + Otsu.
  3. YELLOW (pale or vivid): b*-channel threshold → refine_yellow_roi.
  4. Full frame (last resort).

Yellow strategy:
  The coarse b*-threshold mask gives a geometric cup outline. refine_yellow_roi
  uses that outline as a shape constraint only, then isolates actual smoothie
  pixels inside it via adaptive LAB thresholds (b* channel, a* cap, L* glare
  cutoff, chroma floor) and an inward erosion to drop cup-edge plastic and
  border reflections.

All branches apply a spatial crop (configurable margins) to exclude blender
side panels and motor base before colour analysis, then translate back.

This detector keys on colour, so it is fragile across shades (it can miss
tan/pale fills and occasionally bleeds onto the machine deck) — which is why the
SAM detector in ``sam.py`` is preferred. Shared geometry/colour helpers live in
``common.py``; the SAM→classical dispatcher lives in ``__init__.py``.

Returns a filled binary mask (255 inside the smoothie region, 0 outside) at the
same resolution as the input image, plus a bounding box (x, y, w, h) or None.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from smoothie_cv.detection.common import (
    BBox,
    SmoothieType,
    _classify_smoothie,
    flatten_roi_top,
    top_edge_roughness,
)


@dataclass(frozen=True)
class YellowRefineParams:
    """Tunable knobs for the yellow ROI refinement step."""
    erode_scale: float = 0.018     # fraction of min(H,W) for inward erosion
    delta_b: float = 12.0          # b* units below center median still = smoothie
    a_max: float = 14.0            # hard a* cap (0-centered) to exclude pink/red
    L_max: float = 220.0           # L* ceiling: foam/glare cutoff
    chroma_min: float = 6.0        # min sqrt(a²+b²) to reject neutral metal/white


def detect_classical(
    image: np.ndarray,
    min_area_frac: float = 0.05,
    crop_x_frac: float = 0.10,
    crop_y_bot_frac: float = 0.10,
    yellow_params: YellowRefineParams | None = None,
    flatten_top: bool = True,
    top_tilt_max_deg: float = 10.0,
    squiggle_thresh: float = 3.0,
) -> tuple[np.ndarray, BBox | None]:
    """
    Args:
        image:             BGR image (H x W x 3, uint8)
        min_area_frac:     Reject chroma blobs smaller than this fraction of
                           the frame (filters stray colored specks).
        crop_x_frac:       Fraction of frame width to exclude from each side
                           before colour analysis (blocks blender side panels).
        crop_y_bot_frac:   Fraction of frame height to exclude from the bottom
                           before colour analysis (blocks motor base hardware).
        yellow_params:     Tuning knobs for the yellow ROI refinement step.
                           Pass None to use defaults.
        flatten_top:       Master switch for top-edge straightening. When True,
                           the straight-line prior is applied ONLY to yellow/tan
                           smoothies AND only when the detected top edge is
                           actually squiggly (see ``squiggle_thresh``). Red/pink
                           fills segment cleanly and are never flattened.
        top_tilt_max_deg:  Maximum allowed tilt of the enforced top line from
                           horizontal, in degrees (guards against runaway fits).
        squiggle_thresh:   Roughness gate, in pixels (std of the high-frequency
                           top-edge component; see ``top_edge_roughness``). The
                           flatten only fires when
                           ``top_edge_roughness(mask) >= squiggle_thresh`` — a
                           clean/smoothly-curved top is left untouched. Default
                           3.0 sits in the natural gap between clean (≤2.5) and
                           jagged (≥3.6) tops in the sample set.

    Returns:
        roi_mask:  H x W uint8 (255 inside detected region, 0 outside)
        bbox:      (x, y, w, h) bounding box of the region, or None on last resort
    """
    h, w = image.shape[:2]
    frame_area = h * w
    min_area = min_area_frac * frame_area

    if yellow_params is None:
        yellow_params = YellowRefineParams()

    smoothie_type = _classify_smoothie(image)
    is_yellow = smoothie_type in (SmoothieType.PALE_YELLOW, SmoothieType.VIVID_YELLOW)

    mask: np.ndarray | None = None
    bbox: BBox | None = None

    if is_yellow:
        coarse_mask, coarse_bbox = _yellow_b_channel_roi(
            image, min_area, crop_x_frac, crop_y_bot_frac,
        )
        if coarse_mask is not None:
            refined, rbbox = _refine_yellow_roi(image, coarse_mask, yellow_params)
            if refined is not None:
                mask, bbox = refined, rbbox
            else:
                mask, bbox = coarse_mask, coarse_bbox

    # RED_PINK or fallback from yellow branches
    if mask is None:
        mask, bbox = _chroma_roi(image, min_area, crop_x_frac, crop_y_bot_frac)

    # --- last resort: full frame (top is already a straight line; skip flatten) ---
    if mask is None:
        return np.full((h, w), 255, dtype=np.uint8), None

    # Straight-line prior: yellow/tan only, and only when the top is squiggly.
    if flatten_top and is_yellow and top_edge_roughness(mask) >= squiggle_thresh:
        mask, bbox = flatten_roi_top(mask, top_tilt_max_deg=top_tilt_max_deg)

    return mask, bbox


def _refine_yellow_roi(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    params: YellowRefineParams,
) -> tuple[np.ndarray | None, BBox | None]:
    """Refine a coarse yellow geometry mask into a tight smoothie-content mask.

    The coarse b*-threshold mask defines the cup shape but
    includes cup-edge plastic, foam/froth, and specular glare.  This function:

    1. Re-fills the largest contour of the coarse mask to get a clean geometry.
    2. Erodes inward by ``params.erode_scale * min(H,W)`` pixels so the cup
       wall and boundary reflections are excluded from sampling.
    3. Samples LAB stats from a center window *inside* the eroded geometry to
       get a robust reference point (median b* and a*) for the smoothie color.
    4. Applies adaptive LAB thresholds (b* floor, a* cap, L* glare ceiling,
       chroma floor) inside the eroded geometry to isolate true smoothie pixels.
    5. Morphologically cleans up and keeps the single largest connected blob.

    Returns (refined_mask, bbox) or (None, None) if the result is too small.
    """
    h, w = image.shape[:2]
    min_side = min(h, w)

    # --- Step 1: fill the largest contour of the coarse mask ---
    contours, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    geometry_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(geometry_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # --- Step 2: erode inward to drop cup edge / border artifacts ---
    erode_px = max(3, int(min_side * params.erode_scale))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    inner_mask = cv2.erode(geometry_mask, erode_kernel, iterations=1)

    if not np.any(inner_mask):
        return None, None

    # --- Step 3: compute center-crop LAB stats inside the eroded geometry ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1] - 128.0
    b_ch = lab[:, :, 2] - 128.0

    # Center window: 35% of each dimension around image center, intersected with inner_mask
    cy, cx = h // 2, w // 2
    half_y, half_x = int(h * 0.175), int(w * 0.175)
    center_region = np.zeros((h, w), dtype=np.uint8)
    center_region[
        max(0, cy - half_y) : min(h, cy + half_y),
        max(0, cx - half_x) : min(w, cx + half_x),
    ] = 255
    sample_mask = cv2.bitwise_and(inner_mask, center_region)

    # Fall back to the full inner_mask if the center intersection is empty
    if not np.any(sample_mask):
        sample_mask = inner_mask

    b_sample = b_ch[(inner_mask > 0) & (sample_mask > 0)]
    if len(b_sample) == 0:
        b_sample = b_ch[inner_mask > 0]

    median_b = float(np.median(b_sample))

    # --- Step 4: adaptive LAB thresholding inside the inner geometry ---
    b_floor = median_b - params.delta_b

    smoothie_pixels = (
        (b_ch >= b_floor)           # yellow/beige channel floor
        & (a_ch <= params.a_max)    # not pink/red
        & (L_ch <= params.L_max)    # not foam/glare/specular
        & (np.sqrt(a_ch ** 2 + b_ch ** 2) >= params.chroma_min)  # not neutral metal
    )
    yellow_mask = (smoothie_pixels).astype(np.uint8) * 255

    # keep only inside the eroded geometry
    yellow_mask = cv2.bitwise_and(yellow_mask, inner_mask)

    # --- Step 5: morphological cleanup + largest component ---
    morph_px = max(5, int(min_side * 0.015))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_px, morph_px))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, morph_kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, morph_kernel)

    contours2, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        return None, None

    largest2 = max(contours2, key=cv2.contourArea)
    min_area = h * w * 0.03
    if cv2.contourArea(largest2) < min_area:
        return None, None

    refined = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(refined, [largest2], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(largest2)
    return refined, (x, y, bw, bh)


def _yellow_b_channel_roi(
    image: np.ndarray,
    min_area: float,
    crop_x_frac: float = 0.10,
    crop_y_bot_frac: float = 0.10,
) -> tuple[np.ndarray | None, BBox | None]:
    """Segment yellow smoothies by thresholding the b* channel.

    Uses a low fixed floor (b* > 8) so pale/cream smoothies are included;
    morphological cleanup + largest-blob selection isolate the cup region.
    """
    h, w = image.shape[:2]
    cx = int(w * crop_x_frac)
    cy_bot = int(h * crop_y_bot_frac)
    search = image[0 : h - cy_bot, cx : w - cx]

    lab = cv2.cvtColor(search, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2].astype(np.float32) - 128.0

    # Low floor catches pale yellow (b* ~ 8–15) as well as vivid (b* > 20)
    yellow_mask = (b > 8).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, None

    largest_full = largest + np.array([[[cx, 0]]], dtype=np.int32)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(full_mask, [largest_full], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(largest_full)
    return full_mask, (x, y, bw, bh)


def _chroma_roi(
    image: np.ndarray,
    min_area: float,
    crop_x_frac: float = 0.10,
    crop_y_bot_frac: float = 0.10,
) -> tuple[np.ndarray | None, BBox | None]:
    """Segment colorful regions using a warm-biased LAB chroma signal.

    Three improvements applied in order:

    1. Spatial crop: analyse only the central band of the frame so that blender
       side panels and the motor base are never candidates.  The found contour
       is translated back to full-frame coordinates before the mask is drawn.

    2. Warm-channel bias: add clip(b*, 0) * 0.75 to the chroma signal so that
       light-yellow and tan smoothies (small positive b*, near-zero chroma)
       produce a meaningfully higher signal than neutral gray metal (b*≈0).

    3. Post-Otsu gray exclusion: zero out pixels whose a* and b* are both
       near-neutral (|a*| < 6 and |b*| < 6) after thresholding.
    """
    h, w = image.shape[:2]

    # --- spatial crop ---
    cx = int(w * crop_x_frac)
    cy_bot = int(h * crop_y_bot_frac)
    search = image[0 : h - cy_bot, cx : w - cx]

    lab = cv2.cvtColor(search, cv2.COLOR_BGR2LAB)

    # OpenCV LAB stores a,b in [0,255] centered at 128
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    # warm-channel bias
    warm_signal = np.hypot(a, b) + np.clip(b, 0.0, None) * 0.75

    # Scale to uint8 for Otsu (max theoretical value ~272, clamp to 255)
    signal_u8 = np.clip(warm_signal, 0, 255).astype(np.uint8)

    _, chroma_mask = cv2.threshold(signal_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # gray pixel exclusion
    gray_zone = (np.abs(a) < 6) & (np.abs(b) < 6)
    chroma_mask[gray_zone] = 0

    # close gaps (label text, highlights) then drop specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    chroma_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_CLOSE, kernel)
    chroma_mask = cv2.morphologyEx(chroma_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(chroma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, None

    # Translate contour from cropped space → full-frame (x offset by cx; y unchanged)
    largest_full = largest + np.array([[[cx, 0]]], dtype=np.int32)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(full_mask, [largest_full], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(largest_full)
    return full_mask, (x, y, bw, bh)
