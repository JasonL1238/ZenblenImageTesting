"""
Classify smoothie images as red/pink or yellow based on dominant ROI color.

Uses the same container ROI detector as the blend pipelines so background
metal/white does not skew the shade estimate. Median LAB a*/b* inside the ROI
determines the shade: yellow smoothies sit on the +b* axis; red/pink on +a*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import cv2
import numpy as np

from smoothie_cv.detection.container import detect_container

ShadeLabel = Literal["red_pink", "yellow"]


class SmoothieShade(str, Enum):
    """Supported smoothie shade buckets for dataset organization."""

    RED_PINK = "red_pink"
    YELLOW = "yellow"


@dataclass(frozen=True)
class ShadeResult:
    """Classification output for one image."""

    shade: SmoothieShade
    median_a: float  # LAB a* centered at 0 (OpenCV stores a at 128 = neutral)
    median_b: float  # LAB b* centered at 0
    hue_deg: float   # atan2(b*, a*) in degrees, 0–360
    roi_coverage: float  # fraction of frame covered by detected smoothie ROI


def classify_smoothie_shade(image: np.ndarray) -> ShadeResult:
    """
    Classify a BGR image as red/pink or yellow smoothie.

    Args:
        image: H x W x 3 uint8 BGR array.

    Returns:
        ShadeResult with label and diagnostic LAB stats.
    """
    roi_mask, _ = detect_container(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    roi_bool = roi_mask > 0
    roi_pixels = lab[roi_bool]
    if len(roi_pixels) == 0:
        # Fallback: whole frame (should be rare if container detection fails)
        roi_pixels = lab.reshape(-1, 3)
        roi_coverage = 1.0
    else:
        roi_coverage = float(roi_bool.sum()) / roi_bool.size

    a_star = roi_pixels[:, 1] - 128.0
    b_star = roi_pixels[:, 2] - 128.0
    median_a = float(np.median(a_star))
    median_b = float(np.median(b_star))
    hue_deg = float(np.degrees(np.arctan2(median_b, median_a)) % 360.0)

    shade = SmoothieShade.YELLOW if _is_yellow_shade(median_a, median_b, hue_deg) else SmoothieShade.RED_PINK

    return ShadeResult(
        shade=shade,
        median_a=median_a,
        median_b=median_b,
        hue_deg=hue_deg,
        roi_coverage=roi_coverage,
    )


def _is_yellow_shade(median_a: float, median_b: float, hue_deg: float) -> bool:
    """
    Return True when ROI color reads as a yellow smoothie.

    Yellow blends have low red (a*) and strong yellow (b*). Pink smoothies
    often have similar b* but higher a*; require a clear yellow signal so
    pale pinks near the red/yellow boundary stay in red_pink.
    """
    if median_a <= 6.0 and median_b >= 10.0:
        return True
    if median_b - median_a >= 8.0:
        return True
    if hue_deg >= 75.0 and median_b >= 10.0:
        return True
    return False
