"""
Classical CV pipeline for smoothie blendedness detection.

Detects unblended regions using two complementary signals:
  1. Local color variance in LAB space — high variance → heterogeneous (unblended)
  2. Edge density (Canny) — strong edges → chunk boundaries

Both signals are combined with a logical OR, then morphologically cleaned up.
"""

from __future__ import annotations

import cv2
import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendPipeline, BlendResult
from smoothie_cv.scoring.metrics import compute_blend_score


class ClassicalCVPipeline(BlendPipeline):

    name = "classical"

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        h, w = image.shape[:2]
        if roi_mask is None:
            roi_mask = np.full((h, w), 255, dtype=np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # --- signal 1: local color variance in LAB ---
        variance_mask = self._variance_mask(lab, roi_mask)

        # --- signal 2: edge density (chunk boundaries) ---
        edge_mask = self._edge_mask(image, roi_mask)

        # --- combine ---
        unblended = cv2.bitwise_or(variance_mask, edge_mask)

        # apply ROI
        unblended = cv2.bitwise_and(unblended, roi_mask)

        # morphological cleanup: remove tiny noise, close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        unblended = cv2.morphologyEx(unblended, cv2.MORPH_OPEN, kernel)
        unblended = cv2.morphologyEx(unblended, cv2.MORPH_CLOSE, kernel)

        score = compute_blend_score(unblended, roi_mask)
        passed = score >= self.config.threshold

        return BlendResult(
            blend_score=score,
            passed=passed,
            mask=unblended,
            pipeline_name=self.name,
            metadata={
                "variance_threshold": self.config.variance_threshold,
                "variance_window": self.config.variance_window,
                "canny_lo": self.config.canny_lo,
                "canny_hi": self.config.canny_hi,
            },
        )

    def _variance_mask(self, lab: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel local variance across LAB channels in a sliding window.
        Pixels whose variance exceeds the threshold are marked as unblended.
        """
        w = self.config.variance_window
        ksize = (w, w)

        # work in float32 for accuracy
        lab_f = lab.astype(np.float32)
        variance_sum = np.zeros(lab_f.shape[:2], dtype=np.float32)

        for c in range(3):
            ch = lab_f[:, :, c]
            mean = cv2.blur(ch, ksize)
            sq_mean = cv2.blur(ch * ch, ksize)
            var = sq_mean - mean * mean
            variance_sum += np.maximum(var, 0)

        # normalize by number of channels for a consistent threshold scale
        variance_map = variance_sum / 3.0

        # restrict to ROI before thresholding
        variance_map[roi_mask == 0] = 0

        _, mask = cv2.threshold(
            variance_map, self.config.variance_threshold, 255, cv2.THRESH_BINARY
        )
        return mask.astype(np.uint8)

    def _edge_mask(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Detect strong edges (chunk boundaries) inside the ROI.
        Dilate edges slightly and fill enclosed contours to get filled chunk regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # restrict Canny to ROI
        masked_gray = cv2.bitwise_and(gray, roi_mask)
        edges = cv2.Canny(masked_gray, self.config.canny_lo, self.config.canny_hi)

        # dilate to connect nearby edges, then find + fill contours
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled = np.zeros_like(edges)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # ignore tiny noise and very large regions (likely the full jar edge)
            roi_area = (roi_mask > 0).sum()
            if area < 50 or area > roi_area * 0.30:
                continue
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        return filled
