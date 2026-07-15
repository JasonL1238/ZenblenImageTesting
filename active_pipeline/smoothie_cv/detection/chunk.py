"""
Chunk (unblended lump) detection — YOLO-only public API.

Runs the fine-tuned chunk YOLO (``checkpoints/yolo_chunk_seg.pt``).

``config.chunk_yolo_input``:
  ``full_filter`` — run chunk YOLO on the full frame, keep only pixels inside
                    the smoothie ROI (matches full-frame training labels)
  ``roi_crop``    — crop to the ROI first, run chunk YOLO on the crop

Operates on FULL-FRAME ``image`` / ``roi_mask`` and returns a full-frame mask.
An empty YOLO mask is a real clean verdict; missing weights / inference error
returns an all-zero mask with detector name ``"none"``.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.roi import crop_to_roi, paste_mask

CHUNK_DETECTOR_PRIORITY = ["yolo"]

__all__ = ["detect_chunk", "CHUNK_DETECTOR_PRIORITY"]


def _adapt_yolo(
    image: np.ndarray, roi_mask: np.ndarray, config: Config,
) -> np.ndarray:
    from smoothie_cv.detection.chunk_yolo import detect_chunk_yolo  # lazy

    mode = getattr(config, "chunk_yolo_input", "full_filter")
    if mode == "roi_crop":
        roi = crop_to_roi(image, roi_mask)
        mask = detect_chunk_yolo(roi.image, config)
        mask = cv2.bitwise_and(mask, roi.mask)
        return paste_mask(mask, roi)

    # full_filter (default): full-frame inference, then clip to smoothie ROI
    mask = detect_chunk_yolo(image, config)
    return cv2.bitwise_and(mask, roi_mask)


DETECTORS: dict[str, Callable] = {
    "yolo": _adapt_yolo,
}


def detect_chunk(
    image: np.ndarray,
    roi_mask: np.ndarray,
    config: Config | None = None,
    *,
    prefer: str | list[str] | None = None,
) -> tuple[np.ndarray, str]:
    """Detect unblended chunks inside ``roi_mask`` with YOLO-seg.

    Args:
        image:     Full-frame BGR image (H x W x 3, uint8).
        roi_mask:  Full-frame smoothie ROI mask (same H x W).
        config:    Config (chunk_weights, chunk_conf, chunk_detector_priority,
                   chunk_yolo_input).
        prefer:    Override the order. Only ``"yolo"`` is registered. Defaults
                   to ``config.chunk_detector_priority``.

    Returns:
        (full-frame uint8 chunk mask, detector name that produced it).
        Detector name is ``"none"`` if every detector failed.
    """
    if config is None:
        config = Config()

    if prefer is None:
        order = list(getattr(config, "chunk_detector_priority", CHUNK_DETECTOR_PRIORITY))
    elif isinstance(prefer, str):
        order = [prefer]
    else:
        order = list(prefer)

    for name in order:
        if name not in DETECTORS:
            raise ValueError(
                f"Unknown chunk detector {name!r}. Active: {list(DETECTORS)}."
            )
        try:
            return DETECTORS[name](image, roi_mask, config), name
        except Exception:  # unavailable or failed -> try the next detector
            continue

    return np.zeros(roi_mask.shape, np.uint8), "none"
