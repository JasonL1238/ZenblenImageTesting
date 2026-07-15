"""
Chunk (unblended lump) detection — public API.

Two registered detectors, in priority order (mirrors this package's
``__init__.py`` for container detection):
  yolo       fine-tuned YOLO11n-seg on hand-labeled chunk masks   [PRIORITY]
  classical  LAB local-deviation ensemble                         [FALLBACK]

``config.chunk_detector_priority`` defaults to ``["yolo", "classical"]``.
YOLO input mode is ``config.chunk_yolo_input``:
  ``full_filter`` — run chunk YOLO on the full frame, keep only pixels inside
                    the smoothie ROI (matches full-frame training labels)
  ``roi_crop``    — crop to the ROI first, run chunk YOLO on the crop

Operates on FULL-FRAME ``image`` / ``roi_mask`` and returns a full-frame mask.
Classical falls back only when YOLO raises (missing weights / load / inference
error); an empty YOLO mask is a real clean verdict.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.roi import crop_to_roi, paste_mask

# Order in which detectors are tried when config does not override.
CHUNK_DETECTOR_PRIORITY = ["yolo", "classical"]

__all__ = ["detect_chunk", "CHUNK_DETECTOR_PRIORITY"]


def _adapt_yolo(
    image: np.ndarray, roi_mask: np.ndarray, config: Config,
    logo_mask: np.ndarray | None,
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


def _adapt_classical(
    image: np.ndarray, roi_mask: np.ndarray, config: Config,
    logo_mask: np.ndarray | None,
) -> np.ndarray:
    # Lazy import: classical_cv.py's analyze() calls into this dispatcher, and
    # this adapter calls back into classical_cv.py — a logical cycle, broken by
    # deferring the import to call time (same trick detection/__init__.py uses
    # for its own lazy detector imports).
    from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline

    if logo_mask is None and getattr(config, "dev_logo_yolo_suppress", False):
        from smoothie_cv.detection.logo import detect_logo
        logo_mask = detect_logo(image, config)

    roi = crop_to_roi(image, roi_mask)
    logo_crop = None
    if logo_mask is not None:
        x0, y0 = roi.offset
        ch, cw = roi.mask.shape
        logo_crop = logo_mask[y0:y0 + ch, x0:x0 + cw]
    crop_mask = ClassicalCVPipeline(config)._deviation_mask(
        roi.image, roi.mask, logo_crop,
    )
    return paste_mask(crop_mask, roi)


DETECTORS: dict[str, Callable] = {
    "yolo": _adapt_yolo,
    "classical": _adapt_classical,
}


def detect_chunk(
    image: np.ndarray,
    roi_mask: np.ndarray,
    config: Config | None = None,
    *,
    prefer: str | list[str] | None = None,
    logo_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Detect unblended chunks inside ``roi_mask``.

    Tries each detector in priority order and returns the first one that
    succeeds. A detector that's unavailable (e.g. YOLO weights missing) or
    that raises is skipped and the next one runs — mirrors
    ``detect_container``'s fallback behaviour. If every detector fails,
    returns an all-zero mask (no chunks found) rather than raising, since a
    detection failure should never crash the blend-scoring pipeline.

    Args:
        image:     Full-frame BGR image (H x W x 3, uint8).
        roi_mask:  Full-frame smoothie ROI mask (same H x W).
        config:    Config (chunk_weights, chunk_conf, chunk_detector_priority,
                   chunk_yolo_input).
        prefer:    Override the order. A single name ("yolo"/"classical") or an
                   explicit list. Defaults to ``config.chunk_detector_priority``.
        logo_mask: Optional full-frame trained-logo mask, forwarded to classical.

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
            raise ValueError(f"Unknown chunk detector {name!r}. Registered: {list(DETECTORS)}")
        try:
            return DETECTORS[name](image, roi_mask, config, logo_mask), name
        except Exception:  # unavailable or failed -> try the next detector
            continue

    return np.zeros(roi_mask.shape, np.uint8), "none"
