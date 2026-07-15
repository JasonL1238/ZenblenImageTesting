"""
YOLO-seg container detection — the PRIORITY detector.

A YOLO11n-seg model fine-tuned on our own labelled cups (see the labeling/ tool
and dataset_tools/ export pipeline) segments the smoothie liquid directly —
smoothie-only masks, foam excluded, per the locked labeling standard. Unlike
SAM2 (which segments generic *objectness* and needs the fixed-prompt trickery
plus top-flatten/side-refine priors to carve the liquid out of the cup), the
trained model has learned exactly the region we care about, reaches the true
cup bottom (gasket-complete masks that made Path 6 viable), and runs a 6 MB
nano network instead of a SAM2 checkpoint.

Weights: ``config.yolo_weights`` (default ``checkpoints/yolo_smoothie_seg.pt``).
After retraining (``training/train.py``), promote the new run's ``best.pt`` there:

    cp runs/smoothie-seg/<run>/weights/best.pt checkpoints/yolo_smoothie_seg.pt

Returns the standard detector contract:
  (roi_mask: HxW uint8 with 255 inside the smoothie, bbox: (x, y, w, h) or None).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.detection.common import BBox

# cached model so the weights load once per process, not once per image
_MODEL = None
_MODEL_WEIGHTS: str | None = None


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes: a liquid mask is simply connected, so any enclosed
    hole (e.g. the model excising a printed logo letter) is a segmentation
    artifact. Holes over logo letters break the text-line logo exclusion
    downstream (the word is no longer a row of marks inside the ROI)."""
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def get_yolo_roi(result, shape: tuple[int, int]) -> np.ndarray:
    """Extract the highest-confidence instance mask from an ultralytics result,
    resized to the full frame and hole-filled. Zeros if nothing was detected."""
    h, w = shape
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    confs = result.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    raw = result.masks.data[idx].cpu().numpy()
    m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return fill_holes(((m > 0.5) * 255).astype(np.uint8))


def _get_model(weights: str):
    global _MODEL, _MODEL_WEIGHTS
    if _MODEL is None or _MODEL_WEIGHTS != weights:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights} — train with training/train.py, then "
                f"cp runs/smoothie-seg/<run>/weights/best.pt {weights}"
            )
        from ultralytics import YOLO  # lazy: non-YOLO callers skip the import
        _MODEL = YOLO(weights)
        _MODEL_WEIGHTS = weights
    return _MODEL


def detect_yolo(image: np.ndarray, config: Config) -> tuple[np.ndarray, BBox | None]:
    """Segment the smoothie with the fine-tuned YOLO-seg model."""
    model = _get_model(str(config.yolo_weights))
    result = model(image, verbose=False)[0]
    mask = get_yolo_roi(result, image.shape[:2])
    if not mask.any():
        return mask, None
    x, y, w, h = cv2.boundingRect(mask)
    return mask, (x, y, w, h)
