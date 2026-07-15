"""
YOLO-seg LOGO detection — a chunk-suppression signal (NOT a container detector).

A YOLO11n-seg model fine-tuned on our own labelled "zenblen" wordmarks (see the
labeling/ multi-mode tool: ``training/train_multi.py --mode logo`` →
``checkpoints/yolo_logo_seg.pt``). Returns a full-frame mask of the printed logo
so the classical chunk detector can reject false-positive components that land on
the wordmark — in particular the CLIPPED-wordmark cases (word partly out of frame
/ curved on the cup) that defeat the classical text-line heuristic
(``_logo_text_labels``): too few letters / too short a span to confirm a wordmark,
so the classical band/corner suppression never fires.

This is ADDITIVE: it only removes components; it does not replace the classical
logo handling. Weights: ``config.logo_weights`` (default
``checkpoints/yolo_logo_seg.pt``); confidence floor ``config.logo_conf``.

Unlike ``get_yolo_roi`` (which keeps only the single highest-confidence instance,
because a container is one object), this UNIONS every instance above the floor —
each letter or word fragment can be a separate detection.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smoothie_cv.config import Config

# cached model so the weights load once per process, independent of the smoothie
# container model in yolo.py (both run in the same process; separate globals so
# neither clobbers the other's cache).
_LOGO_MODEL = None
_LOGO_WEIGHTS: str | None = None


def _get_logo_model(weights: str):
    global _LOGO_MODEL, _LOGO_WEIGHTS
    if _LOGO_MODEL is None or _LOGO_WEIGHTS != weights:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Logo YOLO weights not found: {weights} — train with "
                f"training/train_multi.py --mode logo, then "
                f"cp runs/logo-seg/<run>/weights/best.pt {weights}"
            )
        from ultralytics import YOLO  # lazy: non-logo callers skip the import
        _LOGO_MODEL = YOLO(weights)
        _LOGO_WEIGHTS = weights
    return _LOGO_MODEL


def detect_logo(image: np.ndarray, config: Config) -> np.ndarray:
    """Full-frame uint8 mask (255 = printed logo) — the UNION of every logo
    instance above ``config.logo_conf``. Empty mask if the model finds nothing.

    No hole-filling: we want a tight letter mask, not a filled blob (a filled
    wordmark box would swallow real chunks sitting between the letters)."""
    h, w = image.shape[:2]
    model = _get_logo_model(str(config.logo_weights))
    # force CPU: YOLO-seg segfaults on MPS (matches training/train_multi.py / predict_batch.py)
    result = model(image, verbose=False, device="cpu", conf=config.logo_conf)[0]

    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or len(result.masks) == 0:
        return mask
    for raw in result.masks.data.cpu().numpy():
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[m > 0.5] = 255
    return mask
