"""
YOLO-seg SPILL detection — the spill pipeline's segmenter.

A YOLO11n-seg model fine-tuned on our own labelled spills (see the labeling/
multi-mode tool: ``training/train_multi.py --mode spill`` →
``checkpoints/yolo_spill_seg.pt``). Spill = any smoothie material OUTSIDE the
cup: drips/pooling on the holder gasket, splatter on the machine interior.

Unlike ``get_yolo_roi`` (container detector — keeps only the single
highest-confidence instance, because a cup is one object), spill can appear as
several disjoint blobs, so this UNIONS every instance above ``config.spill_conf``
and also returns per-instance areas + the max confidence for reporting.

Weights: ``config.spill_weights`` (default ``checkpoints/yolo_spill_seg.pt``);
confidence floor ``config.spill_conf``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smoothie_cv.config import Config

# cached model — separate global from the container (yolo.py) and logo (logo.py)
# models so all three can coexist in one process without clobbering each other.
_SPILL_MODEL = None
_SPILL_WEIGHTS: str | None = None


def _get_spill_model(weights: str):
    global _SPILL_MODEL, _SPILL_WEIGHTS
    if _SPILL_MODEL is None or _SPILL_WEIGHTS != weights:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Spill YOLO weights not found: {weights} — train with "
                f"training/train_multi.py --mode spill, then "
                f"cp runs/spill-seg/<run>/weights/best.pt {weights}"
            )
        from ultralytics import YOLO  # lazy: non-spill callers skip the import
        _SPILL_MODEL = YOLO(weights)
        _SPILL_WEIGHTS = weights
    return _SPILL_MODEL


def detect_spill(image: np.ndarray, config: Config) -> tuple[np.ndarray, list[float], float]:
    """Segment spilled smoothie material outside the cup.

    Returns ``(mask, instance_confs, max_conf)``:
      - ``mask``:          full-frame uint8, 255 = spilled material (union of every
                           instance above ``config.spill_conf``).
      - ``instance_confs`` confidence of each unioned instance (descending).
      - ``max_conf``:      highest instance confidence (0.0 if nothing detected).

    No hole-filling: spill is not a simply-connected object, so an enclosed gap
    (e.g. a clean patch surrounded by drips) is genuine, not an artifact.
    """
    h, w = image.shape[:2]
    model = _get_spill_model(str(config.spill_weights))
    # force CPU: YOLO-seg segfaults on MPS (matches logo.py / predict_batch.py)
    result = model(image, verbose=False, device="cpu", conf=config.spill_conf)[0]

    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or len(result.masks) == 0:
        return mask, [], 0.0

    confs = result.boxes.conf.cpu().numpy()
    for raw in result.masks.data.cpu().numpy():
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[m > 0.5] = 255
    return mask, sorted(confs.tolist(), reverse=True), float(confs.max())
