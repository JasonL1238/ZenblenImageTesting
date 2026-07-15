"""
YOLO-seg CHUNK detection — the trained "chunk" labeler-mode model
(``checkpoints/yolo_chunk_seg.pt``, ``labeling/chunk_dataset``,
``training/train_multi.py --mode chunk``). See ``smoothie_cv.detection.chunk``
for the priority dispatcher that falls back to the classical ensemble when this
checkpoint is missing or inference fails.

Input may be a full frame or an ROI crop — ``config.chunk_yolo_input`` chooses
which the dispatcher feeds. The model was trained on full-frame labels; the
A/B in ``scripts/eval_chunk_yolo_input.py`` picks the better inference mode.

UNIONS every instance above ``config.chunk_conf`` (a cup can have several
disjoint chunks) — same policy as ``detect_logo``/``detect_spill``, never keeps
only the single highest-confidence instance the way ``get_yolo_roi`` does for
the one-object container.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from smoothie_cv.config import Config

# cached model so the weights load once per process, independent of the other
# per-mode models (all run in the same process; separate globals so none clobber
# each other's cache).
_CHUNK_MODEL = None
_CHUNK_WEIGHTS: str | None = None


def _get_chunk_model(weights: str):
    global _CHUNK_MODEL, _CHUNK_WEIGHTS
    if _CHUNK_MODEL is None or _CHUNK_WEIGHTS != weights:
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Chunk YOLO weights not found: {weights} — label with "
                f"labeling/run_chunk_seed.py + app_multi.py (mode 4 · Chunk), "
                f"export with labeling/export_multi.py --mode chunk, train with "
                f"training/train_multi.py --mode chunk, then "
                f"cp runs/chunk-seg/<run>/weights/best.pt {weights}"
            )
        from ultralytics import YOLO  # lazy: non-chunk callers skip the import
        _CHUNK_MODEL = YOLO(weights)
        _CHUNK_WEIGHTS = weights
    return _CHUNK_MODEL


def detect_chunk_yolo(image: np.ndarray, config: Config) -> np.ndarray:
    """Uint8 mask (255 = unblended chunk) matching ``image``'s H×W — the UNION
    of every chunk instance above ``config.chunk_conf``. Empty mask if the model
    finds nothing. Raises ``FileNotFoundError`` if weights are missing — callers
    should go through ``smoothie_cv.detection.chunk.detect_chunk()``, which
    catches this and falls back to the classical ensemble."""
    h, w = image.shape[:2]
    model = _get_chunk_model(str(config.chunk_weights))
    # force CPU: YOLO-seg segfaults on MPS (matches every other mode's detector)
    result = model(image, verbose=False, device="cpu", conf=config.chunk_conf)[0]

    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or len(result.masks) == 0:
        return mask
    for raw in result.masks.data.cpu().numpy():
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[m > 0.5] = 255
    return mask
