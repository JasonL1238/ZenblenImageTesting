"""
Container detection — public API.

Priority: **SAM2 is the primary detector; classical colour-thresholding is the
fallback.**  SAM is colour-agnostic and position-tolerant and never failed
catastrophically across the sample set; the classical detector is fragile on
tan/pale fills, so it is used only when SAM is unavailable (not installed / no
checkpoint) or returns no plausible mask.

Layout:
  common.py     shared, detector-agnostic helpers (types, classify, geometry)
  sam.py        SAM2 fixed-prompt detector        — detect_sam()   [PRIORITY]
  classical.py  colour-threshold detector         — detect_classical()  [FALLBACK]
  __init__.py   detect_container() dispatcher (this file)

Callers should use ``detect_container`` and let the dispatcher choose; pass
``prefer=`` to force a specific detector (e.g. ``prefer="classical"`` for a fast,
torch-free path).
"""

from __future__ import annotations

import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.detection.classical import YellowRefineParams, detect_classical
from smoothie_cv.detection.common import (
    BBox,
    SmoothieType,
    _classify_smoothie,
    draw_container_overlay,
    flatten_roi_top,
    top_edge_roughness,
)

# Order in which detectors are tried. SAM first, classical as the fallback.
DETECTOR_PRIORITY = ["sam", "classical"]

__all__ = [
    "detect_container",
    "DETECTOR_PRIORITY",
    "draw_container_overlay",
    "top_edge_roughness",
    "flatten_roi_top",
    "_classify_smoothie",
    "SmoothieType",
    "YellowRefineParams",
    "BBox",
]


def _is_plausible(mask: np.ndarray | None) -> bool:
    """A usable ROI mask exists and covers a non-trivial part of the frame."""
    if mask is None:
        return False
    frac = float((mask > 0).sum()) / float(mask.size)
    return 0.02 <= frac <= 0.95


def _run_detector(
    name: str,
    image: np.ndarray,
    config: Config,
    yellow_params: YellowRefineParams | None,
    flatten_top: bool,
) -> tuple[np.ndarray, BBox | None]:
    """Run a single named detector. Raises on unavailability/failure."""
    if name == "sam":
        # imported lazily so classical-only callers never pay the torch import
        from smoothie_cv.detection.sam import detect_sam
        return detect_sam(image, config, flatten_top=flatten_top)
    if name == "classical":
        return detect_classical(image, yellow_params=yellow_params, flatten_top=flatten_top)
    raise ValueError(f"Unknown detector {name!r}. Known: {DETECTOR_PRIORITY}")


def detect_container(
    image: np.ndarray,
    config: Config | None = None,
    *,
    prefer: str | list[str] | None = None,
    yellow_params: YellowRefineParams | None = None,
    flatten_top: bool = True,
    return_meta: bool = False,
):
    """Detect the smoothie ROI. **SAM2 is the priority detector; classical is the fallback.**

    Tries each detector in priority order and returns the first plausible mask.
    A detector that is unavailable (e.g. SAM2 not installed / checkpoint missing)
    or that raises / returns an implausible mask is skipped and the next one runs.

    Args:
        image:        BGR image (H x W x 3, uint8).
        config:       Config (SAM model, detector_priority). Defaults to Config().
        prefer:       Override the order. A single name ("sam"/"classical") or an
                      explicit list. Defaults to ``config.detector_priority``.
        yellow_params: Tuning knobs forwarded to the classical detector.
        flatten_top:  Apply the straight-line top prior (yellow-gated in classical,
                      unconditional in SAM).
        return_meta:  If True, also return a dict
                      ``{"detector", "fallback", "roughness"}``.

    Returns:
        (roi_mask, bbox)               if return_meta is False
        (roi_mask, bbox, meta)         if return_meta is True
    """
    if config is None:
        config = Config()

    if prefer is None:
        order = list(getattr(config, "detector_priority", DETECTOR_PRIORITY))
    elif isinstance(prefer, str):
        order = [prefer]
    else:
        order = list(prefer)

    h, w = image.shape[:2]
    errors: list[str] = []

    for i, name in enumerate(order):
        try:
            mask, bbox = _run_detector(name, image, config, yellow_params, flatten_top)
        except Exception as e:  # unavailable or failed → try the next detector
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if _is_plausible(mask):
            if return_meta:
                meta = {
                    "detector": name,
                    "fallback": i > 0,
                    "roughness": round(top_edge_roughness(mask), 2),
                    "errors": errors,
                }
                return mask, bbox, meta
            return mask, bbox
        errors.append(f"{name}: implausible mask")

    # everything failed — last-resort full frame
    full = np.full((h, w), 255, dtype=np.uint8)
    if return_meta:
        return full, None, {"detector": "full_frame", "fallback": True,
                            "roughness": 0.0, "errors": errors}
    return full, None
