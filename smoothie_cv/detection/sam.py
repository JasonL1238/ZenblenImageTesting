"""
SAM2-based container detection — the PRIORITY detector.

Motivation: the classical detector (``classical.py``) keys on LAB colour, which
is fragile across smoothie shades (red/pink vs vivid yellow vs pale tan) and
produces jagged tops on low-saturation fills. SAM2 segments by general
*objectness* instead, so it is colour-agnostic, and a fixed centre-point prompt
tolerates the slight frame-to-frame position drift of the cup. Across the full
sample it never fails catastrophically, which is why it is the priority detector
(see the SAM→classical dispatcher in ``__init__.py``).

Prompt strategy (fixed, no per-image annotation):
  * Positive points down the vertical centre line — the smoothie always occupies
    the central column even as it shifts slightly.
  * Negative points at the four corners — pushes SAM away from grabbing the
    whole frame (machine body / background).
  * multimask_output=True, then pick the highest-scoring mask whose area is a
    plausible fraction of the frame.

Returns the standard detector contract:
  (roi_mask: HxW uint8 with 255 inside the smoothie, bbox: (x, y, w, h) or None).
"""

from __future__ import annotations

import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.detection.common import (
    BBox,
    _largest_filled,
    extend_roi_to_gasket,
    flatten_roi_top,
    refine_cup_sides,
    top_edge_roughness,
)

# maps Config.sam_model string → (hydra config name, checkpoint filename)
# Config names are resolved relative to pkg://sam2 by Hydra.
_MODEL_MAP: dict[str, tuple[str, str]] = {
    # SAM 2.0
    "sam2_hiera_tiny": ("configs/sam2/sam2_hiera_t", "sam2_hiera_tiny.pt"),
    "sam2_hiera_small": ("configs/sam2/sam2_hiera_s", "sam2_hiera_small.pt"),
    "sam2_hiera_base_plus": ("configs/sam2/sam2_hiera_b+", "sam2_hiera_base_plus.pt"),
    "sam2_hiera_large": ("configs/sam2/sam2_hiera_l", "sam2_hiera_large.pt"),
    # SAM 2.1 (updated weights, same architecture, better boundary quality)
    "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t", "sam2.1_hiera_tiny.pt"),
    "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s", "sam2.1_hiera_small.pt"),
    "sam2.1_hiera_base_plus": ("configs/sam2.1/sam2.1_hiera_b+", "sam2.1_hiera_base_plus.pt"),
    "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l", "sam2.1_hiera_large.pt"),
}

# plausible smoothie area as a fraction of the frame; masks outside this band
# (a tiny speck or the whole frame) are rejected before scoring.
_MIN_AREA_FRAC = 0.05
_MAX_AREA_FRAC = 0.70

# cached predictor so the model loads once per process, not once per image
_PREDICTOR = None
_PREDICTOR_MODEL: str | None = None


def _get_predictor(model_name: str):
    """Lazily build and cache a SAM2 image predictor on MPS (or CPU fallback)."""
    global _PREDICTOR, _PREDICTOR_MODEL
    if _PREDICTOR is not None and _PREDICTOR_MODEL == model_name:
        return _PREDICTOR

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if model_name not in _MODEL_MAP:
        raise ValueError(f"Unknown SAM2 model {model_name!r}. Valid: {list(_MODEL_MAP)}")
    cfg_yaml, ckpt_name = _MODEL_MAP[model_name]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sam2 = build_sam2(cfg_yaml, f"checkpoints/{ckpt_name}", device=device,
                      apply_postprocessing=False)
    _PREDICTOR = SAM2ImagePredictor(sam2)
    _PREDICTOR_MODEL = model_name
    return _PREDICTOR


def _prompt_points(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed positive centre-line points + negative corner points."""
    cx = w // 2
    pos = [
        (cx, int(h * 0.35)),
        (cx, int(h * 0.50)),
        (cx, int(h * 0.65)),
    ]
    margin_x, margin_y = int(w * 0.04), int(h * 0.04)
    neg = [
        (margin_x, margin_y),
        (w - margin_x, margin_y),
        (margin_x, h - margin_y),
        (w - margin_x, h - margin_y),
    ]
    coords = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return coords, labels


def _smoothie_prompt_points(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Prompt tuned for the LIQUID body, not the container.

    Positives sit on the central column plus two lateral points so SAM captures
    the full width of the fill. Negatives sit at the four corners AND high on the
    centre line — the clear headspace / cup rim above the liquid surface — so SAM
    cuts at the liquid top instead of grabbing the empty rim, lid, or straw.
    """
    cx = w // 2
    pos = [
        (cx, int(h * 0.40)),
        (cx, int(h * 0.55)),
        (cx, int(h * 0.70)),
        (int(w * 0.32), int(h * 0.55)),
        (int(w * 0.68), int(h * 0.55)),
    ]
    mx, my = int(w * 0.04), int(h * 0.04)
    neg = [
        (mx, my), (w - mx, my), (mx, h - my), (w - mx, h - my),  # corners: frame/machine
        (cx, int(h * 0.06)),                                     # rim / headspace above liquid
    ]
    coords = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return coords, labels


def detect_smoothie(
    image: np.ndarray,
    config: Config | None = None,
    model_name: str | None = None,
) -> tuple[np.ndarray, BBox | None]:
    """Segment the visible SMOOTHIE/LIQUID region (not the whole container).

    Unlike :func:`detect_sam` (which applies container priors — top flatten,
    gasket extend, side refine — to recover the full cup footprint), this takes
    the RAW highest-scoring SAM mask of the central liquid mass, keeps the largest
    filled blob, and stops there. That leaves the true liquid surface at the top
    for the human labeller to fine-tune, rather than a straightened cup rim.

    Returns the standard detector contract ``(mask uint8 255-inside, bbox|None)``.
    """
    import cv2
    import torch

    if config is None:
        config = Config()
    model_name = model_name or config.sam_model

    h, w = image.shape[:2]
    frame_area = float(h * w)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor = _get_predictor(model_name)
    coords, labels = _smoothie_prompt_points(h, w)

    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=coords, point_labels=labels, multimask_output=True,
        )

    best_idx, best_score = -1, -1.0
    for i, m in enumerate(masks):
        frac = float(m.sum()) / frame_area
        if _MIN_AREA_FRAC <= frac <= _MAX_AREA_FRAC and scores[i] > best_score:
            best_idx, best_score = i, float(scores[i])
    if best_idx < 0:
        best_idx = int(np.argmax(scores))

    raw = (masks[best_idx] > 0).astype(np.uint8) * 255
    filled, bbox = _largest_filled(raw)
    if filled is None:
        return np.zeros((h, w), dtype=np.uint8), None
    return filled, bbox


def detect_sam(
    image: np.ndarray,
    config: Config | None = None,
    model_name: str | None = None,
    flatten_top: bool | None = None,
    top_tilt_max_deg: float = 10.0,
) -> tuple[np.ndarray, BBox | None]:
    """Detect the smoothie region with SAM2 and a fixed centre-point prompt.

    Args:
        image:       BGR image (H x W x 3, uint8)
        config:      Config (used for ``sam_model``); defaults to ``Config()``.
        model_name:  Explicit SAM2 model name; overrides ``config.sam_model``.
        flatten_top: Top-edge prior policy. The RAW SAM mask is primary; the
                     straight-line ``flatten_roi_top`` prior is a fallback for
                     jagged ("weird") tops whose ragged rim would drag the
                     foam/meniscus band into the ROI and misfire the chunk
                     detector. Values:
                       * ``None`` (default) — AUTO: flatten only when the raw
                         top is too jagged, i.e.
                         ``top_edge_roughness(raw) > config.sam_top_roughness_max``.
                       * ``True``  — always flatten (force the prior on).
                       * ``False`` — never flatten (force the raw top).
                     Auto keeps the true, un-straightened surface geometry on
                     clean masks and only intervenes on the squiggly ones.
        top_tilt_max_deg: Max tilt of the enforced top line from horizontal.

    Returns:
        roi_mask:  H x W uint8 (255 inside detected region, 0 outside)
        bbox:      (x, y, w, h) bounding box, or None if no plausible mask found
    """
    import cv2
    import torch

    if config is None:
        config = Config()
    model_name = model_name or config.sam_model

    h, w = image.shape[:2]
    frame_area = float(h * w)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor = _get_predictor(model_name)
    coords, labels = _prompt_points(h, w)

    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )

    # choose the highest-scoring mask whose area is a plausible cup fraction
    best_idx, best_score = -1, -1.0
    for i, m in enumerate(masks):
        frac = float(m.sum()) / frame_area
        if _MIN_AREA_FRAC <= frac <= _MAX_AREA_FRAC and scores[i] > best_score:
            best_idx, best_score = i, float(scores[i])
    if best_idx < 0:  # nothing in-band: fall back to overall best score
        best_idx = int(np.argmax(scores))

    raw = (masks[best_idx] > 0).astype(np.uint8) * 255

    # keep the single largest blob and fill interior holes (glare/foam)
    filled, bbox = _largest_filled(raw)
    if filled is None:
        return np.zeros((h, w), dtype=np.uint8), None

    # Straighten ragged side walls (logo-text scallops / low-confidence jitter on
    # dark fills) so the ROI follows the cup wall and stops dragging thin dark
    # slivers into the chunk detector. Colour-agnostic geometry; a clean wall is
    # left as-is. Done before the top prior, which handles the top edge separately.
    if config.sam_side_refine_win > 0:
        filled = refine_cup_sides(filled, win_frac=config.sam_side_refine_win)
        filled, bbox = _largest_filled(filled)
        if filled is None:
            return np.zeros((h, w), dtype=np.uint8), None

    # Fixed-rig bottom prior: extend the ROI down to the holder gasket so a big
    # bottom chunk on a dark fill (where SAM stops mid-cup) isn't left out and the
    # cup falsely scored clean. Gated on finding the dark gasket, so cups already at
    # their true bottom are untouched.
    if config.sam_bottom_extend_frac > 0:
        filled = extend_roi_to_gasket(
            filled, image,
            max_frac=config.sam_bottom_extend_frac,
            dark_drop=config.sam_gasket_dark_drop,
        )
        filled, bbox = _largest_filled(filled)
        if filled is None:
            return np.zeros((h, w), dtype=np.uint8), None

    # Top-edge prior: raw is primary; flatten only when forced (True) or, in
    # AUTO mode (None), when the raw top is jaggier than the roughness gate.
    if flatten_top is None:
        do_flatten = top_edge_roughness(filled) > config.sam_top_roughness_max
    else:
        do_flatten = flatten_top
    if do_flatten:
        filled, bbox = flatten_roi_top(filled, top_tilt_max_deg=top_tilt_max_deg)
    return filled, bbox
