from __future__ import annotations

import cv2
import numpy as np
import torch

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendPipeline, BlendResult
from smoothie_cv.scoring.metrics import compute_blend_score

# maps Config.sam_model string → (hydra config yaml, checkpoint filename)
_MODEL_MAP: dict[str, tuple[str, str]] = {
    "sam2_hiera_tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "sam2_hiera_small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "sam2_hiera_base_plus": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "sam2_hiera_large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}

# a segment whose mean LAB color differs from the ROI median by more than this
# (Euclidean in CIE-Lab space, scale ~0–100) is marked unblended
_LAB_DIST_THRESHOLD = 20.0

# minimum segment pixels that overlap the ROI before we consider the segment
_MIN_OVERLAP_PX = 200

# at least this fraction of the segment must lie within the ROI
_MIN_ROI_OVERLAP_FRAC = 0.20


class SAMPipeline(BlendPipeline):

    name = "sam"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._mask_generator = None

    def _build_generator(self) -> None:
        if self._mask_generator is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "sam-2 is required: "
                "pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'"
            )

        model_name = self.config.sam_model
        if model_name not in _MODEL_MAP:
            raise ValueError(
                f"Unknown SAM2 model {model_name!r}. "
                f"Valid choices: {list(_MODEL_MAP)}"
            )
        cfg_yaml, ckpt_name = _MODEL_MAP[model_name]
        ckpt_path = f"checkpoints/{ckpt_name}"

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        sam2 = build_sam2(cfg_yaml, ckpt_path, device=device, apply_postprocessing=False)
        self._mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=self.config.sam_points_per_side,
        )

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        h, w = image.shape[:2]
        if roi_mask is None:
            roi_mask = np.full((h, w), 255, dtype=np.uint8)

        self._build_generator()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self._mask_generator.generate(rgb)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # dominant smoothie color = median LAB of all ROI pixels
        roi_bool = roi_mask > 0
        roi_lab_pixels = lab[roi_bool]
        if len(roi_lab_pixels) == 0:
            dominant_lab = np.array([128.0, 128.0, 128.0])
        else:
            dominant_lab = np.median(roi_lab_pixels, axis=0)

        unblended_mask = np.zeros((h, w), dtype=np.uint8)

        for seg_dict in masks:
            seg: np.ndarray = seg_dict["segmentation"]  # bool H×W
            seg_in_roi = seg & roi_bool

            overlap_px = int(seg_in_roi.sum())
            if overlap_px < _MIN_OVERLAP_PX:
                continue
            seg_area = int(seg.sum())
            if seg_area == 0 or overlap_px / seg_area < _MIN_ROI_OVERLAP_FRAC:
                continue

            mean_lab = lab[seg_in_roi].mean(axis=0)
            dist = float(np.linalg.norm(mean_lab - dominant_lab))

            if dist > _LAB_DIST_THRESHOLD:
                unblended_mask[seg] = 255

        unblended_mask = cv2.bitwise_and(unblended_mask, roi_mask)
        score = compute_blend_score(unblended_mask, roi_mask)

        return BlendResult(
            blend_score=score,
            passed=score >= self.config.threshold,
            mask=unblended_mask,
            pipeline_name=self.name,
            metadata={
                "num_segments": len(masks),
                "dominant_lab": dominant_lab.tolist(),
                "lab_dist_threshold": _LAB_DIST_THRESHOLD,
                "sam_model": self.config.sam_model,
            },
        )
