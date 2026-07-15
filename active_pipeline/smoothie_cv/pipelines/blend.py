"""Active blend pipeline — YOLO container ROI + YOLO chunk masks → blend score."""

from __future__ import annotations

import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendPipeline as BlendPipelineBase
from smoothie_cv.pipelines.base import BlendResult
from smoothie_cv.scoring.metrics import compute_blend_score


class BlendPipeline(BlendPipelineBase):
    """YOLO-only unblended-chunk analysis inside a precomputed ROI."""

    name = "blend"

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        h, w = image.shape[:2]
        if roi_mask is None:
            roi_mask = np.full((h, w), 255, dtype=np.uint8)

        from smoothie_cv.detection.chunk import detect_chunk

        unblended, chunk_detector = detect_chunk(image, roi_mask, self.config)
        score = compute_blend_score(unblended, roi_mask)
        passed = score >= self.config.threshold

        return BlendResult(
            blend_score=score,
            passed=passed,
            mask=unblended,
            pipeline_name=self.name,
            metadata={
                "chunk_detector": chunk_detector,
                "chunk_yolo_input": getattr(self.config, "chunk_yolo_input", "full_filter"),
            },
        )
