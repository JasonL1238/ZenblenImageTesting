from __future__ import annotations

import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendPipeline, BlendResult


class SegFormerPipeline(BlendPipeline):

    name = "segformer"

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        raise NotImplementedError(
            "SegFormer pipeline needs a fine-tuned checkpoint. "
            "Steps: (1) run the VLM pipeline on your images to generate pseudo-labels, "
            "(2) train transformers.SegformerForSemanticSegmentation with two classes "
            "(blended / unblended), (3) set Config.segformer_checkpoint to the saved path."
        )
