from __future__ import annotations

import base64

import cv2
import numpy as np
from pydantic import BaseModel

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendPipeline, BlendResult
from smoothie_cv.scoring.metrics import compute_blend_score

_PROMPT = (
    "Analyze this smoothie image for blendedness.\n\n"
    "Determine:\n"
    "1. blend_score: 0 (completely unblended — visible chunks or unmixed color streaks) "
    "to 100 (perfectly smooth and uniform).\n"
    "2. unblended_regions: approximate bounding boxes of any unblended areas. "
    "Express each box as fractions of the image dimensions (0.0–1.0): "
    "x = left edge, y = top edge, w = width, h = height. "
    "Return an empty list if the smoothie appears fully blended.\n"
    "3. reasoning: one or two sentences explaining your assessment."
)


class _BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class _BlendAnalysis(BaseModel):
    blend_score: int
    unblended_regions: list[_BoundingBox]
    reasoning: str


class VLMPipeline(BlendPipeline):

    name = "vlm"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = None

    def _client_instance(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
            api_key = self.config.vlm_api_key()
            if not api_key:
                raise EnvironmentError(
                    f"Set the {self.config.vlm_api_key_env} environment variable."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        h, w = image.shape[:2]
        if roi_mask is None:
            roi_mask = np.full((h, w), 255, dtype=np.uint8)

        _, buf = cv2.imencode(".png", image)
        image_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        client = self._client_instance()
        response = client.messages.parse(
            model=self.config.vlm_model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }],
            output_format=_BlendAnalysis,
        )

        analysis: _BlendAnalysis = response.parsed_output

        # rasterize bounding boxes → binary mask
        unblended_mask = np.zeros((h, w), dtype=np.uint8)
        for box in analysis.unblended_regions:
            x1 = max(0, int(box.x * w))
            y1 = max(0, int(box.y * h))
            x2 = min(w, int((box.x + box.w) * w))
            y2 = min(h, int((box.y + box.h) * h))
            if x2 > x1 and y2 > y1:
                unblended_mask[y1:y2, x1:x2] = 255

        unblended_mask = cv2.bitwise_and(unblended_mask, roi_mask)

        vlm_score = analysis.blend_score / 100.0
        pixel_score = compute_blend_score(unblended_mask, roi_mask)

        # weight: VLM is semantic, pixel coverage is spatial.
        # if the model returned no regions, trust VLM score fully.
        if not analysis.unblended_regions:
            score = vlm_score
        else:
            score = 0.6 * vlm_score + 0.4 * pixel_score

        return BlendResult(
            blend_score=score,
            passed=score >= self.config.threshold,
            mask=unblended_mask,
            pipeline_name=self.name,
            metadata={
                "vlm_raw_score": analysis.blend_score,
                "vlm_score": round(vlm_score, 4),
                "pixel_score": round(pixel_score, 4),
                "num_unblended_regions": len(analysis.unblended_regions),
                "reasoning": analysis.reasoning,
                "model": self.config.vlm_model,
            },
        )
