from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BlendResult:
    blend_score: float          # 0.0 (not blended) → 1.0 (fully blended)
    passed: bool                # blend_score >= threshold
    mask: np.ndarray            # HxW uint8; 255 = unblended pixel
    pipeline_name: str
    metadata: dict = field(default_factory=dict)


class BlendPipeline(ABC):
    """All pipelines implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        """
        Args:
            image:    BGR image (H x W x 3, uint8)
            roi_mask: binary mask (H x W uint8), 255 = inside blender jar.
                      If None the pipeline operates on the full frame.
        Returns:
            BlendResult
        """
        ...
