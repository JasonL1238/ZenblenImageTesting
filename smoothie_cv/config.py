"""
Central config for the smoothie blendedness pipeline.

Priority (highest → lowest):
  CLI flags  >  config.yaml  >  dataclass defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class Config:
    # --- pass/fail ---
    threshold: float = 0.90          # blend_score >= threshold → passed

    # --- classical CV ---
    variance_window: int = 15
    variance_threshold: float = 200.0
    canny_lo: int = 50
    canny_hi: int = 150

    # --- yellow ROI refinement ---
    # Fraction of min(H,W) to erode inward from the coarse geometry mask,
    # removing cup-edge plastic, reflections, and border glare.
    yellow_erode_scale: float = 0.018
    # Adaptive threshold: how many b* units below the center-crop median
    # a pixel can still be considered yellow smoothie.
    yellow_delta_b: float = 12.0
    # Hard cap on a* (0-centered): excludes pink/red contamination.
    yellow_a_max: float = 14.0
    # L* ceiling: pixels brighter than this are foam, glare, or specular — excluded.
    yellow_L_max: float = 220.0
    # Minimum LAB chroma (sqrt(a²+b²)) inside the ROI; rejects neutral metal/white.
    yellow_chroma_min: float = 6.0

    # --- container detection (ROI) ---
    # Order detectors are tried in. SAM2 is the priority detector (colour-agnostic,
    # robust across shades); classical colour-thresholding is the fallback, used
    # only when SAM is unavailable or returns no plausible mask.
    detector_priority: list[str] = field(default_factory=lambda: ["sam", "classical"])

    # --- SAM2 ---
    sam_model: str = "sam2_hiera_tiny"   # tiny preferred for Jetson compatibility
    sam_points_per_side: int = 32

    # --- VLM ---
    vlm_model: str = "claude-sonnet-4-6"
    vlm_api_key_env: str = "ANTHROPIC_API_KEY"

    # --- SegFormer (future) ---
    segformer_checkpoint: str | None = None

    # --- output ---
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml is required to load config from YAML. pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "Config":
        """Load from yaml_path if given, otherwise look for config.yaml in cwd."""
        default_path = Path("config.yaml")
        if yaml_path is None and default_path.exists():
            yaml_path = default_path
        if yaml_path is not None:
            return cls.from_yaml(yaml_path)
        return cls()

    def vlm_api_key(self) -> str | None:
        return os.environ.get(self.vlm_api_key_env)
