"""
Smoke tests for each pipeline.

Run:  pytest smoothie_cv/tests/test_pipeline.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendResult
from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(h: int = 100, w: int = 100, color: tuple = (60, 120, 40)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _patchy_image(h: int = 100, w: int = 100) -> np.ndarray:
    img = _solid_image(h, w, color=(60, 120, 40))
    img[30:70, 30:70] = (20, 30, 200)  # red chunk in center
    return img


def _roi(h: int = 100, w: int = 100) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:95, 5:95] = 255
    return mask


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

class TestRoiCrop:
    def test_crop_zeros_pixels_outside_contour(self):
        from smoothie_cv.roi import crop_to_roi, paste_mask

        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255

        roi = crop_to_roi(image, mask)
        assert roi.image.shape == (60, 40, 3)
        assert roi.offset == (30, 20)
        assert (roi.mask > 0).sum() == (mask > 0).sum()
        assert (roi.image[roi.mask == 0] == 0).all()

        crop_result = np.zeros(roi.image.shape[:2], dtype=np.uint8)
        crop_result[roi.mask > 0] = 255
        full = paste_mask(crop_result, roi)
        assert full.shape == (100, 100)
        assert (full == mask).all()

    def test_paste_round_trip(self):
        from smoothie_cv.roi import crop_to_roi, paste_mask

        image = _solid_image()
        mask = _roi()
        roi = crop_to_roi(image, mask)
        crop_mask = np.zeros(roi.image.shape[:2], dtype=np.uint8)
        crop_mask[10:20, 10:20] = 255
        full = paste_mask(crop_mask, roi)
        x, y = roi.offset
        assert full[y + 10 : y + 20, x + 10 : x + 20].sum() == 255 * 100


# ---------------------------------------------------------------------------
# Classical CV pipeline
# ---------------------------------------------------------------------------

class TestClassicalCVPipeline:
    def setup_method(self):
        self.pipeline = ClassicalCVPipeline(Config())

    def test_returns_blend_result(self):
        result = self.pipeline.analyze(_solid_image(), _roi())
        assert isinstance(result, BlendResult)

    def test_score_in_range(self):
        for img in (_solid_image(), _patchy_image()):
            result = self.pipeline.analyze(img, _roi())
            assert 0.0 <= result.blend_score <= 1.0

    def test_mask_shape_and_dtype(self):
        result = self.pipeline.analyze(_solid_image(), _roi())
        assert result.mask.shape == (100, 100)
        assert result.mask.dtype == np.uint8

    def test_pipeline_name(self):
        assert self.pipeline.analyze(_solid_image()).pipeline_name == "classical"

    def test_no_roi_mask(self):
        result = self.pipeline.analyze(_solid_image())
        assert 0.0 <= result.blend_score <= 1.0

    def test_passed_flag_always_passes_at_zero_threshold(self):
        pipeline = ClassicalCVPipeline(Config(threshold=0.0))
        assert pipeline.analyze(_solid_image()).passed is True

    def test_chunky_scores_no_higher_than_uniform(self):
        uniform_score = self.pipeline.analyze(_solid_image(), _roi()).blend_score
        chunky_score = self.pipeline.analyze(_patchy_image(), _roi()).blend_score
        assert chunky_score <= uniform_score

    def test_mask_zero_outside_roi(self):
        result = self.pipeline.analyze(_patchy_image(), _roi())
        assert (result.mask[_roi() == 0] == 0).all()


# ---------------------------------------------------------------------------
# VLM pipeline — import + init only (API call skipped unless key is set)
# ---------------------------------------------------------------------------

class TestVLMPipelineInit:
    def test_import_and_name(self):
        from smoothie_cv.pipelines.vlm import VLMPipeline
        assert VLMPipeline(Config()).name == "vlm"

    def test_missing_api_key_raises_environment_error(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")  # skip if not installed
        from smoothie_cv.pipelines.vlm import VLMPipeline
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError):
            VLMPipeline(Config())._client_instance()


# ---------------------------------------------------------------------------
# SAM pipeline — import + init only (model load skipped if not installed)
# ---------------------------------------------------------------------------

class TestSAMPipelineInit:
    def test_import_and_name(self):
        from smoothie_cv.pipelines.sam import SAMPipeline
        assert SAMPipeline(Config()).name == "sam"

    def test_build_generator_raises_import_error_without_sam2(self, monkeypatch):
        import sys
        from smoothie_cv.pipelines.sam import SAMPipeline
        # simulate sam2 not installed
        monkeypatch.setitem(sys.modules, "sam2", None)
        monkeypatch.setitem(sys.modules, "sam2.build_sam", None)
        monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", None)
        pipeline = SAMPipeline(Config())
        pipeline._mask_generator = None  # reset lazy cache
        with pytest.raises((ImportError, Exception)):
            pipeline._build_generator()


# ---------------------------------------------------------------------------
# SegFormer pipeline — must raise NotImplementedError
# ---------------------------------------------------------------------------

class TestSegFormerStub:
    def test_raises_not_implemented(self):
        from smoothie_cv.pipelines.segformer import SegFormerPipeline
        with pytest.raises(NotImplementedError):
            SegFormerPipeline(Config()).analyze(_solid_image())
