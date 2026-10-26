"""
test_inference.py
-----------------
Unit/smoke tests for inference.py helper functions.

These tests do NOT run actual YOLO inference (which would require a model
download or a trained checkpoint). Instead, they verify the annotation
drawing and result parsing logic using synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import _draw_annotations  # noqa: E402


class TestDrawAnnotations:
    """Test the annotation drawing helper."""

    @pytest.fixture()
    def blank_image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture()
    def sample_detections(self) -> list[dict]:
        return [
            {"xyxy": [10, 20, 100, 120], "confidence": 0.88, "class_id": 0},
            {"xyxy": [200, 30, 350, 130], "confidence": 0.72, "class_id": 0},
        ]

    def test_returns_ndarray(self, blank_image, sample_detections):
        result = _draw_annotations(blank_image, sample_detections, 0.80, 2)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self, blank_image, sample_detections):
        result = _draw_annotations(blank_image, sample_detections, 0.80, 2)
        assert result.shape == blank_image.shape

    def test_empty_detections_no_error(self, blank_image):
        result = _draw_annotations(blank_image, [], 0.0, 0)
        assert isinstance(result, np.ndarray)

    def test_image_none_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            _draw_annotations(None, [], 0.0, 0)  # type: ignore[arg-type]

    def test_image_is_not_modified_in_place(self, blank_image, sample_detections):
        original_copy = blank_image.copy()
        _draw_annotations(blank_image, sample_detections, 0.80, 2)
        # The original blank_image should not have been mutated
        assert np.array_equal(blank_image, original_copy)
