"""
conftest.py
-----------
Shared pytest fixtures for the retail-shelf-ai test suite.

Tests use only synthetic data: no dataset, GPU, or MPS required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src/ importable without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Synthetic detection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def single_row_detections() -> list[dict]:
    """5 products in a straight horizontal row."""
    return [
        {"xyxy": [  0, 100, 80, 180], "confidence": 0.90, "class_id": 0},
        {"xyxy": [ 90, 102, 170, 182], "confidence": 0.85, "class_id": 0},
        {"xyxy": [180, 99,  260, 179], "confidence": 0.80, "class_id": 0},
        {"xyxy": [350, 101, 430, 181], "confidence": 0.75, "class_id": 0},  # gap ~90 px
        {"xyxy": [440, 100, 520, 180], "confidence": 0.70, "class_id": 0},
    ]


@pytest.fixture()
def two_row_detections() -> list[dict]:
    """Products on two separate shelf rows (top ~y=140, bottom ~y=340)."""
    top_row = [
        {"xyxy": [  0, 100, 80, 180], "confidence": 0.88, "class_id": 0},
        {"xyxy": [ 90, 105, 170, 185], "confidence": 0.82, "class_id": 0},
        {"xyxy": [180, 100, 260, 180], "confidence": 0.79, "class_id": 0},
    ]
    bottom_row = [
        {"xyxy": [  0, 300, 80, 380], "confidence": 0.91, "class_id": 0},
        {"xyxy": [ 90, 305, 170, 385], "confidence": 0.86, "class_id": 0},
        {"xyxy": [330, 300, 410, 380], "confidence": 0.77, "class_id": 0},  # gap ~240 px
    ]
    return top_row + bottom_row


@pytest.fixture()
def low_confidence_detections() -> list[dict]:
    """Mix of high- and low-confidence detections for filtering tests."""
    return [
        {"xyxy": [0, 0, 50, 50],   "confidence": 0.80, "class_id": 0},
        {"xyxy": [60, 0, 110, 50], "confidence": 0.15, "class_id": 0},  # below threshold
        {"xyxy": [120, 0, 170, 50],"confidence": 0.60, "class_id": 0},
        {"xyxy": [180, 0, 230, 50],"confidence": 0.10, "class_id": 0},  # below threshold
        {"xyxy": [240, 0, 290, 50],"confidence": 0.45, "class_id": 0},
    ]


@pytest.fixture()
def empty_detections() -> list[dict]:
    return []
