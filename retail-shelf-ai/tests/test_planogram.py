from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from planogram import evaluate_planogram
from shelf_analysis import analyse_shelf


def _analysis():
    detections = [
        {"xyxy": [0, 100, 50, 160], "confidence": 0.9, "class_id": 0, "class_name": "sku_a"},
        {"xyxy": [55, 100, 105, 160], "confidence": 0.9, "class_id": 0, "class_name": "sku_a"},
        {"xyxy": [110, 100, 160, 160], "confidence": 0.9, "class_id": 1, "class_name": "sku_b"},
    ]
    return analyse_shelf(detections, gap_threshold_px=1000)


def test_planogram_passes_matching_counts():
    result = evaluate_planogram(
        _analysis(),
        {"rows": [{"row_index": 0, "expected_counts": {"sku_a": 2, "sku_b": 1}}]},
    )
    assert result["status"] == "PASS"
    assert result["issue_count"] == 0


def test_planogram_flags_missing_and_unexpected_products():
    result = evaluate_planogram(
        _analysis(),
        {"rows": [{"row_index": 0, "expected_counts": {"sku_a": 3}}]},
    )
    assert result["status"] == "REVIEW"
    assert result["issue_count"] == 1
    assert result["issues"][0]["missing"] == {"sku_a": 1}
    assert result["issues"][0]["misplaced"] == {"sku_b": 1}
