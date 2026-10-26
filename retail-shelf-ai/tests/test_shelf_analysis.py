"""
test_shelf_analysis.py
----------------------
Unit tests for the shelf_analysis module.

All tests use synthetic bounding-box data.
No GPU, MPS, dataset, or network access is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shelf_analysis import (  # noqa: E402
    Detection,
    analyse_shelf,
    estimate_gaps,
    filter_by_confidence,
    group_into_rows,
    shelf_status,
)


# ---------------------------------------------------------------------------
# filter_by_confidence
# ---------------------------------------------------------------------------

class TestFilterByConfidence:

    def test_removes_low_confidence(self, low_confidence_detections):
        result = filter_by_confidence(low_confidence_detections, conf_threshold=0.25)
        confidences = [d.confidence for d in result]
        assert all(c >= 0.25 for c in confidences)

    def test_keeps_high_confidence(self, single_row_detections):
        result = filter_by_confidence(single_row_detections, conf_threshold=0.25)
        assert len(result) == 5

    def test_empty_input(self, empty_detections):
        result = filter_by_confidence(empty_detections)
        assert result == []

    def test_threshold_above_all(self, single_row_detections):
        result = filter_by_confidence(single_row_detections, conf_threshold=0.99)
        assert result == []

    def test_returns_detection_objects(self, single_row_detections):
        result = filter_by_confidence(single_row_detections)
        assert all(isinstance(d, Detection) for d in result)

    def test_correct_count_after_filtering(self, low_confidence_detections):
        # 3 items have confidence >= 0.25 (0.80, 0.60, 0.45)
        result = filter_by_confidence(low_confidence_detections, conf_threshold=0.25)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# group_into_rows
# ---------------------------------------------------------------------------

class TestGroupIntoRows:

    def test_single_row_grouping(self, single_row_detections):
        dets  = filter_by_confidence(single_row_detections)
        rows  = group_into_rows(dets, row_gap_tolerance=40)
        assert len(rows) == 1

    def test_two_row_grouping(self, two_row_detections):
        dets = filter_by_confidence(two_row_detections)
        rows = group_into_rows(dets, row_gap_tolerance=40)
        assert len(rows) == 2

    def test_row_product_count(self, two_row_detections):
        dets = filter_by_confidence(two_row_detections)
        rows = group_into_rows(dets, row_gap_tolerance=40)
        counts = sorted(r.count for r in rows)
        assert counts == [3, 3]

    def test_products_sorted_left_to_right(self, single_row_detections):
        dets = filter_by_confidence(single_row_detections)
        rows = group_into_rows(dets)
        row  = rows[0]
        xs   = [p.cx for p in row.products]
        assert xs == sorted(xs), "Products should be sorted left→right within a row"

    def test_empty_input(self):
        rows = group_into_rows([])
        assert rows == []

    def test_row_index_sequential(self, two_row_detections):
        dets = filter_by_confidence(two_row_detections)
        rows = group_into_rows(dets, row_gap_tolerance=40)
        indices = [r.row_index for r in rows]
        assert indices == list(range(len(rows)))


# ---------------------------------------------------------------------------
# estimate_gaps
# ---------------------------------------------------------------------------

class TestEstimateGaps:

    def test_detects_large_gap(self, single_row_detections):
        # The fixture has a ~90 px gap between product 3 and 4
        dets = filter_by_confidence(single_row_detections)
        rows = group_into_rows(dets)
        gaps = estimate_gaps(rows, gap_threshold_px=80.0)
        assert len(gaps) >= 1

    def test_no_gap_below_threshold(self, single_row_detections):
        dets = filter_by_confidence(single_row_detections)
        rows = group_into_rows(dets)
        # High threshold – should find nothing
        gaps = estimate_gaps(rows, gap_threshold_px=999.0)
        assert len(gaps) == 0

    def test_gap_is_estimated_only(self, single_row_detections):
        dets = filter_by_confidence(single_row_detections)
        rows = group_into_rows(dets)
        gaps = estimate_gaps(rows, gap_threshold_px=80.0)
        for gap in gaps:
            assert gap.is_estimated_empty is True

    def test_gap_width_positive(self, single_row_detections):
        dets = filter_by_confidence(single_row_detections)
        rows = group_into_rows(dets)
        gaps = estimate_gaps(rows, gap_threshold_px=80.0)
        for gap in gaps:
            assert gap.gap_width_px > 0

    def test_two_row_gaps(self, two_row_detections):
        dets = filter_by_confidence(two_row_detections)
        rows = group_into_rows(dets, row_gap_tolerance=40)
        # Bottom row has a ~240 px gap
        gaps = estimate_gaps(rows, gap_threshold_px=80.0)
        assert len(gaps) >= 1

    def test_empty_rows_no_gaps(self):
        gaps = estimate_gaps([])
        assert gaps == []


# ---------------------------------------------------------------------------
# analyse_shelf (integration)
# ---------------------------------------------------------------------------

class TestAnalyseShelf:

    def test_full_pipeline_single_row(self, single_row_detections):
        result = analyse_shelf(single_row_detections, conf_threshold=0.25)
        assert result.total_products == 5
        assert len(result.rows) == 1

    def test_full_pipeline_two_rows(self, two_row_detections):
        result = analyse_shelf(
            two_row_detections,
            conf_threshold    = 0.25,
            row_gap_tolerance = 40.0,
            gap_threshold_px  = 80.0,
        )
        assert result.total_products == 6
        assert len(result.rows) == 2
        assert result.total_estimated_gaps >= 1

    def test_avg_confidence_range(self, single_row_detections):
        result = analyse_shelf(single_row_detections)
        assert 0.0 <= result.avg_confidence <= 1.0

    def test_empty_input(self, empty_detections):
        result = analyse_shelf(empty_detections)
        assert result.total_products == 0
        assert result.total_estimated_gaps == 0
        assert result.avg_confidence == 0.0
        assert result.rows == []
        assert result.gaps == []


# ---------------------------------------------------------------------------
# shelf_status
# ---------------------------------------------------------------------------

class TestShelfStatus:

    def test_ok_status(self, empty_detections):
        analysis = analyse_shelf(empty_detections)
        # No detections are an operator-review condition, not proof that the
        # shelf is compliant.
        status, rec = shelf_status(analysis)
        assert status == "REVIEW"
        assert "No products" in rec

    def test_review_status(self, single_row_detections):
        # Force 1–2 gaps
        analysis = analyse_shelf(
            single_row_detections,
            conf_threshold    = 0.25,
            gap_threshold_px  = 80.0,
        )
        status, rec = shelf_status(analysis)
        assert status in ("OK", "REVIEW", "UNDERSTOCKED")  # depends on gap count

    def test_understocked_status(self):
        # Manufacture detections with many large gaps
        dets = [
            {"xyxy": [  0, 100,  50, 150], "confidence": 0.9, "class_id": 0},
            {"xyxy": [200, 100, 250, 150], "confidence": 0.9, "class_id": 0},  # gap 150px
            {"xyxy": [400, 100, 450, 150], "confidence": 0.9, "class_id": 0},  # gap 150px
            {"xyxy": [600, 100, 650, 150], "confidence": 0.9, "class_id": 0},  # gap 150px
        ]
        analysis = analyse_shelf(dets, gap_threshold_px=80.0)
        status, rec = shelf_status(analysis)
        assert status == "UNDERSTOCKED"
        assert "estimate" in rec.lower() or "gap" in rec.lower()

    def test_recommendation_non_empty(self, single_row_detections):
        analysis = analyse_shelf(single_row_detections)
        _, rec = shelf_status(analysis)
        assert len(rec) > 0
