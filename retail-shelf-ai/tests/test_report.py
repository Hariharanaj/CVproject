"""
test_report.py
--------------
Unit tests for the report generation module.

No GPU, MPS, dataset, or network access required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from report import (  # noqa: E402
    build_report,
    load_report,
    save_report_csv,
    save_report_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_report(**overrides) -> dict:
    defaults = dict(
        source         = "test_shelf.jpg",
        product_count  = 20,
        estimated_gaps = 1,
        avg_confidence = 0.72,
        fps            = None,
        status         = "REVIEW",
        recommendation = "1 estimated gap(s). Consider restocking check.",
    )
    defaults.update(overrides)
    return build_report(**defaults)


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:

    def test_required_keys_present(self):
        r = _sample_report()
        for key in ("source", "timestamp", "product_count",
                    "estimated_gap_count", "avg_confidence",
                    "fps", "status", "recommendation", "disclaimer"):
            assert key in r, f"Missing key: {key}"

    def test_values_match_input(self):
        r = _sample_report(product_count=42, estimated_gaps=3, status="UNDERSTOCKED")
        assert r["product_count"]       == 42
        assert r["estimated_gap_count"] == 3
        assert r["status"]              == "UNDERSTOCKED"

    def test_fps_none_for_image(self):
        r = _sample_report(fps=None)
        assert r["fps"] is None

    def test_fps_set_for_video(self):
        r = _sample_report(fps=24.5)
        assert r["fps"] == 24.5

    def test_avg_confidence_rounded(self):
        r = _sample_report(avg_confidence=0.123456789)
        # build_report rounds to 4dp
        assert r["avg_confidence"] == 0.1235

    def test_frame_number_default_none(self):
        r = _sample_report()
        assert r["frame_number"] is None

    def test_frame_number_set(self):
        r = _sample_report(frame_number=42)
        assert r["frame_number"] == 42

    def test_extra_fields_included(self):
        r = build_report(
            source         = "x.jpg",
            product_count  = 5,
            estimated_gaps = 0,
            avg_confidence = 0.5,
            fps            = None,
            status         = "OK",
            recommendation = "Fine",
            extra          = {"device": "mps", "custom": 99},
        )
        assert r["device"]  == "mps"
        assert r["custom"]  == 99

    def test_disclaimer_contains_estimate_language(self):
        r = _sample_report()
        assert "estimate" in r["disclaimer"].lower()

    def test_schema_version_present(self):
        r = _sample_report()
        assert "schema_version" in r


# ---------------------------------------------------------------------------
# save / load JSON
# ---------------------------------------------------------------------------

class TestJsonReport:

    def test_save_creates_file(self, tmp_path):
        r = _sample_report()
        path = save_report_json(r, output_dir=tmp_path)
        assert path.exists()

    def test_saved_content_is_valid_json(self, tmp_path):
        r = _sample_report()
        path = save_report_json(r, output_dir=tmp_path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["product_count"] == r["product_count"]

    def test_load_report_roundtrip(self, tmp_path):
        r = _sample_report(product_count=77)
        path = save_report_json(r, output_dir=tmp_path)
        loaded = load_report(path)
        assert loaded["product_count"] == 77

    def test_load_report_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_report("/nonexistent/path/report.json")

    def test_custom_filename(self, tmp_path):
        r    = _sample_report()
        path = save_report_json(r, output_dir=tmp_path, filename="custom.json")
        assert path.name == "custom.json"


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCsvReport:

    def test_save_creates_file(self, tmp_path):
        r    = _sample_report()
        path = save_report_csv(r, output_dir=tmp_path)
        assert path.exists()

    def test_csv_has_header(self, tmp_path):
        r    = _sample_report()
        path = save_report_csv(r, output_dir=tmp_path, filename="test.csv")
        with open(path) as f:
            first_line = f.readline()
        assert "product_count" in first_line

    def test_csv_appends_multiple_rows(self, tmp_path):
        r1 = _sample_report(product_count=10)
        r2 = _sample_report(product_count=20)
        path = save_report_csv(r1, output_dir=tmp_path, filename="multi.csv")
        save_report_csv(r2, output_dir=tmp_path, filename="multi.csv")
        lines = Path(path).read_text().splitlines()
        # header + 2 data rows
        assert len(lines) == 3

    def test_list_of_reports(self, tmp_path):
        reports = [_sample_report(product_count=i) for i in range(5)]
        path = save_report_csv(reports, output_dir=tmp_path, filename="batch.csv")
        lines = Path(path).read_text().splitlines()
        assert len(lines) == 6  # header + 5 rows
