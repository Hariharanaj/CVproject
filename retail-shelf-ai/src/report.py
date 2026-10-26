"""
report.py
---------
Generate structured inference reports in JSON and CSV formats.

A report captures the full context of a single inference run:
  source name, timestamp, frame number, product count, estimated empty gaps,
  average confidence, FPS, shelf status, and a human-readable recommendation.

Usage
-----
from report import build_report, save_report_json, save_report_csv, load_report

report = build_report(
    source="shelf.jpg",
    product_count=42,
    estimated_gaps=3,
    avg_confidence=0.71,
    fps=None,            # None for still images
    status="REVIEW",
    recommendation="3 estimated gap(s) detected. Consider restocking review.",
    frame_number=None,
)
path = save_report_json(report)
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_report(
    source: str,
    product_count: int,
    estimated_gaps: int,
    avg_confidence: float,
    fps: float | None,
    status: str,
    recommendation: str,
    frame_number: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a report dictionary.

    Args:
        source:          Image or video filename / webcam identifier.
        product_count:   Number of products detected.
        estimated_gaps:  Number of large gaps estimated (NOT guaranteed stockouts).
        avg_confidence:  Mean confidence of all detections.
        fps:             Frames per second (None for still images).
        status:          "OK" | "REVIEW" | "UNDERSTOCKED".
        recommendation:  Human-readable recommendation string.
        frame_number:    Frame index (None for still images).
        extra:           Any additional key-value pairs to embed.

    Returns:
        Report dict ready for serialisation.
    """
    report: dict[str, Any] = {
        "schema_version"    : "1.0",
        "source"            : source,
        "timestamp"         : datetime.now(timezone.utc).isoformat(),
        "frame_number"      : frame_number,
        "product_count"     : product_count,
        "estimated_gap_count": estimated_gaps,
        "avg_confidence"    : round(avg_confidence, 4),
        "fps"               : round(fps, 2) if fps is not None else None,
        "status"            : status,
        "recommendation"    : recommendation,
        "disclaimer"        : (
            "Gaps are estimates based on bounding-box spacing. "
            "They are NOT confirmed out-of-stock events."
        ),
    }
    if extra:
        report.update(extra)
    return report


# ---------------------------------------------------------------------------
# Save / load JSON
# ---------------------------------------------------------------------------

def save_report_json(
    report: dict[str, Any],
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    """Serialize a report dict to a JSON file and return the path.

    Args:
        report:     Report dict from build_report().
        output_dir: Directory to write to. Defaults to outputs/.
        filename:   Override the auto-generated filename.

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(report.get("source", "report")).stem
        filename = f"{stem}_{ts}.json"

    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[report] JSON report saved: {out_path}")
    return out_path


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a previously saved JSON report."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "schema_version",
    "source",
    "timestamp",
    "frame_number",
    "product_count",
    "estimated_gap_count",
    "avg_confidence",
    "fps",
    "status",
    "recommendation",
    "disclaimer",
]


def save_report_csv(
    reports: list[dict[str, Any]] | dict[str, Any],
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    """Append one or more reports to a CSV file.

    Args:
        reports:    Single report dict or a list of report dicts.
        output_dir: Directory to write to. Defaults to outputs/.
        filename:   Override the auto-generated filename.

    Returns:
        Path to the written CSV file.
    """
    if isinstance(reports, dict):
        reports = [reports]

    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d")
        filename = f"shelf_report_{ts}.csv"

    out_path = out_dir / filename
    write_header = not out_path.exists()

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(reports)

    print(f"[report] CSV report saved : {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Quick CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: build and save a synthetic report
    demo_report = build_report(
        source         = "demo_shelf.jpg",
        product_count  = 47,
        estimated_gaps = 2,
        avg_confidence = 0.72,
        fps            = None,
        status         = "REVIEW",
        recommendation = "2 estimated gap(s). Consider a restocking check.",
    )
    json_path = save_report_json(demo_report)
    csv_path  = save_report_csv(demo_report)
    print(f"\nDemo report  → {json_path}")
    print(f"Demo CSV     → {csv_path}")
    import json as _json
    print("\nReport contents:")
    print(_json.dumps(demo_report, indent=2))
