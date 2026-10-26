"""
shelf_analysis.py
-----------------
Shelf-level analysis of YOLO detection results.

Responsibilities
----------------
1. Group bounding-box detections into approximate shelf rows using the
   vertical (Y) centre of each box and a configurable row-gap tolerance.
2. Within each row, sort products left to right.
3. Estimate unusually large horizontal gaps between adjacent products.
4. Return a structured ShelfAnalysis dataclass.

⚠️  IMPORTANT DISCLAIMER
These gaps are **estimates** derived from bounding-box spacing.
They are NOT guaranteed stockout alerts. Occlusion, labelling errors,
and camera angle can all produce false positives or false negatives.

All thresholds are configurable so callers can tune to their use case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Minimal representation of one YOLO bounding-box detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "product"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class ShelfRow:
    """Products grouped into a single approximate shelf row."""
    row_index: int
    products: list[Detection]        # sorted left → right

    @property
    def count(self) -> int:
        return len(self.products)

    @property
    def y_center_mean(self) -> float:
        if not self.products:
            return 0.0
        return float(np.mean([p.cy for p in self.products]))


@dataclass
class GapEstimate:
    """An estimated large gap between two adjacent products on a row."""
    row_index: int
    left_product_x2: float          # right edge of the left product
    right_product_x1: float         # left edge of the right product
    gap_width_px: float             # gap size in pixels
    is_estimated_empty: bool = True # always True – never a guaranteed stockout


@dataclass
class ShelfAnalysis:
    """Full analysis result for one detection frame."""
    rows: list[ShelfRow]
    gaps: list[GapEstimate]
    total_products: int
    total_estimated_gaps: int
    avg_confidence: float
    conf_threshold: float
    row_gap_tolerance: float
    gap_threshold_px: float


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def filter_by_confidence(
    detections: list[dict[str, Any]],
    conf_threshold: float = 0.25,
) -> list[Detection]:
    """Convert raw YOLO dicts to Detection objects, filtering low-confidence boxes.

    Args:
        detections:      List of dicts with keys 'xyxy', 'confidence', 'class_id'.
        conf_threshold:  Minimum confidence to keep.

    Returns:
        List of Detection objects above the threshold.
    """
    result: list[Detection] = []
    for d in detections:
        if d["confidence"] < conf_threshold:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        result.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=d["confidence"],
                class_id=d.get("class_id", 0),
                class_name=d.get("class_name", "product"),
            )
        )
    return result


def group_into_rows(
    detections: list[Detection],
    row_gap_tolerance: float = 40.0,
) -> list[ShelfRow]:
    """Group detections into shelf rows by vertical proximity of centres.

    Args:
        detections:        List of Detection objects (any order).
        row_gap_tolerance: Max vertical distance (pixels) between box centres
                           to be considered the same row. Tune per image scale.

    Returns:
        List of ShelfRow objects sorted top → bottom, products sorted left → right.
    """
    if not detections:
        return []

    # Sort by Y-centre then X-centre
    sorted_dets = sorted(detections, key=lambda d: (d.cy, d.cx))

    rows: list[ShelfRow] = []
    current_row_dets: list[Detection] = [sorted_dets[0]]
    current_y_ref = sorted_dets[0].cy

    for det in sorted_dets[1:]:
        if abs(det.cy - current_y_ref) <= row_gap_tolerance:
            current_row_dets.append(det)
            # Update running mean of Y centres
            current_y_ref = float(np.mean([d.cy for d in current_row_dets]))
        else:
            rows.append(
                ShelfRow(
                    row_index=len(rows),
                    products=sorted(current_row_dets, key=lambda d: d.cx),
                )
            )
            current_row_dets = [det]
            current_y_ref = det.cy

    # Flush last row
    if current_row_dets:
        rows.append(
            ShelfRow(
                row_index=len(rows),
                products=sorted(current_row_dets, key=lambda d: d.cx),
            )
        )

    return rows


def estimate_gaps(
    rows: list[ShelfRow],
    gap_threshold_px: float = 80.0,
) -> list[GapEstimate]:
    """Identify unusually large horizontal gaps between adjacent products.

    A gap is flagged when the horizontal distance between the right edge of
    one product and the left edge of the next exceeds `gap_threshold_px`.

    ⚠️  Estimated gaps are NOT confirmed stockout positions. They indicate
    unusually large spacing that warrants human review.

    Args:
        rows:             List of ShelfRow objects.
        gap_threshold_px: Gap width in pixels above which to flag an estimate.

    Returns:
        List of GapEstimate objects.
    """
    gaps: list[GapEstimate] = []
    for row in rows:
        products = row.products
        for i in range(len(products) - 1):
            left  = products[i]
            right = products[i + 1]
            gap_w = right.x1 - left.x2
            if gap_w > gap_threshold_px:
                gaps.append(
                    GapEstimate(
                        row_index        = row.row_index,
                        left_product_x2  = left.x2,
                        right_product_x1 = right.x1,
                        gap_width_px     = gap_w,
                    )
                )
    return gaps


def analyse_shelf(
    raw_detections: list[dict[str, Any]],
    conf_threshold: float     = 0.25,
    row_gap_tolerance: float  = 40.0,
    gap_threshold_px: float   = 80.0,
) -> ShelfAnalysis:
    """Full shelf analysis pipeline.

    Args:
        raw_detections:    Output from YOLO inference (list of dicts).
        conf_threshold:    Minimum detection confidence.
        row_gap_tolerance: Max Y-distance to group detections into one row.
        gap_threshold_px:  Min gap width to flag as an estimated empty space.

    Returns:
        ShelfAnalysis dataclass.
    """
    filtered = filter_by_confidence(raw_detections, conf_threshold)
    rows     = group_into_rows(filtered, row_gap_tolerance)
    gaps     = estimate_gaps(rows, gap_threshold_px)

    avg_conf = (
        float(np.mean([d.confidence for d in filtered]))
        if filtered
        else 0.0
    )

    return ShelfAnalysis(
        rows                  = rows,
        gaps                  = gaps,
        total_products        = len(filtered),
        total_estimated_gaps  = len(gaps),
        avg_confidence        = avg_conf,
        conf_threshold        = conf_threshold,
        row_gap_tolerance     = row_gap_tolerance,
        gap_threshold_px      = gap_threshold_px,
    )


def shelf_status(analysis: ShelfAnalysis) -> tuple[str, str]:
    """Derive a shelf status label and recommendation from an analysis result.

    Returns:
        (status, recommendation) strings.
    """
    gaps = analysis.total_estimated_gaps
    if analysis.total_products == 0:
        return (
            "REVIEW",
            "No products were detected. Check the image, model path, and confidence threshold.",
        )
    if gaps == 0:
        return "OK", "Shelf appears adequately stocked."
    elif gaps <= 2:
        return (
            "REVIEW",
            f"{gaps} estimated gap(s) detected. Consider restocking review."
        )
    else:
        return (
            "UNDERSTOCKED",
            f"{gaps} estimated gap(s) detected. Restocking likely needed. "
            "(Note: gaps are estimates, not confirmed stockouts.)"
        )
