"""Small, explicit planogram rule engine for shelf compliance.

The detector supplies observations; this module compares them to a manually
authored JSON planogram. It deliberately does not infer backroom inventory or
claim a product is out of stock from a visual gap alone.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from shelf_analysis import ShelfAnalysis


def load_planogram(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Planogram not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload.get("rows"), list):
        raise ValueError("Planogram must contain a 'rows' list")
    return payload


def evaluate_planogram(
    analysis: ShelfAnalysis,
    planogram: dict[str, Any],
) -> dict[str, Any]:
    """Compare observed SKU counts to expected counts per detected row."""
    issues: list[dict[str, Any]] = []
    row_results: list[dict[str, Any]] = []
    expected_rows = {int(row.get("row_index", index)): row for index, row in enumerate(planogram["rows"])}

    for row_index, expected_row in expected_rows.items():
        actual_row = next((row for row in analysis.rows if row.row_index == row_index), None)
        observed = Counter(product.class_name for product in actual_row.products) if actual_row else Counter()
        expected = Counter({str(k): int(v) for k, v in expected_row.get("expected_counts", {}).items()})
        missing = {sku: expected[sku] - observed[sku] for sku in expected if observed[sku] < expected[sku]}
        extra = {sku: observed[sku] - expected[sku] for sku in observed if observed[sku] > expected[sku]}
        misplaced = {sku: observed[sku] for sku in observed if sku not in expected}
        if missing or extra or misplaced:
            issues.append({
                "row_index": row_index,
                "missing": missing,
                "extra": extra,
                "misplaced": misplaced,
            })
        row_results.append({
            "row_index": row_index,
            "expected_counts": dict(expected),
            "observed_counts": dict(observed),
            "missing": missing,
            "extra": extra,
            "misplaced": misplaced,
        })

    if issues:
        status = "REVIEW"
        recommendation = "Planogram differences found; verify flagged rows manually."
    else:
        status = "PASS"
        recommendation = "Observed product counts match the configured planogram."
    return {
        "status": status,
        "recommendation": recommendation,
        "issue_count": len(issues),
        "issues": issues,
        "rows": row_results,
    }
