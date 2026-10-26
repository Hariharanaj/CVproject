from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_skus_shelves_pl import choose_categories, split_image_ids


def _payload():
    return {
        "images": [{"id": index, "file_name": f"{index}.jpg"} for index in range(1, 7)],
        "categories": [
            {"id": 1, "name": "sku_a"},
            {"id": 2, "name": "sku_b"},
            {"id": 3, "name": "unknown_product"},
        ],
        "annotations": [
            *({"image_id": index, "category_id": 1, "bbox": [0, 0, 10, 10]} for index in range(1, 7)),
            *({"image_id": index, "category_id": 2, "bbox": [0, 0, 10, 10]} for index in range(1, 4)),
        ],
    }


def test_selects_frequent_categories_and_excludes_ambiguous():
    selected = choose_categories(_payload(), top_k=2, min_instances=2)
    assert [item["name"] for item in selected] == ["sku_a", "sku_b"]


def test_splits_are_deterministic_and_disjoint():
    payload = _payload()
    annotations_by_image = {image_id: [] for image_id in range(1, 7)}
    for annotation in payload["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    first = split_image_ids(list(range(1, 7)), annotations_by_image, {1, 2}, 0.2, 0.2, 42)
    second = split_image_ids(list(range(1, 7)), annotations_by_image, {1, 2}, 0.2, 0.2, 42)
    assert first == second
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])
