#!/usr/bin/env python3
"""Prepare SKUs on Shelves PL for a practical exact-SKU detector.

The public dataset is distributed as COCO annotations with roughly 8,000
product categories and one listed split.  Training all classes immediately is
not a useful first milestone, so this script:

* selects the most frequent SKU categories;
* maps every remaining/ambiguous product to ``other_product``;
* creates deterministic train/val/test splits;
* converts annotations to YOLO format; and
* writes a ready-to-use dataset YAML and class map.

The script is intentionally independent of Hugging Face.  Download and
extract the archive first with ``download_skus_shelves_pl.py`` or provide a
local extracted directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AMBIGUOUS_NAMES = {
    "unreadable",
    "unknown_product",
    "unknown product",
    "n/a",
    "na",
}


def find_coco_json(root: Path) -> Path:
    """Find the first JSON file that has COCO image/annotation/category keys."""
    candidates = sorted(root.rglob("*.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(key in payload for key in ("images", "annotations", "categories")):
            return path
    raise FileNotFoundError(f"No COCO annotation JSON found below {root}")


def load_coco(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"COCO file is missing keys: {sorted(missing)}")
    return payload


def category_counts(payload: dict[str, Any]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for annotation in payload["annotations"]:
        bbox = annotation.get("bbox", [])
        if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
            counts[int(annotation["category_id"])] += 1
    return counts


def choose_categories(
    payload: dict[str, Any],
    top_k: int,
    min_instances: int,
) -> list[dict[str, Any]]:
    counts = category_counts(payload)
    categories = {
        int(category["id"]): category
        for category in payload["categories"]
    }
    eligible = []
    for category_id, count in counts.items():
        category = categories.get(category_id, {})
        name = str(category.get("name", category_id)).strip()
        if name.lower() in AMBIGUOUS_NAMES or count < min_instances:
            continue
        eligible.append({"id": category_id, "name": name, "instances": count})
    eligible.sort(key=lambda item: (-item["instances"], item["name"].lower()))
    return eligible[:top_k]


def dominant_group(
    image_id: int,
    annotations_by_image: dict[int, list[dict[str, Any]]],
    selected_ids: set[int],
) -> int | str:
    counts: Counter[int | str] = Counter()
    for annotation in annotations_by_image.get(image_id, []):
        category_id = int(annotation["category_id"])
        counts[category_id if category_id in selected_ids else "other"] += 1
    return counts.most_common(1)[0][0] if counts else "other"


def split_image_ids(
    image_ids: list[int],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    selected_ids: set[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    """Make deterministic approximate-stratified image-level splits."""
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    groups: dict[int | str, list[int]] = defaultdict(list)
    for image_id in image_ids:
        groups[dominant_group(image_id, annotations_by_image, selected_ids)].append(image_id)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for group_ids in groups.values():
        rng.shuffle(group_ids)
        total = len(group_ids)
        n_test = max(1, round(total * test_ratio)) if total >= 3 else 0
        n_val = max(1, round(total * val_ratio)) if total >= 3 else 0
        n_test = min(n_test, max(0, total - 2))
        n_val = min(n_val, max(0, total - n_test - 1))
        splits["test"].extend(group_ids[:n_test])
        splits["val"].extend(group_ids[n_test:n_test + n_val])
        splits["train"].extend(group_ids[n_test + n_val:])

    for split_ids in splits.values():
        split_ids.sort()
    return splits


def resolve_image(root: Path, file_name: str) -> Path:
    candidates = [root / file_name, root / Path(file_name).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    basename = Path(file_name).name
    matches = list(root.rglob(basename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image referenced by COCO annotations was not found: {file_name}")


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)


def yolo_line(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
    class_id: int,
) -> str:
    x, y, width, height = [float(value) for value in annotation["bbox"][:4]]
    x = max(0.0, min(x, image_width - 1))
    y = max(0.0, min(y, image_height - 1))
    width = max(1.0, min(width, image_width - x))
    height = max(1.0, min(height, image_height - y))
    cx = (x + width / 2) / image_width
    cy = (y + height / 2) / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {width / image_width:.6f} {height / image_height:.6f}"


def write_yaml(output: Path, names: list[str]) -> None:
    lines = [
        "# Generated by scripts/prepare_skus_shelves_pl.py",
        f"path: {output.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(names))
    (output / "SKUs_on_Shelves_PL.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(
    source_root: Path,
    coco_path: Path,
    output: Path,
    top_k: int,
    min_instances: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    link_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}. Use --overwrite to rebuild it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    payload = load_coco(coco_path)
    selected = choose_categories(payload, top_k=top_k, min_instances=min_instances)
    selected_ids = {int(item["id"]) for item in selected}
    categories = {int(item["id"]): str(item.get("name", item["id"])) for item in payload["categories"]}
    names = [item["name"] for item in selected] + ["other_product"]
    class_by_original = {int(item["id"]): index for index, item in enumerate(selected)}
    other_class_id = len(selected)

    images = {int(image["id"]): image for image in payload["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        bbox = annotation.get("bbox", [])
        if len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    splits = split_image_ids(
        list(images), annotations_by_image, selected_ids, val_ratio, test_ratio, seed
    )
    counts_by_class = Counter()

    for split, split_ids in splits.items():
        for image_id in split_ids:
            image = images[image_id]
            source_image = resolve_image(source_root, str(image["file_name"]))
            stem = f"{image_id:08d}_{Path(source_image).stem}{source_image.suffix.lower()}"
            destination_image = output / "images" / split / stem
            destination_label = output / "labels" / split / f"{Path(stem).stem}.txt"
            materialize_image(source_image, destination_image, link_mode)

            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                from PIL import Image
                with Image.open(source_image) as opened:
                    width, height = opened.size

            labels = []
            for annotation in annotations_by_image.get(image_id, []):
                original_id = int(annotation["category_id"])
                class_id = class_by_original.get(original_id, other_class_id)
                labels.append(yolo_line(annotation, width, height, class_id))
                counts_by_class[names[class_id]] += 1
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

    class_map = {
        "classes": [
            {"class_id": index, "name": name, "source_category_id": int(selected[index]["id"])}
            for index, name in enumerate(names[:-1])
        ] + [{"class_id": other_class_id, "name": "other_product", "source_category_id": None}],
        "top_k": top_k,
        "min_instances": min_instances,
        "seed": seed,
    }
    (output / "class_map.json").write_text(json.dumps(class_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "source_root": str(source_root),
        "coco_path": str(coco_path),
        "total_images": len(images),
        "total_annotations": sum(counts_by_class.values()),
        "selected_skus": selected,
        "class_instance_counts": dict(counts_by_class),
        "split_image_counts": {split: len(ids) for split, ids in splits.items()},
        "link_mode": link_mode,
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_yaml(output, names)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Extracted dataset directory")
    parser.add_argument("--coco", type=Path, default=None, help="COCO JSON; auto-detected when omitted")
    parser.add_argument("--output", type=Path, default=Path("data/SKUs_on_Shelves_PL"))
    parser.add_argument("--top-k", type=int, default=50, help="Number of frequent SKU classes to retain")
    parser.add_argument("--min-instances", type=int, default=100, help="Minimum annotated instances for a retained SKU")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    coco_path = (args.coco or find_coco_json(source_root)).resolve()
    summary = prepare_dataset(
        source_root=source_root,
        coco_path=coco_path,
        output=args.output.resolve(),
        top_k=args.top_k,
        min_instances=args.min_instances,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
