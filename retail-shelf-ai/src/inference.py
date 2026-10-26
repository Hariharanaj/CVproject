"""
inference.py
------------
Single-image inference for the retail shelf product detector.

Loads a trained YOLO model, runs detection on one image, draws annotated
bounding boxes with confidence scores, saves the result to outputs/, and
returns a structured result dict.

Usage
-----
python src/inference.py --image path/to/shelf.jpg
python src/inference.py --image shelf.jpg --model models/skus_pl_top50/weights/best.pt
python src/inference.py --image shelf.jpg --device 0 --conf 0.3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from device_utils import get_device  # noqa: E402

# Default paths
DEFAULT_MODEL   = ROOT / "models" / "skus_pl_top50" / "weights" / "best.pt"
OUTPUTS_DIR     = ROOT / "outputs"


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def load_model(model_path: str | Path, device: str):
    """Load a trained YOLO model from disk without a misleading fallback."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed. Run: pip install ultralytics>=8.3.0")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Prepare SKUs on Shelves PL and train a checkpoint first."
        )

    model = YOLO(str(model_path))
    # Ultralytics accepts GPU index strings during predict/train, while
    # torch.nn.Module.to() expects a torch device object.
    if device in {"cpu", "mps", "cuda", "cuda:0"}:
        model.to(device)
    return model


def run_inference(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
    device: str | None = None,
    conf_threshold: float = 0.25,
    save_output: bool = True,
) -> dict[str, Any]:
    """Run YOLO inference on a single image.

    Args:
        image_path:      Path to the input image.
        model_path:      Path to YOLO weights (.pt file).
        device:          "0"/"cuda" | "mps" | "cpu". Auto-detected if None.
        conf_threshold:  Minimum confidence to keep a detection.
        save_output:     Whether to write the annotated image to outputs/.

    Returns:
        dict with keys: image_path, product_count, avg_confidence,
                        detections, annotated_image_path, inference_time_ms.
    """
    if device is None:
        device = get_device("auto")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(model_path, device)

    # ---- run inference --------------------------------------------------
    t0 = time.perf_counter()
    results = model.predict(
        source    = str(image_path),
        conf      = conf_threshold,
        device    = device,
        verbose   = False,
        save      = False,   # we handle saving ourselves
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result  = results[0]
    boxes   = result.boxes

    # ---- parse detections -----------------------------------------------
    detections: list[dict[str, Any]] = []
    names = getattr(result, "names", {}) or {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy  = box.xyxy[0].cpu().numpy().tolist()
            conf  = float(box.conf[0].cpu())
            cls   = int(box.cls[0].cpu())
            detections.append({
                "xyxy": xyxy,
                "confidence": conf,
                "class_id": cls,
                "class_name": _class_name(names, cls),
            })

    product_count  = len(detections)
    avg_confidence = (
        float(np.mean([d["confidence"] for d in detections]))
        if detections
        else 0.0
    )

    # ---- draw annotations -----------------------------------------------
    img = cv2.imread(str(image_path))
    annotated = _draw_annotations(img, detections, avg_confidence, product_count)

    # ---- save output image ----------------------------------------------
    annotated_path: str = ""
    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        out_file = OUTPUTS_DIR / f"{stem}_annotated.jpg"
        cv2.imwrite(str(out_file), annotated)
        annotated_path = str(out_file)
        print(f"[inference] Annotated image saved: {out_file}")

    print(
        f"[inference] Products detected : {product_count}\n"
        f"[inference] Avg confidence    : {avg_confidence:.3f}\n"
        f"[inference] Inference time    : {elapsed_ms:.1f} ms\n"
        f"[inference] Device            : {device}"
    )

    return {
        "image_path"         : str(image_path),
        "product_count"      : product_count,
        "avg_confidence"     : avg_confidence,
        "detections"         : detections,
        "annotated_image_path": annotated_path,
        "inference_time_ms"  : elapsed_ms,
        "device"             : device,
    }


def _draw_annotations(
    img: np.ndarray,
    detections: list[dict[str, Any]],
    avg_conf: float,
    count: int,
) -> np.ndarray:
    """Draw bounding boxes, confidence labels, and summary overlay."""
    if img is None:
        raise ValueError("Could not read image (cv2.imread returned None).")

    vis = img.copy()
    BOX_COLOR  = (0, 200, 50)    # green
    TEXT_COLOR = (255, 255, 255)
    FONT       = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        conf = det["confidence"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.get('class_name', 'product')} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), BOX_COLOR, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - 3), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    # Summary banner
    h, w = vis.shape[:2]
    banner_h = 36
    cv2.rectangle(vis, (0, 0), (w, banner_h), (20, 20, 20), -1)
    summary = (
        f"Products: {count}  |  Avg conf: {avg_conf:.2f}  "
        f"|  SKU-aware detector"
    )
    cv2.putText(vis, summary, (8, 24), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retail shelf image inference")
    p.add_argument("--image",   required=True, help="Path to input image")
    p.add_argument("--model",   default=str(DEFAULT_MODEL), help="Path to YOLO .pt weights")
    p.add_argument("--device",  default=None,  help="'0'/'cuda' | 'mps' | 'cpu'")
    p.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--no-save", action="store_true", help="Do not save annotated image")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        image_path    = args.image,
        model_path    = args.model,
        device        = args.device,
        conf_threshold= args.conf,
        save_output   = not args.no_save,
    )
