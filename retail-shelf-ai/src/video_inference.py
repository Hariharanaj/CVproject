"""
video_inference.py
------------------
Video and webcam real-time inference for the retail shelf product detector.

Features
--------
* Supports webcam (source=0) and prerecorded video (path string).
* Displays product count and FPS overlay on each frame.
* Press 'q' to exit the live window.
* Configurable frame-skip to keep pace on slower hardware.
* CUDA/MPS-accelerated when available; CPU fallback otherwise.
* Efficient pipeline: only annotated frames are drawn; raw frames are forwarded.

Usage
-----
# Webcam (default device 0)
python src/video_inference.py --source 0

# Prerecorded video
python src/video_inference.py --source path/to/shelf_video.mp4

# Skip every other frame (faster on CPU)
python src/video_inference.py --source 0 --skip 2

# Save output video
python src/video_inference.py --source shelf.mp4 --save

# Force CPU
python src/video_inference.py --source 0 --device cpu
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
from inference    import load_model, DEFAULT_MODEL, _class_name  # noqa: E402

OUTPUTS_DIR = ROOT / "outputs"


def _draw_video_overlay(
    frame: np.ndarray,
    product_count: int,
    fps: float,
    detections: list[dict[str, Any]],
    conf_threshold: float,
    inference_ran: bool,
) -> np.ndarray:
    """Draw bounding boxes and HUD overlay on a video frame."""
    vis = frame.copy()
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    BOX_COLOR  = (0, 200, 50)
    TEXT_COLOR = (255, 255, 255)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        conf = det["confidence"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.get('class_name', 'product')} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 3), (x1 + tw + 2, y1), BOX_COLOR, -1)
        cv2.putText(vis, label, (x1 + 1, y1 - 2), FONT, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

    # HUD banner
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 36), (20, 20, 20), -1)
    hud = (
        f"Products: {product_count}  |  FPS: {fps:.1f}  "
        f"|  Conf ≥ {conf_threshold:.2f}  |  Press 'q' to quit"
    )
    if not inference_ran:
        hud += " | frame skipped"
    cv2.putText(vis, hud, (8, 24), FONT, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return vis


def run_video_inference(
    source: str | int = 0,
    model_path: str | Path = DEFAULT_MODEL,
    device: str | None = None,
    conf_threshold: float = 0.25,
    frame_skip: int = 1,
    save_output: bool = False,
    display: bool = True,
) -> None:
    """Run real-time inference on webcam or video file.

    Args:
        source:         0 for webcam, or a file path string.
        model_path:     Path to trained YOLO .pt weights.
        device:         "mps" | "cpu". Auto-detected if None.
        conf_threshold: Minimum confidence to keep detections.
        frame_skip:     Process 1 out of every N frames (1 = all frames).
        save_output:    Whether to write annotated video to outputs/.
        display:        Whether to show the live OpenCV window.
    """
    if device is None:
        device = get_device("auto")

    model = load_model(model_path, device)

    # Open capture
    cap_source = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"[ERROR] Cannot open video source: {source}\n"
            "  • For webcam: ensure no other app is using it.\n"
            "  • For file:   check the path exists and codec is supported."
        )

    # Video properties
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Video writer (optional)
    writer: cv2.VideoWriter | None = None
    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        src_name = Path(str(source)).stem if isinstance(source, str) else "webcam"
        out_path = OUTPUTS_DIR / f"{src_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_native, (w, h))
        print(f"[video] Saving output video to: {out_path}")

    print(f"[video] Source  : {source}")
    print(f"[video] Device  : {device}")
    print(f"[video] Frame skip: every {frame_skip} frame(s)")
    print(f"[video] Press 'q' in the display window to quit.\n")

    frame_idx     = 0
    fps_measured  = 0.0
    t_fps         = time.perf_counter()
    fps_frame_cnt = 0
    last_detections: list[dict[str, Any]] = []
    last_count      = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[video] Stream ended or frame read failed.")
                break

            frame_idx    += 1
            fps_frame_cnt += 1

            # FPS calculation (updated every 30 frames)
            if fps_frame_cnt >= 30:
                elapsed = time.perf_counter() - t_fps
                fps_measured  = fps_frame_cnt / elapsed
                fps_frame_cnt = 0
                t_fps         = time.perf_counter()

            # Frame-skip: run inference only on selected frames
            inference_ran = frame_idx % max(1, frame_skip) == 0
            if inference_ran:
                results = model.predict(
                    source  = frame,
                    conf    = conf_threshold,
                    device  = device,
                    verbose = False,
                )
                res = results[0]
                boxes = res.boxes
                last_detections = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu())
                        cls  = int(box.cls[0].cpu())
                        names = getattr(res, "names", {}) or {}
                        last_detections.append({
                            "xyxy": xyxy,
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": _class_name(names, cls),
                        })
                last_count = len(last_detections)

            # Do not display detections from a previous frame: after camera
            # motion they can be visibly wrong. Skipped frames show a clear
            # status banner and no boxes.
            display_detections = last_detections if inference_ran else []
            display_count = last_count if inference_ran else 0
            annotated = _draw_video_overlay(
                frame, display_count, fps_measured, display_detections,
                conf_threshold, inference_ran
            )

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Retail Shelf Detector", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[video] 'q' pressed – exiting.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        print(f"[video] Processed {frame_idx} frames. Final FPS ≈ {fps_measured:.1f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retail shelf video / webcam inference")
    p.add_argument("--source",  default="0",  help="Video source: 0=webcam or file path")
    p.add_argument("--model",   default=str(DEFAULT_MODEL), help="YOLO .pt weights path")
    p.add_argument("--device",  default=None, help="'0'/'cuda' | 'mps' | 'cpu'")
    p.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--skip",    type=int,   default=1,    help="Frame skip factor (1=all)")
    p.add_argument("--save",    action="store_true",      help="Save annotated video")
    p.add_argument("--no-display", action="store_true",   help="Disable OpenCV window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run_video_inference(
        source         = source,
        model_path     = args.model,
        device         = args.device,
        conf_threshold = args.conf,
        frame_skip     = args.skip,
        save_output    = args.save,
        display        = not args.no_display,
    )
