"""
app.py
------
Streamlit dashboard for Real-Time Retail Shelf Product Detection.

Features
--------
• Upload a shelf image.
• Run YOLO inference with one click.
• Display annotated image with bounding boxes.
• Show product count, average confidence, estimated gaps, and shelf status.
• Download a JSON report.
• Clear disclaimer that only selected frequent SKUs are supported.

Run
---
    cd retail-shelf-ai
    streamlit run src/app.py

CPU-only environments (Docker):
    DEVICE=cpu streamlit run src/app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---- lazy imports (so the page loads even if ultralytics isn't installed) --
from device_utils import get_device, print_device_info  # noqa: E402
from shelf_analysis import analyse_shelf, shelf_status   # noqa: E402
from report import build_report, save_report_json         # noqa: E402
from planogram import evaluate_planogram, load_planogram  # noqa: E402
from inference import _class_name                         # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title   = "Retail Shelf AI – Product Detector",
    page_icon    = "🛒",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

device_env = os.environ.get("DEVICE", None)
device     = device_env if device_env else get_device("auto")

st.sidebar.markdown(f"**Active device:** `{device}`")

default_model = str(ROOT / "models" / "skus_pl_top50" / "weights" / "best.pt")
model_path = st.sidebar.text_input(
    "Model weights (.pt)",
    value = default_model,
    help  = "Path to the trained SKUs on Shelves PL YOLO checkpoint.",
)

planogram_path = st.sidebar.text_input(
    "Planogram JSON (optional)",
    value="",
    help="Use configs/example_planogram.json as a template after replacing SKU names.",
)

conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value = 0.10,
    max_value = 0.90,
    value     = 0.25,
    step      = 0.05,
    help      = "Only detections above this confidence are shown.",
)

row_gap_tol = st.sidebar.slider(
    "Row gap tolerance (px)",
    min_value = 10,
    max_value = 100,
    value     = 40,
    step      = 5,
    help      = "Max vertical distance to group detections into one shelf row.",
)

gap_threshold = st.sidebar.slider(
    "Empty-gap threshold (px)",
    min_value = 30,
    max_value = 300,
    value     = 80,
    step      = 10,
    help      = "Horizontal gap wider than this is flagged as an estimated empty space.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**📖 Dataset note**\n\n"
    "This system is designed for the prepared **SKUs on Shelves PL** dataset. "
    "It recognizes selected frequent SKUs and maps unsupported products to "
    "`other_product`."
)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("🛒 Real-Time Retail Shelf Product Detection")
st.markdown(
    """
    Upload a shelf image to detect **supported SKU products**, estimate **empty gaps**,
    and optionally compare the result with a manual planogram.

    > ⚠️ **Disclaimer**: `other_product` and low-confidence detections require
    > human review. Visual shelf gaps are not inventory confirmations.
    > Gap estimates are based on bounding-box spacing and are **not** guaranteed
    > out-of-stock confirmations.
    """
)

# File uploader
uploaded = st.file_uploader(
    "Upload a shelf image",
    type   = ["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files = False,
)

if uploaded is not None:
    col1, col2 = st.columns(2)

    # Convert to PIL
    pil_img = Image.open(uploaded).convert("RGB")
    with col1:
        st.subheader("📷 Original Image")
        st.image(pil_img, use_container_width=True)

    # Analyse button
    run_btn = st.button("🔍 Analyse Shelf", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running inference … (first run may download model weights)"):
            try:
                from ultralytics import YOLO
            except ImportError:
                st.error(
                    "Ultralytics is not installed. Run:\n"
                    "`pip install ultralytics>=8.3.0`"
                )
                st.stop()

            # ---- Load model ------------------------------------------------
            mp = Path(model_path)
            if not mp.exists():
                st.error(
                    f"Trained model not found at `{model_path}`. "
                    "Train the SKUs on Shelves PL model in MoLab and copy `best.pt` here."
                )
                st.stop()
            model_path_use = model_path

            model = YOLO(model_path_use)

            # ---- Run inference on PIL image --------------------------------
            import numpy as _np
            img_np = _np.array(pil_img)
            import cv2
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            t0 = time.perf_counter()
            results = model.predict(
                source  = img_bgr,
                conf    = conf_threshold,
                device  = device,
                verbose = False,
                save    = False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            result = results[0]
            boxes  = result.boxes

            # ---- Parse detections -----------------------------------------
            raw_dets: list[dict] = []
            names = getattr(result, "names", {}) or {}
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu())
                    cls  = int(box.cls[0].cpu())
                    raw_dets.append({
                        "xyxy": xyxy,
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": _class_name(names, cls),
                    })

            # ---- Shelf analysis -------------------------------------------
            analysis = analyse_shelf(
                raw_detections    = raw_dets,
                conf_threshold    = conf_threshold,
                row_gap_tolerance = float(row_gap_tol),
                gap_threshold_px  = float(gap_threshold),
            )
            status, recommendation = shelf_status(analysis)

            planogram_result = None
            if planogram_path.strip():
                try:
                    planogram_result = evaluate_planogram(
                        analysis, load_planogram(planogram_path.strip())
                    )
                    if planogram_result["status"] == "REVIEW":
                        status = "REVIEW"
                        recommendation = planogram_result["recommendation"]
                except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                    st.warning(f"Planogram was not applied: {exc}")

            # ---- Draw annotated image ------------------------------------
            annotated_bgr = img_bgr.copy()
            for det in raw_dets:
                x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 200, 50), 2)
                label = f"{det.get('class_name', 'product')} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated_bgr, (x1, y1 - th - 3), (x1 + tw + 2, y1), (0, 200, 50), -1)
                cv2.putText(annotated_bgr, label, (x1 + 1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw gap markers (red rectangles in gap regions)
            h_img, w_img = annotated_bgr.shape[:2]
            for gap in analysis.gaps:
                gx1 = int(gap.left_product_x2)
                gx2 = int(gap.right_product_x1)
                cv2.rectangle(annotated_bgr, (gx1, 40), (gx2, h_img - 10), (0, 0, 220), 2)
                cv2.putText(annotated_bgr, "~gap",
                            (gx1 + 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1, cv2.LINE_AA)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("🔍 Annotated Image")
                st.image(annotated_rgb, use_container_width=True)

        # ---- Metrics -------------------------------------------------------
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🛍️ Products Detected", analysis.total_products)
        m2.metric("📊 Avg Confidence",    f"{analysis.avg_confidence:.2%}")
        m3.metric("⚠️ Est. Gaps",         analysis.total_estimated_gaps)
        m4.metric("📈 Shelf Rows",        len(analysis.rows))
        m5.metric("⚡ Inference",         f"{elapsed_ms:.0f} ms")

        # ---- Status banner ------------------------------------------------
        if status == "OK":
            st.success(f"✅ **{status}** — {recommendation}")
        elif status == "REVIEW":
            st.warning(f"🔶 **{status}** — {recommendation}")
        else:
            st.error(f"🔴 **{status}** — {recommendation}")

        # ---- Row breakdown -----------------------------------------------
        if analysis.rows:
            with st.expander("📋 Row Breakdown", expanded=False):
                for row in analysis.rows:
                    st.markdown(f"**Row {row.row_index + 1}** — {row.count} product(s)")

        # ---- Gap details --------------------------------------------------
        if analysis.gaps:
            with st.expander("⚠️ Estimated Gap Details", expanded=False):
                st.info(
                    "These are estimated gaps based on bounding-box spacing.  \n"
                    "They are NOT confirmed out-of-stock events."
                )
                for gap in analysis.gaps:
                    st.markdown(
                        f"• Row {gap.row_index + 1}: gap ≈ **{gap.gap_width_px:.0f} px** "
                        f"(between x={gap.left_product_x2:.0f} and x={gap.right_product_x1:.0f})"
                    )

        if planogram_result is not None:
            with st.expander("📐 Planogram Compliance", expanded=True):
                st.write(planogram_result["recommendation"])
                for issue in planogram_result["issues"]:
                    st.json(issue)

        # ---- Build & download report --------------------------------------
        report = build_report(
            source         = uploaded.name,
            product_count  = analysis.total_products,
            estimated_gaps = analysis.total_estimated_gaps,
            avg_confidence = analysis.avg_confidence,
            fps            = None,
            status         = status,
            recommendation = recommendation,
            extra          = {
                "inference_time_ms": round(elapsed_ms, 1),
                "device"           : device,
                "conf_threshold"   : conf_threshold,
                "row_count"        : len(analysis.rows),
                "planogram"        : planogram_result,
            },
        )

        report_json = json.dumps(report, indent=2)
        st.download_button(
            label     = "⬇️ Download JSON Report",
            data      = report_json,
            file_name = f"shelf_report_{Path(uploaded.name).stem}.json",
            mime      = "application/json",
        )

        # ---- Dataset disclaimer ------------------------------------------
        st.markdown("---")
        st.caption(
            "ℹ️ The model supports only the frequent SKUs selected during dataset "
            "preparation. Products outside that set appear as `other_product`; "
            "price OCR is not yet part of this prototype."
        )

else:
    st.info("👆 Upload a shelf image to get started.")

    # Show device info when idle
    active_device = get_device("auto")
    if active_device in {"0", "cuda", "cuda:0", "mps"}:
        st.success(f"GPU acceleration available — selected device: `{active_device}`")
    else:
        st.info("Running on CPU. Use MoLab with the RTX GPU for training.")
