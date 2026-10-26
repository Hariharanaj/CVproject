"""
train.py
--------
Train a YOLO model on the prepared SKUs on Shelves PL dataset.

Apple Silicon (M4) notes
------------------------
* Device is automatically set to CUDA, MPS, or CPU.
* Set PYTORCH_ENABLE_MPS_FALLBACK=1 before running if you hit unsupported ops:
      export PYTORCH_ENABLE_MPS_FALLBACK=1
* Defaults target the MoLab RTX 6000. Override them for local M4 training.

Model choice
------------
Ultralytics ≥ 8.3 ships the YOLO11 and YOLO12 families.
The default checkpoint is **yolo11m.pt**, which is a reasonable accuracy
starting point on the MoLab RTX 6000. Use a smaller checkpoint for local tests.

Usage
-----
# Activate your virtual environment first.
python src/train.py                        # CUDA / MPS / CPU auto-detect
python src/train.py --device 0             # MoLab CUDA GPU 0
python src/train.py --device mps           # local Mac GPU
python src/train.py --epochs 30 --batch 2  # reduce batch for low memory
python src/train.py --resume               # resume from last checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from device_utils import get_device, print_device_info  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The medium checkpoint is a useful accuracy/speed starting point on an RTX 6000.
MODEL_NAME: str = "yolo11m.pt"

# Path to dataset YAML (relative to repo root)
DATA_YAML: str = str(ROOT / "data" / "SKUs_on_Shelves_PL" / "SKUs_on_Shelves_PL.yaml")

# Output directory for training runs
RUNS_DIR: str = str(ROOT / "models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the prepared SKUs on Shelves PL dataset"
    )
    parser.add_argument("--model",   default=MODEL_NAME, help="YOLO checkpoint name")
    parser.add_argument("--data",    default=DATA_YAML,   help="Path to generated dataset YAML")
    parser.add_argument("--epochs",  type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz",   type=int, default=960, help="Image size (px)")
    parser.add_argument("--batch",   type=int, default=-1, help="Batch size; -1 auto-fits GPU memory")
    parser.add_argument("--workers", type=int, default=8,  help="DataLoader workers")
    parser.add_argument("--device",  default=None,
                        help="Compute device: 0/cuda, mps, or cpu. Auto-detected if omitted.")
    parser.add_argument("--project", default=RUNS_DIR,    help="Output project directory")
    parser.add_argument("--name",    default="skus_pl_top50", help="Run name")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--resume",  action="store_true",  help="Resume last training run")
    parser.add_argument("--cfg",     default=str(ROOT / "configs" / "train_config.yaml"),
                        help="Hyperparameter YAML override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- device selection ------------------------------------------------
    device = args.device if args.device else get_device("auto")
    print_device_info()
    print(f"\n[train] Using device : {device}")
    print(f"[train] Model        : {args.model}")
    print(f"[train] Data config  : {args.data}")
    print(f"[train] Epochs       : {args.epochs}")
    print(f"[train] Batch size   : {args.batch}")
    print(f"[train] Image size   : {args.imgsz}")
    print(f"[train] Workers      : {args.workers}")

    # ---- verify data config exists ---------------------------------------
    if not os.path.isfile(args.data):
        print(
            f"\n[ERROR] Dataset config not found: {args.data}\n"
            "Prepare SKUs on Shelves PL first with scripts/download_skus_shelves_pl.py\n"
            "and scripts/prepare_skus_shelves_pl.py. See README.md for MoLab steps."
        )
        sys.exit(1)

    # ---- import Ultralytics here (after path setup) ---------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[ERROR] Ultralytics is not installed.\n"
            "Run: pip install ultralytics>=8.3.0"
        )
        sys.exit(1)

    # ---- load model ------------------------------------------------------
    model = YOLO(args.model)

    # ---- train -----------------------------------------------------------
    train_kwargs: dict = dict(
        data    = args.data,
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = device,
        workers = args.workers,
        cache   = False,
        patience = args.patience,
        project = args.project,
        name    = args.name,
        exist_ok= True,
        resume  = args.resume,
    )

    # Include hyperparameter cfg only if the file exists
    if os.path.isfile(args.cfg):
        train_kwargs["cfg"] = args.cfg

    print("\n[train] Starting training …")
    results = model.train(**train_kwargs)

    # ---- post-training info ----------------------------------------------
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[train] Training complete.")
    if best_weights.exists():
        print(f"[train] Best weights saved to: {best_weights}")
    else:
        print("[train] Training results saved under:", args.project)

    return results


if __name__ == "__main__":
    main()
