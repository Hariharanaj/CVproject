"""
device_utils.py
---------------
Shared utility: select the best available compute device.

Priority for automatic selection: CUDA → MPS (Apple Silicon) → CPU.
The same project can therefore train in MoLab on the RTX 6000 and run locally
on the user's M4 Mac.

Set the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1 before running
if you encounter ops that are not yet implemented on MPS:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

from __future__ import annotations

import os
import torch


def get_device(prefer: str = "auto") -> str:
    """Return the best available device string.

    Args:
        prefer: Preferred device ("auto", "cuda", "mps", or "cpu").
            Default is "auto". The DEVICE environment variable takes
            precedence when set.

    Returns:
        A device string accepted by Ultralytics/PyTorch.
    """
    requested = os.environ.get("DEVICE", prefer).strip().lower()

    if requested in {"0", "cuda", "cuda:0"} and torch.cuda.is_available():
        return "0" if requested == "0" else "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cpu":
        return "cpu"

    if requested in {"auto", ""}:
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # A requested accelerator is unavailable. Fall back safely rather than
    # failing at import time; the caller still receives a usable device.
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_info() -> None:
    """Print a human-readable device availability report."""
    mps_built   = torch.backends.mps.is_built()
    mps_avail   = torch.backends.mps.is_available()
    cuda_avail  = torch.cuda.is_available()
    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    print("=" * 50)
    print("  Compute Device Report")
    print("=" * 50)
    print(f"  PyTorch version      : {torch.__version__}")
    print(f"  MPS built-in support : {mps_built}")
    print(f"  MPS available        : {mps_avail}")
    print(f"  CUDA available       : {cuda_avail}")
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK : {fallback_env}")
    selected = get_device("auto")
    if selected == "0":
        print("  → Using device       : cuda:0 (NVIDIA GPU)")
    elif selected == "mps":
        print("  → Using device       : mps (Apple Metal)")
    else:
        print("  → Using device       : cpu (fallback)")
    print("=" * 50)


if __name__ == "__main__":
    print_device_info()
    device = get_device("auto")
    print(f"\nSelected device for this session: {device}")
