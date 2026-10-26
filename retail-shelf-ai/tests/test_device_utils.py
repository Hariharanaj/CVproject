"""
test_device_utils.py
--------------------
Unit tests for device selection logic.

These tests are intentionally lightweight and do NOT require MPS to be
available. They verify the selection logic works correctly on all platforms
including CPU-only environments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from device_utils import get_device  # noqa: E402


class TestGetDevice:

    def test_returns_mps_when_available_and_preferred(self):
        with patch("torch.backends.mps.is_available", return_value=True):
            device = get_device(prefer="mps")
        assert device == "mps"

    def test_falls_back_to_cpu_when_mps_unavailable(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_device(prefer="mps")
        assert device == "cpu"

    def test_cpu_preferred_always_returns_cpu(self):
        # Even when MPS is available, CPU preference must be respected
        with patch("torch.backends.mps.is_available", return_value=True):
            device = get_device(prefer="cpu")
        assert device == "cpu"

    def test_default_prefer_is_mps(self):
        # Default call → prefers MPS
        with patch("torch.backends.mps.is_available", return_value=True):
            device = get_device()
        assert device == "mps"

    def test_returns_string(self):
        device = get_device()
        assert isinstance(device, str)

    def test_valid_device_values(self):
        for avail in (True, False):
            with patch("torch.backends.mps.is_available", return_value=avail):
                device = get_device()
            assert device in ("mps", "cpu", "0")

    def test_auto_prefers_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            assert get_device("auto") == "0"

    def test_explicit_cuda_falls_back_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            assert get_device("cuda") == "mps"
