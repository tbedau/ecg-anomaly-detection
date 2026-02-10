"""Tests for ECG preprocessing functions."""

import numpy as np

from src.preprocess import bandpass_filter, detect_r_peaks


def test_bandpass_filter_preserves_length():
    """Test that bandpass filter preserves signal length."""
    sig = np.random.randn(1000)
    filtered = bandpass_filter(sig, 0.5, 40.0, 360.0)
    assert len(filtered) == len(sig)


def test_detect_r_peaks_returns_list():
    """Test that R-peak detection returns a list."""
    sig = np.sin(np.linspace(0, 10 * np.pi, 3600))
    peaks = detect_r_peaks(sig, 360.0)
    assert isinstance(peaks, list)
