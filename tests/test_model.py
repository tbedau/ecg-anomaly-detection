"""Tests for the ECG model architecture."""

import torch

from src.model import ECGAnomalyDetector


def test_model_output_shape():
    """Test that model produces correct output dimensions."""
    model = ECGAnomalyDetector(num_classes=5)
    x = torch.randn(4, 1, 360)
    output = model(x)
    assert output.shape == (4, 5)


def test_model_different_classes():
    """Test model with a different number of output classes."""
    model = ECGAnomalyDetector(num_classes=3, input_length=250)
    x = torch.randn(2, 1, 250)
    output = model(x)
    assert output.shape == (2, 3)
