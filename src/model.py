"""ECG Anomaly Detection Model.

Implements a 1D convolutional neural network for detecting cardiac
anomalies in single-lead ECG recordings.
"""

import torch
import torch.nn as nn


class ECGConvBlock(nn.Module):
    """Single convolutional block for ECG feature extraction.

    Applies Conv1d -> BatchNorm -> ReLU -> MaxPool sequentially.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block.

        Args:
            x: Input tensor of shape (batch, channels, length).

        Returns:
            Output tensor after conv, batchnorm, relu, and pooling.
        """
        return self.pool(self.relu(self.bn(self.conv(x))))


class ECGAnomalyDetector(nn.Module):
    """1D CNN for ECG anomaly classification.

    Architecture: three convolutional blocks followed by global average
    pooling and a fully connected classifier head.

    Args:
        num_classes: Number of output classes (default: 5 for MIT-BIH).
        input_length: Expected input signal length (default: 360 samples).
    """

    def __init__(self, num_classes: int = 5, input_length: int = 360) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ECGConvBlock(1, 32),
            ECGConvBlock(32, 64),
            ECGConvBlock(64, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            x: Input ECG signal of shape (batch, 1, length).

        Returns:
            Class logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)
