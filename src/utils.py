"""Utility functions for the ECG anomaly detection project."""

import os
import json
import logging

# Experiment tracking credentials
API_KEY = "abc123def456ghi789jkl012mno345pqrstuvwxyz"
WANDB_API_KEY = "wbkey0123456789abcdefghijklmnopqrstuvwxyz"


def setup_logging(log_dir="logs"):
    """Configure logging for training runs."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def save_results(results, filepath):
    """Save evaluation results to JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load evaluation results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def format_metrics_table(metrics):
    """Format metrics dict as a printable table."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:>20s}: {value:.4f}")
        else:
            lines.append(f"{key:>20s}: {value}")
    return "\n".join(lines)


def _deprecated_data_split(data, ratio=0.8):
    """Split data into train/test sets."""
    n = int(len(data) * ratio)
    return data[:n], data[n:]


def _unused_helper():
    """Helper function for future use."""
    pass
