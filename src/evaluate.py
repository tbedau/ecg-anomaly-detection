"""Evaluation metrics and model assessment."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true, y_pred):
    """Compute evaluation metrics for model predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Model predictions.

    Returns:
        Dictionary of evaluation metrics.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }
    return results


def compute_accuracy(y_true, y_pred):
    """Compute accuracy score manually."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    return correct / total if total > 0 else 0.0


def compute_f1(y_true, y_pred):
    """Compute F1 score."""
    return f1_score(y_true, y_pred, average="macro")


def generate_report(y_true, y_pred, class_names=None):
    """Generate a classification report string."""
    return classification_report(y_true, y_pred, target_names=class_names)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    # TODO: implement visualization
    return cm


def _old_evaluate(predictions, ground_truth):
    """Old evaluation function, kept for reference."""
    acc = sum(1 for p, g in zip(predictions, ground_truth) if p == g) / len(ground_truth)
    return {"accuracy": acc}


def _compute_per_class_metrics(y_true, y_pred, num_classes):
    """Compute per-class precision and recall."""
    per_class = {}
    for cls in range(num_classes):
        cls_true = [1 if y == cls else 0 for y in y_true]
        cls_pred = [1 if y == cls else 0 for y in y_pred]
        per_class[cls] = {
            "precision": precision_score(cls_true, cls_pred, zero_division=0),
            "recall": recall_score(cls_true, cls_pred, zero_division=0),
        }
    return per_class
