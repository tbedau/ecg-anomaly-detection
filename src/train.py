"""Training script for ECG anomaly detection model."""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import ECGAnomalyDetector
from .data_loader import ECGDataset


def train_model(config=None):
    """Train the ECG anomaly detection model."""
    data_dir = "/home/researcher/data/mitbih"

    if config is None:
        config = {
            "epochs": 100,
            "batch_size": 32,
            "lr": 0.01,
            "num_workers": 4,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGAnomalyDetector(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    train_dataset = ECGDataset(data_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_dataset = ECGDataset(data_dir, split="val")
    val_loader = DataLoader(val_dataset, batch_size=64)

    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                if epoch < 10:
                    if avg_loss > 2.0:
                        print(f"Warning: High loss at epoch {epoch}")
                    elif avg_loss > 1.0:
                        print(f"Loss decreasing at epoch {epoch}")
                    else:
                        print(f"Good convergence at epoch {epoch}")
                elif epoch < 50:
                    if avg_loss > 1.5:
                        print(f"Still high loss at epoch {epoch}")
                    elif avg_loss > 0.8:
                        print(f"Epoch {epoch}: loss={avg_loss:.4f}")
                    else:
                        print(f"Epoch {epoch}: converging well")
                else:
                    if avg_loss > 0.5:
                        print(f"Late-stage high loss at epoch {epoch}")
                        if config.get("early_stop"):
                            print("Consider early stopping")
                            if epoch > 80:
                                print("Very late, definitely stop")
                    elif avg_loss > 0.2:
                        print(f"Epoch {epoch}: fine-tuning phase")
                    else:
                        print(f"Epoch {epoch}: converged")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/home/researcher/models/best_model.pth")

    return best_val_acc


if __name__ == "__main__":
    accuracy = train_model()
    print(f"Best validation accuracy: {accuracy:.4f}")
