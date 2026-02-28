"""Validation metrics and confusion matrix generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
    output_dir: str,
) -> Dict[str, str | float]:
    """Run validation and save confusion matrix + classification report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            probs = model(images)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    cm_path = output_path / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    report_path = output_path / "classification_report.txt"
    report_path.write_text(report)

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    return {
        "val_accuracy": round(accuracy, 4),
        "confusion_matrix_path": str(cm_path),
        "classification_report_path": str(report_path),
    }
