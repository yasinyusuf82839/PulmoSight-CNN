"""Utilities for dataset and batch inspection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder


INV_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
INV_NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def plot_class_distribution(dataset_dir: str, save_path: str | None = None) -> Counter:
    """Plot and optionally save class distribution from folder labels."""
    dataset = ImageFolder(dataset_dir)
    counts = Counter(dataset.targets)

    labels = [dataset.classes[i] for i in sorted(counts)]
    values = [counts[i] for i in sorted(counts)]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    return counts


def inspect_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    save_path: str,
    max_images: int = 9,
) -> None:
    """Save a grid preview of a batch with class labels."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n = min(max_images, images.size(0))
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(n):
        img = images[i].detach().cpu() * INV_NORM_STD + INV_NORM_MEAN
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i].item()])
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
