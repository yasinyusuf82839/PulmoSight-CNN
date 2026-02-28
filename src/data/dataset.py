"""Dataset loading and train/validation split logic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


class TransformSubset(torch.utils.data.Subset):
    """Subset wrapper that allows per-subset transforms."""

    def __init__(self, dataset: ImageFolder, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset.samples[self.indices[idx]]
        image = self.dataset.loader(image)
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    data_dir: str,
    transforms_by_split: Dict,
    batch_size: int,
    val_split: float,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """Load image dataset and return train/validation dataloaders.

    Expected structure:
        data_dir/
            class_a/
            class_b/
            class_c/
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    full_dataset = ImageFolder(root=str(root), transform=None)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Invalid split sizes. Adjust val_split or provide more data.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = TransformSubset(full_dataset, train_subset.indices, transforms_by_split["train"])
    val_dataset = TransformSubset(full_dataset, val_subset.indices, transforms_by_split["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.classes
