"""Preprocessing and augmentation transforms."""

from __future__ import annotations

from typing import Dict, Tuple

from torchvision import transforms


def build_transforms(image_size: Tuple[int, int]) -> Dict[str, transforms.Compose]:
    """Build train/validation transforms.

    Args:
        image_size: Target image size as (height, width).

    Returns:
        Dictionary with ``train`` and ``val`` transform pipelines.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {"train": train_transform, "val": val_transform}
