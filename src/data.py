
"""TV1 - CIFAR-10 data pipeline."""

from copy import copy
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms


def get_transforms(image_size: int = 224):
    """Create separate transforms for training and evaluation."""
    if type(image_size) is not int or image_size < 32:
        raise ValueError("image_size must be an integer >= 32.")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def build_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Return train/val/test loaders and class metadata."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary.")

    batch_size = config.get("batch_size", 32)
    seed = config.get("seed", 42)
    num_workers = config.get("num_workers", 0)
    val_ratio = config.get("val_ratio", 0.1)
    image_size = config.get("image_size", 224)
    data_dir = Path(config.get("data_dir", "./data"))

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    if type(val_ratio) not in (int, float) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    train_transform, eval_transform = get_transforms(image_size)

    full_train = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=eval_transform,
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError("Invalid train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_indices = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    val_full = copy(full_train)
    val_full.transform = eval_transform
    val_dataset = Subset(val_full, val_indices.indices)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return {
        "train": DataLoader(
            train_subset, shuffle=True, **loader_options
        ),
        "val": DataLoader(
            val_dataset, shuffle=False, **loader_options
        ),
        "test": DataLoader(
            test_dataset, shuffle=False, **loader_options
        ),
        "class_names": full_train.classes,
        "num_classes": len(full_train.classes),
    }


__all__ = ["get_transforms", "build_dataloaders"]
