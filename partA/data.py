"""Data loading and preprocessing
This module handles all data-related operations including:
- Data augmentation
- Dataset splitting
- DataLoader creation
"""

import os
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_transforms(cfg):
    """Create data transformations
    Args:
        cfg: Configuration dictionary
    Returns:
        Composition of image transformations
    """
    if cfg["augmentation"]:
        return transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])),  # Resize to target
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomRotation(30),  # Random rotation up to 30 degrees
            transforms.ColorJitter(0.2, 0.2, 0.2),  # Random color adjustments
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])
    return transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_dataloaders(cfg):
    """Create train/val/test dataloaders
    Args:
        cfg: Configuration dictionary
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transforms = get_transforms(cfg)
    val_transforms = get_transforms(cfg)
    
    # Load datasets from directory structure
    train_dataset = datasets.ImageFolder(
        Path(cfg["base_dir"]) / "train",
        transform=train_transforms
    )
    test_dataset = datasets.ImageFolder(
        Path(cfg["base_dir"]) / "val",
        transform=val_transforms
    )
    
    # Split training set into train/validation (80/20)
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=train_dataset.targets,  # Maintain class distribution
        random_state=42  # For reproducibility
    )
    
    # Determine number of workers for data loading
    num_workers = min(2, os.cpu_count() or 1)  # Use up to 2 workers
    
    return (
        DataLoader(Subset(train_dataset, train_idx),
                  batch_size=cfg["batch_size"],
                  shuffle=True,  # Shuffle training data
                  num_workers=num_workers,
                  pin_memory=True),  # Faster transfer to GPU
        DataLoader(Subset(train_dataset, val_idx),
                  batch_size=cfg["batch_size"],
                  shuffle=False,  # No need to shuffle validation
                  num_workers=num_workers,
                  pin_memory=True),
        DataLoader(test_dataset,
                  batch_size=cfg["batch_size"],
                  shuffle=False,  # No need to shuffle test
                  num_workers=num_workers,
                  pin_memory=True)
    )