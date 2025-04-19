"""
Data and model utilities for Part B: Data loading and model setup.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


def get_data_loaders(cfg):
    """
    Prepare DataLoaders for training, validation, and testing.
    Args:
        cfg (dict): Configuration dictionary
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        *(([transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.2, 0.2, 0.2)] if cfg['augmentation'] else [])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(cfg['base_dir'], 'train'),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(cfg['base_dir'], 'val'),
        transform=val_transforms
    )

    # Split train into train + validation
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=train_dataset.targets,
        random_state=42
    )

    # DataLoaders
    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


def get_model(strategy):
    """
    Load pretrained ResNet-50 and apply fine-tuning strategy.
    Args:
        strategy (int): 1 => freeze backbone, 2 => train all layers
    Returns:
        model (nn.Module), trainable_params (iterator)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # Replace classifier head
    model.fc = nn.Linear(num_ftrs, 10)
    if strategy == 1:
        # Freeze all layers except final fc
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    # Collect parameters to optimize
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    return model, trainable_params