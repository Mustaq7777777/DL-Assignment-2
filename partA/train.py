"""Main training routine
This script handles the core training loop, validation, and testing procedures.
It integrates with Weights & Biases for experiment tracking and logging.
"""

import wandb
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from config import get_config
from model import initialize_model
from data import get_dataloaders
import torch.optim as optim
import torch
import torch.nn as nn

def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    """Single training epoch
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimization algorithm
        criterion: Loss function
        scaler: Gradient scaler for mixed precision training
        device: Device to run training on (CPU/GPU)
    Returns:
        Average training loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        # Mixed precision training for faster computation
        with autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=='cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
    return running_loss/len(train_loader), 100*correct/total

def validate(model, val_loader, criterion, device):
    """Validation phase
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on (CPU/GPU)
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
    return val_loss/len(val_loader), 100*correct/total

def main():
    """Main execution flow
    Handles the complete training pipeline including:
    - Configuration setup
    - W&B initialization
    - Model and data loading
    - Training loop
    - Validation and testing
    - Model saving
    """
    config, entity, project = get_config()
    
    # Initialize W&B with a descriptive run name containing all key parameters
    run_name = (
    f"optimizer={config['optimizer']} "
    f"activation={config['activation']} "
    f"num_filters={config['num_filters']} "
    f"batch_size={config['batch_size']} "
    f"learning_rate={config['learning_rate']} "
    f"filter_sizes={config['filter_sizes']} "
    f"batch_norm={config['batch_norm']} "
    f"augmentation={config['augmentation']} "
    f"weight_decay={config['weight_decay']} "
    f"img_size={config['img_size']}"
    )

    wandb.init(project=project, entity=entity, name=run_name, config=config)
    
    # Setup device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Get data and model
    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = initialize_model(config, device)
    
    # Optimizer setup - currently only NAdam is implemented
    opt_name = config["optimizer"].lower()
    optimizer = optim.NAdam(model.parameters(), 
                           lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))  # For mixed precision training
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
    
    # Final evaluation on test set using best model
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    wandb.log({"test_accuracy": test_acc})
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    wandb.finish()  # Close W&B run

if __name__ == "__main__":
    main()