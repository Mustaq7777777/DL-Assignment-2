"""
Training script for Part B: Fine-tune pretrained ResNet-50.
"""
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from config import get_config
from data_model import get_data_loaders, get_model


def train():
    # Load configuration
    cfg = get_config()

    #run name
    run_name = (
    f"batch_size={cfg['batch_size']} "
    f"learning_rate={cfg['learning_rate']} "
    f"augmentation={cfg['augmentation']} "
    )

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # Prepare model
    model, params = get_model(cfg['strategy'])
    model = nn.DataParallel(model).to(device)

    # Optimizer, loss, scaler
    optimizer = optim.NAdam(params, lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Initialize W&B
    wandb.login()
    wandb.init(
        entity=cfg['wandb_entity'],
        name = run_name,
        project=cfg['wandb_project'],
        config=cfg
    )

    best_val_acc = 0.0
    for epoch in range(1, cfg['num_epochs'] + 1):
        # Training phase
        model.train()
        running_loss = correct = total = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['num_epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=='cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['num_epochs']}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                preds = outputs.argmax(1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Log metrics & checkpoint
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })

        # printing accuracy  values

        print(f"Epoch {epoch}/{cfg['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # Testing phase
    model.load_state_dict(torch.load('best_model.pth'))
    test_correct = test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            test_correct += (preds == targets).sum().item()
            test_total += targets.size(0)
    test_acc = 100 * test_correct / test_total
    wandb.log({'test_acc': test_acc})
    print(f"Test Accuracy: {test_acc:.2f}%")
    wandb.finish()


if __name__ == '__main__':
    train()
