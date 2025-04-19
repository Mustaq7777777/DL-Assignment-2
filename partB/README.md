# Fine‑Tuning ResNet‑50 on iNaturalist‑12K (Assignment 2 Part B)

This repository fine‑tunes a pretrained ResNet‑50 model on the iNaturalist‑12K dataset using PyTorch and Weights & Biases.

---

## 📂 Repository Structure

```
├── config.py        # Parses CLI args and fixed default hyperparameters
├── data_model.py    # Data loading, augmentation, and pretrained model setup
├── train.py         # Training, validation, and testing logic with W&B run name
├── dl_assignment2_partb.ipynb #colab note book
└── README.md        # Instructions and argument reference
```

➡️ **Ensure all three Python files (`config.py`, `data_model.py`, `train.py`) are placed in the same directory before running.**

---

## ⚙️ Prerequisites

- **Python** 3.7+  
- **PyTorch** 1.8+  
- **torchvision** 0.9+  
- **scikit‑learn** 0.24+  
- **tqdm** 4.x  
- **Weights & Biases** (`wandb`) 0.12+  

Install dependencies via:

```bash
pip install torch torchvision scikit-learn tqdm wandb
```

---

## Dataset Layout

!!!This is mandatory

Organize your iNaturalist‑12K dataset under a base directory:  

```
<BASE_DIR>/
├── train/    # subfolders per class label
└── val/      # subfolders per class label
```

By default, `config.py` uses:

```python
DEFAULT_CONFIG['base_dir'] = 'inaturalist_12K'
```

Override with `--base_dir` when running.

---

## How to Run

1. **Authenticate with W&B** (once per environment):
   ```python
   import wandb
   wandb.login(key="YOUR_WANDB_API_KEY")
   ```

2. **Run training** (image size fixed at 224 inside code):
   ```bash
   python train.py \
     --wandb_entity YOUR_WANDB_ENTITY \
     --wandb_project YOUR_PROJECT_NAME \
     --base_dir /path/to/inaturalist_12K \
     --batch_size 64 \
     --learning_rate 1e-4 \
     --augmentation true \
     --strategy 1 \
     --num_epochs 10
   ```

3. **Inspect options & defaults**:
   ```bash
   python train.py --help
   ```

4. **Outputs**:
   - **W&B Run Name**: auto-generated as `resnet50_strat{strategy}_bs{batch}_lr{lr}_aug{aug}_ep{epochs}`  
   - **Metrics**: logged to W&B under your entity/project  
   - **Checkpoint**: best model saved as `best_model.pth` in working directory  
   - **Final Test Accuracy**: printed at the end of the run

---

## 🎛️ Command‑Line Arguments

| Argument             | Description                                         | Type   | Default       | Choices       |
|----------------------|-----------------------------------------------------|--------|---------------|---------------|
| `--wandb_entity`     | W&B account/entity name                             | string | _none_        | any valid     |
| `--wandb_project`    | W&B project name                                    | string | _none_        | any valid     |
| `--base_dir`, `-br`  | Path to dataset root (train/ and val/ folders)      | string | `inaturalist_12K` | valid path |
| `--batch_size`, `-b` | Mini‑batch size                                     | int    | 64            | ≥1            |
| `--learning_rate`, `-lr` | Initial learning rate                          | float  | 1e‑4          | >0            |
| `--augmentation`, `-au` | Enable data augmentation                         | bool   | true          | `true`, `false` |
| `--strategy`, `-s`   | Fine‑tuning strategy (1=freeze backbone)            | int    | 1             | 1, 2          |
| `--num_epochs`, `-e` | Number of training epochs                           | int    | 10            | ≥1            |

---

## 📖 Notes

- **Image size** is fixed at 224×224 inside code; `--img_size` argument has been removed.  
- The **run name** logged to W&B includes strategy, batch size, learning rate, augmentation flag, and epochs for clarity.  
- Ensure your dataset directory matches the layout and prerequisites are installed.


