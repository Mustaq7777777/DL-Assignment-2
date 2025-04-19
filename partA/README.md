# Multi-Layer CNN (Assignment 2 Part A)

This repository implements a **configurable** convolutional neural network (CNN) in PyTorch, designed to classify images from the iNaturalist‑12K dataset (10 classes). You can adjust architecture and training hyperparameters via command‑line arguments or by editing the `DEFAULT_CONFIG` in `config.py`.

---

## 📂 Repository Structure

```
├── config.py        # Parses CLI args and defines default hyperparameters
├── data.py          # Data loading, augmentation, and train/val/test split
├── model.py         # CNN architecture and model initialization
├── train.py        # Main training, validation, testing loop & W&B logging
├── da6401_assignment2_parta.ipynb #google colab file
└── README.md        # Usage instructions and argument reference
```

➡️ **Make sure to download all four Python files (`config.py`, `data.py`, `model.py`, `train.py`) and place them in the same directory before running the training command.**

---

## ⚙️ Prerequisites

- **Python** 3.7+  
- **PyTorch** 1.8+  
- **torchvision** 0.9+  
- **scikit‑learn** 0.24+  
- **tqdm** 4.x  
- **Weights & Biases** (`wandb`) 0.12+  

Install via:

```bash
pip install torch torchvision scikit-learn tqdm wandb
```

---

##  Dataset Layout

!!! This is mandatory

Place your iNaturalist‑12K data under a base directory with subfolders:

```
<BASE_DIR>/
├── train/    # one subdirectory per class label
└── val/      # one subdirectory per class label
```

By default, `config.py` points to:
```python
DEFAULT_CONFIG["base_dir"] = "/kaggle/input/nature-12k/inaturalist_12K"
```
Override with `--base_dir` when running.

---

##  Usage

➡️ **If you're running this on Kaggle**, replace the login step with:
```python
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")
```
Make sure to replace `YOUR_WANDB_API_KEY` with your actual key.


1. **Authenticate with W&B** (once per machine):
   ```bash
   wandb login
   ```
   Paste your API key when prompted.

2. **Run training**:
   ```bash
   python train.py \
     --wandb_entity YOUR_WANDB_ENTITY \
     --wandb_project YOUR_PROJECT_NAME \
     --base_dir /path/to/inaturalist_12K \
     --batch_size 64 \
     --num_epochs 10 \
     --learning_rate 1e-4 \
     --optimizer_name nadam \
     --activation silu \
     --augmentation true \
     --dropout 0.4 \
     --weight_decay 1e-4 \
     --batch_norm true
   ```

3. **Inspect options & defaults**:
   ```bash
   python train.py --help
   ```

4. **Outputs**  
   - **Metrics**: logged live to W&B under your entity/project  
   - **Checkpoint**: best model saved as `best_model.pth`  
   - **Final Test Accuracy**: printed at end of run  

---

## 🎛️ Command-Line Arguments

| Argument             | Description                                     | Type         | Default                      | Choices                         |
|----------------------|-------------------------------------------------|--------------|------------------------------|---------------------------------|
| `--wandb_entity`     | W&B account/entity name                         | string       | `cs24m045-indian‑institute‑of‑technology‑madras` | any valid W&B username         |
| `--wandb_project`    | W&B project name                                | string       | `DA6401‑Assignment‑2`         | any valid W&B project name      |
| `--base_dir`         | Dataset root directory                          | string       | see `DEFAULT_CONFIG`         | valid filesystem path           |
| `--batch_size`, `-b` | Mini‑batch size                                 | int          | 32                           | ≥1                              |
| `--num_epochs`, `-e` | Number of training epochs                       | int          | 8                            | ≥1                              |
| `--learning_rate`, `-lr` | Initial learning rate                      | float        | 1e‑4                         | >0                              |
| `--weight_decay`, `-w_d` | L2 regularization strength                  | float        | 0                            | ≥0                              |
| `--optimizer_name`, `-o` | Optimizer algorithm                         | string       | `nadam`                      | `nadam`, `adam`, `rmsprop`      |
| `--activation`, `-ac`| Activation function                             | string       | `silu`                       | `relu`, `gelu`, `silu`, `mish`, `elu`, `selu` |
| `--num_filters`, `-nf`| List of conv‑layer filter counts (5 values)     | int list     | `[32,64,128,256,512]`        | any list of 5 positive ints     |
| `--filter_sizes`, `-fs`| List of conv‑layer kernel sizes (5 values)     | int list     | `[5,5,5,5,5]`                | any list of 5 positive ints     |
| `--input_dim`, `-in` | Input image height & width                      | int          | 400                          | ≥32                             |
| `--batch_norm`, `-bn`| Enable batch normalization                      | bool         | true                         | `true`, `false`                 |
| `--augmentation`, `-au` | Enable data augmentation                     | bool         | false                        | `true`, `false`                 |
| `--hidden_size`, `-dl`| Hidden units in classifier                      | int          | 512                          | ≥1                              |
| `--dropout`, `-dp`   | Dropout probability in classifier                | float        | 0.4                          | 0–1                             |

---

## 📖 Notes

- The model architecture is defined in **`model.py`**, data logic in **`data.py`**, and hyperparameters in **`config.py`**.  
- Feel free to tweak `DEFAULT_CONFIG` or supply different CLI flags for experimentation.  
- Ensure your dataset directory structure matches the expected layout and that you’ve installed all prerequisites.  


