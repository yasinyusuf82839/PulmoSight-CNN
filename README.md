# PulmoSight-CNN

A deep-learning histopathology classifier that distinguishes **normal lung tissue** from **adenocarcinoma** and **squamous cell carcinoma**.

## 1) Project Objective and Class Definitions

### Objective
PulmoSight-CNN is designed to train and evaluate a convolutional neural network for **3-class lung histopathology image classification**.

### Classes
- `normal`: Histologically normal lung tissue.
- `adenocarcinoma`: Lung adenocarcinoma tissue.
- `squamous`: Lung squamous cell carcinoma tissue.

## 2) Dataset Layout and File Naming Assumptions

Expected dataset root structure (class-per-folder):

```text
dataset/
├── normal/
│   ├── normal_0001.png
│   ├── normal_0002.png
│   └── ...
├── adenocarcinoma/
│   ├── adenocarcinoma_0001.png
│   ├── adenocarcinoma_0002.png
│   └── ...
└── squamous/
    ├── squamous_0001.png
    ├── squamous_0002.png
    └── ...
```

### Assumptions
- Images are readable by common Python imaging backends (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`).
- Folder names map directly to class labels (`normal`, `adenocarcinoma`, `squamous`).
- Filenames are unique per class directory.
- Corrupted/non-image files are removed before training.

## 3) Environment Setup

### Python version
Use **Python 3.10+** (recommended: Python 3.10 or 3.11).

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

If a dependency file exists:

```bash
pip install -r requirements.txt
```

Otherwise install the typical stack manually:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy pillow
```

## 4) Train and Evaluate Commands

> Adjust paths/flags to match your local scripts.

### Train

```bash
python train.py \
  --data-dir dataset \
  --output-dir outputs \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --seed 42
```

### Evaluate

```bash
python evaluate.py \
  --data-dir dataset \
  --checkpoint outputs/best_model.pth \
  --output-dir outputs \
  --batch-size 32
```

## 5) Expected Output Artifacts

After training/evaluation, users should expect artifacts similar to:

```text
outputs/
├── best_model.pth                # model checkpoint (best validation metric)
├── train_history.csv             # epoch-wise metrics
├── loss_curve.png                # train/val loss vs epoch
├── accuracy_curve.png            # train/val accuracy vs epoch
├── confusion_matrix.png          # class-wise confusion matrix on eval set
└── classification_report.txt     # precision/recall/F1 summary
```

## 6) Reproducibility Notes

For reproducible experiments:

- **Seed setting**: fix all relevant random seeds (Python, NumPy, PyTorch).
- **Split ratio**: document and keep fixed (e.g., `70/15/15` for train/val/test).
- **Key hyperparameters to log**:
  - optimizer (e.g., Adam)
  - learning rate (e.g., `1e-4`)
  - batch size (e.g., `32`)
  - epochs (e.g., `30`)
  - image size/input preprocessing
  - augmentation settings
  - loss function (e.g., CrossEntropyLoss)

Recommended deterministic flags for PyTorch workflows:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 7) Optional: Module-to-Workflow Mapping

Use this as a reference when organizing implementation files:

1. **Data ingestion & labeling** → `data_loader.py` / `dataset.py`
2. **Preprocessing & augmentation** → `transforms.py`
3. **Model definition** → `model.py`
4. **Training loop** → `train.py`
5. **Validation & checkpointing** → `trainer.py` / `train.py`
6. **Evaluation & confusion matrix** → `evaluate.py` / `metrics.py`
7. **Visualization/reporting** → `plot_utils.py` / `reports.py`

If your repository uses different filenames, keep this mapping conceptually aligned with the same workflow stages.
