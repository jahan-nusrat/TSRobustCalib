# 🧠 Improving the robustness and calibration of neural networks in time series analysis

## 📁 Project Structure

```
├── code_projects/
│   ├── corruption/
│   │   ├── __init__.py
│   │   └── corrupt_dataset.py           # Script to generate corrupted EEG segments
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py                # Raw EEG preprocessing to standardized format
│   │   └── tusz_utilities.py            # TUH channel/montage handling
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_loader.py               # DataLoader with variable-channel support
│   │   ├── models.py                    # ResNet-based model definitions
│   │   └── trainer.py                   # Training loop and RobustTrainer class
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py                 # Global constants and paths
│   │   ├── metrics.py                   # Accuracy, F1, ECE, etc.
│   │   └── visualization/
│   │       ├── __init__.py
│   │       ├── visualize_corruption.py  # Visualize corrupted segments
│   │       └── visualize_preprocess.py  # Visualize raw/preprocessed segments
│
├── job.sh                               # Shell script for cluster execution (optional)
├── main_train.py                        # Train a model (edit config inside)
├── main_evaluate.py                     # Evaluate model checkpoint (with TTA support)
├── requirements.txt                     # Python package dependencies
└── README.md
```

---

## ⚙️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Default Training Configuration

| Parameter     | Value                             |
| ------------- | --------------------------------- |
| Batch Size    | `32`                              |
| Epochs        | `120`                             |
| Optimizer     | `Adam`                            |
| Learning Rate | `5e-4`                            |
| Weight Decay  | `1e-5`                            |
| Scheduler     | `StepLR(gamma=0.1, step_size=10)` |

These values are set in `main_train.py` and can be modified for experimentation.

---

## 🧹 Step 1: Preprocess the EEG Data

To generate preprocessed `.mat` files from raw TUH EEG recordings:

```bash
python preprocess.py
```

This will:

- Load and clean EEG recordings
- Select a standardized set of channels
- Resample to a fixed sampling rate (e.g., 250 Hz)
- Save the data in a training-ready format

## 🔧 Step 2: Generate Corrupted Versions

To simulate real-world signal distortions (noise, dropout, etc.):

```bash
python corrupt_dataset.py
```

This script applies multiple corruption types across severity levels (1-5) and saves them to disk for later evaluation.

## 🧠 Step 3: Train a Model

To train a model, open main_train.py and configure the following:

```bash
model_type = "ResNet18Robust"     # Options: ResNet18Robust, ResNet18MIMO, ResNet18ManifoldMixup, MIMOManifoldMixup
ensemble_size = 1                 # >1 for ensemble or MIMO
use_manifoldmixup = True          # Only for ManifoldMixup models
```

Then launch training: python main_train.py

Model checkpoints and logs will be saved in a directory named runs/

## 📊 Step 4: Evaluate the Trained Model

To evaluate performance on the corrupted test set with optional test-time adaptation (TTA), configure main_evaluate.py:

```bash
model_type = "ResNet18ManifoldMixup" # Options: ResNet18Robust, ResNet18MIMO, ResNet18ManifoldMixup, MIMOManifoldMixup
ensemble_size = 1 # >1 for multi-head
checkpoint_path = "/path/to/checkpoint.pt"
apply_tta = True #or False if you dont want to apply TTA
```

Then run: python main_evaluate.py. The evaluation script reports metrics such as:

- F1 score
- Accuracy
- Expected Calibration Error (ECE)

## 📉 Data Loading and Subsampling

Both training and validation datasets are loaded using a fraction argument

```bash
train_dataset = EEGDataset(data_dir=train_data_path, fraction=0.2)
val_dataset = EEGDataset(data_dir=val_data_path, fraction=0.2)
```

- Only 20% of the dataset is used for training/validation by default.
- Set fraction=1.0 to use the full dataset.
