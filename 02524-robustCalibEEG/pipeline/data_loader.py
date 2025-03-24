import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

############################
# Data Augmentation Functions
############################
def random_channel_drop(spectrogram: torch.Tensor, drop_prob=0.3) -> torch.Tensor:
    """
    Randomly drops (zeros out) some channels with probability drop_prob.
    Args:
      spectrogram: shape [C, F, T]
      drop_prob: probability of dropping each channel

    Returns:
      spectrogram: shape [C, F, T] with some channels zeroed out
    """
    c, _, _ = spectrogram.shape
    for ch in range(c):
        if random.random() < drop_prob:
            spectrogram[ch, :, :] = 0.0
    return spectrogram


############################
# EEGDataset
############################
class EEGDataset(Dataset):
    """
    Loads .mat files under data_dir, each .mat containing:
      'spectrogram' -> shape [C, F, T]
      'label'       -> shape [C], [C,1], or [1] (we reduce to a single scalar 0/1).

    fraction: fraction of files to load
    verbose: if True, prints details about loaded files
    is_train: if True, apply data augmentations
    """

    def __init__(self, data_dir: str, fraction: float = 1.0, verbose: bool = False, is_train: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.verbose = verbose
        self.is_train = is_train

        # 1) Gather all .mat files
        mat_paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".mat"):
                    mat_paths.append(os.path.join(root, f))

        if not mat_paths:
            raise ValueError(f"No .mat files found under {data_dir}")

        # 2) Shuffle and keep fraction
        random.shuffle(mat_paths)
        cutoff = int(len(mat_paths) * fraction)
        mat_paths = mat_paths[:cutoff]

        if self.verbose:
            print(f"[EEGDataset] Found {len(mat_paths)} .mat files in '{data_dir}' (fraction={fraction}).")
            for i, p in enumerate(mat_paths[:5]):
                print(f"  -> Example file #{i+1}: {p}")
            if len(mat_paths) > 5:
                print("  ...")

        # 3) Build samples
        self.samples = []
        for path in mat_paths:
            mat_dict = loadmat(path)
            label_arr = mat_dict["label"]  # shape [C], [C,1], or [1]

            # Convert to single label (0 or 1)
            has_seizure = (label_arr > 0).any()
            label_scalar = 1 if has_seizure else 0

            self.samples.append((path, label_scalar))

        # Summaries
        self.num_seizure = sum(lbl for (_, lbl) in self.samples)
        self.num_normal = len(self.samples) - self.num_seizure
        print(f"[EEGDataset] {data_dir}: Seizure samples = {self.num_seizure}, Normal samples = {self.num_normal}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return: (spectrogram [C,F,T], label_scalar)
        """
        path, label_scalar = self.samples[idx]
        mat_dict = loadmat(path)
        spectrogram = mat_dict["spectrogram"]  # shape [C, F, T]
        spectrogram = torch.from_numpy(spectrogram).float()

        # Apply augmentations only in training mode
        if self.is_train:
            spectrogram = random_channel_drop(spectrogram, drop_prob=0.3)

        return spectrogram, label_scalar


############################
# variable_channel_collate
############################
def variable_channel_collate(batch, ensemble_size=1):
    """
    Collate function that:
      1) Zero-pads the channel dimension to maxC => shape [B, maxC, F, T]
      2) If ensemble_size>1 (MIMO), replicates => [ensemble_size, B, maxC, F, T]
      3) Returns a single label per sample => shape [B]
      4) Returns a channel mask => shape [B, maxC]

    Args:
      batch: list of (spectrogram [C,F,T], label_scalar)
      ensemble_size: if >1, replicate the batch dimension for MIMO.

    Output:
      If ensemble_size=1:
        X => [B, maxC, F, T]
        Y => [B]
        M => [B, maxC]
      If ensemble_size>1:
        X => [ensemble_size, B, maxC, F, T]
        Y => [B]
        M => [B, maxC]
    """
    # 1) Find maxC among samples
    max_channels = max(item[0].shape[0] for item in batch)
    # Also assume same F,T across samples
    _, F, T = batch[0][0].shape

    # 2) Build lists
    X_list = []
    Y_list = []
    M_list = []

    for (spectrogram, label_scalar) in batch:
        c = spectrogram.shape[0]
        pad_spect = torch.zeros((max_channels, F, T), dtype=spectrogram.dtype)
        pad_spect[:c] = spectrogram

        # Build channel mask
        mask = torch.zeros((max_channels,), dtype=torch.bool)
        mask[:c] = True

        X_list.append(pad_spect)
        Y_list.append(label_scalar)
        M_list.append(mask)

    # Stack => shape [B, maxC, F, T]
    X = torch.stack(X_list, dim=0)
    Y = torch.tensor(Y_list, dtype=torch.long)  # [B]
    M = torch.stack(M_list, dim=0)             # [B, maxC]

    # 3) If ensemble_size>1, replicate => shape [ensemble_size, B, maxC, F, T]
    if ensemble_size > 1:
        X = X.unsqueeze(0).expand(ensemble_size, -1, -1, -1, -1).contiguous()
    return X, Y, M


############################
# evaluate_model
############################
def evaluate_model(model, loader, device, metrics_dict):
    """
    Runs the model on 'loader', collects logits, and computes each metric
    in 'metrics_dict'. Each metric can have its own threshold internally
    (e.g. f1_score(..., threshold=...)).

    Returns a dict of {metric_name: value}.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                data, labels, mask = batch_data
                mask = mask.to(device)
            else:
                data, labels = batch_data
                mask = None

            data = data.to(device)
            labels = labels.to(device)
            logits = model(data, mask=mask)

            # If MIMO => average ensemble dimension
            if logits.ndim == 3 and logits.shape[0] > 1:
                logits = logits.mean(dim=0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    results = {}
    for mname, mfn in metrics_dict.items():
        val_ = mfn(all_logits, all_labels)
        if isinstance(val_, torch.Tensor):
            val_ = val_.item()
        results[mname] = val_
    return results
