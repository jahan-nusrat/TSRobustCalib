"""
data_loader.py

Loads EEG data stored as .mat files.
Each file must contain:
  - "spectrogram": a tensor of shape [channels, frequency, time]
  - "label": a binary label (0 for normal, 1 for seizure)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

def collate_eeg_samples(batch):
    """
    Collate a list of (spectrogram, label) pairs.
    Zero-pads the channel dimension to the maximum found in the batch.
    
    Returns:
      X: Tensor of shape [B, max_channels, F, T]
      Y: Tensor of shape [B]
    """
    max_channels = max(spectrogram.shape[0] for spectrogram, _ in batch)
    _, F, T = batch[0][0].shape

    X_list, Y_list = [], []
    for spectrogram, label in batch:
        padded_spec = torch.zeros((max_channels, F, T), dtype=spectrogram.dtype)
        padded_spec[:spectrogram.shape[0]] = spectrogram
        X_list.append(padded_spec)
        Y_list.append(label)
    
    X = torch.stack(X_list, dim=0)
    Y = torch.tensor(Y_list, dtype=torch.long)
    return X, Y

def apply_random_augmentation(x_batch, noise_prob=0.3, noise_std=0.01):
    """
    Randomly add noise to the data.
    """
    B, C, F, T = x_batch.shape
    for i in range(B):
        for c in range(C):
            if random.random() < noise_prob:
                noise = torch.randn((F, T), dtype=x_batch.dtype, device=x_batch.device) * noise_std
                x_batch[i, c] += noise
    return x_batch

def variable_channel_collate(batch, ensemble_size=1, augment_per_head=False):
    """
    Collate the batch and, if using an ensemble (MIMO), replicate the batch.
    Optionally, apply random augmentation per ensemble head.
    """
    X, Y = collate_eeg_samples(batch)
    if ensemble_size > 1:
        X_ensemble = []
        for _ in range(ensemble_size):
            x_copy = X.clone()
            if augment_per_head:
                x_copy = apply_random_augmentation(x_copy)
            X_ensemble.append(x_copy)
        X = torch.stack(X_ensemble, dim=0)  # [ensemble_size, B, channels, F, T]
    return X, Y

class EEGDataset(Dataset):
    """
    EEGDataset loads .mat files from a directory.
    
    Args:
      data_dir: Directory containing .mat files.
      fraction: Fraction of files to load.
      verbose: If True, prints additional info.
    """
    def __init__(self, data_dir: str, fraction: float = 1.0, verbose: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.verbose = verbose
        
        mat_paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".mat"):
                    mat_paths.append(os.path.join(root, f))
                    
        if not mat_paths:
            raise ValueError(f"No .mat files found in {data_dir}")
            
        random.shuffle(mat_paths)
        cutoff = int(len(mat_paths) * fraction)
        mat_paths = mat_paths[:cutoff]
        
        if verbose:
            print(f"Found {len(mat_paths)} .mat files in {data_dir}")

        self.samples = []
        for path in mat_paths:
            mat = loadmat(path)
            label_arr = mat["label"]
            label = 1 if (label_arr > 0).any() else 0
            self.samples.append((path, label))
            
        self.num_seizure = sum(label for _, label in self.samples)
        self.num_normal = len(self.samples) - self.num_seizure
        
        if verbose:
            print(f"Seizure samples: {self.num_seizure}, Normal samples: {self.num_normal}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mat = loadmat(path)
        spectrogram = torch.from_numpy(mat["spectrogram"]).float()
        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(-1)
        return spectrogram, label

