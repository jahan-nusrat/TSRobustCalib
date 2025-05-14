"""
main_train.py

Main training script for robust EEG classification.
Select one model variant and one training strategy per run.
"""
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
import matplotlib.pyplot as plt
from models.data_loader import EEGDataset, variable_channel_collate
from models import build_resnet_model
from models.trainer import RobustTrainer
from code_projects.utils.metrics import f1_score, ece, accuracy

def save_plot(fig, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, format="svg")
    print(f"[Saved] {filepath}")

def plot_class_distribution(dataset, title, filename):
    counts = [dataset.num_normal, dataset.num_seizure]
    labels = ['Normal', 'Seizure']
    colors = ['skyblue', 'salmon']
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Number of Samples")
    for i, count in enumerate(counts):
        percent = count / sum(counts) * 100
        ax.text(i, count + 100, f"{percent:.2f}%", ha='center')
    fig.tight_layout()
    save_plot(fig, "plots/data_distribution", filename)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main_train] Using device: {device}")

    # Data paths
    train_data_path = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/ProcessedData/train"
    val_data_path   = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/ProcessedData/dev"

    # Load full datasets
    print(f"[main_train] Loading train dataset from {train_data_path}")
    train_dataset = EEGDataset(data_dir=train_data_path, fraction=0.2, verbose=True) #use fraction=1.0 for full dataset

    print(f"[main_train] Loading full validation dataset from {val_data_path}")
    val_dataset = EEGDataset(data_dir=val_data_path, fraction=0.2, verbose=True) #use fraction=1.0 for full dataset
    # Plot class distributions
    plot_class_distribution(train_dataset, "Training Data Class Distribution", "train_distribution.svg")
    plot_class_distribution(val_dataset, "Validation Data Class Distribution", "val_distribution.svg")

    # Create a weighted sampler to handle class imbalance.
    labels = [lbl for (_, lbl) in train_dataset.samples]
    class_counts = np.bincount(labels)
    weights = [1.0 / class_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    ensemble_size = 1  # Set to 1 for single-head model, >1 for multi-head model
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        collate_fn=lambda b: variable_channel_collate(b, ensemble_size=ensemble_size, augment_per_head=True)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: variable_channel_collate(b, ensemble_size=ensemble_size, augment_per_head=False)
    )

    # Model selection
    # Choose one of the following model types: 
    # "ResNet18Robust", "ResNet18MIMO", "ResNet18ManifoldMixup", "MIMOManifoldMixup"
    # For MIMOManifoldMixup, set ensemble_size > 1
    # For ResNet18ManifoldMixup, set use_manifoldmixup=True in the trainer
    model_type = "ResNet18Robust"
    model = build_resnet_model(model_type, in_channels=20, ensemble_size=ensemble_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    metrics_dict = {
        "f1": f1_score,
        "ece": ece,
        "accuracy": accuracy
    }

    trainer = RobustTrainer(
        device=device,
        model=model,
        metrics=metrics_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=5.0,
        fp16_precision=False,
        log_every_n_steps=10,
        save_every_n_epochs=5,
        epochs=120,
        seed=42,
        verbose=True,
        comment=model_type,
        use_mixup=False,
        mixup_alpha=0.2,
        use_manifoldmixup=False,  # Use manifold mixup for the ResNet18ManifoldMixup model.
        manifoldmixup_alpha=1.0
    )

    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
