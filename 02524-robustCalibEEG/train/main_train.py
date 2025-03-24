import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pipeline.data_loader import EEGDataset, variable_channel_collate, evaluate_model
from models import ResNetEEG, MIMOResNetEEG
from trainer import GenericTrainer
from evaluation.metrics import f1_score, ece, auprc, sensitivity, specificity

def tune_threshold(model, val_loader, device, thresholds=None):
    """
    Gathers model predictions (probabilities) on val_loader,
    tests a range of thresholds, and returns the best threshold for F1.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.1, ... 1.0

    model.eval()
    all_probs = []
    all_labels = []

    # 1) Collect probabilities and labels
    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 3:
                data, labels, mask = batch_data
                mask = mask.to(device)
            else:
                data, labels = batch_data
                mask = None

            data = data.to(device)
            labels = labels.to(device)

            logits = model(data, mask=mask)
            # If MIMO => [ensemble_size, B, 1], average across ensemble dimension
            if logits.ndim == 3 and logits.shape[0] > 1:
                logits = logits.mean(dim=0)  # => [B,1]

            probs = torch.sigmoid(logits.view(-1))  # => [B]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    best_threshold = 0.0
    best_f1 = 0.0

    # 2) For each threshold, compute F1 manually
    for t in thresholds:
        preds = (all_probs > t).long()
        tp = ((preds == 1) & (all_labels == 1)).sum().float()
        fp = ((preds == 1) & (all_labels == 0)).sum().float()
        fn = ((preds == 0) & (all_labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"[Threshold Tuning] Best threshold={best_threshold:.2f}, F1={best_f1:.4f}")
    return best_threshold

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")

    # fraction of test and val dataset
    train_fraction = 1.0
    val_fraction   = 1.0

    # Paths
    train_data_path = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Processed_data/train"
    val_data_path   = "/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Processed_data/dev"

    # === Datasets ===
    print(f"[main] Loading train dataset from {train_data_path} with fraction={train_fraction}")
    train_dataset = EEGDataset(data_dir=train_data_path, fraction=train_fraction, verbose=True, is_train=True)

    print(f"[main] Loading val dataset from {val_data_path} with fraction={val_fraction}")
    val_dataset = EEGDataset(data_dir=val_data_path, fraction=val_fraction, verbose=True, is_train=False)

    # === WeightedRandomSampler (Optional) ===
    labels = [lbl for (_, lbl) in train_dataset.samples]
    class_counts = np.bincount(labels)  # e.g. [N_normal, N_seizure]
    print(f"[main] class_counts = {class_counts}")
    weights = [1.0 / class_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # === DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,
        collate_fn=variable_channel_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=variable_channel_collate
    )

    # === Model ===
    # model = ResNetEEG(pretrained=False, max_channels=22).to(device)
    model = MIMOResNetEEG(pretrained=False, max_channels=22, ensemble_size=3)

    # === Optimizer, Scheduler ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # === Metrics ===
    custom_threshold = 0.7
    metrics_dict = {
        "f1": lambda preds, labels: f1_score(preds, labels, threshold=custom_threshold),
        "ece": ece,
        "auprc": auprc,
        "sensitivity": lambda preds, labels: sensitivity(preds, labels, threshold=custom_threshold),
        "specificity": lambda preds, labels: specificity(preds, labels, threshold=custom_threshold)
    }

    # === Trainer ===
    trainer = GenericTrainer(
        device=device,
        model=model,
        metrics=metrics_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=5.0,
        fp16_precision=False,
        log_every_n_steps=10,
        save_every_n_epochs=5,
        epochs=30,
        seed=42,
        verbose=True,
        comment="mimo_with_threshold"
    )

    # === Train ===
    trainer.train(train_loader, val_loader)

    # === After Training: Tune Threshold on Validation ===
    best_thresh = tune_threshold(model, val_loader, device)
    print(f"[main] Best threshold found: {best_thresh:.2f}")

    # === Evaluate Final Metrics at best_thresh ===
    final_metrics_dict = {
        "f1": lambda preds, labels: f1_score(preds, labels, threshold=best_thresh),
        "ece": ece,
        "auprc": auprc,
        "sensitivity": lambda preds, labels: sensitivity(preds, labels, threshold=best_thresh),
        "specificity": lambda preds, labels: specificity(preds, labels, threshold=best_thresh)
    }
    final_scores = evaluate_model(model, val_loader, device, final_metrics_dict)
    print("Final Scores with tuned threshold:", final_scores)

if __name__ == "__main__":
    main()
