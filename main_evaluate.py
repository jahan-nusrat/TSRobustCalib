"""
main_evaluate.py

This script loads a trained model and evaluates it on the test set.
It supports evaluating:
  - The model without test time adaptation (TTA)
  - The same model with test time adaptation (TTA)
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.data_loader import EEGDataset, variable_channel_collate
from models import build_resnet_model
from models.trainer import test_time_adaptation
from code_projects.utils.metrics import f1_score, ece, accuracy

def run_inference(model, data_loader, device):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Inference Batches"):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.ndim == 3:
                outputs = outputs.mean(dim=0)
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)

def compute_and_print_metrics(predictions, labels):
    """
    Computes and prints evaluation metrics.
    """
    metrics_dict = {
        "f1": f1_score,
        "ece": ece,
        "accuracy": accuracy
    }
    print("=== Evaluation Metrics ===")
    for name, fn in metrics_dict.items():
        val = fn(predictions, labels)
        print(f"{name}: {val:.4f}")

def evaluate_model(
    model_type="ResNet18MIMO", 
    in_channels=22,
    ensemble_size=3, # Set to 1 for single-head model, >1 for multi-head model
    test_data_path="/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Processed_data/eval",
    fraction=1.0,
    batch_size=32,
    checkpoint_path="/work/home/nj31voho/thesis/eeg/models/runs/Apr15_07-00-54_mpqc0001ResNet18MIMO/checkpoint_00010.pt",
    device=None,
    apply_tta=True, # Set to True to apply test time adaptation
    tta_lr=1e-3,
    tta_steps=1
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Evaluating model_type={model_type}, checkpoint={checkpoint_path}")
    print(f"TTA: {apply_tta}, TTA lr: {tta_lr}, TTA steps: {tta_steps}")
    
    # Build and load the model
    model = build_resnet_model(model_type, in_channels=in_channels, ensemble_size=ensemble_size)
    model = model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from epoch: {ckpt.get('epoch', 'N/A')}")
    
    # Create test DataLoader
    test_dataset = EEGDataset(data_dir=test_data_path, fraction=fraction, verbose=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: variable_channel_collate(b, ensemble_size=ensemble_size, augment_per_head=False)
    )
    
    # Evaluate without TTA
    print("Evaluating WITHOUT test time adaptation...")
    preds, labels = run_inference(model, test_loader, device)
    compute_and_print_metrics(preds, labels)
    
    # Evaluate with TTA (if desired)
    if apply_tta:
        print("Evaluating WITH test time adaptation...")
        # Reload a fresh copy of the model to avoid contamination by previous adaptation
        model_tta = build_resnet_model(model_type, in_channels=in_channels, ensemble_size=ensemble_size).to(device)
        model_tta.load_state_dict(ckpt["state_dict"])
        preds_tta = test_time_adaptation(model_tta, test_loader, adaptation_steps=tta_steps, lr=tta_lr, device=device)
        compute_and_print_metrics(preds_tta, labels)

def main():
    evaluate_model(
        model_type="ResNet18ManifoldMixup",   # Change as needed: "ResNet18Robust", "ResNet18MIMO", "ResNet18ManifoldMixup", "MIMOManifoldMixup"
        in_channels=20,  # Number of EEG channels
        ensemble_size=1,
        test_data_path="/work/groups/kismed/Datasets/TUH_Seizure_Corpus/Processed_data/eval",
        fraction=1.0,
        batch_size=32,
        checkpoint_path="/work/home/nj31voho/thesis/eeg/models/runs/Apr15_13-45-14_gaoc0001ResNet18ManifoldMixup/checkpoint_00010.pt",
        apply_tta=True,
        tta_lr=1e-3,
        tta_steps=1
    )

if __name__ == "__main__":
    main()
