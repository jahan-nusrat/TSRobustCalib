"""
metrics.py
Defines evaluation metrics for the model.
"""

import torch

def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    if predictions.ndim == 3:
        predictions = predictions.mean(dim=0)
    probs = torch.sigmoid(predictions.view(-1))
    preds = (probs > 0.5).long()
    labels = labels.view(-1)
    correct = (preds == labels).sum().float()
    total = labels.numel()
    return (correct / total).item()

def f1_score(predictions: torch.Tensor, labels: torch.Tensor, eps=1e-9) -> float:
    if predictions.ndim == 3:
        predictions = predictions.mean(dim=0)
    probs = torch.sigmoid(predictions.view(-1))
    preds = (probs > 0.5).long()
    labels = labels.view(-1)
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return (2 * precision * recall / (precision + recall + eps)).item()

def ece(predictions: torch.Tensor, labels: torch.Tensor, n_bins=10) -> float:
    if predictions.ndim == 3:
        predictions = predictions.mean(dim=0)
    probs = torch.sigmoid(predictions.view(-1)).detach().cpu()
    labels = labels.view(-1).cpu()
    confidences, indices = torch.sort(probs)
    sorted_labels = labels[indices]
    bin_size = len(labels) // n_bins
    ece_val = 0.0
    start = 0
    for b in range(n_bins):
        end = start + bin_size
        if b == n_bins - 1:
            end = len(labels)
        if start == end:
            break
        bin_labels = sorted_labels[start:end]
        bin_confs = confidences[start:end]
        avg_conf = bin_confs.mean().item()
        acc = bin_labels.float().mean().item()
        ece_val += (len(bin_labels) / len(labels)) * abs(avg_conf - acc)
        start = end
    return ece_val
