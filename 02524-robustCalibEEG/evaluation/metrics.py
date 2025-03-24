import torch
import torch.nn.functional as F

def f1_score(predictions: torch.Tensor, labels: torch.Tensor, threshold=0.5, eps=1e-9) -> torch.Tensor:
    """
    predictions: shape [N,1] => single logit for each sample
    labels: shape [N] => 0 or 1
    threshold: used after sigmoid for binary classification
    """
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions.view(-1))  # => [N]
        preds = (probs > threshold).long()
    else:
        # fallback for multi-class
        preds = predictions.argmax(dim=-1)

    labels = labels.view(-1)
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1

def ece(predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """
    Expected Calibration Error for single-logit or multi-class.
    """
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions.view(-1))
    else:
        # multi-class fallback
        probs = F.softmax(predictions, dim=-1)
        preds = probs.argmax(dim=-1)
        probs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

    labels = labels.view(-1)
    probs = probs.view(-1)

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
        avg_conf = bin_confs.mean()
        acc = bin_labels.float().mean()
        ece_val += (len(bin_labels) / len(labels)) * torch.abs(avg_conf - acc)
        start = end
    return ece_val

def auprc(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Area Under the Precision-Recall Curve for binary classification.
    """
    from sklearn.metrics import average_precision_score

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions.view(-1)).detach().cpu().numpy()
    else:
        # multi-class fallback, assume 2-class
        probs = F.softmax(predictions, dim=-1)[:, 1].detach().cpu().numpy()

    labels = labels.cpu().numpy()
    import numpy as np
    if (labels == 1).sum() == 0:
        # no positives => define AUPRC=0 or 1
        return torch.tensor(0.0)
    return torch.tensor(average_precision_score(labels, probs))

def sensitivity(predictions: torch.Tensor, labels: torch.Tensor, threshold=0.5, eps=1e-9) -> torch.Tensor:
    """
    Recall for the positive (seizure) class.
    """
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions.view(-1))
        preds = (probs > threshold).long()
    else:
        preds = predictions.argmax(dim=-1)

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    return tp / (tp + fn + eps)

def specificity(predictions: torch.Tensor, labels: torch.Tensor, threshold=0.5, eps=1e-9) -> torch.Tensor:
    """
    True Negative Rate for the negative (normal) class.
    """
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions.view(-1))
        preds = (probs > threshold).long()
    else:
        preds = predictions.argmax(dim=-1)

    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    return tn / (tn + fp + eps)
