import torch
import torch.nn as nn
import logging
import os
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss

# ---------- Loss and Mixup Utilities ----------

def focal_loss_fn(logits, labels, alpha=0.9, gamma=2.0, reduction="mean"):
    """
    Computes focal loss for binary classification.
    Averages ensemble outputs if necessary.
    """
    if logits.ndim == 3:
        logits = logits.mean(dim=0)
    logits = logits.view(-1)
    labels = labels.float()
    return sigmoid_focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction=reduction)


def mixup_data(x, y, alpha=0.2):
    """
    Standard mixup on raw inputs.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion_fn, preds, y_a, y_b, lam):
    """
    Computes mixup loss.
    """
    loss_a = criterion_fn(preds, y_a)
    loss_b = criterion_fn(preds, y_b)
    return lam * loss_a + (1 - lam) * loss_b


def manifold_mixup_hidden(h, labels, alpha=1.0):
    """
    Apply manifold mixup on hidden features.
    """
    B = h.size(0)
    if B < 2:
        return h, labels
    lam = torch.distributions.Beta(alpha, alpha).sample().to(h.device)
    perm = torch.randperm(B, device=h.device)
    h2 = h[perm]
    y2 = labels[perm].float()
    h_mix = lam * h + (1 - lam) * h2
    y_mix = lam * labels.float() + (1 - lam) * y2
    return h_mix, y_mix


def entropy_loss(logits):
    """
    Computes entropy loss to encourage confident predictions.
    """
    probs = torch.sigmoid(logits)
    loss = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
    return loss.mean()


def test_time_adaptation(model, test_loader, adaptation_steps=1, lr=1e-3, device=torch.device("cpu")):
    """
    Test time adaptation (TTA): For each test batch, perform a small unsupervised adaptation step
    by minimizing the entropy of the model’s predictions, then collect the adapted predictions.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    adapted_preds = []

    for data, _ in test_loader:
        data = data.to(device)
        for _ in range(adaptation_steps):
            raw_output = model(data)
            logits = raw_output[0] if isinstance(raw_output, tuple) else raw_output
            if logits.ndim == 3:
                logits = logits.mean(dim=0)
            loss = entropy_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            raw_output = model(data)
            final_logits = raw_output[0] if isinstance(raw_output, tuple) else raw_output
            if final_logits.ndim == 3:
                final_logits = final_logits.mean(dim=0)
        adapted_preds.append(final_logits.cpu())

    model.eval()
    return torch.cat(adapted_preds, dim=0)


# ---------- Robust Trainer ----------

class RobustTrainer:
    """
    Trainer for a single model experiment.
    Only one augmentation strategy is used per run.
    If using manifold mixup, a random hidden layer is selected per iteration.
    """
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        metrics: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        max_grad_norm: float,
        fp16_precision: bool,
        log_every_n_steps: int,
        save_every_n_epochs: int,
        epochs: int,
        seed: int,
        verbose: bool,
        comment: str = "",
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        use_manifoldmixup: bool = False,
        manifoldmixup_alpha: float = 1.0
    ):
        self.device = device
        self.model = model.to(device)
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.fp16_precision = fp16_precision
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose

        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_manifoldmixup = use_manifoldmixup
        self.manifoldmixup_alpha = manifoldmixup_alpha

        torch.manual_seed(seed)
        config = {
            "device": str(device),
            "optimizer": str(optimizer),
            "max_grad_norm": max_grad_norm,
            "fp16_precision": fp16_precision,
            "log_every_n_steps": log_every_n_steps,
            "save_every_n_epochs": save_every_n_epochs,
            "epochs": epochs,
            "seed": seed,
            "use_mixup": use_mixup,
            "mixup_alpha": mixup_alpha,
            "use_manifoldmixup": use_manifoldmixup,
            "manifoldmixup_alpha": manifoldmixup_alpha
        }
        self.writer = SummaryWriter(comment=comment) if verbose else SummaryWriter(comment=comment, write_to_disk=False)
        log_dir = self.writer.log_dir
        logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.DEBUG)
        with open(os.path.join(log_dir, 'config.json'), "w") as f:
            json.dump(config, f)

    def focal_loss_fn(self, logits, labels):
        return focal_loss_fn(logits, labels)

    def train(self, train_loader, val_loader=None):
        scaler = GradScaler(enabled=self.fp16_precision)
        global_iter = 0
        training_loss_history = []
        validation_loss_history = []
        training_metrics = {m: [] for m in self.metrics}
        validation_metrics = {m: [] for m in self.metrics}

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            metric_sums = {m: 0.0 for m in self.metrics}
            num_batches = len(train_loader)

            for batch_idx, (data, labels) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=not self.verbose)):
                data = data.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.fp16_precision):
                    # ------ Manifold Mixup ------
                    if self.use_manifoldmixup:
                        M = getattr(self.model, "ensemble_size", 1)
                        B = data.size(0)
                        # 1) Build M distinct views via permutations
                        perms = [torch.randperm(B, device=data.device) for _ in range(M)]
                        data_mimo = torch.stack([data[p] for p in perms], dim=0)  # [M,B,C,F,T]
                        # 2) Choose random mixup layer
                        layer = random.choice([0,1,2,3,4])
                        # 3) Forward through model with manifold mixup
                        logits_mimo, labels_mix = self.model(
                            data_mimo,
                            mix_layer=layer,
                            mix_fn=manifold_mixup_hidden,
                            labels=labels,
                            alpha=self.manifoldmixup_alpha
                        )  # logits_mimo: [M,B,1]
                        logits_mimo = logits_mimo.squeeze(-1)  # [M,B]
                        # 4) Head-specific losses on original labels
                        head_losses = [self.focal_loss_fn(logits_mimo[i], labels) for i in range(M)]
                        L_head = sum(head_losses) / M
                        # 5) Mix-specific loss on mixed labels
                        logits_mix = logits_mimo.mean(dim=0)  # [B]
                        L_mix   = self.focal_loss_fn(logits_mix, labels_mix)
                        # 6) Total loss
                        loss = L_head + L_mix
                        final_preds, final_labels = logits_mix, labels

                    # ------ Standard Mixup ------
                    elif self.use_mixup:
                        mixed_data, y_a, y_b, lam = mixup_data(data, labels, alpha=self.mixup_alpha)
                        preds = self.model(mixed_data)
                        loss = mixup_criterion(self.focal_loss_fn, preds, y_a, y_b, lam)
                        final_preds, final_labels = preds, labels

                    # ------ Plain MIMO or Single-Head ------
                    else:
                        M = getattr(self.model, "ensemble_size", 1)
                        if M > 1:
                            B = data.size(0)
                            perms = [torch.randperm(B, device=data.device) for _ in range(M)]
                            data_mimo = torch.stack([data[p] for p in perms], dim=0)
                            logits, _ = self.model(data_mimo)
                            logits = logits.squeeze(-1)  # [M,B]
                            preds = logits.mean(dim=0)   # [B]
                        else:
                            out = self.model(data)
                            preds = out[0].squeeze(-1) if isinstance(out, tuple) else out.squeeze(-1)
                        loss = self.focal_loss_fn(preds, labels)
                        final_preds, final_labels = preds, labels
                # Backpropagation
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.detach().item()
                # metrics
                if final_preds.ndim == 3:
                    final_preds = final_preds.mean(dim=0)
                for m, mfn in self.metrics.items():
                    val = mfn(final_preds, final_labels)
                    metric_sums[m] += (val.item() if isinstance(val, torch.Tensor) else val)

                if self.verbose and global_iter % self.log_every_n_steps == 0:
                    self.writer.add_scalar("train_batch_loss", loss, global_iter)
                global_iter += 1

            # epoch end
            self.scheduler.step()
            epoch_loss /= num_batches
            for m in metric_sums:
                metric_sums[m] /= num_batches
            training_loss_history.append(epoch_loss)
            for m in metric_sums:
                training_metrics[m].append(metric_sums[m])

            # validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_sums = {m:0.0 for m in self.metrics}
                with torch.no_grad():
                    for data, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=not self.verbose):
                        data, labels = data.to(self.device), labels.to(self.device)
                        with autocast(enabled=self.fp16_precision):
                            preds = self.model(data)
                            if isinstance(preds, tuple):
                                preds = preds[0]
                            loss_val = self.focal_loss_fn(preds, labels)
                        val_loss += loss_val.detach().item()
                        preds_proc = preds.mean(dim=0) if preds.ndim==3 else preds
                        for m, mfn in self.metrics.items():
                            v = mfn(preds_proc, labels)
                            val_sums[m] += (v.item() if isinstance(v, torch.Tensor) else v)
                val_loss /= len(val_loader)
                for m in val_sums:
                    val_sums[m] /= len(val_loader)
                validation_loss_history.append(val_loss)
                for m in val_sums:
                    validation_metrics[m].append(val_sums[m])
                if self.verbose:
                    print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")
                    for m,v in val_sums.items(): print(f"[Epoch {epoch}] Val {m}: {v:.4f}")
                self.writer.add_scalar("epoch_validation_loss", val_loss, epoch)
                for m,v in val_sums.items(): self.writer.add_scalar(f"val_{m}", v, epoch)

            # checkpointing
            if epoch>0 and self.save_every_n_epochs>0 and epoch % self.save_every_n_epochs==0:
                ck = os.path.join(self.writer.log_dir, f"checkpoint_{epoch:05d}.pt")
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'seed': self.seed
                }, ck)

        # final save
        final_ckpt = os.path.join(self.writer.log_dir, f"checkpoint_{self.epochs:05d}.pt")
        torch.save({
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'seed': self.seed
        }, final_ckpt)
        if self.verbose:
            print("Training finished. Final checkpoint saved.")

        # write metrics and plots
        df = pd.DataFrame({
            'training_loss': training_loss_history,
            'validation_loss': validation_loss_history,
            **{f'training_{m}':training_metrics[m] for m in self.metrics},
            **{f'validation_{m}':validation_metrics[m] for m in self.metrics}
        })
        out_csv = os.path.join(self.writer.log_dir, 'metrics.csv')
        df.to_csv(out_csv, index=False)

        def save_plot(fig, folder, fname):
            os.makedirs(folder, exist_ok=True)
            fig.savefig(os.path.join(folder, fname), format='svg')

        def plot_curve(df, ct, cv, ylbl, title, fname):
            fig, ax = plt.subplots()
            ax.plot(df[ct], label='Train')
            ax.plot(df[cv], label='Val')
            ax.set(xlabel='Epoch', ylabel=ylbl, title=title)
            ax.legend(); fig.tight_layout()
            save_plot(fig, os.path.join(self.writer.log_dir, 'plots/metrics'), fname)

        plot_curve(df, 'training_loss','validation_loss', 'Focal Loss', 'Loss Curve', 'loss_curve.svg')
        for m in self.metrics:
            plot_curve(df, f'training_{m}', f'validation_{m}', m.upper(), f'{m.upper()} Curve', f'{m}_curve.svg')
        fig, ax = plt.subplots()
        ax.bar(['Validation ECE'], [df['validation_ece'].iloc[-1] if 'validation_ece' in df else 0], color='orange')
        ax.set_ylim(0,1); ax.set_title('Final ECE')
        save_plot(fig, os.path.join(self.writer.log_dir, 'plots/metrics'), 'ece_val.svg')
