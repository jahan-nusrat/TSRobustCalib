import torch
import torch.nn as nn
import logging
import os
import json
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision.ops import sigmoid_focal_loss

def focal_loss_fn(logits, labels, alpha=0.95, gamma=2.0, reduction="mean"):
    logits = logits.view(-1)
    labels = labels.float()
    return sigmoid_focal_loss(logits, labels, alpha=alpha, gamma=gamma, reduction=reduction)

class GenericTrainer:
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
        comment: str = ""
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

        # Focal loss hyperparams
        self.alpha = 0.6
        self.gamma = 1.0
        self.reduction = "mean"

        torch.manual_seed(seed)

        config_dict = {
            "device": str(device),
            "optimizer": str(optimizer),
            "max_grad_norm": max_grad_norm,
            "fp16_precision": fp16_precision,
            "log_every_n_steps": log_every_n_steps,
            "save_every_n_epochs": save_every_n_epochs,
            "epochs": epochs,
            "verbose": verbose,
            "seed": seed,
        }
        self.writer = SummaryWriter(comment=comment) if verbose else SummaryWriter(comment=comment, write_to_disk=False)
        log_dir = self.writer.log_dir
        logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.DEBUG)
        with open(os.path.join(log_dir, 'config.json'), "w") as f:
            json.dump(config_dict, f)

    def _forward_with_mimo_check(self, data: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        If the model is MIMO and returns shape [ensemble_size, B, 1],
        average over ensemble_size => shape [B,1].
        Otherwise shape [B,1].
        """
        logits = self.model(data, mask=mask)
        # e.g. MIMO => [ensemble_size,B,1], single => [B,1]
        if logits.ndim == 3 and logits.shape[0] > 1:
            logits = logits.mean(dim=0)  # => [B,1]
        return logits

    def train(self, train_loader, val_loader=None):
        scaler = GradScaler(enabled=self.fp16_precision)
        n_iter = 0

        training_history = {
            "loss": [],
        }
        validation_history = {
            "loss": [],
        }
        
        for mname in self.metrics.keys():
            training_history[mname] = []
            validation_history[mname] = []

        for epoch in range(self.epochs):
            ################################
            # TRAIN PHASE
            ################################
            self.model.train()
            epoch_train_loss = 0.0
            # We'll accumulate sums for each metric, then average at epoch end
            metric_sums = {m: 0.0 for m in self.metrics.keys()}
            num_train_batches = len(train_loader)

            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=not self.verbose)):
                if len(batch_data) == 3:
                    data, labels, mask = batch_data
                    mask = mask.to(self.device)
                else:
                    data, labels = batch_data
                    mask = None

                data = data.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.fp16_precision):
                    logits = self._forward_with_mimo_check(data, mask)
                    loss_val = focal_loss_fn(logits, labels, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)

                self.optimizer.zero_grad()
                scaler.scale(loss_val).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()

                epoch_train_loss += loss_val.detach().item()

                # Compute each metric for this batch
                for mname, mfn in self.metrics.items():
                    mval = mfn(logits, labels)
                    # If the metric returns a tensor, call .item()
                    # If it returns float, skip .item()
                    if isinstance(mval, torch.Tensor):
                        mval = mval.item()
                    metric_sums[mname] += mval

                if self.verbose and (n_iter % self.log_every_n_steps == 0):
                    self.writer.add_scalar('train_batch_loss', loss_val, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                n_iter += 1

            self.scheduler.step()
            epoch_train_loss /= num_train_batches

            for mname in metric_sums:
                metric_sums[mname] /= num_train_batches

            # Save to training_history
            training_history["loss"].append(epoch_train_loss)
            for mname in metric_sums:
                training_history[mname].append(metric_sums[mname])

            if self.verbose:
                print(f"[Epoch {epoch}] Training Loss: {epoch_train_loss:.4f}")
                for mname, val_ in metric_sums.items():
                    print(f"[Epoch {epoch}] Training {mname}: {val_:.4f}")

            self.writer.add_scalar('epoch_training_loss', epoch_train_loss, epoch)
            for mname, val_ in metric_sums.items():
                self.writer.add_scalar(f'train_{mname}', val_, epoch)

            ################################
            # VALIDATION PHASE
            ################################
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = 0.0
                val_metric_sums = {m: 0.0 for m in self.metrics.keys()}
                num_val_batches = len(val_loader)

                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=not self.verbose)):
                        if len(batch_data) == 3:
                            data, labels, mask = batch_data
                            mask = mask.to(self.device)
                        else:
                            data, labels = batch_data
                            mask = None

                        data = data.to(self.device)
                        labels = labels.to(self.device)

                        with autocast(enabled=self.fp16_precision):
                            logits = self._forward_with_mimo_check(data, mask)
                            val_loss = focal_loss_fn(logits, labels, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)

                        epoch_val_loss += val_loss.detach().item()

                        for mname, mfn in self.metrics.items():
                            mval = mfn(logits, labels)
                            if isinstance(mval, torch.Tensor):
                                mval = mval.item()
                            val_metric_sums[mname] += mval

                epoch_val_loss /= num_val_batches
                for mname in val_metric_sums:
                    val_metric_sums[mname] /= num_val_batches

                validation_history["loss"].append(epoch_val_loss)
                for mname in val_metric_sums:
                    validation_history[mname].append(val_metric_sums[mname])

                if self.verbose:
                    print(f"[Epoch {epoch}] Validation Loss: {epoch_val_loss:.4f}")
                    for mname, val_ in val_metric_sums.items():
                        print(f"[Epoch {epoch}] Validation {mname}: {val_:.4f}")

                self.writer.add_scalar('epoch_validation_loss', epoch_val_loss, epoch)
                for mname, val_ in val_metric_sums.items():
                    self.writer.add_scalar(f'val_{mname}', val_, epoch)

            # Save checkpoint
            if epoch > 0 and self.save_every_n_epochs > 0 and (epoch % self.save_every_n_epochs == 0):
                ckpt_path = os.path.join(self.writer.log_dir, f"checkpoint_{epoch:05d}.pt")
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'seed': self.seed,
                }, ckpt_path)
                if self.verbose:
                    logging.info(f"Checkpoint saved: {ckpt_path}")

        # Final checkpoint
        final_ckpt = os.path.join(self.writer.log_dir, f"checkpoint_{self.epochs:05d}.pt")
        torch.save({
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'seed': self.seed,
        }, final_ckpt)
        if self.verbose:
            logging.info(f"Final checkpoint saved: {final_ckpt}")
            print("Training finished. Final checkpoint saved.")

        df_dict = {}
        for key, values in training_history.items():
            df_dict[f"training_{key}"] = values
        if val_loader is not None:
            for key, values in validation_history.items():
                df_dict[f"validation_{key}"] = values

        # Write out metrics.csv
        out_csv = os.path.join(self.writer.log_dir, "metrics.csv")
        pd.DataFrame(df_dict).to_csv(out_csv, index=False)
        if self.verbose:
            print(f"Saved metrics to {out_csv}")
