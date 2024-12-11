import torch
import torch.nn as nn
import logging
import os
from typing import Callable, Dict
from torch.amp import GradScaler, autocast
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import pandas as pd

class GenericTrainer():

    def __init__(self, device:torch.device, 
                 model:nn.Module,
                 loss:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 metrics:Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                 optimizer:torch.optim.Optimizer, 
                 scheduler:torch.optim.lr_scheduler.LRScheduler,
                 max_grad_norm:float,
                 fp16_precision:bool, 
                 log_every_n_steps:int, 
                 save_every_n_epochs:int, 
                 epochs:int,
                 seed:int,
                 verbose:bool,
                 comment:str="") -> None:
        
        self.device = device
        self.model = model.to(device)

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

        self.metrics = metrics

        self.fp16_precision = fp16_precision
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.epochs = epochs

        self.seed = seed

        self.verbose = verbose

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

        if(self.verbose): # Create a directory to save the config and logs in
            self.writer = SummaryWriter(comment=comment)
            logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
            with open(os.path.join(self.writer.log_dir, 'config.json'), "w") as outfile: 
                json.dump(config_dict, outfile)
        elif(self.save_every_n_epochs > 0): # Still create the directory to save the model checkpoints in
            self.writer = SummaryWriter(comment=comment)
            with open(os.path.join(self.writer.log_dir, 'config.json'), "w") as outfile: 
                json.dump(config_dict, outfile)

    def train(self, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader|None=None) -> None:
        scaler = GradScaler(device=self.device, enabled=self.fp16_precision)
        n_iter = 0
        n_metrics = len(self.metrics)
        if(self.verbose):
            logging.info(f"Start training for %d epochs."%self.epochs)

        # Create tensors to store the metrics and losses for each epoch
        training_loss_tensor = torch.zeros(self.epochs).to(self.device)
        validation_loss_tensor = torch.zeros(self.epochs).to(self.device)
        if(n_metrics > 0):
            training_metrics_tensor = torch.zeros(self.epochs, n_metrics).to(self.device)
            validation_metrics_tensor = torch.zeros(self.epochs, n_metrics).to(self.device)

        for epoch_counter in range(self.epochs):
            self.model.train() # Training phase
            epoch_loss = 0
            metrics = torch.zeros(n_metrics).to(self.device)
            for data, labels in tqdm(train_loader, disable=not self.verbose):
                data: torch.Tensor = data.to(self.device)
                labels: torch.Tensor = labels.to(self.device).to(torch.int64)# Ensure labels are LongTensor

                with autocast(device_type=str(self.device), enabled=self.fp16_precision):
                    predictions = self.model(data)
                    loss = self.loss(predictions, labels)

                    epoch_loss += loss.detach()  # Add current batch loss to epoch loss in order to take the mean later

                    # Calculate metrics
                    for i, metric in enumerate(self.metrics.values()):
                        metrics[i] += metric(predictions, labels).detach()

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                    scaler.step(self.optimizer)
                    scaler.update()

                    if self.verbose and (n_iter % self.log_every_n_steps == 0):
                        self.writer.add_scalar('training_loss', loss, global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                    n_iter += 1

            self.scheduler.step() # Step scheduler each epoch

            # Take the mean of the losses and metrics
            epoch_loss = epoch_loss / len(train_loader) 
            training_loss_tensor[epoch_counter] = epoch_loss

            if(n_metrics > 0):
                metrics = metrics / len(train_loader)
                training_metrics_tensor[epoch_counter, :] = metrics

            if(self.verbose):
                logging.debug(f"Epoch: {epoch_counter}\tTraining Loss: {epoch_loss}")
                print(f"Epoch: {epoch_counter}\tTraining Loss: {epoch_loss}")
                self.writer.add_scalar('epoch_training_loss', loss, global_step=epoch_counter)

                for i, metric_name in enumerate(self.metrics.keys()):
                    self.writer.add_scalar(f'training_epoch_{metric_name}', metrics[i], global_step=epoch_counter)
                    logging.debug(f"Epoch: {epoch_counter}\tTraining {metric_name}: {metrics[i]}")
                    print(f"Epoch: {epoch_counter}\tTraining {metric_name}: {metrics[i]}")
    	    
            # Save model checkpoint
            if epoch_counter > 0 and self.save_every_n_epochs > 0 and epoch_counter % self.save_every_n_epochs == 0:
                checkpoint_name = 'checkpoint_{:05d}.pt'.format(epoch_counter)
                torch.save({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler' : scaler.state_dict(),
                    'scheduler' : self.scheduler.state_dict(),
                    'seed': self.seed,
                    'max_grad_norm': self.max_grad_norm,
                }, os.path.join(self.writer.log_dir, checkpoint_name))
                if(self.verbose):
                    logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")   
            
            # Validation phase
            if(val_loader is not None):
                with torch.no_grad():
                    self.model.eval()
                    epoch_loss = 0
                    metrics = torch.zeros(n_metrics).to(self.device)
                    for data, labels in tqdm(val_loader, disable=not self.verbose):
                        data:torch.Tensor = data.to(self.device)
                        labels:torch.Tensor = labels.to(self.device)

                        with autocast(device_type=str(self.device), enabled=self.fp16_precision):
                            predictions = self.model(data)
                            loss = self.loss(predictions, labels)
                        
                        epoch_loss += loss.detach()
                        # Calculate metrics
                        for i, metric in enumerate(self.metrics.values()):
                            metrics[i] += metric(predictions, labels).detach()
                    
                    epoch_loss = epoch_loss / len(val_loader) # Take the mean of the losses
                    validation_loss_tensor[epoch_counter] = epoch_loss

                    if(n_metrics > 0):
                        metrics = metrics / len(val_loader)
                        validation_metrics_tensor[epoch_counter, :] = metrics

                    if(self.verbose):
                        logging.debug(f"Epoch: {epoch_counter}\tValidation Loss: {epoch_loss}")
                        print(f"Epoch: {epoch_counter}\tValidation Loss: {epoch_loss}")
                        self.writer.add_scalar('epoch_validation_loss', loss, global_step=epoch_counter)

                        for i, metric_name in enumerate(self.metrics.keys()):
                            self.writer.add_scalar(f'validation_epoch_{metric_name}', metrics[i], global_step=epoch_counter)
                            logging.debug(f"Epoch: {epoch_counter}\tValidation {metric_name}: {metrics[i]}")
                            print(f"Epoch: {epoch_counter}\tValidation {metric_name}: {metrics[i]}")
                
        if(self.verbose):
            logging.info("Training has finished.")
        
        if(self.save_every_n_epochs > 0):
            # save model checkpoints
            checkpoint_name = 'checkpoint_{:05d}.pt'.format(self.epochs)
            torch.save({
                'epoch': self.epochs,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scaler' : scaler.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
                'seed': self.seed,
                'max_grad_norm': self.max_grad_norm,
            }, os.path.join(self.writer.log_dir, checkpoint_name))
            if(self.verbose):
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
        # Save metrics and loss to a csv file
        loss_dict = {"training loss" : training_loss_tensor.cpu().numpy(), "validation loss" : validation_loss_tensor.cpu().numpy()}
        if(n_metrics > 0):
            training_metrics_tensor = training_metrics_tensor.cpu().numpy()
            validation_metrics_tensor = validation_metrics_tensor.cpu().numpy()
            for i, metric_name in enumerate(self.metrics.keys()):
                loss_dict[f"training {metric_name}"] = training_metrics_tensor[:, i]
                loss_dict[f"validation {metric_name}"] = validation_metrics_tensor[:, i]
        pd.DataFrame(loss_dict).to_csv(path_or_buf=os.path.join(self.writer.log_dir, 'metrics.csv'), index=False)