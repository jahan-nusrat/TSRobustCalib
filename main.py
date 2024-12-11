import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import GenericTrainer
from models.resNet import EEGNet  # Adjust the import path if needed

# Load Preprocessed Data
def load_data(data_dir):
    data, labels = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".npz"):
            loaded = np.load(os.path.join(data_dir, file))
            data.append(loaded['data'])  # Spectrograms or raw signals
            labels.append(loaded['labels'])
    data = np.vstack(data)
    labels = np.hstack(labels).astype(np.int64)

    # Remove the last dimension to make it compatible with the model
    data = torch.tensor(data).float().squeeze(-1)  # Remove the last dimension
    labels = torch.tensor(labels)
    return TensorDataset(torch.tensor(data).float(), torch.tensor(labels, dtype=torch.int64))




# Instantiate EEGNet
model = EEGNet(n_classes=2)  # 2 classes: normal and seizure

# Ensure data dimensions
train_dataset = load_data("data/processed/train")  # Preprocessed spectrograms
val_dataset = load_data("data/processed/eval")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Training
trainer = GenericTrainer(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    model=model,
    loss=criterion,
    metrics={"accuracy": lambda preds, labels: (preds.argmax(dim=1) == labels).float().mean()},
    optimizer=optimizer,
    scheduler=scheduler,
    max_grad_norm=5.0,
    fp16_precision=True,
    log_every_n_steps=10,
    save_every_n_epochs=5,
    epochs=20,
    seed=42,
    verbose=True,
    comment="Training EEGNet with spectrogram data"
)

# Train the Model
trainer.train(train_loader, val_loader)
