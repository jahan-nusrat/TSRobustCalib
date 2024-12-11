from models.resnet import EEGNet
from trainer import GenericTrainer

# Dataset and Dataloader
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_dir):
    data, labels = [], []
    for file in os.listdir(data_dir):
        if file.endswith("_segments.npy"):
            segments = np.load(os.path.join(data_dir, file))
            label = 1 if "seizure" in file else 0
            data.append(segments)
            labels.extend([label] * segments.shape[0])
    data = np.vstack(data)
    labels = np.array(labels)
    return TensorDataset(torch.tensor(data).unsqueeze(1).float(), torch.tensor(labels))

train_dataset = load_data("preprocessed_data/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, Optimizer, Loss
model = EEGNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = GenericTrainer(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                         model=model,
                         loss=criterion,
                         metrics={"accuracy": accuracy},
                         optimizer=optimizer,
                         scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
                         max_grad_norm=5.0,
                         fp16_precision=False,
                         log_every_n_steps=10,
                         save_every_n_epochs=5,
                         epochs=20,
                         seed=42,
                         verbose=True)

trainer.train(train_loader)
