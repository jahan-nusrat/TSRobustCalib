import torch
import torch.nn as nn
from torchvision.models import resnet18

class EEGNet(nn.Module):
    def __init__(self, n_classes=2):
        super(EEGNet, self).__init__()
        self.base_model = resnet18(pretrained=False)  # Use a fresh ResNet18 model
        # Modify the first convolutional layer to accept 1 channel
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Update the final fully connected layer for the correct number of classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, n_classes)

    def forward(self, x):
        return self.base_model(x)
