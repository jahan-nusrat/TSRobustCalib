import torch.nn as nn
from torchvision.models import densenet121

class EEGDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGDenseNet, self).__init__()
        self.base_model = densenet121(pretrained=False)
        self.base_model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
