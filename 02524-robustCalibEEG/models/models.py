import torch
import torch.nn as nn
from torchvision import models

class ResNetEEG(nn.Module):
    def __init__(self, pretrained=False, max_channels=22):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=max_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        nn.init.kaiming_normal_(self.resnet.conv1.weight, nonlinearity="relu")
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)  # single logit for binary classification

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).unsqueeze(-1).float()
        logits = self.resnet(x)  # shape [B,1]
        return logits


class MIMOResNetEEG(nn.Module):
    """
    A MIMO (Multi-Input Multi-Output) ResNet for binary classification.
    - We create `ensemble_size` separate first conv layers
      and separate final fc layers, but share the middle blocks
      for better memory efficiency.
    - Output shape: [ensemble_size, B, 1], or we can average
      to shape [B,1] if you want a single final logit.
    """
    def __init__(self, pretrained=False, max_channels=22, ensemble_size=3):
        super().__init__()
        self.ensemble_size = ensemble_size

        backbone = models.resnet18(pretrained=pretrained)

        self.shared_layers = nn.Sequential(
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool
        )

        # Each ensemble member gets its own conv1
        self.individual_convs = nn.ModuleList([
            nn.Conv2d(max_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            for _ in range(ensemble_size)
        ])
        for conv in self.individual_convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")

        # Each ensemble member also gets its own FC
        in_features = backbone.fc.in_features
        self.individual_fcs = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(ensemble_size)
        ])

    def forward(self, x, mask=None):

        # Optional mask
        if mask is not None:
            x = x * mask.unsqueeze(-1).unsqueeze(-1).float()

        # shape => [B, C, F, T]
        outputs = []
        for i in range(self.ensemble_size):
            # conv1 => BN => ReLU => MaxPool => layer1..4 => avgpool
            xi = self.individual_convs[i](x)           # [B,64,F/2,T/2]
            xi = self.shared_layers(xi)                # [B,512,1,1] at end
            xi = torch.flatten(xi, 1)                  # [B,512]
            logit = self.individual_fcs[i](xi)         # [B,1]
            outputs.append(logit)

        # outputs => list of [B,1], one per ensemble member
        outputs = torch.stack(outputs, dim=0)  # [ensemble_size, B, 1]

        return outputs