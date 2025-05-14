"""
models.py
"""

import torch
import torch.nn as nn
from torchvision import models

class ResNet18Robust(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        # Modify first convolution to support desired channels.
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.resnet(x)

class ResNet18MIMO(nn.Module):
    def __init__(self, in_channels=1, ensemble_size=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        base = models.resnet18(pretrained=False)
        # Shared layers (from bn1 to avgpool)
        self.shared_layers = nn.Sequential(
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool
        )
        in_feats = base.fc.in_features
        # Separate head-specific initial conv and final FC for each ensemble head.
        self.head_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            for _ in range(ensemble_size)
        ])
        self.head_fcs = nn.ModuleList([nn.Linear(in_feats, 1) for _ in range(ensemble_size)])

    def forward(self, x):
        if x.ndim == 5:
            outputs = []
            for i in range(self.ensemble_size):
                xi = x[i]
                out = self.head_convs[i](xi)
                out = self.shared_layers(out)
                out = torch.flatten(out, 1)
                out = self.head_fcs[i](out)
                outputs.append(out)
            return torch.stack(outputs, dim=0)
        else:
            out = self.head_convs[0](x)
            out = self.shared_layers(out)
            out = torch.flatten(out, 1)
            out = self.head_fcs[0](out)
            return out

class ResNet18ManifoldMixup(nn.Module):
    """
    ResNet-18 that allows manifold mixup on ONE of multiple hidden layers.
    The network is split into:
      - layer0: conv1, bn1, relu, maxpool
      - layer1: base.layer1
      - layer2: base.layer2
      - layer3: base.layer3
      - layer4: base.layer4
      - avgpool and fc for classification
    The forward method accepts parameters to apply mixup at a specific layer.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        in_feats = base.fc.in_features
        self.fc = nn.Linear(in_feats, 1)

    def forward(self, x, mix_layer=None, mix_fn=None, labels=None, alpha=1.0):
        """
        Forward pass with optional manifold mixup at exactly one layer.
        """
        x = self.layer0(x)
        if mix_layer == 0 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer1(x)
        if mix_layer == 1 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer2(x)
        if mix_layer == 2 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer3(x)
        if mix_layer == 3 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer4(x)
        if mix_layer == 4 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, labels

class MIMOManifoldMixup(nn.Module):
    """
    MIMOManifoldMixup combines multi-head (MIMO) architecture with manifold mixup.
    For each head, we use an individual first convolution and final classifier.
    The intermediate layers are shared and are split into:
      - shared_layer0: bn1, relu, maxpool (after head_conv)
      - layer1, layer2, layer3, layer4, avgpool
    The forward method supports applying manifold mixup at one chosen hidden layer.
    """
    def __init__(self, in_channels=1, ensemble_size=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        # Head-specific initial conv layers (acting as conv1)
        self.head_convs = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            for _ in range(ensemble_size)
        ])
        # Shared layers (after conv1)
        base = models.resnet18(pretrained=False)
        self.shared_layer0 = nn.Sequential(
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        in_feats = base.fc.in_features
        # Head-specific final classifiers
        self.head_fcs = nn.ModuleList([
            nn.Linear(in_feats, 1)
            for _ in range(ensemble_size)
        ])

    def shared_forward(self, x, mix_layer, mix_fn, labels, alpha):
        """
        Forward pass through the shared layers with optional manifold mixup.
        """
        x = self.shared_layer0(x)
        if mix_layer == 0 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer1(x)
        if mix_layer == 1 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer2(x)
        if mix_layer == 2 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer3(x)
        if mix_layer == 3 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.layer4(x)
        if mix_layer == 4 and mix_fn is not None and labels is not None:
            x, labels = mix_fn(x, labels, alpha=alpha)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, labels

    def forward(self, x, mix_layer=None, mix_fn=None, labels=None, alpha=1.0):
        """
        For multi-head input, x should have shape [ensemble_size, B, C, F, T].
        Otherwise, a single-head forward is performed.
        """
        if x.ndim == 5:
            outputs = []
            # For each head, apply the head-specific initial conv and shared forward.
            # The shared forward applies mixup at the specified layer.
            # The final output is a stack of logits from each head.
            # The labels_out are the labels after mixup.
            for i in range(self.ensemble_size):
                xi = x[i]
                f0 = self.head_convs[i](xi)  # head-specific initial features
                out, labels_out = self.shared_forward(f0, mix_layer, mix_fn, labels, alpha)
                logit = self.head_fcs[i](out)
                outputs.append(logit)
            return torch.stack(outputs, dim=0), labels_out
        else:
            f0 = self.head_convs[0](x)
            out, labels_out = self.shared_forward(f0, mix_layer, mix_fn, labels, alpha)
            logit = self.head_fcs[0](out)
            return logit, labels_out

def build_resnet_model(model_type, in_channels=1, ensemble_size=1):
    """
    Factory function to build a ResNet-based model.
    Options:
      - "ResNet18Robust"
      - "ResNet18MIMO"
      - "ResNet18ManifoldMixup"
      - "MIMOManifoldMixup"
    """
    if model_type == "ResNet18Robust":
        return ResNet18Robust(in_channels)
    elif model_type == "ResNet18MIMO":
        return ResNet18MIMO(in_channels, ensemble_size)
    elif model_type == "ResNet18ManifoldMixup":
        return ResNet18ManifoldMixup(in_channels)
    elif model_type == "MIMOManifoldMixup":
        return MIMOManifoldMixup(in_channels, ensemble_size)
    else:
        raise ValueError("Invalid model type")
