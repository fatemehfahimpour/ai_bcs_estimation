import torch
import torch.nn as nn
from torchvision import models


class OrdinalResNet18(nn.Module):

    def __init__(self, trainable_layers="layer4_fc", num_thresholds=4):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        # ResNet18 feature size = 512
        self.model.fc = nn.Linear(
            in_features=512,
            out_features=num_thresholds
        )

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        self._set_trainable_layers(trainable_layers)

    def _set_trainable_layers(self, trainable_layers):

        if trainable_layers == "fc":

            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif trainable_layers == "layer4_fc":

            for param in self.model.layer4.parameters():
                param.requires_grad = True

            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif trainable_layers == "layer3_layer4_fc":

            for param in self.model.layer3.parameters():
                param.requires_grad = True

            for param in self.model.layer4.parameters():
                param.requires_grad = True

            for param in self.model.fc.parameters():
                param.requires_grad = True

        else:
            raise ValueError(
                f"Unknown trainable_layers: {trainable_layers}"
            )

    def forward(self, x):
        return self.model(x)