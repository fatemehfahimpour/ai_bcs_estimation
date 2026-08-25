import torch.nn as nn
from torchvision import models


class BCSResNet18(nn.Module):

    def __init__(self, trainable_layers="fc"):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        self.model.fc = nn.Linear(
            in_features=512,
            out_features=5
        )

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze selected layers
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