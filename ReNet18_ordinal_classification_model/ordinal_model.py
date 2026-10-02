"""
Ordinal ResNet-18 Neural Network Architecture Module.

Implements a transfer learning ResNet-18 model tailored for ordinal classification
tasks (e.g., Body Condition Scoring) using cumulative binary thresholds. Supports
configurable layer freezing and fine-tuning strategies.
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

TrainableLayersType = Literal[
    "fc",
    "layer4_fc",
    "layer3_layer4_fc",
    "layer2_layer3_layer4_fc",
]


class OrdinalResNet18(nn.Module):
    """
    Ordinal Classification Model based on a pretrained ResNet-18 backbone.

    Replaces the standard classification head with an ordinal output layer predicting
    logits for binary cumulative thresholds (K - 1 thresholds for K classes).
    """

    def __init__(
        self,
        trainable_layers: TrainableLayersType = "layer4_fc",
        num_thresholds: int = 4,
    ) -> None:
        """
        Initialize the Ordinal ResNet-18 architecture.

        Args:
            trainable_layers (str): Strategy specifying which layers remain unfreezed/trainable.
                Options: 'fc', 'layer4_fc', 'layer3_layer4_fc', 'layer2_layer3_layer4_fc'.
            num_thresholds (int): Number of ordinal threshold outputs (classes - 1). Default is 4.
        """
        super().__init__()

        # Load pretrained ResNet-18 weights
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        # ResNet-18 feature extraction dimension = 512
        self.model.fc = nn.Linear(
            in_features=512,
            out_features=num_thresholds,
        )

        # Freeze all backbone layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze designated subset of layers for fine-tuning
        self._set_trainable_layers(trainable_layers)

    def _set_trainable_layers(self, trainable_layers: str) -> None:
        """
        Configure parameter gradient updates based on selected fine-tuning strategy.

        Args:
            trainable_layers (str): Specification string for trainable layer blocks.

        Raises:
            ValueError: If an unrecognized trainable layer strategy is provided.
        """
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

        elif trainable_layers == "layer2_layer3_layer4_fc":
            for param in self.model.layer2.parameters():
                param.requires_grad = True
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input batch tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits for binary thresholds with shape (B, num_thresholds).
        """
        return self.model(x)
