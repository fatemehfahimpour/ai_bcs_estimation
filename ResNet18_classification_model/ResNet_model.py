"""
ResNet18-based Multi-class Classification Model for Cow Body Condition Scoring (BCS).

Provides a fine-tunable ResNet18 architecture with configurable trainable layers
and explicit control over batch normalization and frozen sub-modules during training.
"""

import torch
import torch.nn as nn
from torchvision import models


class BCSResNet18(nn.Module):
    """
    ResNet18 architecture tailored for standard multi-class BCS classification.

    Replaces the default fully connected layer with a 5-class linear head
    and unfreezes specified backbone layers based on the requested fine-tuning depth.

    Attributes:
        model (models.ResNet): Underlying torchvision ResNet18 model.
        trainable_layers (str): Identifies the unfrozen layer configuration.
    """

    def __init__(self, trainable_layers: str = "fc") -> None:
        """
        Initialize the BCSResNet18 model with pretrained weights and custom head.

        Args:
            trainable_layers (str): Layer unfreezing policy. Options include:
                "fc", "layer4_fc", or "layer3_layer4_fc". Defaults to "fc".
        """
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.model: models.ResNet = models.resnet18(weights=weights)

        # Replace classification head for 5 BCS classes
        self.model.fc = nn.Linear(
            in_features=512,
            out_features=5,
        )

        # Freeze all pretrained layers by default
        for param in self.model.parameters():
            param.requires_grad = False

        self.trainable_layers: str = trainable_layers

        # Unfreeze selected layers according to configuration
        self._set_trainable_layers(trainable_layers)

    def _set_trainable_layers(self, trainable_layers: str) -> None:
        """
        Set gradients requirement for parameters according to the unfreezing strategy.

        Args:
            trainable_layers (str): Unfreezing policy name.

        Raises:
            ValueError: If an unsupported layer configuration name is provided.
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

        else:
            raise ValueError(f"Unknown trainable_layers: {trainable_layers}")

    def train(self, mode: bool = True) -> "BCSResNet18":
        """
        Set the training mode while keeping frozen layers in evaluation mode.

        Ensures that BatchNorm statistics and frozen residual stages are not
        inadvertently updated during fine-tuning.

        Args:
            mode (bool): Whether to activate training mode. Defaults to True.

        Returns:
            BCSResNet18: Self reference instance.
        """
        super().train(mode)

        if mode:
            if self.trainable_layers == "fc":
                self.model.conv1.eval()
                self.model.bn1.eval()
                self.model.layer1.eval()
                self.model.layer2.eval()
                self.model.layer3.eval()
                self.model.layer4.eval()

            elif self.trainable_layers == "layer4_fc":
                self.model.conv1.eval()
                self.model.bn1.eval()
                self.model.layer1.eval()
                self.model.layer2.eval()
                self.model.layer3.eval()

            elif self.trainable_layers == "layer3_layer4_fc":
                self.model.conv1.eval()
                self.model.bn1.eval()
                self.model.layer1.eval()
                self.model.layer2.eval()

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the ResNet18 network.

        Args:
            x (torch.Tensor): Input batch of images with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, 5).
        """
        return self.model(x)
