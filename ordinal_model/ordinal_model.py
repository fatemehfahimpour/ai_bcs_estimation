from torch import nn
from torchvision import models


class BSCOrdinal_ResNet18(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes
        self.backbone = models.resnet18(weights=models.ResNet18_weights.DEFAULT if pretrained else None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes - 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x