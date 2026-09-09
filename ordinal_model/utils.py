import torch

import preprocess

def labels_to_levels(labels, num_classes):
    """
    labels: [B] with class indices in [0, ..., K-1]
    returns: [B, K-1]
    """
    levels = []
    for label in labels:
        level = [1] * label.item() + [0] * (num_classes - 1 - label.item())
        levels.append(level)
    return torch.tensor(levels, dtype=torch.float32, device=labels.device)


def levels_to_labels(probs, threshold=0.5):
    """
    probs: [B, K-1] after sigmoid
    returns predicted class indices [B]
    """
    return torch.sum(probs > threshold, dim=1)
