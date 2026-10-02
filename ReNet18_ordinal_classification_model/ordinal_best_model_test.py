"""
Evaluation and Testing Suite for the Best Ordinal ResNet18 Model.

Loads the best model checkpoint from training, evaluates performance on the test set,
computes classification and regression metrics (Accuracy, MAE, RMSE, F1-score),
and exports visual plots and reports (confusion matrices, distributions, CSV, and JSON).
"""

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
import torch

from ReNet18_ordinal_classification_model.ordinal_dataset_dataloader import get_ordinal_data_loader
from ReNet18_ordinal_classification_model.ordinal_model import OrdinalResNet18

SAVE_DIR: str = "test_results"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH: str = "model_results_round2/best_ordinal_resnet18.pth"
CRITERION: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
CLASS_NAMES: List[str] = ["3.25", "3.50", "3.75", "4.00", "4.25"]
BCS_VALUES: npt.NDArray[np.float64] = np.array([3.25, 3.50, 3.75, 4.00, 4.25])


def logits_to_class(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw ordinal logits into predicted discrete class indices.

    Applies a sigmoid activation to cumulative binary thresholds and counts
    the number of thresholds exceeding the 0.5 probability cutoff.

    Args:
        logits (torch.Tensor): Unnormalized ordinal logits of shape (batch_size, num_thresholds).

    Returns:
        torch.Tensor: Predicted class indices of shape (batch_size,).
    """
    probabilities = torch.sigmoid(logits)
    class_indices = (probabilities >= 0.5).sum(dim=1)
    return class_indices.long()


def test_model() -> None:
    """
    Load the best model checkpoint, run test inference, and save all evaluation artifacts.

    Computes comprehensive loss, accuracy, MAE, and RMSE metrics, generates
    confusion matrices and distribution figures, and exports prediction records
    to CSV and JSON.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint: Dict[str, Any] = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

    trainable_layers: str = checkpoint["trainable_layers"]

    print(f"Trainable layers: {trainable_layers}")

    if "optimizer" in checkpoint:
        print(f"Optimizer: {checkpoint['optimizer']}")

    if "learning_rate" in checkpoint:
        print(f"Learning rate: {checkpoint['learning_rate']}")

    if "best_epoch" in checkpoint:
        print(f"Best epoch: {checkpoint['best_epoch']}")

    model = OrdinalResNet18(trainable_layers=trainable_layers, num_thresholds=4).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loader, val_loader, test_loader = get_ordinal_data_loader()

    criterion = CRITERION

    total_loss = 0.0
    total_samples = 0

    all_predictions_list: List[np.ndarray] = []
    all_targets_list: List[np.ndarray] = []
    all_probabilities_list: List[np.ndarray] = []

    print("\n" + "=" * 60)
    print("Testing best model on TEST set")
    print("=" * 60)

    with torch.no_grad():
        for images, ordinal_targets, class_indices in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            ordinal_targets = ordinal_targets.to(DEVICE, non_blocking=True)
            class_indices = class_indices.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, ordinal_targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits)
            predictions = logits_to_class(logits)

            all_predictions_list.extend(predictions.cpu().numpy())
            all_targets_list.extend(class_indices.cpu().numpy())
            all_probabilities_list.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions_list)
    all_targets = np.array(all_targets_list)
    all_probabilities = np.array(all_probabilities_list)

    test_loss = total_loss / total_samples
    test_accuracy = float(np.mean(all_predictions == all_targets))
    test_mae_class = float(mean_absolute_error(all_targets, all_predictions))

    actual_bcs = BCS_VALUES[all_targets]
    predicted_bcs = BCS_VALUES[all_predictions]

    test_mae_bcs = float(mean_absolute_error(actual_bcs, predicted_bcs))
    test_rmse_bcs = float(np.sqrt(np.mean((actual_bcs - predicted_bcs) ** 2)))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss       : {test_loss:.4f}")
    print(f"Test Accuracy   : {test_accuracy:.4f}")
    print(f"Test MAE (class): {test_mae_class:.4f}")
    print(f"Test MAE (BCS)  : {test_mae_bcs:.4f}")
    print(f"Test RMSE (BCS) : {test_rmse_bcs:.4f}")

    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_targets, all_predictions, labels=np.arange(5), zero_division=0
    )

    print("\n" + "=" * 60)
    print("PER-CLASS PRECISION, RECALL AND F1-SCORE")
    print("=" * 60)

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"\nBCS {class_name}")
        print(f"Precision : {precision[i]:.4f}")
        print(f"Recall    : {recall[i]:.4f}")
        print(f"F1-score  : {f1_score[i]:.4f}")
        print(f"Support   : {support[i]}")

    report = classification_report(
        all_targets,
        all_predictions,
        labels=np.arange(5),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=np.arange(5))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted BCS")
    plt.ylabel("Actual BCS")
    plt.title("Test Set - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "test_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Normalized Confusion Matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted BCS")
    plt.ylabel("Actual BCS")
    plt.title("Test Set - Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, "test_confusion_matrix_normalized.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_bcs, predicted_bcs, alpha=0.25)
    plt.plot(BCS_VALUES, BCS_VALUES, linestyle="--", linewidth=2, label="Perfect prediction")
    plt.xticks(BCS_VALUES)
    plt.yticks(BCS_VALUES)
    plt.xlabel("Actual BCS")
    plt.ylabel("Predicted BCS")
    plt.title("Test Set - Actual vs Predicted BCS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "test_actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Actual vs Predicted Distribution Histogram
    actual_counts = np.bincount(all_targets, minlength=5)
    predicted_counts = np.bincount(all_predictions, minlength=5)
    x = np.arange(5)
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, actual_counts, width, label="Actual")
    plt.bar(x + width / 2, predicted_counts, width, label="Predicted")
    plt.xticks(x, CLASS_NAMES)
    plt.xlabel("BCS")
    plt.ylabel("Number of images")
    plt.title("Test Set - Actual vs Predicted Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "test_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Export predictions to CSV
    predictions_df = pd.DataFrame({
        "actual_class_index": all_targets,
        "predicted_class_index": all_predictions,
        "actual_bcs": actual_bcs,
        "predicted_bcs": predicted_bcs,
    })

    for i in range(4):
        predictions_df[f"threshold_{i + 1}_probability"] = all_probabilities[:, i]

    predictions_df.to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)

    # Export summary metrics to JSON
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for i, class_name in enumerate(CLASS_NAMES):
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1_score[i]),
            "support": int(support[i]),
        }

    metrics: Dict[str, Any] = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_mae_class": float(test_mae_class),
        "test_mae_bcs": float(test_mae_bcs),
        "test_rmse_bcs": float(test_rmse_bcs),
        "num_test_samples": int(total_samples),
        "per_class_metrics": per_class_metrics,
        "trainable_layers": trainable_layers,
        "optimizer": checkpoint.get("optimizer", None),
        "learning_rate": checkpoint.get("learning_rate", None),
        "best_epoch": checkpoint.get("best_epoch", None),
    }

    with open(os.path.join(SAVE_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("Test evaluation completed.")
    print(f"Results saved to: {SAVE_DIR}")
    print("=" * 60)

    print("\nGenerated files:")
    print("1. test_confusion_matrix.png")
    print("2. test_confusion_matrix_normalized.png")
    print("3. test_actual_vs_predicted.png")
    print("4. test_distribution.png")
    print("5. test_predictions.csv")
    print("6. test_metrics.json")
    print("7. classification_report.txt")


if __name__ == "__main__":
    test_model()
