"""
Evaluation and Testing Suite for the Best ResNet18 BCS Classification Model.

Loads the best trained ResNet18 checkpoint, evaluates performance on the test dataset,
computes classification and regression metrics (Accuracy, MAE, RMSE, Tolerance Accuracy,
Per-class F1/Precision/Recall), generates visualization charts, and saves summary artifacts.
"""

import json
import os
from typing import Any, Dict, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from data.preprocess import get_data_loader
from ResNet_model import BCSResNet18

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH: str = "model_results/best_resnet18.pth"
CRITERION: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

CLASS_NAMES: List[str] = ["3.25", "3.50", "3.75", "4.00", "4.25"]
BCS_VALUES: np.ndarray = np.array([3.25, 3.50, 3.75, 4.00, 4.25])

SAVE_DIR: str = "test_results"


def plot_confusion_matrix(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    class_names: List[str],
    save_path: str,
) -> None:
    """
    Generate and save a standard count-based confusion matrix heatmap.

    Args:
        y_true (Union[np.ndarray, Sequence[int]]): Ground truth class indices.
        y_pred (Union[np.ndarray, Sequence[int]]): Predicted class indices.
        class_names (List[str]): List of textual labels for each class.
        save_path (str): Destination file path to save the generated figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted BCS")
    plt.ylabel("Actual BCS")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_normalized_confusion_matrix(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    class_names: List[str],
    save_path: str,
) -> None:
    """
    Generate and save a row-normalized confusion matrix heatmap.

    Args:
        y_true (Union[np.ndarray, Sequence[int]]): Ground truth class indices.
        y_pred (Union[np.ndarray, Sequence[int]]): Predicted class indices.
        class_names (List[str]): List of textual labels for each class.
        save_path (str): Destination file path to save the generated figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted BCS")
    plt.ylabel("Actual BCS")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    class_names: List[str],
    save_path: str,
) -> None:
    """
    Plot and save a comparative bar chart of actual vs predicted class frequencies.

    Args:
        y_true (Union[np.ndarray, Sequence[int]]): Ground truth class indices.
        y_pred (Union[np.ndarray, Sequence[int]]): Predicted class indices.
        class_names (List[str]): List of textual labels for each class.
        save_path (str): Destination file path to save the generated figure.
    """
    true_counts = [np.sum(np.array(y_true) == i) for i in range(len(class_names))]
    pred_counts = [np.sum(np.array(y_pred) == i) for i in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, true_counts, width, label="Actual")
    plt.bar(x + width / 2, pred_counts, width, label="Predicted")
    plt.xticks(x, class_names)
    plt.xlabel("BCS Class")
    plt.ylabel("Count")
    plt.title("Actual vs Predicted Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distribution(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    save_path: str,
) -> None:
    """
    Plot and save a histogram showing the distribution of prediction index errors.

    Args:
        y_true (Union[np.ndarray, Sequence[int]]): Ground truth class indices.
        y_pred (Union[np.ndarray, Sequence[int]]): Predicted class indices.
        save_path (str): Destination file path to save the generated figure.
    """
    errors = np.array(y_pred) - np.array(y_true)

    plt.figure(figsize=(8, 5))
    bins = np.arange(errors.min() - 0.5, errors.max() + 1.5, 1)
    plt.hist(errors, bins=bins, edgecolor="black")
    plt.xlabel("Prediction Error (Predicted Index - Actual Index)")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    y_true: Union[np.ndarray, Sequence[int]],
    y_pred: Union[np.ndarray, Sequence[int]],
    save_path: str,
) -> None:
    """
    Plot and save a scatter plot of actual vs predicted continuous BCS scores.

    Args:
        y_true (Union[np.ndarray, Sequence[int]]): Ground truth class indices.
        y_pred (Union[np.ndarray, Sequence[int]]): Predicted class indices.
        save_path (str): Destination file path to save the generated figure.
    """
    actual_bcs = BCS_VALUES[np.array(y_true)]
    predicted_bcs = BCS_VALUES[np.array(y_pred)]

    plt.figure(figsize=(8, 6))
    plt.scatter(actual_bcs, predicted_bcs, alpha=0.25)
    plt.plot(BCS_VALUES, BCS_VALUES, linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.xticks(BCS_VALUES)
    plt.yticks(BCS_VALUES)
    plt.xlabel("Actual BCS")
    plt.ylabel("Predicted BCS")
    plt.title("Actual vs Predicted BCS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def test_model() -> None:
    """
    Run the end-to-end test evaluation pipeline for the best ResNet18 model.

    Loads the saved model weights, evaluates on the test data loader, logs
    comprehensive metrics (loss, accuracy, MAE, RMSE, tolerance thresholds),
    exports plots, and saves prediction dataframes and metric summaries.
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

    model = BCSResNet18(trainable_layers=trainable_layers).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = get_data_loader()

    criterion = CRITERION

    total_loss: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    print("\n" + "=" * 60)
    print("Testing best ResNet18 model on TEST set")
    print("=" * 60)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx % 100 == 0:
                print(f"Batch: {idx}")

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, dtype=torch.long, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

            all_preds.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_loss: float = total_loss / total_samples
    test_accuracy: float = correct_predictions / total_samples

    all_preds_arr: np.ndarray = np.array(all_preds)
    all_labels_arr: np.ndarray = np.array(all_labels)

    actual_bcs: np.ndarray = BCS_VALUES[all_labels_arr]
    predicted_bcs: np.ndarray = BCS_VALUES[all_preds_arr]

    mae_classes: float = float(mean_absolute_error(all_labels_arr, all_preds_arr))
    mae_bcs: float = float(mean_absolute_error(actual_bcs, predicted_bcs))
    rmse_bcs: float = float(np.sqrt(mean_squared_error(actual_bcs, predicted_bcs)))

    tolerance_1_acc: float = float(np.mean(np.abs(all_preds_arr - all_labels_arr) <= 1))
    tolerance_2_acc: float = float(np.mean(np.abs(all_preds_arr - all_labels_arr) <= 2))

    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_labels_arr,
        all_preds_arr,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"MAE (Class Index): {mae_classes:.4f}")
    print(f"MAE (BCS): {mae_bcs:.4f}")
    print(f"RMSE (BCS): {rmse_bcs:.4f}")
    print(f"Tolerance Accuracy (±1 class / ±0.25 BCS): {tolerance_1_acc:.4f}")
    print(f"Tolerance Accuracy (±2 classes / ±0.50 BCS): {tolerance_2_acc:.4f}")

    print("\n" + "=" * 60)
    print("PER-CLASS PRECISION, RECALL AND F1-SCORE")
    print("=" * 60)

    per_class_metrics: Dict[str, Dict[str, Union[float, int]]] = {}

    for i, class_name in enumerate(CLASS_NAMES):
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1_score[i]),
            "support": int(support[i]),
        }

        print(f"\nBCS {class_name}")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1-score: {f1_score[i]:.4f}")
        print(f"Support: {support[i]}")

    report = classification_report(
        all_labels_arr,
        all_preds_arr,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Generate and export visualization plots
    plot_confusion_matrix(
        all_labels_arr,
        all_preds_arr,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "confusion_matrix.png"),
    )
    plot_normalized_confusion_matrix(
        all_labels_arr,
        all_preds_arr,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "normalized_confusion_matrix.png"),
    )
    plot_class_distribution(
        all_labels_arr,
        all_preds_arr,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "class_distribution.png"),
    )
    plot_error_distribution(
        all_labels_arr,
        all_preds_arr,
        os.path.join(SAVE_DIR, "error_distribution.png"),
    )
    plot_actual_vs_predicted(
        all_labels_arr,
        all_preds_arr,
        os.path.join(SAVE_DIR, "actual_vs_predicted.png"),
    )

    # Save aggregated test metrics
    metrics: Dict[str, Any] = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "mae_class_index": float(mae_classes),
        "mae_bcs": float(mae_bcs),
        "rmse_bcs": float(rmse_bcs),
        "tolerance_accuracy_0.25": float(tolerance_1_acc),
        "tolerance_accuracy_0.50": float(tolerance_2_acc),
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

    # Export individual sample predictions to CSV
    predictions_df = pd.DataFrame(
        {
            "actual_class_index": all_labels_arr,
            "predicted_class_index": all_preds_arr,
            "actual_bcs": actual_bcs,
            "predicted_bcs": predicted_bcs,
        }
    )
    predictions_df.to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {SAVE_DIR}")
    print("\nGenerated files:")
    print("1. confusion_matrix.png")
    print("2. normalized_confusion_matrix.png")
    print("3. class_distribution.png")
    print("4. error_distribution.png")
    print("5. actual_vs_predicted.png")
    print("6. test_metrics.json")
    print("7. classification_report.txt")
    print("8. test_predictions.csv")


if __name__ == "__main__":
    test_model()
