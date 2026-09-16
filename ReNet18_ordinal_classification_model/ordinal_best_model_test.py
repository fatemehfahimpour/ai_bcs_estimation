import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from ReNet18_ordinal_classification_model.ordinal_dataset_dataloader import get_ordinal_data_loader
from ReNet18_ordinal_classification_model.ordinal_model import OrdinalResNet18

BEST_MODEL_PATH = "model_results_round2/best_ordinal_resnet18.pth"
SAVE_DIR = "test_results"
PREDICTIONS_PATH = os.path.join(SAVE_DIR, "test_predictions.csv")
METRICS_PATH = os.path.join(SAVE_DIR, "test_metrics.json")
REPORT_PATH = os.path.join(SAVE_DIR, "classification_report.txt")
CM_PATH = os.path.join(SAVE_DIR, "confusion_matrix.png")
CM_NORMALIZED_PATH = os.path.join(SAVE_DIR, "confusion_matrix_normalized.png")
SCATTER_PATH = os.path.join(SAVE_DIR, "actual_vs_predicted.png")
DISTRIBUTION_PATH = os.path.join(SAVE_DIR, "bcs_distribution.png")
ERROR_PATH = os.path.join(SAVE_DIR, "error_distribution.png")
CLASS_NAMES = ["3.25", "3.50", "3.75", "4.00", "4.25"]
BCS_VALUES = np.array([3.25, 3.50, 3.75, 4.00, 4.25])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.BCEWithLogitsLoss()

def calculate_metrics(actual_bcs, predicted_bcs, tolerance=0.25):
    absolute_errors = np.abs(actual_bcs - predicted_bcs)
    correct_without_tolerance = absolute_errors == 0
    correct_with_tolerance = absolute_errors <= tolerance
    accuracy_with_tolerance = np.mean(correct_with_tolerance)
    accuracy_without_tolerance = np.mean(correct_without_tolerance)
    mae = mean_absolute_error(actual_bcs, predicted_bcs)
    rmse = np.sqrt(mean_squared_error(actual_bcs, predicted_bcs))
    correct_count_with_tolerance = np.sum(correct_with_tolerance)
    correct_count_without_tolerance = np.sum(correct_without_tolerance)
    incorrect_count_with_tolerance = len(correct_with_tolerance) - correct_count_with_tolerance
    incorrect_count_without_tolerance = len(correct_without_tolerance) - correct_count_without_tolerance
    max_error = np.max(absolute_errors)
    mean_error = np.mean(absolute_errors)
    return {"accuracy_without_tolerance": float(accuracy_without_tolerance), "accuracy_with_tolerance": float(accuracy_with_tolerance), "correct_count_without_tolerance": int(correct_count_without_tolerance), "correct_count_with_tolerance": int(correct_count_with_tolerance), "incorrect_count_without_tolerance": int(incorrect_count_without_tolerance), "incorrect_count_with_tolerance": int(incorrect_count_with_tolerance), "mae": float(mae), "rmse": float(rmse), "max_error": float(max_error), "mean_error": float(mean_error)}

def print_metrics(metrics, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"accuracy_without_tolerance: {metrics['accuracy_without_tolerance']:.4f} ({metrics['accuracy_without_tolerance'] * 100:.2f}%)")
    print(f"accuracy_with_tolerance: {metrics['accuracy_with_tolerance']:.4f} ({metrics['accuracy_with_tolerance'] * 100:.2f}%)")
    print(f"correct_count_without_tolerance: {metrics['correct_count_without_tolerance']}")
    print(f"correct_count_with_tolerance: {metrics['correct_count_with_tolerance']}")
    print(f"incorrect_count_without_tolerance: {metrics['incorrect_count_without_tolerance']}")
    print(f"incorrect_count_with_tolerance: {metrics['incorrect_count_with_tolerance']}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Max Error: {metrics['max_error']:.4f}")
    print(f"Mean Absolute Error: {metrics['mean_error']:.4f}")

def logits_to_class(logits):
    probabilities = torch.sigmoid(logits)
    class_indices = (probabilities >= 0.5).sum(dim=1)
    return class_indices.long()

def plot_confusion_matrix(actual_classes, predicted_classes):
    cm = confusion_matrix(actual_classes, predicted_classes, labels=np.arange(5))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted BCS")
    plt.ylabel("Actual BCS")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=300)
    plt.close()

def plot_normalized_confusion_matrix(actual_classes, predicted_classes):
    cm = confusion_matrix(actual_classes, predicted_classes, labels=np.arange(5))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted BCS")
    plt.ylabel("Actual BCS")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_NORMALIZED_PATH, dpi=300)
    plt.close()

def plot_actual_vs_predicted(actual_bcs, predicted_bcs):
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_bcs, predicted_bcs, alpha=0.35)
    plt.plot([3.25, 4.25], [3.25, 4.25], linestyle="--")
    plt.xticks(BCS_VALUES)
    plt.yticks(BCS_VALUES)
    plt.xlabel("Actual BCS")
    plt.ylabel("Predicted BCS")
    plt.title("Actual vs Predicted BCS")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=300)
    plt.close()

def plot_bcs_distribution(actual_bcs, predicted_bcs):
    actual_counts = [np.sum(actual_bcs == bcs) for bcs in BCS_VALUES]
    predicted_counts = [np.sum(predicted_bcs == bcs) for bcs in BCS_VALUES]
    x = np.arange(len(BCS_VALUES))
    width = 0.35
    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, actual_counts, width, label="Actual")
    plt.bar(x + width / 2, predicted_counts, width, label="Predicted")
    plt.xticks(x, CLASS_NAMES)
    plt.xlabel("BCS")
    plt.ylabel("Number of Images")
    plt.title("Actual vs Predicted BCS Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PATH, dpi=300)
    plt.close()

def plot_error_distribution(actual_bcs, predicted_bcs):
    errors = predicted_bcs - actual_bcs
    plt.figure(figsize=(9, 6))
    plt.hist(errors, bins=np.arange(-1.375, 1.376, 0.125), edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.axvline(0.25, linestyle="--")
    plt.axvline(-0.25, linestyle="--")
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Number of Images")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(ERROR_PATH, dpi=300)
    plt.close()

def test():
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    trainable_layers = checkpoint["trainable_layers"]
    print(f"Device: {DEVICE}")
    print(f"Trainable layers: {trainable_layers}")
    model = OrdinalResNet18(trainable_layers=trainable_layers, num_thresholds=4).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, _, test_loader = get_ordinal_data_loader()
    total_loss = 0.0
    total_samples = 0
    actual_bcs = []
    predicted_bcs = []
    actual_classes = []
    predicted_classes = []
    threshold_probabilities = []
    with torch.no_grad():
        for images, ordinal_targets, class_indices in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            ordinal_targets = ordinal_targets.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = CRITERION(logits, ordinal_targets)
            probabilities = torch.sigmoid(logits)
            predictions = logits_to_class(logits)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            actual_classes.extend(class_indices.cpu().numpy())
            predicted_classes.extend(predictions.cpu().numpy())
            actual_bcs.extend(BCS_VALUES[class_indices.cpu().numpy()])
            predicted_bcs.extend(BCS_VALUES[predictions.cpu().numpy()])
            threshold_probabilities.extend(probabilities.cpu().numpy())
    actual_bcs = np.array(actual_bcs)
    predicted_bcs = np.array(predicted_bcs)
    actual_classes = np.array(actual_classes)
    predicted_classes = np.array(predicted_classes)
    threshold_probabilities = np.array(threshold_probabilities)
    test_loss = total_loss / total_samples
    metrics = calculate_metrics(actual_bcs, predicted_bcs, tolerance=0.25)
    metrics["test_loss"] = float(test_loss)
    metrics["num_test_samples"] = int(total_samples)
    print_metrics(metrics, "Test Metrics")
    print(f"test_loss: {test_loss:.4f}")
    plot_confusion_matrix(actual_classes, predicted_classes)
    plot_normalized_confusion_matrix(actual_classes, predicted_classes)
    plot_actual_vs_predicted(actual_bcs, predicted_bcs)
    plot_bcs_distribution(actual_bcs, predicted_bcs)
    plot_error_distribution(actual_bcs, predicted_bcs)
    predictions_df = pd.DataFrame({"actual_class_index": actual_classes, "predicted_class_index": predicted_classes, "actual_bcs": actual_bcs, "predicted_bcs": predicted_bcs, "threshold_1_probability": threshold_probabilities[:, 0], "threshold_2_probability": threshold_probabilities[:, 1], "threshold_3_probability": threshold_probabilities[:, 2], "threshold_4_probability": threshold_probabilities[:, 3]})
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    report = classification_report(actual_classes, predicted_classes, labels=np.arange(5), target_names=CLASS_NAMES, digits=4, zero_division=0)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Classification report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {CM_PATH}")
    print(f"Normalized confusion matrix saved to: {CM_NORMALIZED_PATH}")
    print(f"Actual vs predicted plot saved to: {SCATTER_PATH}")
    print(f"BCS distribution plot saved to: {DISTRIBUTION_PATH}")
    print(f"Error distribution plot saved to: {ERROR_PATH}")

if __name__ == "__main__":
    test()