import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from ResNet_model import BCSResNet18
from preprocess import get_data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "model_results/best_resnet18.pth"
CRITERION = torch.nn.CrossEntropyLoss()

CLASS_NAMES = ["3.25", "3.50", "3.75", "4.00", "4.25"]
SAVE_DIR = "test_results"


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_class_distribution(y_true, y_pred, class_names, save_path):
    true_counts = [sum(np.array(y_true) == i) for i in range(len(class_names))]
    pred_counts = [sum(np.array(y_pred) == i) for i in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, true_counts, width, label="True", color="skyblue")
    plt.bar(x + width/2, pred_counts, width, label="Predicted", color="orange")

    plt.xticks(x, class_names)
    plt.xlabel("BCS Class")
    plt.ylabel("Count")
    plt.title("True vs Predicted Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_error_distribution(y_true, y_pred, save_path):
    errors = np.array(y_pred) - np.array(y_true)

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=np.arange(errors.min() - 0.5, errors.max() + 1.5, 1), edgecolor='black')
    plt.xlabel("Prediction Error (Predicted Index - True Index)")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def test_model():
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    trainable_layers = checkpoint["trainable_layers"]
    print(f"Trainable layers: {trainable_layers}")

    model = BCSResNet18(trainable_layers=trainable_layers).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = get_data_loader()
    criterion = CRITERION

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx % 100 == 0:
                print(f"Batch: {idx}")

            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

            all_preds.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_loss = total_loss / total_samples
    test_accuracy = correct_predictions / total_samples

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae_classes = np.mean(np.abs(all_preds - all_labels))
    tolerance_1_acc = np.mean(np.abs(all_preds - all_labels) <= 1)   # ±1 class = ±0.25 BCS
    tolerance_2_acc = np.mean(np.abs(all_preds - all_labels) <= 2)   # ±2 class = ±0.50 BCS

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"MAE (class index): {mae_classes:.4f}")
    print(f"Tolerance Accuracy (±1 class / ±0.25 BCS): {tolerance_1_acc:.4f}")
    print(f"Tolerance Accuracy (±2 classes / ±0.50 BCS): {tolerance_2_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(
        all_labels,
        all_preds,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "confusion_matrix.png")
    )

    plot_class_distribution(
        all_labels,
        all_preds,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "class_distribution.png")
    )

    plot_error_distribution(
        all_labels,
        all_preds,
        os.path.join(SAVE_DIR, "error_distribution.png")
    )


if __name__ == "__main__":
    test_model()
