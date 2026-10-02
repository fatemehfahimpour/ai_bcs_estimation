"""
Hyperparameter Search Pipeline for Standard ResNet18 BCS Classification.

Performs a grid search over specified trainable backbone layer configurations,
optimizers, and learning rates. Automatically tracks validation loss/accuracy,
saves evaluation metrics to JSON, saves the best model checkpoint, and generates
training history visualizations.
"""

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from data.preprocess import get_data_loader
from ResNet_model import BCSResNet18
from ResNet_trainer import Trainer

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

RESULTS_PATH: str = "model_results/resnet_hyperparameter_results.json"
BEST_MODEL_PATH: str = "model_results/best_resnet18.pth"

TRAINABLE_LAYERS: List[str] = ["layer4_fc", "layer3_layer4_fc"]
OPTIMIZERS: List[str] = ["adam"]
ADAM_LRS: List[float] = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
EPOCHS: int = 100
PATIENCE: int = 20


def show_best_model_result(best_history: Dict[str, List[float]]) -> None:
    """
    Plot and save training and validation metrics for the optimal hyperparameter run.

    Generates two plots (loss and accuracy progression over epochs) and exports them
    to PNG files alongside the best model checkpoint.

    Args:
        best_history (Dict[str, List[float]]): History dictionary containing training
            and validation losses and accuracies across all epochs.
    """
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    epochs = range(1, len(best_history["train_loss"]) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, best_history["train_loss"], label="Train Loss")
    plt.plot(epochs, best_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Best Model - Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(
        os.path.dirname(BEST_MODEL_PATH),
        "best_model_loss.png",
    )
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, best_history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, best_history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Best Model - Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = os.path.join(
        os.path.dirname(BEST_MODEL_PATH),
        "best_model_accuracy.png",
    )
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def find_parameters() -> List[Dict[str, Any]]:
    """
    Execute systematic grid search across candidate training hyperparameter combinations.

    Iterates over layer unfreezing depths, optimization algorithms, and learning rates.
    Selects the best performing model based on minimum validation loss, serializes
    the full grid search summary into JSON, saves the winning weights, and creates plots.

    Returns:
        List[Dict[str, Any]]: List containing metric dictionaries for all completed experiments.
    """
    train_loader, val_loader, _ = get_data_loader()
    results: List[Dict[str, Any]] = []

    total_experiments: int = len(TRAINABLE_LAYERS) * len(ADAM_LRS)
    experiment_number: int = 0

    best_result: Optional[Dict[str, Any]] = None
    best_model_state: Optional[Dict[str, Any]] = None
    best_history: Optional[Dict[str, List[float]]] = None
    best_total_val_loss: float = float("inf")

    for trainable_layers in TRAINABLE_LAYERS:
        for optimizer_name in OPTIMIZERS:

            learning_rates: Optional[List[float]] = None
            if optimizer_name == "adam":
                learning_rates = ADAM_LRS

            if learning_rates is None:
                continue

            for learning_rate in learning_rates:
                experiment_number += 1
                print(f"experiment number: {experiment_number}/{total_experiments}")

                model = BCSResNet18(trainable_layers=trainable_layers)
                model = model.to(DEVICE)

                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=DEVICE,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    criterion_name="cross_entropy",
                    epoch=EPOCHS,
                    patience=PATIENCE,
                )

                (
                    history,
                    best_experiment_train_loss,
                    best_experiment_val_loss,
                    best_experiment_train_accuracy,
                    best_val_experiment_accuracy,
                    best_experiment_model_state,
                ) = trainer.fit()

                best_epoch: int = history["val_loss"].index(best_experiment_val_loss) + 1

                result: Dict[str, Any] = {
                    "trainable_layers": trainable_layers,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "best_epoch": best_epoch,
                    "best_train_loss": best_experiment_train_loss,
                    "best_val_loss": best_experiment_val_loss,
                    "best_train_accuracy": best_experiment_train_accuracy,
                    "best_val_accuracy": best_val_experiment_accuracy,
                }

                if best_experiment_val_loss < best_total_val_loss:
                    best_total_val_loss = best_experiment_val_loss
                    best_result = result
                    best_model_state = best_experiment_model_state
                    best_history = history

                results.append(result)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save summary of all experiments
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as file:
        json.dump(results, file, indent=4)

    # Save best checkpoint and corresponding hyperparameter config
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    if best_result is not None and best_model_state is not None:
        torch.save(
            {
                "model_state_dict": best_model_state,
                "trainable_layers": best_result["trainable_layers"],
                "optimizer": best_result["optimizer"],
                "learning_rate": best_result["learning_rate"],
                "best_epoch": best_result["best_epoch"],
                "best_val_loss": best_result["best_val_loss"],
                "best_val_accuracy": best_result["best_val_accuracy"],
            },
            BEST_MODEL_PATH,
        )

        print("\nBEST HYPERPARAMETERS")
        print(f"Trainable layers: {best_result['trainable_layers']}")
        print(f"Optimizer: {best_result['optimizer']}")
        print(f"Learning rate: {best_result['learning_rate']}")
        print(f"Best epoch: {best_result['best_epoch']}")
        print(f"Best val loss: {best_result['best_val_loss']:.4f}")
        print(f"Best val accuracy: {best_result['best_val_accuracy']:.4f}")
        print(f"Best train accuracy: {best_result['best_train_accuracy']:.4f}")
        print(f"\nResults saved to: {RESULTS_PATH}")

    if best_history is not None:
        show_best_model_result(best_history)

    return results


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE}")
    find_parameters()
