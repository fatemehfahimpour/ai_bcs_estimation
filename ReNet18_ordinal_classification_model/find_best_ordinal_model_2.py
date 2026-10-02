"""
Round 2 Hyperparameter Tuning and Grid Search for Ordinal ResNet18.

Focuses on deeper trainable layer configurations, refined learning rate grids,
and backbone differential learning rate scaling. Evaluates candidate models,
generates comparison plots, and saves checkpoints and summary JSON reports.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ReNet18_ordinal_classification_model.ordinal_dataset_dataloader import get_ordinal_data_loader
from ReNet18_ordinal_classification_model.ordinal_model import OrdinalResNet18
from ReNet18_ordinal_classification_model.ordinal_trainer import OrdinalTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV: str = "../meta_data/splits/train.csv"
VAL_CSV: str = "../meta_data/splits/val.csv"

# Round 2 storage paths
MODEL_PATH: str = "model_results_round2/best_ordinal_resnet18.pth"
RESULT_PATH: str = "model_results_round2/ordinal_training_result_2.json"
PLOTS_DIR: str = "model_results_round2/plots"

EPOCHS: int = 100
PATIENCE: int = 20
MIN_DELTA: float = 0.001
BACKBONE_LR_RATIO: float = 0.1  # Backbone learning rate scaling factor

# ============================================================
# Optimized hyperparameters for Round 2 (focusing on deeper layers and refined LRs)
# ============================================================
TRAINABLE_LAYERS: List[str] = ["layer4_fc", "layer3_layer4_fc", "layer2_layer3_layer4_fc"]
OPTIMIZERS: List[str] = ["adam", "momentum"]
ADAM_LRS: List[float] = [5e-6, 1e-5, 3e-5]
MOMENTUM_LRS: List[float] = [3e-4, 1e-3, 3e-3]


def plot_results(
    results: List[Dict[str, Any]],
    best_result: Dict[str, Any],
    save_dir: str = PLOTS_DIR,
) -> None:
    """
    Generate and save evaluation plots for the best model and experiment comparisons.

    Creates separate line plots for the best model's training and validation metrics
    (Loss, Accuracy, MAE) over epochs, as well as comparison curves across all experiments.

    Args:
        results (List[Dict[str, Any]]): Metrics summary for each executed experiment.
        best_result (Dict[str, Any]): Dictionary containing configuration and history of the best run.
        save_dir (str): Directory where generated plot images are saved. Defaults to PLOTS_DIR.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # 1. Best Model - Loss
    # ============================================================
    history = best_result["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Best Model - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_model_loss.png"), dpi=300)
    plt.close()

    # ============================================================
    # 2. Best Model - Accuracy
    # ============================================================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Best Model - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_model_accuracy.png"), dpi=300)
    plt.close()

    # ============================================================
    # 3. Best Model - MAE
    # ============================================================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    plt.plot(epochs, history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Best Model - MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_model_mae.png"), dpi=300)
    plt.close()

    # ============================================================
    # Experiment labels
    # ============================================================
    experiment_labels: List[str] = []
    for i, result in enumerate(results):
        label = (
            f"{result['trainable_layers']}\n"
            f"{result['optimizer']}, "
            f"lr={result['learning_rate']}"
        )
        experiment_labels.append(label)

    x = range(len(results))

    # ============================================================
    # 4. Validation Loss - All Experiments
    # ============================================================
    val_losses = [result["best_val_loss"] for result in results]

    plt.figure(figsize=(14, 6))
    plt.plot(x, val_losses, marker="o", color="tab:red")
    plt.xticks(list(x), experiment_labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("Experiment")
    plt.ylabel("Best Validation Loss")
    plt.title("Validation Loss - All Experiments")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "experiments_validation_loss.png"), dpi=300)
    plt.close()

    # ============================================================
    # 5. Validation Accuracy - All Experiments
    # ============================================================
    val_accuracies = [result["best_val_accuracy"] for result in results]

    plt.figure(figsize=(14, 6))
    plt.plot(x, val_accuracies, marker="o", color="tab:blue")
    plt.xticks(list(x), experiment_labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("Experiment")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Validation Accuracy - All Experiments")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "experiments_validation_accuracy.png"), dpi=300)
    plt.close()

    # ============================================================
    # 6. Validation MAE - All Experiments
    # ============================================================
    val_maes = [result["best_val_mae"] for result in results]

    plt.figure(figsize=(14, 6))
    plt.plot(x, val_maes, marker="o", color="tab:green")
    plt.xticks(list(x), experiment_labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("Experiment")
    plt.ylabel("Best Validation MAE")
    plt.title("Validation MAE - All Experiments")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "experiments_validation_mae.png"), dpi=300)
    plt.close()

    print(f"\nAll plots saved to:\n{save_dir}")


def search(
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute Round 2 grid search across fine-grained model and optimizer parameters.

    Iterates through candidate layer depths and learning rates, enforces differential
    backbone learning rates, records validation metrics, preserves GPU VRAM,
    checkpoints the best configuration, and generates evaluation plots.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: List of all experiment results
            and the dictionary containing the optimal configuration.

    Raises:
        ValueError: If an unrecognized optimizer is specified.
    """
    results: List[Dict[str, Any]] = []

    best_result: Optional[Dict[str, Any]] = None
    best_model_state: Any = None
    best_val_loss = float("inf")

    total_experiments = len(TRAINABLE_LAYERS) * (len(ADAM_LRS) + len(MOMENTUM_LRS))
    current_experience = 0

    for trainable_layers in TRAINABLE_LAYERS:
        for optimizer in OPTIMIZERS:

            if optimizer == "adam":
                learning_rates = ADAM_LRS
            elif optimizer == "momentum":
                learning_rates = MOMENTUM_LRS
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            for learning_rate in learning_rates:
                print(f"\nExperiment: {current_experience + 1}/{total_experiments}")
                print(f"trainable_layers: {trainable_layers} | optimizer: {optimizer} | learning_rate: {learning_rate}")
                current_experience += 1

                model = OrdinalResNet18(
                    trainable_layers=trainable_layers,
                    num_thresholds=4,
                )
                model = model.to(DEVICE)

                trainer = OrdinalTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=DEVICE,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer,
                    epochs=EPOCHS,
                    patience=PATIENCE,
                    min_delta=MIN_DELTA,
                    backbone_lr_ratio=BACKBONE_LR_RATIO,  # Pass backbone scaling factor
                )

                result = trainer.fit()

                experiment_result: Dict[str, Any] = {
                    "trainable_layers": trainable_layers,
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                    "best_epoch": result["best_epoch"],
                    "best_train_loss": result["best_train_loss"],
                    "best_val_loss": result["best_val_loss"],
                    "best_train_accuracy": result["best_train_accuracy"],
                    "best_val_accuracy": result["best_val_accuracy"],
                    "best_train_mae": result["best_train_mae"],
                    "best_val_mae": result["best_val_mae"],
                    "history": result["history"],
                }

                results.append(experiment_result)

                # Check and update the best model
                if result["best_val_loss"] < best_val_loss:
                    best_val_loss = result["best_val_loss"]
                    best_result = experiment_result
                    best_model_state = result["best_model_state"]

                # Explicit GPU memory cleanup at the end of each experiment to avoid VRAM fragmentation
                del model
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save weights of the best performing model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "model_state_dict": best_model_state,
        "trainable_layers": best_result["trainable_layers"],
        "optimizer": best_result["optimizer"],
        "learning_rate": best_result["learning_rate"],
        "best_epoch": best_result["best_epoch"],
        "best_val_loss": best_result["best_val_loss"],
        "best_val_accuracy": best_result["best_val_accuracy"],
        "best_val_mae": best_result["best_val_mae"],
    }

    torch.save(checkpoint, MODEL_PATH)

    # Save results to a JSON file
    search_result: Dict[str, Any] = {
        "best_result": best_result,
        "all_results": results,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(search_result, f, indent=4)

    print("\n" + "=" * 50)
    print(f"Best trainable layers: {best_result['trainable_layers']}")
    print(f"Best optimizer: {best_result['optimizer']}")
    print(f"Best learning rate: {best_result['learning_rate']}")
    print(f"Best epoch: {best_result['best_epoch']}")
    print(f"Best validation loss: {best_result['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {best_result['best_val_accuracy']:.4f}")
    print(f"Best validation MAE: {best_result['best_val_mae']:.4f}")
    print(f"\nBest model saved to:\n{MODEL_PATH}")
    print(f"\nResults saved to:\n{RESULT_PATH}")
    print("=" * 50 + "\n")

    # Plot and save curves at the end of search process
    plot_results(results, best_result, save_dir=PLOTS_DIR)

    return results, best_result


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_ordinal_data_loader()

    results, best_result = search(train_loader, val_loader)
