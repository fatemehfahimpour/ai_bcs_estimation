import json
import os

import torch

from ReNet18_ordinal_classification_model.ordinal_model import OrdinalResNet18
from ReNet18_ordinal_classification_model.ordinal_trainer import OrdinalTrainer
from data.preprocess import get_data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "../meta_data/splits/train.csv"
VAL_CSV = "../meta_data/splits/val.csv"

MODEL_PATH = "model_results/best_ordinal_resnet18.pth"
RESULT_PATH = "model_results/ordinal_training_result.json"

EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 0.001

TRAINABLE_LAYERS = ["fc", "layer4_fc", "layer3_layer4_fc"]
OPTIMIZERS = ["adam", "momentum"]
ADAM_LRS = [1e-5, 1e-4, 1e-3]
MOMENTUM_LRS = [1e-4, 1e-3, 1e-2]


import matplotlib.pyplot as plt


def plot_results(results, best_result, save_dir="model_results/plots"):

    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # 1. Best Model - Loss
    # ============================================================

    history = best_result["history"]

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Best Model - Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "best_model_loss.png"),
        dpi=300
    )

    plt.close()


    # ============================================================
    # 2. Best Model - Accuracy
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Best Model - Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "best_model_accuracy.png"),
        dpi=300
    )

    plt.close()


    # ============================================================
    # 3. Best Model - MAE
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_mae"],
        label="Train MAE"
    )

    plt.plot(
        epochs,
        history["val_mae"],
        label="Validation MAE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Best Model - MAE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "best_model_mae.png"),
        dpi=300
    )

    plt.close()


    # ============================================================
    # Experiment labels
    # ============================================================

    experiment_labels = []

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

    val_losses = [
        result["best_val_loss"]
        for result in results
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        val_losses,
        marker="o"
    )

    plt.xticks(
        list(x),
        experiment_labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Experiment")
    plt.ylabel("Best Validation Loss")
    plt.title("Validation Loss - All Experiments")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "experiments_validation_loss.png"),
        dpi=300
    )

    plt.close()


    # ============================================================
    # 5. Validation Accuracy - All Experiments
    # ============================================================

    val_accuracies = [
        result["best_val_accuracy"]
        for result in results
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        val_accuracies,
        marker="o"
    )

    plt.xticks(
        list(x),
        experiment_labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Experiment")
    plt.ylabel("Best Validation Accuracy")
    plt.title("Validation Accuracy - All Experiments")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "experiments_validation_accuracy.png"),
        dpi=300
    )

    plt.close()


    # ============================================================
    # 6. Validation MAE - All Experiments
    # ============================================================

    val_maes = [
        result["best_val_mae"]
        for result in results
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        val_maes,
        marker="o"
    )

    plt.xticks(
        list(x),
        experiment_labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Experiment")
    plt.ylabel("Best Validation MAE")
    plt.title("Validation MAE - All Experiments")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "experiments_validation_mae.png"),
        dpi=300
    )

    plt.close()


    print(f"\nAll plots saved to:")
    print(save_dir)

def search(train_loader, val_loader):
    results = []

    best_result = None
    best_model_state = None
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
                print(f"experiment: {current_experience}/{total_experiments}")
                print(f"trainable_layers: {trainable_layers} | optimizer: {optimizer} | learning_rate: {learning_rate}")
                current_experience += 1

                model = OrdinalResNet18(
                    trainable_layers=trainable_layers,
                    num_thresholds=4
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
                    min_delta=MIN_DELTA
                )

                result = trainer.fit()

                experiment_result = {
                    "trainable_layers":
                        trainable_layers,
                    "optimizer":
                        optimizer,
                    "learning_rate":
                        learning_rate,
                    "best_epoch":
                        result["best_epoch"],
                    "best_train_loss":
                        result["best_train_loss"],
                    "best_val_loss":
                        result["best_val_loss"],
                    "best_train_accuracy":
                        result["best_train_accuracy"],
                    "best_val_accuracy":
                        result["best_val_accuracy"],
                    "best_train_mae":
                        result["best_train_mae"],
                    "best_val_mae":
                        result["best_val_mae"],
                    "history":
                        result["history"]
                }

                results.append(experiment_result)
                if result["best_val_loss"] < best_val_loss:
                    best_val_loss = (
                        result["best_val_loss"]
                    )

                    best_result = (
                        experiment_result
                    )

                    best_model_state = (
                        result["best_model_state"]
                    )

                    del model
                    del trainer

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    checkpoint = {

        "model_state_dict":
            best_model_state,

        "trainable_layers":
            best_result["trainable_layers"],

        "optimizer":
            best_result["optimizer"],

        "learning_rate":
            best_result["learning_rate"],

        "best_epoch":
            best_result["best_epoch"],

        "best_val_loss":
            best_result["best_val_loss"],

        "best_val_accuracy":
            best_result["best_val_accuracy"],

        "best_val_mae":
            best_result["best_val_mae"]
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    search_result = {
        "best_result":
            best_result,
        "all_results":
            results
    }

    with open(
            RESULT_PATH,
            "w"
    ) as f:
        json.dump(
            search_result,
            f,
            indent=4
        )

    print(
        f"Best trainable layers: "
        f"{best_result['trainable_layers']}"
    )

    print(
        f"Best optimizer: "
        f"{best_result['optimizer']}"
    )

    print(
        f"Best learning rate: "
        f"{best_result['learning_rate']}"
    )

    print(
        f"Best epoch: "
        f"{best_result['best_epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{best_result['best_val_loss']:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_result['best_val_accuracy']:.4f}"
    )

    print(
        f"Best validation MAE: "
        f"{best_result['best_val_mae']:.4f}"
    )

    print(
        f"\nBest model saved to:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULT_PATH}"
    )
    return results, best_result


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loader()

    results, best_result = search(train_loader, val_loader)
