import json
import os
import torch
from matplotlib import pyplot as plt

from ResNet_model import BCSResNet18
from ResNet_trainer import Trainer
from preprocess import get_data_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = "meta_data/resnet_hyperparameter_results.json"
BEST_MODEL_PATH = "model_results/best_resnet18.pth"

TRAINABLE_LAYERS = ["fc", "layer4_fc", "layer3_layer4_fc"]
OPTIMIZERS = ["adam", "sgd", "momentum"]
ADAM_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
SGD_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
MOMENTUM_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
EPOCHS = 30
PATIENCE = 5
# TRAINABLE_LAYERS = ["fc"]
# OPTIMIZERS = ["adam"]
# LEARNING_RATES = [1e-4]
# EPOCHS = 1
# PATIENCE = 2


def show_best_model_result(best_history):
    os.makedirs(
        os.path.dirname(BEST_MODEL_PATH),
        exist_ok=True
    )

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
        "best_model_loss.png"
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
        "best_model_accuracy.png"
    )

    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def find_parameters():
    train_loader, val_loader, test_loader = get_data_loader()
    results = []

    total_experiments = (len(TRAINABLE_LAYERS) * (len(ADAM_LRS) + len(SGD_LRS) + len(MOMENTUM_LRS)))
    experiment_number = 0

    best_result = None
    best_model_state = None
    best_history = None
    best_total_val_loss = float("inf")

    for trainable_layers in TRAINABLE_LAYERS:
        for optimizer_name in OPTIMIZERS:

            LEARNING_RATES = None
            if optimizer_name == 'adam':
                LEARNING_RATES = ADAM_LRS
            elif optimizer_name == 'sgd':
                LEARNING_RATES = SGD_LRS
            elif optimizer_name == 'momentum':
                LEARNING_RATES = MOMENTUM_LRS

            for learning_rate in LEARNING_RATES:
                experiment_number += 1
                print(f'experiment number: {experiment_number}/{total_experiments}')

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
                    patience=PATIENCE
                )

                (history, best_experiment_train_loss, best_experiment_val_loss,
                 best_experiment_train_accuracy, best_val_experiment_accuracy, best_experiment_model_state) = trainer.fit()

                best_epoch = (history["val_loss"].index(best_experiment_val_loss) + 1)

                result = {
                    "trainable_layers": trainable_layers,
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "best_epoch": best_epoch,
                    "best_train_loss": best_experiment_train_loss,
                    "best_val_loss": best_experiment_val_loss,
                    "best_train_accuracy": best_experiment_train_accuracy,
                    "best_val_accuracy": best_val_experiment_accuracy
                }

                if best_experiment_val_loss < best_total_val_loss:
                    best_total_val_loss = best_experiment_val_loss
                    best_result = result
                    best_model_state = best_experiment_model_state
                    best_history = history

                results.append(result)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as file:
        json.dump(results, file, indent=4)

    # save best model
    os.makedirs(
        os.path.dirname(BEST_MODEL_PATH),
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": best_model_state,
            "trainable_layers": best_result["trainable_layers"],
            "optimizer": best_result["optimizer"],
            "learning_rate": best_result["learning_rate"],
            "best_epoch": best_result["best_epoch"],
            "best_val_loss": best_result["best_val_loss"],
            "best_val_accuracy": best_result["best_val_accuracy"]
        },
        BEST_MODEL_PATH
    )


    print("BEST HYPERPARAMETERS")
    print(f"Trainable layers: "f"{best_result['trainable_layers']}")
    print(f"Optimizer: "f"{best_result['optimizer']}")
    print(f"Learning rate: "f"{best_result['learning_rate']}")
    print(f"Best epoch: "f"{best_result['best_epoch']}")
    print(f"Best val loss: "f"{best_result['best_val_loss']:.4f}")
    print(f"Best val accuracy:"f"{best_result['best_val_accuracy']:.4f}")
    print(f"Best train accuracy: {best_result['best_train_accuracy']:.4f}")
    print(f"\nResults saved to: {RESULTS_PATH}")

    show_best_model_result(best_history)

    return results


if __name__ == "__main__":
    find_parameters()
