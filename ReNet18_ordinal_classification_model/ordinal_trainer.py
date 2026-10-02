"""
Ordinal Model Training and Evaluation Pipeline.

Implements the OrdinalTrainer class responsible for training, validating,
and early-stopping deep neural network models with binary threshold logits.
Supports differential learning rates between the backbone and classification head.
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class OrdinalTrainer:
    """
    Trainer for ordinal classification models utilizing BCEWithLogitsLoss.

    Manages the training loop, validation evaluation, metric logging (Loss, Accuracy, MAE),
    learning rate partitioning between backbone and head layers, and model checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        optimizer_name: str = "adam",
        epochs: int = 30,
        patience: int = 5,
        min_delta: float = 0.001,
        backbone_lr_ratio: float = 0.1,  # 1. Added backbone learning rate scaling factor to input
    ) -> None:
        """
        Initialize the OrdinalTrainer with datasets, optimizer settings, and hyperparameters.

        Args:
            model (nn.Module): PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            device (torch.device): Compute device ('cuda' or 'cpu').
            learning_rate (float): Base learning rate for head layers. Defaults to 1e-4.
            optimizer_name (str): Name of the optimizer ('adam' or 'momentum'). Defaults to "adam".
            epochs (int): Maximum number of training epochs. Defaults to 30.
            patience (int): Number of epochs to wait for improvement before early stopping. Defaults to 5.
            min_delta (float): Minimum change in validation loss to qualify as an improvement. Defaults to 0.001.
            backbone_lr_ratio (float): Ratio to scale down learning rate for backbone parameters. Defaults to 0.1.
        """
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.backbone_lr_ratio = backbone_lr_ratio  # 2. Store ratio in class attribute

        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta

        # Binary Cross Entropy for each ordinal threshold
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = self.build_optimizer()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_mae": [],
            "val_mae": [],
        }

    # def build_optimizer(self):
    #
    #     trainable_params = filter(
    #         lambda p: p.requires_grad,
    #         self.model.parameters()
    #     )
    #
    #     if self.optimizer_name == "adam":
    #
    #         optimizer = torch.optim.Adam(
    #             trainable_params,
    #             lr=self.learning_rate
    #         )
    #
    #     elif self.optimizer_name == "momentum":
    #
    #         optimizer = torch.optim.SGD(
    #             trainable_params,
    #             lr=self.learning_rate,
    #             momentum=0.9
    #         )
    #
    #     else:
    #
    #         raise ValueError(
    #             "Unknown optimizer"
    #         )
    #
    #     return optimizer
    def build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build an optimizer with parameter groups supporting differential learning rates.

        Separates parameters of the fully connected head ('fc') from backbone layers
        to apply a reduced learning rate to the backbone.

        Returns:
            torch.optim.Optimizer: Initialized optimizer instance.

        Raises:
            ValueError: If an unsupported optimizer name is passed.
        """
        # 1. Separate output layer parameters (fc) from backbone layers
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        # 2. Construct parameter groups with dedicated learning rates
        param_groups: List[Dict[str, Any]] = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": self.learning_rate * self.backbone_lr_ratio,  # Lower rate for backbone layers
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": self.learning_rate,  # Base rate for output layer
            })

        # 3. Pass parameter groups to optimizer instead of a single list
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(param_groups)

        elif self.optimizer_name == "momentum":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9)

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        return optimizer

    @staticmethod
    def logits_to_class(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted threshold logits into discrete class indices.

        Applies sigmoid activation and sums the number of satisfied thresholds (>= 0.5).

        Args:
            logits (torch.Tensor): Model raw outputs of shape (batch_size, num_thresholds).

        Returns:
            torch.Tensor: Discrete ordinal class predictions of shape (batch_size,).
        """
        probabilities = torch.sigmoid(logits)

        # Number of thresholds passed
        class_indices = (
            probabilities >= 0.5
        ).sum(dim=1)

        return class_indices.long()

    def train_one_epoch(self) -> Tuple[float, float, float]:
        """
        Run a single training epoch across all batches in train_loader.

        Returns:
            Tuple[float, float, float]: Average training loss, accuracy, and mean absolute error (MAE).
        """
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        total_mae = 0.0

        for images, ordinal_targets, class_indices in self.train_loader:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            ordinal_targets = ordinal_targets.to(
                self.device,
                non_blocking=True,
            )

            class_indices = class_indices.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                ordinal_targets,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            predictions = self.logits_to_class(
                outputs
            )

            batch_size = images.size(0)
            total_loss += (
                loss.item() * batch_size
            )
            correct_predictions += (
                predictions == class_indices
            ).sum().item()
            total_mae += torch.abs(
                predictions.float()
                -
                class_indices.float()
            ).sum().item()

            total_samples += batch_size

        epoch_loss = (
            total_loss / total_samples
        )

        epoch_accuracy = (
            correct_predictions /
            total_samples
        )

        epoch_mae = (
            total_mae /
            total_samples
        )

        return (
            epoch_loss,
            epoch_accuracy,
            epoch_mae,
        )

    def validate(self) -> Tuple[float, float, float]:
        """
        Run evaluation on all batches in val_loader without computing gradients.

        Returns:
            Tuple[float, float, float]: Average validation loss, accuracy, and mean absolute error (MAE).
        """
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        total_mae = 0.0

        with torch.no_grad():
            for images, ordinal_targets, class_indices in self.val_loader:
                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                ordinal_targets = ordinal_targets.to(
                    self.device,
                    non_blocking=True,
                )

                class_indices = class_indices.to(
                    self.device,
                    non_blocking=True,
                )

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    ordinal_targets,
                )

                predictions = self.logits_to_class(
                    outputs
                )

                batch_size = images.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                correct_predictions += (
                    predictions == class_indices
                ).sum().item()

                total_mae += torch.abs(
                    predictions.float()
                    -
                    class_indices.float()
                ).sum().item()

                total_samples += batch_size

        epoch_loss = (
            total_loss / total_samples
        )

        epoch_accuracy = (
            correct_predictions /
            total_samples
        )

        epoch_mae = (
            total_mae /
            total_samples
        )

        return (
            epoch_loss,
            epoch_accuracy,
            epoch_mae,
        )

    def fit(self) -> Dict[str, Any]:
        """
        Execute full training routine across epochs with early stopping.

        Monitors validation loss to trigger early stopping if improvement is below min_delta
        for a consecutive number of epochs specified by patience.

        Returns:
            Dict[str, Any]: Dictionary containing training history, best model state dict,
                best epoch, and corresponding best metrics.
        """
        best_val_loss = float(np.inf)
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        best_epoch = 0

        best_train_loss: Optional[float] = None
        best_val_accuracy: Optional[float] = None
        best_train_accuracy: Optional[float] = None
        best_val_mae: Optional[float] = None
        best_train_mae: Optional[float] = None

        for epoch in range(self.epochs):
            epoch_start = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_start = time.time()

            (
                train_loss,
                train_accuracy,
                train_mae,
            ) = self.train_one_epoch()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_time = (
                time.time() - train_start
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            val_start = time.time()

            (
                val_loss,
                val_accuracy,
                val_mae,
            ) = self.validate()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            val_time = (
                time.time() - val_start
            )

            epoch_time = (
                time.time() - epoch_start
            )

            self.history["train_loss"].append(
                train_loss
            )

            self.history["val_loss"].append(
                val_loss
            )

            self.history["train_accuracy"].append(
                train_accuracy
            )

            self.history["val_accuracy"].append(
                val_accuracy
            )

            self.history["train_mae"].append(
                train_mae
            )

            self.history["val_mae"].append(
                val_mae
            )

            print(
                f"\nEpoch {epoch + 1}/{self.epochs}"
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )

            print(
                f"Train Time: {train_time:.2f}s | "
                f"Val Time: {val_time:.2f}s | "
                f"Total time: {epoch_time:.2f}s"
            )

            if val_loss < (
                best_val_loss -
                self.min_delta
            ):
                best_val_loss = val_loss

                patience_counter = 0

                best_epoch = epoch + 1

                best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )

                best_train_loss = train_loss
                best_train_accuracy = train_accuracy
                best_val_accuracy = val_accuracy
                best_train_mae = train_mae
                best_val_mae = val_mae

            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        return {
            "history": self.history,
            "best_model_state": best_model_state,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_train_accuracy": best_train_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "best_train_mae": best_train_mae,
            "best_val_mae": best_val_mae,
        }
