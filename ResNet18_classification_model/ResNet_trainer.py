"""
Model Training and Validation Pipeline for Multi-class ResNet Models.

Implements the Trainer class managing forward/backward passes, metric tracking,
configurable optimizers and loss criteria, epoch logging, and early stopping.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Standard training manager for multi-class classification architectures.

    Handles epoch iteration, optimizer construction, loss computation, validation,
    metric history recording, and early stopping based on validation loss.

    Attributes:
        model (nn.Module): Neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Compute device (CPU or CUDA).
        learning_rate (float): Initial learning rate for the optimizer.
        min_delta (float): Minimum improvement in validation loss to reset early stopping.
        epoch (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        criterion_name (str): Name of the loss function.
        criterion (Optional[nn.Module]): Loss function instance.
        optimizer_name (str): Name of the optimization algorithm.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer instance.
        history (Dict[str, List[float]]): Training and validation metric history.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float,
        optimizer_name: str = "adam",
        criterion_name: str = "cross_entropy",
        min_delta: float = 0.001,
        epoch: int = 30,
        patience: int = 5,
    ) -> None:
        """
        Initialize the Trainer with datasets, hyperparameters, and training settings.

        Args:
            model (nn.Module): Target classification model.
            train_loader (DataLoader): DataLoader providing training batches.
            val_loader (DataLoader): DataLoader providing validation batches.
            device (torch.device): Compute device for tensor operations.
            learning_rate (float): Learning rate parameter.
            optimizer_name (str): Optimizer algorithm ("adam", "sgd", "momentum"). Defaults to "adam".
            criterion_name (str): Loss function identifier ("cross_entropy"). Defaults to "cross_entropy".
            min_delta (float): Minimum change to qualify as validation improvement. Defaults to 0.001.
            epoch (int): Total training epochs. Defaults to 30.
            patience (int): Early stopping patience counter. Defaults to 5.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.device: torch.device = device

        self.learning_rate: float = learning_rate
        self.min_delta: float = min_delta
        self.epoch: int = epoch
        self.patience: int = patience

        self.criterion_name: str = criterion_name
        self.criterion: Optional[nn.Module] = None
        self.build_criterion()

        self.optimizer_name: str = optimizer_name
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.build_optimizer()

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def build_optimizer(self) -> None:
        """
        Instantiate the optimizer using only parameters requiring gradients.

        Raises:
            ValueError: If an unsupported optimizer name is supplied.
        """
        trainable_params = filter(
            lambda p: p.requires_grad,
            self.model.parameters(),
        )
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate)
        elif self.optimizer_name == "momentum":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError("unknown optimizer_name")

    def build_criterion(self) -> None:
        """
        Instantiate the loss function specified by criterion_name.

        Raises:
            ValueError: If an unsupported loss function name is supplied.
        """
        if self.criterion_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("unknown criterion_name")

    def train_one_epoch(self) -> Tuple[float, float]:
        """
        Execute one training epoch across all mini-batches in train_loader.

        Returns:
            Tuple[float, float]: Average training loss and accuracy for the epoch.
        """
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)

            # Forward pass
            outputs = self.model(images)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backward pass and parameter update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics accumulation
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        return epoch_loss, epoch_accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the validation dataset without gradient tracking.

        Returns:
            Tuple[float, float]: Average validation loss and accuracy.
        """
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                # Forward pass
                outputs = self.model(images)

                # Loss computation
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size

                # Predictions and accuracy accumulation
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += batch_size

        epoch_loss = total_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        return epoch_loss, epoch_accuracy

    def fit(
        self,
    ) -> Tuple[Dict[str, List[float]], float, float, float, float, Optional[Dict[str, Any]]]:
        """
        Execute full training loop with validation checks and early stopping.

        Monitors validation loss improvement, clones the best model weights,
        and halts training if no improvement occurs within the patience window.

        Returns:
            Tuple[Dict[str, List[float]], float, float, float, float, Optional[Dict[str, Any]]]:
                - history: Dictionary containing epoch metrics.
                - best_train_loss: Lowest recorded training loss.
                - best_val_loss: Lowest recorded validation loss.
                - best_train_accuracy: Highest training accuracy.
                - best_val_accuracy: Highest validation accuracy.
                - best_model_state: Cloned state dictionary of the optimal model checkpoint.
        """
        patience_counter = 0
        best_val_loss = np.inf
        best_train_loss = np.inf
        best_train_accuracy = 0.0
        best_val_accuracy = 0.0
        best_model_state: Optional[Dict[str, Any]] = None

        for epoch in range(self.epoch):
            epoch_start_time = time.time()

            train_start_time = time.time()
            train_loss, train_accuracy = self.train_one_epoch()
            train_time = time.time() - train_start_time

            val_start_time = time.time()
            val_loss, val_accuracy = self.validate()
            val_time = time.time() - val_start_time

            # Total epoch time
            epoch_time = time.time() - epoch_start_time

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            print(
                f"Epoch {epoch + 1}/{self.epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Train Time: {train_time:.2f}s | "
                f"Val Time: {val_time:.2f}s | "
                f"Total Time: {epoch_time:.2f}s"
            )

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            if train_loss < best_train_loss:
                best_train_loss = train_loss

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"early stopping at epoch {epoch}")
                break

        return (
            self.history,
            best_train_loss,
            best_val_loss,
            best_train_accuracy,
            best_val_accuracy,
            best_model_state,
        )
