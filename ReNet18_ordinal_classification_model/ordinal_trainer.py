import copy
import time

import numpy as np
import torch
import torch.nn as nn


class OrdinalTrainer:

    def __init__(self, model, train_loader, val_loader,
                 device, learning_rate=1e-4, optimizer_name="adam",
                 epochs=30, patience=5, min_delta=0.001):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta

        # Binary Cross Entropy for each ordinal threshold
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = self.build_optimizer()

        self.history = {

            "train_loss": [],
            "val_loss": [],

            "train_accuracy": [],
            "val_accuracy": [],

            "train_mae": [],
            "val_mae": []
        }

    def build_optimizer(self):

        trainable_params = filter(
            lambda p: p.requires_grad,
            self.model.parameters()
        )

        if self.optimizer_name == "adam":

            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate
            )

        elif self.optimizer_name == "momentum":

            optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.learning_rate,
                momentum=0.9
            )

        else:

            raise ValueError(
                "Unknown optimizer"
            )

        return optimizer

    @staticmethod
    def logits_to_class(logits):

        probabilities = torch.sigmoid(logits)

        # Number of thresholds passed
        class_indices = (
                probabilities >= 0.5
        ).sum(dim=1)

        return class_indices.long()

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        total_mae = 0.0

        for images, ordinal_targets, class_indices in self.train_loader:
            images = images.to(
                self.device,
                non_blocking=True
            )

            ordinal_targets = ordinal_targets.to(
                self.device,
                non_blocking=True
            )

            class_indices = class_indices.to(
                self.device,
                non_blocking=True
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                ordinal_targets
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
            epoch_mae
        )

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        total_mae = 0.0

        with torch.no_grad():
            for images, ordinal_targets, class_indices in self.val_loader:
                images = images.to(
                    self.device,
                    non_blocking=True
                )

                ordinal_targets = ordinal_targets.to(
                    self.device,
                    non_blocking=True
                )

                class_indices = class_indices.to(
                    self.device,
                    non_blocking=True
                )

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    ordinal_targets
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
            epoch_mae
        )

    def fit(self):
        best_val_loss = np.inf
        best_model_state = None
        patience_counter = 0
        best_epoch = 0

        best_train_loss = None
        best_val_accuracy = None
        best_train_accuracy = None
        best_val_mae = None
        best_train_mae = None

        for epoch in range(self.epochs):
            epoch_start = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            train_start = time.time()

            (
                train_loss,
                train_accuracy,
                train_mae
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
                val_mae
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
                f"Val Loss: {val_loss:.4f}"
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
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
            "best_val_mae": best_val_mae
        }
