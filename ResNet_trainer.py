import numpy as np
import torch
from torch import nn


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate, optimizer_name='adam',
                 criterion_name='cross_entropy',
                 min_delta=0.001, epoch=30, patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.learning_rate = learning_rate
        self.min_delta = min_delta
        self.epoch = epoch
        self.patience = patience

        self.criterion_name = criterion_name
        self.criterion = None
        self.build_criterion()

        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.build_optimizer()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def build_optimizer(self):
        trainable_params = filter(
            lambda p: p.requires_grad,
            self.model.parameters()
        )
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate)
        elif self.optimizer_name == 'momentum':
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError("unknown optimizer_name")

    def build_criterion(self):
        if self.criterion_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("unknown criterion_name")

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # if batch_idx == 10:
            #     break

            images = images.to(self.device)

            labels = labels.to(self.device, dtype=torch.long)

            # Forward pass
            outputs = self.model(images)

            # Loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

        epoch_loss = (total_loss / total_samples)
        epoch_accuracy = (correct_predictions / total_samples)

        return epoch_loss, epoch_accuracy

    def validate(self):
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
                # Loss
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                total_loss += (loss.item() * batch_size)

                # Predictions
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += batch_size

        epoch_loss = (total_loss / total_samples)
        epoch_accuracy = (correct_predictions / total_samples)

        return epoch_loss, epoch_accuracy

    def fit(self):
        patience_counter = 0
        best_val_loss = np.inf
        best_train_loss = np.inf
        best_train_accuracy = np.inf
        best_val_accuracy = np.inf
        best_model_state = None

        for epoch in range(self.epoch):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            print(f"{epoch}. train loss: {train_loss}, val loss: {val_loss}")

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

        return self.history, best_train_loss, best_val_loss, best_train_accuracy, best_val_accuracy, best_model_state
