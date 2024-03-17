from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from typing import Any

import torch

from modelstealing.our_dataset import OurDataset
from tools.fs_tools import FsTools
from tools.logger import get_logger
from tools.path_organizer import PathOrganizer


@dataclass
class TrainingParams:
    """
    Class containing all settable training parameters
    """
    num_epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int

    def log_params(self, path_to: str) -> None:
        """
        :param path_to: path to save params
        """
        FsTools.ensure_dir(path_to)
        with open(path_to, mode="w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """
        :return: dict representation
        """
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size
        }


class Model(abc.ABC):

    def __init__(self, prefix: str | None = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger("Model " + self.name)
        self.path_organizer = PathOrganizer(prefix)

        self.model: torch.nn.Module | None = None

    @abc.abstractmethod
    def initialize_model(self) -> None:
        """
        initializuje model
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        :return: name of the model
        """

    def export_to_onxx(self, tag: str, epoch: int) -> None:
        path_to = self.path_organizer.get_onnx_model_path(self.name, tag, epoch)
        FsTools.ensure_dir(path_to)
        torch.onnx.export(
            self.model,
            torch.randn(1, 3, 32, 32),
            path_to,
            export_params=True,
            input_names=["x"],
        )

    def _train_epoch(self, train_loader, optimizer, criterion) -> float:
        #self.model.train_contrastive_with_mse()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        # Calculate the train loss and accuracy
        train_loss = running_loss / total_samples  # total loss divided by total number of samples

        return train_loss

    def train(self, trainset: OurDataset, params: TrainingParams,
              tag_to_save: str | None = None) -> None:
        """
        Train model
        :param trainset: OurDataset object for training set
        :param params: TrainingParams object
        :param tag_to_save: training tag to save under after each epoch if not None, if None then will not save
        """
        if self.model is None:
            raise ValueError("Model is not initialized properly")

        self.logger.info(f"Starting training model {self.name}")
        self.logger.info(f"Training set len: {len(trainset)}")
        train_loader = trainset.get_dataloader_standard(batch_size=params.batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device that will be used: {device}")
        self.model.to(device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

        for epoch in range(params.num_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            train_metrics_str = f"train loss: {train_loss:.4f}"
            metric_to_save = {
                "epoch": epoch, "train loss": f"{train_loss:.4f}",
            }

            # Print the epoch results
            self.logger.info(f'Epoch [{epoch}/{params.num_epochs}]: {train_metrics_str}')

            if tag_to_save:
                self.export_to_onxx(tag_to_save, epoch)
                self.append_metrics(tag_to_save, metric_to_save)

    # def train_contrastive_with_mse(self, trainset: OurDataset, params: TrainingParams,
    #                            tag_to_save: str | None = None) -> None:
    #     """
    #     Train model using a contrastive learning approach with MSE Loss
    #     :param trainset: OurDataset object for training set
    #     :param params: TrainingParams object
    #     :param tag_to_save: training tag to save under after each epoch if not None, if None then will not save
    #     """
    #     if self.model is None:
    #         raise ValueError("Model is not initialized properly")
    #
    #     self.logger.info(f"Starting contrastive training model with MSE loss {self.name}")
    #     self.logger.info(f"Training set len: {len(trainset)}")
    #     train_loader = trainset.get_dataloader_contrastive(batch_size=params.batch_size)
    #
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     self.logger.info(f"Device that will be used: {device}")
    #     self.model.to(device)
    #
    #     criterion = torch.nn.MSELoss()  # Using MSELoss for contrastive training
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    #
    #     for epoch in range(params.num_epochs):
    #         self.model.train()
    #
    #         running_loss = 0.0
    #         total_samples = 0
    #
    #         for inputs1, inputs2, labels in train_loader:
    #             inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.float().to(device)  # Make sure labels are floats
    #
    #             optimizer.zero_grad()
    #
    #             outputs1 = self.model(inputs1)
    #             outputs2 = self.model(inputs2)
    #
    #             # Calculate the mean squared error between the outputs as a measure of similarity
    #             loss = criterion(outputs1, outputs2)
    #
    #             # In the case of contrastive learning, you may need to adjust how you interpret 'labels' with MSE Loss
    #             # For example, you might need to define what 'similar' (1) and 'dissimilar' (0) mean in terms of MSE.
    #             # Below, I assume 'labels' indicate similarity (1 = similar, 0 = dissimilar), and hence adjust the loss accordingly:
    #             adjusted_loss = torch.mean(labels * loss + (1 - labels) * torch.clamp(1 - loss, min=0.0))
    #
    #             adjusted_loss.backward()
    #             optimizer.step()
    #
    #             running_loss += adjusted_loss.item() * inputs1.size(0)
    #             total_samples += inputs1.size(0)
    #
    #         train_loss = running_loss / total_samples  # total loss divided by total number of samples
    #         train_metrics_str = f"train contrastive MSE loss: {train_loss:.4f}"
    #         metric_to_save = {
    #             "epoch": epoch, "train contrastive MSE loss": f"{train_loss:.4f}",
    #         }
    #
    #         self.logger.info(f'Epoch [{epoch}/{params.num_epochs}]: {train_metrics_str}')
    #
    #         if tag_to_save:
    #             self.export_to_onxx(tag_to_save, epoch)
    #             self.append_metrics(tag_to_save, metric_to_save)

    def append_metrics(self, tag: str, epoch_metrics: dict) -> None:
        """
        Appends metrics to a file after each epoch.
        :param tag: training tag
        :param epoch_metrics: dictionary containing the metrics for the current epoch
        """
        metrics_path = self.path_organizer.get_jsonl_metrics_path(self.name, tag)
        self.logger.info(f"Appending metrics to path {metrics_path}")
        FsTools.ensure_dir(metrics_path)
        with open(metrics_path, 'a') as file:
            file.write(str(epoch_metrics) + '\n')
