"""Fully connected neural network with different number of layers and units.

This script is used to define a fully connected neural network with different
number of layers and units.
"""
# %% [markdown]
# # Imports

# %%
import os
from typing import Any

import lightning.pytorch as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelROC,
)

import config
from data import AudioDataset

# %% [markdown]
# # Constants

# %%
torch.set_float32_matmul_precision("high")
RANDOM_SEED: int = 42

N_LAYERS: list[int] = [2, 3, 4, 5, 6]
M_UNITS: list[int] = [500, 1000, 2000, 3000, 4000]

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-2
EXPERIMENT_NAME: str = "fc"

# %% [markdown]
# # Classes

# %%


class FullyConnected(pl.LightningModule):
    """Fully connected neural network with audio frames as input and labels as output.

    Attributes:
        num_layers (int): number of layers
        num_units (int): number of units in each layer
        num_layers (nn.Sequential): layers of the neural network

    References:
        - https://doi.org/10.1109/ICASSP.2017.7952132
    """

    def __init__(
        self,
        num_layers: int,
        num_units: int,
        num_features: int,
        num_classes: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        """Initialize the fully connected neural network.

        Args:
            n_layers (int): number of layers
            m_units (int): number of units in each layer
        """
        super().__init__()
        self.num_layers: int = num_layers
        self.num_units: int = num_units
        self.num_features: int = num_features
        self.num_classes: int = num_classes
        self.activation = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }[activation]
        self.dropout = dropout

        self.layers: nn.Sequential = self._create_layers()
        # Initialize training metrics
        self.train_acc = MultilabelAccuracy(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.train_auroc = MultilabelAUROC(
            num_labels=self.num_classes, average="weighted"
        )
        self.train_f1 = MultilabelF1Score(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.train_hamming = MultilabelHammingDistance(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.train_precision = MultilabelPrecision(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.train_recall = MultilabelRecall(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.train_roc = MultilabelROC(num_labels=self.num_classes)
        # Initialize validation metrics
        self.val_acc = MultilabelAccuracy(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.val_auroc = MultilabelAUROC(
            num_labels=self.num_classes, average="weighted"
        )
        self.val_f1 = MultilabelF1Score(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.val_hamming = MultilabelHammingDistance(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.val_precision = MultilabelPrecision(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.val_recall = MultilabelRecall(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.val_roc = MultilabelROC(num_labels=self.num_classes)
        # Initialize test metrics
        self.test_acc = MultilabelAccuracy(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.test_auroc = MultilabelAUROC(
            num_labels=self.num_classes, average="weighted"
        )
        self.test_f1 = MultilabelF1Score(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.test_hamming = MultilabelHammingDistance(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.test_precision = MultilabelPrecision(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.test_recall = MultilabelRecall(
            num_labels=self.num_classes, threshold=0.5, average="weighted"
        )
        self.test_roc = MultilabelROC(num_labels=self.num_classes)

    def _create_layers(self) -> nn.Sequential:
        """Create the layers of the neural network.

        Returns:
            nn.Sequential: layers of the neural network
        """
        layers: list = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(
                    nn.Linear(
                        in_features=self.num_features, out_features=self.num_units
                    )
                )
            else:
                layers.append(
                    nn.Linear(in_features=self.num_units, out_features=self.num_units)
                )
            layers.append(self.activation())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(
            nn.Linear(in_features=self.num_units, out_features=self.num_classes)
        )
        # layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.layers(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step. Returns the loss value.

        Args:
            batch (torch.Tensor): batch of inputs
            batch_idx (int): index of the batch

        Returns:
            torch.Tensor: loss value
        """
        inputs, targets = batch
        outputs = self(inputs)
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # Log loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # Calculate metrics
        self.train_acc(F.sigmoid(outputs), targets.int())
        self.train_auroc(F.sigmoid(outputs), targets.int())
        self.train_f1(F.sigmoid(outputs), targets.int())
        self.train_hamming(F.sigmoid(outputs), targets.int())
        self.train_precision(F.sigmoid(outputs), targets.int())
        self.train_recall(F.sigmoid(outputs), targets.int())
        # self.train_roc(F.sigmoid(outputs), targets.int())
        # Log metrics
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_hamming", self.train_hamming, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        # self.log("train_roc", self.train_roc, on_step=False, on_epoch=True)
        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        # Return loss value
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validation step. Returns loss.

        Args:
            batch (torch.Tensor): Batch of images.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        inputs, targets = batch
        outputs = self(inputs)
        # calculate loss
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # calculate metrics
        self.val_acc(F.sigmoid(outputs), targets.int())
        self.val_auroc(F.sigmoid(outputs), targets.int())
        self.val_f1(F.sigmoid(outputs), targets.int())
        self.val_hamming(F.sigmoid(outputs), targets.int())
        self.val_precision(F.sigmoid(outputs), targets.int())
        self.val_recall(F.sigmoid(outputs), targets.int())
        # self.val_roc(F.sigmoid(outputs), targets.int())
        # log metrics
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val_hamming", self.val_hamming, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        # self.log("val_roc", self.val_roc, on_step=False, on_epoch=True)
        # return loss
        return loss

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Test step. Returns loss.

        Args:
            batch (torch.Tensor): Batch of images.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        inputs, targets = batch
        outputs = self(inputs)
        # calculate loss
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # log loss
        self.log("test_loss", loss)
        # calculate metrics
        self.test_acc(F.sigmoid(outputs), targets.int())
        self.test_auroc(F.sigmoid(outputs), targets.int())
        self.test_f1(F.sigmoid(outputs), targets.int())
        self.test_hamming(F.sigmoid(outputs), targets.int())
        self.test_precision(F.sigmoid(outputs), targets.int())
        self.test_recall(F.sigmoid(outputs), targets.int())
        # self.test_roc(F.sigmoid(outputs), targets.int())
        # log metrics
        self.log("test_acc", self.test_acc)
        self.log("test_auroc", self.test_auroc)
        self.log("test_f1", self.test_f1)
        self.log("test_hamming", self.test_hamming)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)
        # self.log("test_roc", self.test_roc)
        # return loss
        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """Predict step. Returns predictions.

        Args:
            batch (torch.Tensor): Batch of images.
            batch_idx (int): Batch index.

        Returns:
            Any: Predictions.
        """
        inputs, targets = batch
        outputs = self(inputs)
        return outputs

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=config.EARLY_STOPPING_PATIENCE // 2,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}
        ]


# %% [markdown]
# # Training and Testing

# %%
if __name__ == "__main__":
    for n_layers in N_LAYERS:
        for m_units in M_UNITS:
            experiment_name = f"{EXPERIMENT_NAME}_{n_layers}_{m_units}"
            print("Starting training.")
            # Split annotations into train and val sets
            annotation = pd.read_csv(config.ANNOTATION_FILE)
            train_annotation, val_annotation = train_test_split(
                annotation, test_size=0.2, random_state=RANDOM_SEED
            )
            # Sort by Timestamp
            train_annotation = train_annotation.sort_values(by="Timestamp")
            val_annotation = val_annotation.sort_values(by="Timestamp")
            # Save annotation files to csv
            train_annotation.to_csv(config.TRAIN_ANNOTATION_FILE, index=False)
            val_annotation.to_csv(config.VAL_ANNOTATION_FILE, index=False)
            # Create model
            # Search for checkpoint file in models directory
            # file starts with experiment name
            # file ends with .ckpt
            checkpoint_file: str | None = None
            for file in os.listdir(config.MODELS_DIRECTORY):
                if file.startswith(experiment_name) and file.endswith(".ckpt"):
                    # Checkpoint file found
                    # get file with highest val_auroc
                    if checkpoint_file is None:
                        checkpoint_file = file
                    elif file > checkpoint_file:
                        checkpoint_file = file
            if checkpoint_file is not None:
                # prepend models directory
                checkpoint_file = os.path.join(config.MODELS_DIRECTORY, checkpoint_file)
                model = FullyConnected.load_from_checkpoint(
                    checkpoint_file,
                    num_layers=n_layers,
                    num_units=m_units,
                    num_features=96 * 64,
                    num_classes=config.NUM_CLASSES,
                    dropout=0.5,
                )
            else:
                model = FullyConnected(
                    num_layers=n_layers,
                    num_units=m_units,
                    num_features=96 * 64,
                    num_classes=config.NUM_CLASSES,
                    dropout=0.5,
                )
            # Create datasets
            train_dataset = AudioDataset(
                config.TRAIN_ANNOTATION_FILE,
                config.DATA_DIRECTORY,
                transform=lambda x: torch.flatten(x[None, :, :64]),
                target_transform=lambda x: x[0, :].float(),
            )
            val_dataset = AudioDataset(
                config.VAL_ANNOTATION_FILE,
                config.DATA_DIRECTORY,
                transform=lambda x: torch.flatten(x[None, :, :64]),
                target_transform=lambda x: x[0, :].float(),
            )
            # Create dataloaders
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
            )
            early_stopping = pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, mode="min"
            )
            loggers = [
                pl.loggers.CSVLogger(config.LOG_DIRECTORY, name=experiment_name),
                pl.loggers.TensorBoardLogger(
                    config.LOG_DIRECTORY, name=experiment_name
                ),
            ]
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=config.MODELS_DIRECTORY,
                filename=experiment_name
                + "-{val_auroc:.2f}-{val_loss:.2f}-{epoch:02d}",
                monitor="val_auroc",
                verbose=True,
                save_top_k=1,
                save_weights_only=False,
                mode="max",
                auto_insert_metric_name=True,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            )
            trainer = pl.Trainer(
                callbacks=[early_stopping, checkpoint_callback],
                max_epochs=1000,
                logger=loggers,
                log_every_n_steps=min(50, len(train_dataloader)),
            )
            trainer.fit(model, train_dataloader, val_dataloader)
            print("Finished training.")

# %%
