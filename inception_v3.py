"""Inception v3 model adapted from PyTorch.

This file contains a partial implementation of the Inception v3 model
from https://arxiv.org/abs/1512.00567. It is adapted from PyTorch
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
import torch.nn as nn
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
from torchvision.models.inception import (
    BasicConv2d,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)


import config
from data import AudioDataset

# %% [markdown]
# # Constants

# %%
torch.set_float32_matmul_precision("high")
RANDOM_SEED: int = 42

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-2
EXPERIMENT_NAME: str = "inception_v3"

# %% [markdown]
# # Classes

# %%


class InceptionV3(pl.LightningModule):
    """Inception v3 model adapted from PyTorch.

    This file contains a partial implementation of the Inception v3 model
    from https://arxiv.org/abs/1512.00567. It is adapted from PyTorch
    """

    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.0,
        transform_input: bool = False,
    ) -> None:
        """Initialize the Inception v3 model.

        Args:
            num_classes (int): number of classes
            dropout (float): dropout rate
            transform_input (bool): whether to transform the input
        """
        super().__init__()
        self.transform_input = transform_input
        # self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        # self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        # self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_3b_1x1 = BasicConv2d(1, 80, kernel_size=1)
        self.conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mixed_5b = InceptionA(192, pool_features=32)
        self.mixed_5c = InceptionA(256, pool_features=64)
        self.mixed_5d = InceptionA(288, pool_features=64)
        self.mixed_6a = InceptionB(288)
        self.mixed_6b = InceptionC(768, channels_7x7=128)
        self.mixed_6c = InceptionC(768, channels_7x7=160)
        self.mixed_6d = InceptionC(768, channels_7x7=160)
        self.mixed_6e = InceptionC(768, channels_7x7=192)
        self.mixed_7a = InceptionD(768)
        self.mixed_7b = InceptionE(1280)
        self.mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((10, 6))  # Changed in Hershey2017
        self.dropout = nn.Dropout(p=dropout)
        # Changed number of features from 2048 to 122880
        self.fc = nn.Linear(122880, num_classes)  # pylint: disable=invalid-name
        # self.sigmoid = nn.Sigmoid()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Inception v3 model.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # N x 1 x 96 x 64
        x = self.conv2d_3b_1x1(x)
        # N x 80 x 96 x 64
        x = self.conv2d_4a_3x3(x)
        # N x 192 x 94 x 62
        x = self.maxpool2(x)
        # N x 192 x 46 x 30
        x = self.mixed_5b(x)
        # N x 256 x 46 x 30
        x = self.mixed_5c(x)
        # N x 288 x 46 x 30
        x = self.mixed_5d(x)
        # N x 288 x 46 x 30
        x = self.mixed_6a(x)
        # N x 768 x 23 x 15
        x = self.mixed_6b(x)
        # N x 768 x 23 x 15
        x = self.mixed_6c(x)
        # N x 768 x 23 x 15
        x = self.mixed_6d(x)
        # N x 768 x 23 x 15
        x = self.mixed_6e(x)
        # N x 768 x 23 x 15
        x = self.mixed_7a(x)
        # N x 1280 x 11 x 7
        x = self.mixed_7b(x)
        # N x 2048 x 11 x 7
        x = self.mixed_7c(x)
        # N x 2048 x 11 x 7
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 10 x 6
        x = self.dropout(x)
        # N x 2048 x 10 x 6
        x = torch.flatten(x, 1)
        # N x 122880
        x = self.fc(x)
        # N x 1000 (num_classes)
        # x = self.sigmoid(x)
        return x

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the input.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: transformed input tensor
        """
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

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
        if file.startswith(EXPERIMENT_NAME) and file.endswith(".ckpt"):
            # Checkpoint file found
            # get file with highest val_auroc
            if checkpoint_file is None:
                checkpoint_file = file
            elif file > checkpoint_file:
                checkpoint_file = file
    if checkpoint_file is not None:
        # prepend models directory
        checkpoint_file = os.path.join(config.MODELS_DIRECTORY, checkpoint_file)
        model = InceptionV3.load_from_checkpoint(
            checkpoint_file, num_classes=config.NUM_CLASSES, dropout=0.5
        )
    else:
        model = InceptionV3(
            num_classes=config.NUM_CLASSES,
            dropout=0.5,
        )
    # Create datasets
    train_dataset = AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :].float(),
    )
    val_dataset = AudioDataset(
        config.VAL_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
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
        pl.loggers.CSVLogger(config.LOG_DIRECTORY, name=EXPERIMENT_NAME),
        pl.loggers.TensorBoardLogger(config.LOG_DIRECTORY, name=EXPERIMENT_NAME),
    ]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.MODELS_DIRECTORY,
        filename=EXPERIMENT_NAME + "-{val_auroc:.2f}-{val_loss:.2f}-{epoch:02d}",
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
