"""Adaptation of the ViT model for audio data."""
# %% [mardkown]
# # Imports

# %%
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
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_b_32,
    ViT_B_32_Weights,
    vit_l_16,
    ViT_L_16_Weights,
    vit_l_32,
    ViT_L_32_Weights,
    vit_h_14,
    ViT_H_14_Weights,
)

import config
from data import AudioDataset


# %% [markdown]
# # Constants

# %%
torch.set_float32_matmul_precision("medium")
RANDOM_SEED: int = 42

# %% [markdown]
# # Classes

# %%


class ViT(pl.LightningModule):
    """ViT model for keypoint detection."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        weights: ViT_B_16_Weights
        | ViT_B_32_Weights
        | ViT_L_16_Weights
        | ViT_L_32_Weights
        | ViT_H_14_Weights = None,
    ) -> None:
        """Initialize model.

        Args:
            num_keypoints (int): Number of keypoints to detect.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            weights (
                ViT_B_16_Weights |
                ViT_B_32_Weights |
                ViT_L_16_Weights |
                ViT_L_32_Weights |
                ViT_H_14_Weights, optional
            ) : Weights to load. Defaults to None.

        Raises:
            ValueError: If weights is not None and not one of the supported weight types.
        """
        super().__init__()
        self.num_classes = num_classes
        # Load pretrained model based on weights
        # If weights is None, use default weights with vit_b_16
        if weights is None:
            self.model = vit_b_16(dropout=dropout, num_classes=num_classes)
        elif isinstance(weights, ViT_B_16_Weights) or weights is None:
            # If weights is None, use default weights with vit_b_16
            self.model = vit_b_16(weights=weights, dropout=dropout)
            self.transforms = (
                weights.transforms(antialias=True)
                if weights is not None
                else ViT_B_16_Weights.DEFAULT.transforms(antialias=True)
            )
        elif isinstance(weights, ViT_B_32_Weights):
            self.model = vit_b_32(weights=weights, dropout=dropout)
            self.transforms = weights.transforms(antialias=True)

        elif isinstance(weights, ViT_L_16_Weights):
            self.model = vit_l_16(weights=weights, dropout=dropout)
            self.transforms = weights.transforms(antialias=True)
        elif isinstance(weights, ViT_L_32_Weights):
            self.model = vit_l_32(weights=weights, dropout=dropout)
            self.transforms = weights.transforms(antialias=True)
        elif isinstance(weights, ViT_H_14_Weights):
            self.model = vit_h_14(weights=weights, dropout=dropout)
            self.transforms = weights.transforms(antialias=True)
        else:
            raise ValueError(f"Unknown weights: {weights}")
        # Update model heads to match number of classes
        self._update_heads()
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

    def _update_heads(self) -> None:
        # get number of input features for the classifier
        in_features = self.model.heads[-1].in_features
        # replace the pre-trained head with a new one
        self.model.heads[-1] = nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Predicted keypoints.
        """
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step. Returns loss.

        Args:
            batch (torch.Tensor): Batch of images.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        images, targets = batch
        outputs = self(images)
        # calculate loss
        loss = F.binary_cross_entropy(F.sigmoid(outputs), targets)
        # log loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # calculate metrics
        self.train_acc(F.sigmoid(outputs), targets)
        self.train_auroc(F.sigmoid(outputs), targets)
        self.train_f1(F.sigmoid(outputs), targets)
        self.train_hamming(F.sigmoid(outputs), targets)
        self.train_precision(F.sigmoid(outputs), targets)
        self.train_recall(F.sigmoid(outputs), targets)
        self.train_roc(F.sigmoid(outputs), targets)
        # log metrics
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_hamming", self.train_hamming, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train_roc", self.train_roc, on_step=False, on_epoch=True)
        # return loss
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
        images, targets = batch
        outputs = self(images)
        # calculate loss
        loss = F.binary_cross_entropy(F.sigmoid(outputs), targets)
        # log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # calculate metrics
        self.val_acc(F.sigmoid(outputs), targets)
        self.val_auroc(F.sigmoid(outputs), targets)
        self.val_f1(F.sigmoid(outputs), targets)
        self.val_hamming(F.sigmoid(outputs), targets)
        self.val_precision(F.sigmoid(outputs), targets)
        self.val_recall(F.sigmoid(outputs), targets)
        self.val_roc(F.sigmoid(outputs), targets)
        # log metrics
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val_hamming", self.val_hamming, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_roc", self.val_roc, on_step=False, on_epoch=True)
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
        images, targets = batch
        outputs = self(images)
        # calculate loss
        loss = F.binary_cross_entropy(F.sigmoid(outputs), targets)
        # log loss
        self.log("test_loss", loss)
        # calculate metrics
        self.test_acc(F.sigmoid(outputs), targets)
        self.test_auroc(F.sigmoid(outputs), targets)
        self.test_f1(F.sigmoid(outputs), targets)
        self.test_hamming(F.sigmoid(outputs), targets)
        self.test_precision(F.sigmoid(outputs), targets)
        self.test_recall(F.sigmoid(outputs), targets)
        self.test_roc(F.sigmoid(outputs), targets)
        # log metrics
        self.log("test_acc", self.test_acc)
        self.log("test_auroc", self.test_auroc)
        self.log("test_f1", self.test_f1)
        self.log("test_hamming", self.test_hamming)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)
        self.log("test_roc", self.test_roc)
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
        images, targets = batch
        outputs = self(images)
        return outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999))


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
    model = ViT(
        num_classes=config.NUM_CLASSES,
        dropout=0.0,
    )
    # Create datasets
    train_dataset = AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    )
    val_dataset = AudioDataset(
        config.VAL_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    )
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=24
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=24
    )
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    loggers = [
        pl.loggers.CSVLogger(config.LOG_DIRECTORY, name="vit_b_16_test"),
        pl.loggers.TensorBoardLogger(config.LOG_DIRECTORY, name="vit_b_16_test"),
    ]
    trainer = pl.Trainer(
        callbacks=[early_stopping],
        max_epochs=1000,
        logger=loggers,
        log_every_n_steps=min(50, len(train_dataloader)),
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Finished training.")

# %%
