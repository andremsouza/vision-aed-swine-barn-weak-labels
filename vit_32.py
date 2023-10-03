"""Adaptation of the ViT model for audio data."""
# %% [mardkown]
# # Imports

# %%
import math
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
from torchvision.models import (
    vit_b_32,
    ViT_B_32_Weights,
    vit_l_32,
    ViT_L_32_Weights,
)

import config
from data import AudioDataset


# %% [markdown]
# # Constants

# %%
torch.set_float32_matmul_precision("high")
RANDOM_SEED: int = 42

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 3e-2
EXPERIMENT_NAME: str = "vit_b_32"

# %% [markdown]
# # Classes

# %%


class ViT32(pl.LightningModule):
    """ViT model for keypoint detection."""

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        weights: ViT_B_32_Weights | ViT_L_32_Weights = None,
    ) -> None:
        """Initialize model.

        Args:
            num_keypoints (int): Number of keypoints to detect.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            weights (ViT_B_32_Weights | ViT_L_32_Weights, optional): Weights to use for
                the model. Defaults to None.

        Raises:
            ValueError: If weights is not None and not one of the supported weight types
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_size = (96, 64)
        # Load pretrained model based on weights
        if isinstance(weights, ViT_B_32_Weights) or weights is None:
            # If weights is None, use default weights with vit_b_32
            self.model = vit_b_32(weights=weights, dropout=dropout)
            self.transforms = (
                weights.transforms(antialias=True)
                if weights is not None
                else ViT_B_32_Weights.DEFAULT.transforms(antialias=True)
            )
        elif isinstance(weights, ViT_L_32_Weights):
            self.model = vit_l_32(weights=weights, dropout=dropout)
            self.transforms = weights.transforms(antialias=True)
        else:
            raise ValueError(f"Unknown weights: {weights}")
        # Update conv_proj to match input dimensions
        self._update_conv_proj()
        # Update pos_embedding to match input dimensions
        self._update_pos_embedding()
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

    def _update_conv_proj(self) -> None:
        """Update conv_proj to match input dimensions."""
        self.model.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.model.hidden_dim,
            kernel_size=self.model.patch_size,
            stride=self.model.patch_size,
        )
        # Init the patchify stem conv with the same init as the original ViT
        fan_in: int = (
            self.model.conv_proj.in_channels
            * self.model.conv_proj.kernel_size[0]
            * self.model.conv_proj.kernel_size[1]
        )
        nn.init.trunc_normal_(self.model.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.model.conv_proj.bias is not None:
            nn.init.zeros_(self.model.conv_proj.bias)

    def _update_pos_embedding(self) -> None:
        """Update pos_embedding to match input dimensions."""
        # Set sequence length to match number of input patches to the encoder
        seq_length = (
            self.image_size[0]
            // self.model.patch_size
            * (self.image_size[1] // self.model.patch_size)
        )
        # Add class token
        seq_length += 1
        # Update pos_embedding
        self.model.encoder.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, self.model.hidden_dim).normal_(std=0.02)
        )
        # Update model sequence length
        self.model.seq_length = seq_length
        self.seq_length = seq_length

    def _update_heads(self) -> None:
        """Update heads to match number of classes."""
        # get number of input features for the classifier
        in_features = self.model.heads[-1].in_features
        # replace the pre-trained head with a new one
        self.model.heads.head = nn.Linear(in_features, self.num_classes)
        # Init the new classifier head with the same init as the original ViT
        nn.init.zeros_(self.model.heads.head.weight)
        nn.init.zeros_(self.model.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape  # pylint: disable=invalid-name
        p = self.model.patch_size  # pylint: disable=invalid-name
        torch._assert(
            h == self.image_size[0],
            f"Wrong image height! Expected {self.image_size[0]} but got {h}!",
        )
        torch._assert(
            w == self.image_size[1],
            f"Wrong image width! Expected {self.image_size[1]} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.model.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, n_h * n_w, hidden_dim)
        x = x.reshape(n, self.model.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Predicted keypoints.
        """
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.model.heads(x)

        return x

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
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        # log loss
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # calculate metrics
        self.train_acc(F.sigmoid(outputs), targets.int())
        self.train_auroc(F.sigmoid(outputs), targets.int())
        self.train_f1(F.sigmoid(outputs), targets.int())
        self.train_hamming(F.sigmoid(outputs), targets.int())
        self.train_precision(F.sigmoid(outputs), targets.int())
        self.train_recall(F.sigmoid(outputs), targets.int())
        # self.train_roc(F.sigmoid(outputs), targets.int())
        # log metrics
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_hamming", self.train_hamming, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        # self.log("train_roc", self.train_roc, on_step=False, on_epoch=True)
        # log learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Run Validation step. Returns loss.

        Args:
            batch (torch.Tensor): Batch of images.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        images, targets = batch
        outputs = self(images)
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
        images, targets = batch
        outputs = self(images)
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
        images, targets = batch
        outputs = self(images)
        return outputs

    def configure_optimizers(self) -> Any:
        """Configure optimizers and schedulers.

        Returns:
            Any: Optimizers and schedulers.
        """
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
        model = ViT32.load_from_checkpoint(
            checkpoint_file, num_classes=config.NUM_CLASSES, dropout=0.5
        )
    else:
        model = ViT32(
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
