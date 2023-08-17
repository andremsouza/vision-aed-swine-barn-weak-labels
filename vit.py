"""Adaptation of the ViT model for audio data."""
# %% [mardkown]
# # Imports

# %%
from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
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


# %% [markdown]
# # Constants

# %%

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
        loss = F.mse_loss(outputs, targets)
        # log loss
        self.log("train_loss", loss)
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
        loss = F.mse_loss(outputs, targets)
        # log loss
        self.log("val_loss", loss)
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
        loss = F.mse_loss(outputs, targets)
        # log loss
        self.log("test_loss", loss)
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


# %%
