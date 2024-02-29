"""Adaptation of the ASTModel for PyTorch Lightning."""

# %% [markdown]
# # Imports

# %%
# Add ast/src/models to path
import argparse
from datetime import datetime
import os
import sys
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelROC,
)

sys.path.append("ast/src/models/")
from ast_models import ASTModel  # noqa: E402 # pylint: disable=wrong-import-position
import config  # noqa: E402 # pylint: disable=wrong-import-position
from datasets import (  # noqa: E402 # pylint: disable=wrong-import-position
    SpectrogramDataset,
)


# %% [markdown]
# # Constants and arguments

# %%
torch.set_float32_matmul_precision("high")

# Initialize argument parser
parser = argparse.ArgumentParser()
# Add arguments
# Parse annotations file
parser.add_argument(
    "--annotations_file",
    type=str,
    default=config.ANNOTATION_FILE,
    help="Path to annotations file",
)
# Parse data directory
parser.add_argument(
    "--data_dir",
    type=str,
    default=config.DATA_DIRECTORY,
    help="Path to data directory",
)
# Parse transformed data directory
parser.add_argument(
    "--transformed_dir",
    type=str,
    default=config.TRANSFORMED_DATA_DIRECTORY,
    help="Path to transformed data directory",
)
# Parse num classes
parser.add_argument(
    "--num_classes",
    type=int,
    default=config.NUM_CLASSES,
    help="Number of classes",
)
# Parse sample seconds
parser.add_argument(
    "--sample_seconds",
    type=float,
    default=config.SAMPLE_SECONDS,
    help="Sample seconds",
)
# Parse sample rate
parser.add_argument(
    "--sample_rate",
    type=int,
    default=config.SAMPLE_RATE,
    help="Sample rate",
)
# Parse prune invalid
parser.add_argument(
    "--prune_invalid",
    type=bool,
    default=True,
    help="Prune invalid",
)
# Parse verbose
parser.add_argument(
    "--verbose",
    type=bool,
    default=True,
    help="Verbose",
)
# Parse batch size
parser.add_argument(
    "--batch_size",
    type=int,
    default=config.BATCH_SIZE,
    help="Batch size",
)
# Parse epochs
parser.add_argument(
    "--max_epochs",
    type=int,
    default=config.MAX_EPOCHS,
    help="Max Epochs",
)
# Parse learning rate
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Learning rate",
)
# Parse weight decay
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-7,
    help="Weight decay",
)
# Parse patience
parser.add_argument(
    "--patience",
    type=int,
    default=config.EARLY_STOPPING_PATIENCE,
    help="Patience",
)
# Parse model directory
parser.add_argument(
    "--model_dir",
    type=str,
    default=config.MODELS_DIRECTORY,
    help="Model directory",
)
# Parse log directory
parser.add_argument(
    "--log_dir",
    type=str,
    default=config.LOG_DIRECTORY,
    help="Log directory",
)
# Parse model name
parser.add_argument(
    "--model_name",
    type=str,
    default="ast",
    help="Model name",
)
# Parse use_pretrained
parser.add_argument(
    "--use_pretrained",
    type=bool,
    default=False,
    help="Use pretrained",
)
# Parse skip_trained
parser.add_argument(
    "--skip_trained",
    type=bool,
    default=False,
    help="Skip trained",
)
# Parse --f argument for Jupyter Notebook (ignored by argparse)
parser.add_argument(
    "--f",
    type=str,
    default="",
    help="",
)
# Parse num_workers (default is number of cpu threads)
parser.add_argument(
    "--num_workers",
    type=int,
    default=os.cpu_count() if os.cpu_count() is not None else 1,
    help="Number of workers",
)
# Parse seed
parser.add_argument(
    "--seed",
    type=int,
    default=config.RANDOM_SEED,
    help="Seed",
)
# Parse device
parser.add_argument(
    "--device",
    type=str,
    default=config.DEVICE,
    help="Device",
)
# Parse pred_threshold
parser.add_argument(
    "--pred_threshold",
    type=float,
    default=config.PRED_THRESHOLD,
    help="Prediction threshold",
)
# Parse num_bands
parser.add_argument(
    "--num_bands",
    type=int,
    default=config.NUM_BANDS,
    help="Number of bands",
)
# Parse arguments
args = parser.parse_args()
# Print all arguments
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
# Set argument constants
ANNOTATIONS_FILE: str = args.annotations_file
TRAIN_ANNOTATIONS_FILE: str = args.annotations_file.replace(".csv", "_train.csv")
TEST_ANNOTATIONS_FILE: str = args.annotations_file.replace(".csv", "_test.csv")
DATA_DIRECTORY: str = args.data_dir
TRANSFORMED_DATA_DIRECTORY: str = args.transformed_dir
NUM_CLASSES: int = args.num_classes
SAMPLE_SECONDS: float = args.sample_seconds
SAMPLE_RATE: int = args.sample_rate
PRUNE_INVALID: bool = args.prune_invalid
VERBOSE: bool = args.verbose
BATCH_SIZE: int = args.batch_size
MAX_EPOCHS: int = args.max_epochs
LEARNING_RATE: float = args.learning_rate
WEIGHT_DECAY: float = args.weight_decay
EARLY_STOPPING_PATIENCE: int = args.patience
MODELS_DIRECTORY: str = args.model_dir
LOG_DIRECTORY: str = args.log_dir
MODEL_NAME: str = args.model_name
USE_PRETRAINED: bool = args.use_pretrained
SKIP_TRAINED: bool = args.skip_trained
NUM_WORKERS: int = args.num_workers
RANDOM_SEED: int = args.seed
DEVICE: str = args.device
PRED_THRESHOLD: float = args.pred_threshold
NUM_BANDS: int = args.num_bands
# Set experiment prefix
EXPERIMENT_PREFIX: str = f"{MODEL_NAME}"

# %%
# Create model directory if not exists
if not os.path.exists(MODELS_DIRECTORY):
    os.makedirs(MODELS_DIRECTORY)
# Create log directory if not exists
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

# %% [markdown]
# # Classes

# %%


class AST(ASTModel, pl.LightningModule):
    """Adaptation of the ASTModel for PyTorch Lightning."""

    def __init__(
        self,
        label_dim: int = 527,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        imagenet_pretrain: bool = True,
        audioset_pretrain: bool = False,
        model_size: str = "base384",
        verbose: bool = True,
    ) -> None:
        """Initialize AST model.

        Args:
            label_dim (int, optional): Number of labels. Defaults to 527.
            fstride (int, optional): Stride of frequency dimension. Defaults to 10.
            tstride (int, optional): Stride of time dimension. Defaults to 10.
            input_fdim (int, optional): Frequency dimension of input. Defaults to 128.
            input_tdim (int, optional): Time dimension of input. Defaults to 1024.
            imagenet_pretrain (bool, optional): Whether to use imagenet pretrained weights.
                Defaults to True.
            audioset_pretrain (bool, optional): Whether to use audioset pretrained weights.
                Defaults to False.
            model_size (str, optional): Model size. Defaults to "base384".
            verbose (bool, optional): Whether to print model information. Defaults to True.
        """
        # init lightning module
        pl.LightningModule.__init__(self)
        # init AST model
        ASTModel.__init__(
            self,
            label_dim,
            fstride,
            tstride,
            input_fdim,
            input_tdim,
            imagenet_pretrain,
            audioset_pretrain,
            model_size,
            verbose,
        )
        self.num_classes = label_dim  # just for compliance with experiments
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
        # Initialize per class metrics
        # Initialize training metrics
        self.train_acc_per_class = MultilabelAccuracy(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.train_auroc_per_class = MultilabelAUROC(
            num_labels=self.num_classes, average="none"
        )
        self.train_f1_per_class = MultilabelF1Score(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.train_hamming_per_class = MultilabelHammingDistance(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.train_precision_per_class = MultilabelPrecision(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.train_recall_per_class = MultilabelRecall(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        # Initialize validation metrics
        self.val_acc_per_class = MultilabelAccuracy(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.val_auroc_per_class = MultilabelAUROC(
            num_labels=self.num_classes, average="none"
        )
        self.val_f1_per_class = MultilabelF1Score(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.val_hamming_per_class = MultilabelHammingDistance(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.val_precision_per_class = MultilabelPrecision(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.val_recall_per_class = MultilabelRecall(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        # Initialize test metrics
        self.test_acc_per_class = MultilabelAccuracy(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.test_auroc_per_class = MultilabelAUROC(
            num_labels=self.num_classes, average="none"
        )
        self.test_f1_per_class = MultilabelF1Score(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.test_hamming_per_class = MultilabelHammingDistance(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.test_precision_per_class = MultilabelPrecision(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )
        self.test_recall_per_class = MultilabelRecall(
            num_labels=self.num_classes, threshold=0.5, average="none"
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (torch.Tensor): Batch of data.
            batch_idx (int): Index of batch.

        Returns:
            torch.Tensor: Loss tensor.
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
        # Calculate per class metrics
        self.train_acc_per_class(F.sigmoid(outputs), targets.int())
        self.train_auroc_per_class(F.sigmoid(outputs), targets.int())
        self.train_f1_per_class(F.sigmoid(outputs), targets.int())
        self.train_hamming_per_class(F.sigmoid(outputs), targets.int())
        self.train_precision_per_class(F.sigmoid(outputs), targets.int())
        self.train_recall_per_class(F.sigmoid(outputs), targets.int())
        # Log metrics
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_hamming", self.train_hamming, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True)
        # self.log("train_roc", self.train_roc, on_step=False, on_epoch=True)
        # log learning rate
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        # Per class metrics need to be logged separately
        # Log per class metrics
        for i in range(self.num_classes):
            self.log(
                f"train_acc_{i}",
                self.train_acc_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="train_acc_per_class",
            )
            self.log(
                f"train_auroc_{i}",
                self.train_auroc_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="train_auroc_per_class",
            )
            self.log(
                f"train_f1_{i}",
                self.train_f1_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="train_f1_per_class",
            )
            self.log(
                f"train_hamming_{i}",
                self.train_hamming_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="train_hamming_per_class",
            )
            self.log(
                f"train_precision_{i}",
                self.train_precision_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="train_precision_per_class",
            )
            self.log(
                f"train_recall_{i}",
                self.train_recall_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="train_recall_per_class",
            )
        # Return loss value
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validate step. Returns loss.

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
        # calculate per class metrics
        self.val_acc_per_class(F.sigmoid(outputs), targets.int())
        self.val_auroc_per_class(F.sigmoid(outputs), targets.int())
        self.val_f1_per_class(F.sigmoid(outputs), targets.int())
        self.val_hamming_per_class(F.sigmoid(outputs), targets.int())
        self.val_precision_per_class(F.sigmoid(outputs), targets.int())
        self.val_recall_per_class(F.sigmoid(outputs), targets.int())
        # log metrics
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val_hamming", self.val_hamming, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        # self.log("val_roc", self.val_roc, on_step=False, on_epoch=True)
        # Per class metrics need to be logged separately
        # Log per class metrics
        for i in range(self.num_classes):
            self.log(
                f"val_acc_{i}",
                self.val_acc_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="val_acc_per_class",
            )
            self.log(
                f"val_auroc_{i}",
                self.val_auroc_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="val_auroc_per_class",
            )
            self.log(
                f"val_f1_{i}",
                self.val_f1_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="val_f1_per_class",
            )
            self.log(
                f"val_hamming_{i}",
                self.val_hamming_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="val_hamming_per_class",
            )
            self.log(
                f"val_precision_{i}",
                self.val_precision_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="val_precision_per_class",
            )
            self.log(
                f"val_recall_{i}",
                self.val_recall_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="val_recall_per_class",
            )
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
        # calculate per class metrics
        self.test_acc_per_class(F.sigmoid(outputs), targets.int())
        self.test_auroc_per_class(F.sigmoid(outputs), targets.int())
        self.test_f1_per_class(F.sigmoid(outputs), targets.int())
        self.test_hamming_per_class(F.sigmoid(outputs), targets.int())
        self.test_precision_per_class(F.sigmoid(outputs), targets.int())
        self.test_recall_per_class(F.sigmoid(outputs), targets.int())
        # log metrics
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_hamming", self.test_hamming, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        # self.log("test_roc", self.test_roc)
        # Per class metrics need to be logged separately
        # Log per class metrics
        for i in range(self.num_classes):
            self.log(
                f"test_acc_{i}",
                self.test_acc_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="test_acc_per_class",
            )
            self.log(
                f"test_auroc_{i}",
                self.test_auroc_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="test_auroc_per_class",
            )
            self.log(
                f"test_f1_{i}",
                self.test_f1_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="test_f1_per_class",
            )
            self.log(
                f"test_hamming_{i}",
                self.test_hamming_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="test_hamming_per_class",
            )
            self.log(
                f"test_precision_{i}",
                self.test_precision_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="test_precision_per_class",
            )
            self.log(
                f"test_recall_{i}",
                self.test_recall_per_class[i],
                on_step=False,
                on_epoch=True,
                metric_attribute="test_recall_per_class",
            )
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
        outputs = F.sigmoid(self(inputs)).round().tolist()
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
            patience=EARLY_STOPPING_PATIENCE // 2,
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
# # Main

# %%
if __name__ == "__main__":
    # Load annotations
    annotation: pd.DataFrame = pd.read_csv(ANNOTATIONS_FILE)
    # Split data into train and test
    train_annotation, test_annotation = train_test_split(
        annotation, test_size=0.2, random_state=RANDOM_SEED
    )
    # Sort by Timestamp
    train_annotation = train_annotation.sort_values(by=["Timestamp"])
    test_annotation = test_annotation.sort_values(by=["Timestamp"])
    # Save to csv
    train_annotation.to_csv(TRAIN_ANNOTATIONS_FILE, index=False)
    test_annotation.to_csv(TEST_ANNOTATIONS_FILE, index=False)
    # Load train and test dataset
    dataset_train_spec: SpectrogramDataset = SpectrogramDataset(
        annotations_file=TRAIN_ANNOTATIONS_FILE,
        data_dir=DATA_DIRECTORY,
        num_classes=NUM_CLASSES,
        sample_seconds=SAMPLE_SECONDS,
        sample_rate=SAMPLE_RATE,
        prune_invalid=PRUNE_INVALID,  # Set to True to remove invalid files
        num_mel_filters=NUM_BANDS,
        transform=lambda x: x.transpose(0, 1),
    )
    dataset_test_spec: SpectrogramDataset = SpectrogramDataset(
        annotations_file=TEST_ANNOTATIONS_FILE,
        data_dir=DATA_DIRECTORY,
        num_classes=NUM_CLASSES,
        sample_seconds=SAMPLE_SECONDS,
        sample_rate=SAMPLE_RATE,
        prune_invalid=PRUNE_INVALID,  # Set to True to remove invalid files
        num_mel_filters=NUM_BANDS,
        transform=lambda x: x.transpose(0, 1),
    )
    print(f"[{datetime.now()}]: Loaded swine datasets")
    # Calculate mean and std for normalization
    # Iterate over train dataset
    train_mean_spec: torch.Tensor = torch.zeros(1)
    train_std_spec: torch.Tensor = torch.zeros(1)
    for spectrogram, _ in dataset_train_spec:  # type: ignore
        train_mean_spec += spectrogram.mean()
        train_std_spec += spectrogram.std()
    train_mean_spec /= len(dataset_train_spec)
    train_std_spec /= len(dataset_train_spec)
    print(f"Train mean (spec): {train_mean_spec}")
    print(f"Train std (spec): {train_std_spec}")
    # Normalize datasets (update transform)
    dataset_train_spec.transform = lambda x: (
        (x - train_mean_spec) / train_std_spec
    ).transpose(0, 1)
    dataset_test_spec.transform = lambda x: (
        (x - train_mean_spec) / train_std_spec
    ).transpose(0, 1)
    train_dataloader_spec = DataLoader(
        dataset_train_spec,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_dataloader_spec = DataLoader(
        dataset_test_spec,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"[{datetime.now()}]: Created data loaders")
    # Create model
    # Set dataloaders (for generic use)
    train_dataloaders: list[DataLoader] = [train_dataloader_spec]
    test_dataloaders: list[DataLoader] = [test_dataloader_spec]
    for train_dataloader, test_dataloader in zip(train_dataloaders, test_dataloaders):
        # Get number of input channels from train dataloader
        input_shape: tuple = next(iter(train_dataloader))[0].shape
        print(f"Input shape: {input_shape}")
        # get dataset type (wave/spectrogram/mfcc)
        dataset_type: str = train_dataloader.dataset.__class__.__name__
        # transform type to string in ["wave", "spec", "mfcc"]
        dataset_type = {
            "WaveformDataset": "wave",
            "SpectrogramDataset": "spec",
            "MFCCDataset": "mfcc",
        }[dataset_type]
        experiment_name: str = (
            f"{EXPERIMENT_PREFIX}_"
            f"{dataset_type}_{NUM_CLASSES}-classes_{NUM_BANDS}-bands"
        )
        print(f"[{datetime.now()}]: Training {experiment_name}")
        model: AST = AST(
            label_dim=NUM_CLASSES,
            fstride=10,
            tstride=10,
            input_fdim=NUM_BANDS,
            input_tdim=96,
            imagenet_pretrain=True,
            audioset_pretrain=True,
            model_size="base384",
            verbose=VERBOSE,
        )
        if USE_PRETRAINED or SKIP_TRAINED:
            # Search for checkpoint file in models directory
            # file starts with experiment name
            # file ends with .ckpt
            checkpoint_file: str | None = None
            for file in os.listdir(MODELS_DIRECTORY):
                if file.startswith(experiment_name) and file.endswith(".ckpt"):
                    # Checkpoint file found
                    # get file with highest val_auroc
                    if checkpoint_file is None:
                        checkpoint_file = file
                    elif file > checkpoint_file:
                        checkpoint_file = file
            if checkpoint_file is not None:
                print(f"[{datetime.now()}]: Found checkpoint {checkpoint_file}")
                if SKIP_TRAINED:
                    print(f"[{datetime.now()}]: Skipping {experiment_name}")
                    continue
                # Load checkpoint
                checkpoint_file = os.path.join(MODELS_DIRECTORY, checkpoint_file)
                model = AST.load_from_checkpoint(  # type: ignore
                    checkpoint_path=checkpoint_file,
                    label_dim=NUM_CLASSES,
                    fstride=10,
                    tstride=10,
                    input_fdim=NUM_BANDS,
                    input_tdim=96,
                    imagenet_pretrain=True,
                    audioset_pretrain=True,
                    model_size="base384",
                    verbose=VERBOSE,
                )
        # Train model
        early_stopping: EarlyStopping = EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min"
        )
        loggers: list = [
            CSVLogger(save_dir=LOG_DIRECTORY, name=experiment_name),
            TensorBoardLogger(save_dir=LOG_DIRECTORY, name=experiment_name),
        ]
        checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
            dirpath=MODELS_DIRECTORY,
            filename=experiment_name + "-{val_auroc:.2f}-{val_loss:.2f}-{epoch:02d}",
            monitor="val_auroc",
            verbose=True,
            save_top_k=1,
            save_weights_only=False,
            mode="max",
            auto_insert_metric_name=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
        trainer: pl.Trainer = pl.Trainer(
            callbacks=[early_stopping, checkpoint_callback],
            max_epochs=1000,
            logger=loggers,
            log_every_n_steps=min(50, len(train_dataloader)),
        )
        trainer.fit(model, train_dataloader, test_dataloader)
        print(f"[{datetime.now()}]: Finished training {experiment_name}")
    print(f"[{datetime.now()}]: Finished training")

# %%
