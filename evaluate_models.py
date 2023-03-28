"""Evaluate trained models."""

# %% [markdown]
# # Imports

# %%
import pickle as pkl
import sys
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

import data
import models
import config
import utils

# %% [markdown]
# # Data

# %%
# Load data
device = torch.device(config.DEVICE)

# Get length of dataset
dataset_length = len(
    data.AudioDataset(
        config.ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=None,
        target_transform=None,
    )
)

# Use seed for selecting training and test data
RANDOM_SEED = 42
# Load annotation file
# If train_annotation and test_annotation files do not exist, exit
try:
    train_annotation = pd.read_csv(config.TRAIN_ANNOTATION_FILE)
    test_annotation = pd.read_csv(config.TEST_ANNOTATION_FILE)
except FileNotFoundError:
    print("Train and test annotation files do not exist. Run train_models.py first.")
    sys.exit(1)

# %% [markdown]
# # Fully Connected Neural Network
# Evaluating the trained models with multi-label accuracy, f1-score, and ROC AUC.

# %%
test_loader = DataLoader(
    data.AudioDataset(
        config.TEST_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: torch.flatten(x[None, :, :64]),
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)
# Load model, for each number of layers and units, and learning rate
fc_results = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for n_layers in models.fully_connected.N_LAYERS:
        for m_units in models.fully_connected.M_UNITS:
            for learning_rate in models.fully_connected.LEARNING_RATES:
                # Create model and load state dict if it exists
                model = models.fully_connected.FullyConnected(
                    n_layers=n_layers,
                    m_units=m_units,
                    n_features=96 * 64,
                    m_labels=9,
                ).to(device)
                # Load state dict if it exists
                try:
                    model.load_state_dict(
                        torch.load(
                            f"models/fc_{n_layers}_{m_units}_{learning_rate}.pt",
                        )
                    )
                except (FileNotFoundError, EOFError):
                    # If model does not exist, skip
                    continue
                # Evaluate model
                print(f"Evaluating FC model: {n_layers}, {m_units}, {learning_rate}")
                fc_results[(n_layers, m_units, learning_rate)] = utils.test(
                    model,
                    test_loader,
                    criterion=torch.nn.BCELoss(),
                    device=device,
                    verbose=True,
                )

# %% [markdown]
# # AlexNet

# %%
# Create data loaders
test_loader = DataLoader(
    data.AudioDataset(
        config.TEST_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)

# Load model, for each learning rate
alexnet_results = {}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for learning_rate in models.alexnet.LEARNING_RATES:
        # Create model and load state dict if it exists
        model = models.alexnet.AlexNet(
            num_classes=9,
            dropout=0.5,
        ).to(device)
        # Load state dict if it exists
        try:
            model.load_state_dict(
                torch.load(
                    f"models/alexnet_{learning_rate}.pt",
                )
            )
        except (FileNotFoundError, EOFError):
            # If model does not exist, skip
            continue
        # Evaluate model
        print(f"Evaluating AlexNet model: {learning_rate}")
        alexnet_results[learning_rate] = utils.test(
            model,
            test_loader,
            criterion=torch.nn.BCELoss(),
            device=device,
            verbose=True,
        )

# %% [markdown]
# # Save results

# %%
# Save results with pickle
with open("results.pkl", "wb") as f:
    pkl.dump(
        {
            "fc": fc_results,
            "alexnet": alexnet_results,
        },
        f,
    )

# %%
