"""Setup and run models.

This script is used to setup and run the models. It is used to train the models
and save the trained models.
"""
# %%
import datetime
import warnings

# import pandas as pd
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

# %% [markdown]
# # Fully Connected Neural Network

# %%
# Create a model for each number of layers and units, and learning rate
fc_models = {}

for n_layers in models.fully_connected.N_LAYERS:
    for m_units in models.fully_connected.M_UNITS:
        for learning_rate in models.fully_connected.LEARNING_RATES:
            # Create model and load state dict if it exists
            fc_models[
                (n_layers, m_units, learning_rate)
            ] = models.fully_connected.FullyConnected(
                n_layers=n_layers,
                m_units=m_units,
                n_features=96 * 64,
                m_labels=9,
            ).to(
                device
            )
            # Load state dict if it exists
            try:
                fc_models[(n_layers, m_units, learning_rate)].load_state_dict(
                    torch.load(
                        f"models/fc_{n_layers}_{m_units}_{learning_rate}.pt",
                    )
                )
            except FileNotFoundError:
                pass


train_loader = DataLoader(
    data.AudioDataset(
        config.ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: torch.flatten(x[None, :, :64]),
        target_transform=lambda x: x[0, :],
    ),
    batch_size=24,
    shuffle=True,
    num_workers=24,
)

# %%
# Train model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for (n_layers, m_units, learning_rate), model in fc_models.items():
        print(
            f"{datetime.datetime.now()}: "
            f"Training model w/ {n_layers} layers, {m_units} units, {learning_rate} learning rate"
        )
        utils.train(
            model,
            train_loader,
            criterion=torch.nn.BCELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
            device=device,
            num_epochs=10,
            n_classes=9,
            verbose=True,
        )
        # Save model state dict
        print(
            f"{datetime.datetime.now()}: "
            f"Saving model w/ {n_layers} layers, {m_units} units, {learning_rate} learning rate"
        )
        torch.save(
            model.state_dict(),
            f"models/fc_{n_layers}_{m_units}_{learning_rate}.pt",
        )

# %%

# %% [markdown]
# # AlexNet

# %%
model = models.alexnet.AlexNet(num_classes=9, dropout=0.5).to(device)
# Load state dict if it exists
try:
    model.load_state_dict(torch.load("models/alexnet.pt"))
except FileNotFoundError:
    pass

train_loader = DataLoader(
    data.AudioDataset(
        config.ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=24,
    shuffle=True,
    num_workers=24,
)

# %%
# Train model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"{datetime.datetime.now()}: Training AlexNet model")
    utils.train(
        model,
        train_loader,
        criterion=torch.nn.BCELoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        device=device,
        num_epochs=10,
        n_classes=9,
        verbose=True,
    )
# Save model state dict
print(f"{datetime.datetime.now()}: Saving AlexNet model")
torch.save(model.state_dict(), "models/alexnet.pt")

# %%
