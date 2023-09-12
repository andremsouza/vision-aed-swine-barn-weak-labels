"""Setup and run models.

This script is used to setup and run the models. It is used to train the models
and save the trained models.
"""
# %%
import datetime
import gc
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
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
# Load annotation file
# If train_annotation and test_annotation files do not exist, create them
try:
    train_annotation = pd.read_csv(config.TRAIN_ANNOTATION_FILE)
    test_annotation = pd.read_csv(config.TEST_ANNOTATION_FILE)
except FileNotFoundError:
    # Create train and test annotation files with random sampling
    annotation = pd.read_csv(config.ANNOTATION_FILE)
    # # Merge stress and disputes
    # annotation["Estresse"] = annotation["Estresse"] | annotation["Disputas"]
    # # Drop disputes
    # annotation = annotation.drop(columns=["Disputas"])
    # # Merge cough and sneeze
    # annotation["Tosse"] = annotation["Tosse"] | annotation["Espirro"]
    # # Drop sneeze
    # annotation = annotation.drop(columns=["Espirro"])
    # # rename cough to cough_sneeze
    # annotation = annotation.rename(columns={"Tosse": "Tosse_Espirro"})
    train_annotation, test_annotation = train_test_split(
        annotation,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
    )
    # Sort by Timestamp
    train_annotation = train_annotation.sort_values(by="Timestamp")
    test_annotation = test_annotation.sort_values(by="Timestamp")
    # Save annotation files
    train_annotation.to_csv(config.TRAIN_ANNOTATION_FILE, index=False)
    test_annotation.to_csv(config.TEST_ANNOTATION_FILE, index=False)

# %% [markdown]
# # Fully Connected Neural Network

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: torch.flatten(x[None, :, :64]),
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)
# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for n_layers in models.fully_connected.N_LAYERS:
        for m_units in models.fully_connected.M_UNITS:
            for learning_rate in models.fully_connected.LEARNING_RATES:
                # # Only train specific models
                # if config.FC_BEST_MODELS_ONLY and (
                #     n_layers,
                #     m_units,
                #     learning_rate,
                # ) not in [
                #     (3, 4000, 1e-5),
                #     (3, 3000, 1e-5),
                #     (3, 2000, 1e-5),
                #     (4, 3000, 1e-5),
                #     (2, 4000, 1e-4),
                #     (3, 1000, 1e-5),
                # ]:
                #     continue
                # Create model and load state dict if it exists
                model = models.fully_connected.FullyConnected(
                    num_layers=n_layers,
                    num_units=m_units,
                    num_features=96 * 64,
                    num_classes=9,
                ).to(device)
                try:
                    torch.load(
                        f"{config.MODELS_DIRECTORY}fc_{n_layers}_{m_units}_{learning_rate}.pt",
                    )
                    if config.SKIP_TRAINED_MODELS:
                        continue
                except FileNotFoundError:
                    # Touch file so it exists
                    open(
                        f"{config.MODELS_DIRECTORY}fc_{n_layers}_{m_units}_{learning_rate}.pt",
                        "a",
                        encoding="utf-8",
                    ).close()
                except EOFError:
                    continue
                print(
                    f"{datetime.datetime.now()}: "
                    f"Training FC model w/ {n_layers} layers, {m_units} units, {learning_rate} lr"
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
                    f"Saving model w/ {n_layers} layers, {m_units} units, {learning_rate} lr"
                )
                torch.save(
                    model.state_dict(),
                    f"{config.MODELS_DIRECTORY}fc_{n_layers}_{m_units}_{learning_rate}.pt",
                )
                # free memory
                del model
                # garbage collect
                gc.collect()
                # free cuda memory
                torch.cuda.empty_cache()

# %%
# TODO: Test model and select samples with highest entropy

# %% [markdown]
# # AlexNet

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)

# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for learning_rate in models.alexnet.LEARNING_RATES:
        # Create model and load state dict if it exists
        model = models.alexnet.AlexNet(  # type: ignore
            num_classes=9,
            dropout=0.5,
        ).to(device)
        # Load state dict if it exists
        try:
            model.load_state_dict(
                torch.load(f"{config.MODELS_DIRECTORY}alexnet_{learning_rate}.pt")
            )
            if config.SKIP_TRAINED_MODELS:
                continue
        except FileNotFoundError:
            # Touch file so it exists
            open(
                f"{config.MODELS_DIRECTORY}alexnet_{learning_rate}.pt",
                "a",
                encoding="utf-8",
            ).close()
        except EOFError:
            continue
        print(
            f"{datetime.datetime.now()}: "
            f"Training AlexNet model w/ {learning_rate} learning rate"
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
            f"Saving AlexNet model w/ {learning_rate} learning rate"
        )
        torch.save(
            model.state_dict(),
            f"{config.MODELS_DIRECTORY}alexnet_{learning_rate}.pt",
        )
        # free memory
        del model
        # garbage collect
        gc.collect()
        # free cuda memory
        torch.cuda.empty_cache()

# %%
# TODO: Test model and select samples with highest entropy

# %% [markdown]
# # Inception_v3

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)

# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for learning_rate in models.inception_v3.LEARNING_RATES:
        # Create model and load state dict if it exists
        model = models.inception_v3.InceptionV3(  # type: ignore
            num_classes=9,
            dropout=0.5,
        ).to(device)
        # Load state dict if it exists
        try:
            model.load_state_dict(
                torch.load(f"{config.MODELS_DIRECTORY}inception_v3_{learning_rate}.pt")
            )
            if config.SKIP_TRAINED_MODELS:
                continue
        except FileNotFoundError:
            # Touch file so it exists
            open(
                f"{config.MODELS_DIRECTORY}inception_v3_{learning_rate}.pt",
                "a",
                encoding="utf-8",
            ).close()
        except EOFError:
            continue
        print(
            f"{datetime.datetime.now()}: "
            f"Training Inception_v3 model w/ {learning_rate} learning rate"
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
            f"Saving Inception_v3 model w/ {learning_rate} learning rate"
        )
        torch.save(
            model.state_dict(),
            f"{config.MODELS_DIRECTORY}inception_v3_{learning_rate}.pt",
        )
        # free memory
        del model
        # garbage collect
        gc.collect()
        # free cuda memory
        torch.cuda.empty_cache()

# %%
# TODO: Test model and select samples with highest entropy

# %% [markdown]
# # ResNet

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)

# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for learning_rate in models.resnet50.LEARNING_RATES:
        # Create model and load state dict if it exists
        model = models.resnet50.ResNet50(num_classes=9, dropout=0.5).to(device)  # type: ignore
        # Load state dict if it exists
        try:
            model.load_state_dict(
                torch.load(f"{config.MODELS_DIRECTORY}resnet50_{learning_rate}.pt")
            )
            if config.SKIP_TRAINED_MODELS:
                continue
        except FileNotFoundError:
            # Touch file so it exists
            open(
                f"{config.MODELS_DIRECTORY}resnet50_{learning_rate}.pt",
                "a",
                encoding="utf-8",
            ).close()
        except EOFError:
            continue
        print(
            f"{datetime.datetime.now()}: "
            f"Training ResNet50 model w/ {learning_rate} learning rate"
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
            f"Saving ResNet50 model w/ {learning_rate} learning rate"
        )
        torch.save(
            model.state_dict(),
            f"{config.MODELS_DIRECTORY}resnet50_{learning_rate}.pt",
        )
        # free memory
        del model
        # garbage collect
        gc.collect()
        # free cuda memory
        torch.cuda.empty_cache()

# %%
# TODO: Test model and select samples with highest entropy

# %% [markdown]
# # VGG

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)

# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for learning_rate in models.vgg.LEARNING_RATES:
        # Create model and load state dict if it exists
        model = models.vgg.VGG(num_classes=9, dropout=0.5).to(device)  # type: ignore
        # Load state dict if it exists
        try:
            model.load_state_dict(
                torch.load(f"{config.MODELS_DIRECTORY}vgg_{learning_rate}.pt")
            )
            if config.SKIP_TRAINED_MODELS:
                continue
        except FileNotFoundError:
            # Touch file so it exists
            open(
                f"{config.MODELS_DIRECTORY}vgg_{learning_rate}.pt",
                "a",
                encoding="utf-8",
            ).close()
        except EOFError:
            continue
        print(
            f"{datetime.datetime.now()}: "
            f"Training VGG model w/ {learning_rate} learning rate"
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
            f"Saving VGG model w/ {learning_rate} learning rate"
        )
        torch.save(
            model.state_dict(),
            f"{config.MODELS_DIRECTORY}vgg_{learning_rate}.pt",
        )
        # free memory
        del model
        # garbage collect
        gc.collect()
        # free cuda memory
        torch.cuda.empty_cache()
