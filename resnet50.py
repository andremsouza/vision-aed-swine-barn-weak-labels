"""ResNet-50 model adapted from PyTorch.

This file contains a partial implementation of the ResNet-50 model
from https://arxiv.org/abs/1512.03385. It is adapted from PyTorch
"""
# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
import torchvision

# %% [markdown]
# # Classes

# %%


class ResNet50(torchvision.models.resnet.ResNet):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        # remove stride 2 in the first 7x7 conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # change avgpool size to 6x4
        self.avgpool = nn.AdaptiveAvgPool2d((6, 4))
        # adapt fc layer
        self.fc = nn.Linear(49152, num_classes)


# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    model = ResNet50()
    print(model)
    # generate a batch of 128 (96x64) inputs
    batch_X = torch.randn(128, 1, 96, 64)
    batch_y = model(batch_X)
    print(batch_y.shape)


# %%
