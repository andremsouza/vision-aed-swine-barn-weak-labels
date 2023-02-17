"""Inception v3 model adapted from PyTorch.

This file contains a partial implementation of the Inception v3 model
from https://arxiv.org/abs/1512.00567. It is adapted from PyTorch
"""
# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
from torchvision.models.inception import (
    BasicConv2d,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)

# %% [markdown]
# # Classes

# %%


class InceptionV3(nn.Module):
    """Inception v3 model adapted from PyTorch.

    This file contains a partial implementation of the Inception v3 model
    from https://arxiv.org/abs/1512.00567. It is adapted from PyTorch
    """

    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
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


# %% [markdown]
# # Test

# %%
if __name__ == "__main__":
    model = InceptionV3()
    print(model)
    # generate a batch of 128 (96x64) inputs
    batch_X = torch.randn(128, 1, 96, 64)
    batch_y = model(batch_X)
    print(batch_y.shape)

# %%
