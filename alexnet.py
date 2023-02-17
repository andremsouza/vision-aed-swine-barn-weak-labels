"""Adaptation of the AlexNet PyTorch model for audio data."""
# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn

# %% [markdown]
# # Classes

# %%


class AlexNet(nn.Module):
    """Adaptation of the AlexNet PyTorch model for audio data.

    Attributes:
        features (nn.Sequential): layers of the neural network
        avgpool (nn.AdaptiveAvgPool2d): average pooling layer
        classifier (nn.Sequential): fully connected layers

    References:
        - https://pytorch.org/hub/pytorch_vision_alexnet/
        - https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html
        - https://doi.org/10.1109/ICASSP.2017.7952132
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        """Initialize the AlexNet model.

        Args:
            num_classes (int): number of classes
            dropout (float): dropout rate
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=(2, 1), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# %% [markdown]
# # Test

# %%
if __name__ == "__main__":
    model = AlexNet(num_classes=10)
    print(model)
    # generate a batch of 128 (96x64) inputs
    batch_X = torch.randn(128, 1, 96, 64)
    batch_y = model(batch_X)
    print(batch_y.shape)

# %%
