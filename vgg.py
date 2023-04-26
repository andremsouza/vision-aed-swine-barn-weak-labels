"""Adaptation of the VGG (configuration E) PyTorch model for audio data."""
# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn

# %% [markdown]
# # Constants

# %%
LEARNING_RATES = [0.001, 0.0001, 0.00001]

# %% [markdown]
# # Classes

# %%


class VGG(nn.Module):
    """Adaptation of the VGG (configuration E) PyTorch model for audio data."""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        """Initialize the VGG (configuration E) model.

        Args:
            num_classes (int): number of classes
            dropout (float): dropout rate
        """
        super().__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv9
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv15
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # pool5
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

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
        """Extract features from the model.

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
    model = VGG()
    print(model)
    # generate a batch of 128 (96x64) inputs
    batch_X = torch.randn(128, 1, 96, 64)
    # forward pass
    batch_y = model(batch_X)
    print(batch_y.shape)

# %%
