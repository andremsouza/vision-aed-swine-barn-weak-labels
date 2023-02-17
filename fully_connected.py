"""Fully connected neural network with different number of layers and units.

This script is used to define a fully connected neural network with different
number of layers and units.
"""
# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn

# %% [markdown]
# # Constants

# %%
N_LAYERS = [2, 3, 4, 5, 6]
M_UNITS = [500, 1000, 2000, 3000, 4000]
LEARNING_RATES = [1e-3, 1e-4, 1e-5]

# %% [markdown]
# # Classes

# %%


class FullyConnected(nn.Module):
    """Fully connected neural network with audio frames as input and labels as output.

    Attributes:
        n_layers (int): number of layers
        m_units (int): number of units in each layer
        layers (nn.Sequential): layers of the neural network

    References:
        - https://doi.org/10.1109/ICASSP.2017.7952132
    """

    def __init__(
        self,
        n_layers: int,
        m_units: int,
        n_features: int,
        m_labels: int,
        activation: str = "relu",
    ) -> None:
        """Initialize the fully connected neural network.

        Args:
            n_layers (int): number of layers
            m_units (int): number of units in each layer
        """
        super().__init__()
        self.n_layers: int = n_layers
        self.m_units: int = m_units
        self.n_features: int = n_features
        self.m_labels: int = m_labels
        self.activation = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }[activation]

        self.layers: nn.Sequential = self._create_layers()

    def _create_layers(self) -> nn.Sequential:
        """Create the layers of the neural network.

        Returns:
            nn.Sequential: layers of the neural network
        """
        layers: list = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(
                    nn.Linear(in_features=self.n_features, out_features=self.m_units)
                )
            else:
                layers.append(
                    nn.Linear(in_features=self.m_units, out_features=self.m_units)
                )
            layers.append(self.activation)
        layers.append(nn.Linear(in_features=self.m_units, out_features=self.m_labels))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.layers(x)

    def __repr__(self) -> str:
        """Return the string representation of the neural network.

        Returns:
            str: string representation of the neural network
        """
        return (
            f"FullyConnected(n_layers={self.n_layers}, "
            + f"m_units={self.m_units}, n_features={self.n_features}, m_labels={self.m_labels})"
        )

    def __str__(self) -> str:
        """Return the string representation of the neural network.

        Returns:
            str: string representation of the neural network
        """
        return (
            f"FullyConnected(n_layers={self.n_layers}, "
            + f"m_units={self.m_units}, n_features={self.n_features}, m_labels={self.m_labels})"
        )

    def __len__(self) -> int:
        """Return the number of parameters of the neural network.

        Returns:
            int: number of parameters of the neural network
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __eq__(self, other: object) -> bool:
        """Check if two neural networks are equal.

        Args:
            other (object): other neural network

        Returns:
            bool: whether the two neural networks are equal
        """
        if not isinstance(other, FullyConnected):
            return NotImplemented
        return (
            self.n_layers == other.n_layers
            and self.m_units == other.m_units
            and self.n_features == other.n_features
            and self.m_labels == other.m_labels
        )

    def __hash__(self) -> int:
        """Return the hash of the neural network.

        Returns:
            int: hash of the neural network
        """
        return hash((self.n_layers, self.m_units, self.n_features, self.m_labels))

    def __ne__(self, other: object) -> bool:
        """Check if two neural networks are not equal.

        Args:
            other (object): other neural network

        Returns:
            bool: whether the two neural networks are not equal
        """
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """Check if the current neural network is smaller than the other.

        Args:
            other (object): other neural network

        Returns:
            bool: whether the current neural network is smaller than the other
        """
        if not isinstance(other, FullyConnected):
            return NotImplemented
        return len(self) < len(other)

    def __le__(self, other: object) -> bool:
        """Check if the current neural network is smaller or equal to the other.

        Args:
            other (object): other neural network

        Returns:
            bool: whether the current neural network is smaller or equal to the other
        """
        if not isinstance(other, FullyConnected):
            return NotImplemented
        return len(self) <= len(other)

    def __gt__(self, other: object) -> bool:
        """Check if the current neural network is greater than the other.

        Args:
            other (object): other neural network

        Returns:
            bool: whether the current neural network is greater than the other
        """
        if not isinstance(other, FullyConnected):
            return NotImplemented
        return len(self) > len(other)

    def __ge__(self, other: object) -> bool:
        """Check if the current neural network is greater or equal to the other.

        Args:
            other (object): other neural network

        Returns:
            bool: whether the current neural network is greater or equal to the other
        """
        if not isinstance(other, FullyConnected):
            return NotImplemented
        return len(self) >= len(other)


# %% [markdown]
# # Functions

# %%

# %% [markdown]
# # Main
# Testing with Iris data

# %%
if __name__ == "__main__":
    # Instantiate and train a neural network with iris data for testing
    from sklearn.datasets import load_iris  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    from config import DEVICE  # type: ignore
    from utils import train, test  # type: ignore

    # Set pytorch device
    device = torch.device(DEVICE)

    # Load iris data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=True)

    # Dictionary to store results
    results = {}

    # For each combination of number of layers and number of units
    for n in N_LAYERS:
        for m in M_UNITS:
            for lr in LEARNING_RATES:
                # Instantiate neural network
                net = FullyConnected(n, m, X_train.shape[1], 3, activation="relu")
                net = net.to(device)  # pylint: disable=invalid-name

                # Train neural network
                train(
                    model=net,
                    train_loader=train_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=torch.optim.Adam(net.parameters(), lr=1e-4, eps=1e-8),
                    device=device,
                    num_epochs=1000,
                    verbose=True,
                )

                # Test neural network, store results
                result = test(
                    model=net,
                    test_loader=test_loader,
                    criterion=nn.CrossEntropyLoss(),
                    device=device,
                    verbose=True,
                )
                results[(n, m, lr)] = {
                    "loss": result[0],
                    "accuracy": result[1],
                    "f1": result[2],
                    "n_params": len(net),
                }

                # Print neural network
                print(net)

    # print results sorted by f1-score descending, loss ascending, number of parameters ascending
    for key, value in sorted(
        results.items(),
        key=lambda item: (-item[1]["f1"], item[1]["loss"], item[1]["n_params"]),
    ):
        print(f"{key}: {value}")

# %%
