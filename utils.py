"""Functions for PyTorch utility."""

# %% [markdown]
# # Imports

# %%
import torch
import sklearn.metrics

import config

# %% [markdown]
# # Functions

# %%


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    n_classes: int = 1,
    verbose: bool = True,
) -> dict:
    """Train a PyTorch model.

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): data loader for training data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): device to train on
        num_epochs (int): number of epochs

    Returns:
        dict: dictionary of metrics
    """
    model.train()  # set model to training mode
    metrics: dict = {
        "loss": [],
        "accuracy": [],
        "hamming": [],
        "f1-score": [],
        "roc_auc": [],
    }
    # loop over epochs
    for epoch in range(num_epochs):
        running_loss: float = 0.0
        running_corrects: int = 0
        # initialize inputs and labels tensors
        inputs: torch.Tensor = torch.empty(0)
        labels: torch.Tensor = torch.empty(0)
        # Acumulate labels and predictions for F1 score
        result_labels: torch.Tensor = torch.empty(0)
        result_preds: torch.Tensor = torch.empty(0)
        # loop over data
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                # multiclass classification
                # get probability for each class based on threshold
                # if probability >= threshold, then class = 1, else class = 0
                preds = (outputs >= config.PRED_THRESHOLD).float()
                # calculate loss with predictions and labels
                loss = criterion(outputs, labels)
                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += int(
                torch.sum((outputs >= config.PRED_THRESHOLD).float() == labels.data)
            )
            result_labels = torch.cat((result_labels, labels.detach().cpu()), dim=0)
            result_preds = torch.cat((result_preds, outputs.detach().cpu()), dim=0)
            # running_corrects += int(torch.sum(preds == labels.data))
        # Calculate epoch metrics
        epoch_loss: float = running_loss / float(len(train_loader.dataset))  # type: ignore
        epoch_accuracy = sklearn.metrics.accuracy_score(
            y_true=result_labels.cpu().numpy().astype(int),
            y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        )
        epoch_hamming = sklearn.metrics.hamming_loss(
            y_true=result_labels.cpu().numpy().astype(int),
            y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        )
        epoch_f1 = sklearn.metrics.f1_score(
            y_true=result_labels.cpu().numpy().astype(int),
            y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
            # average="weighted",
            zero_division="warn",
        )
        epoch_roc_auc = sklearn.metrics.roc_auc_score(
            y_true=result_labels.cpu().numpy().astype(int),
            y_score=result_preds.cpu().numpy(),
            average="weighted",
            multi_class="ovr",
        )
        # Append epoch metrics
        metrics["loss"].append(epoch_loss)
        metrics["accuracy"].append(epoch_accuracy)
        metrics["hamming"].append(epoch_hamming)
        metrics["f1-score"].append(epoch_f1)
        metrics["roc_auc"].append(epoch_roc_auc)
        # Print epoch metrics
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"Training Loss: "
                f"{epoch_loss:.4f} "
                f"Accuracy: {epoch_accuracy:.4f} "
                f"Hamming: {epoch_hamming:.4f} "
                f"F1: {epoch_f1} "
                f"ROC AUC: {epoch_roc_auc:.4f}"
            )
    return metrics


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Test a PyTorch model.

    Args:
        model (torch.nn.Module): model to test
        test_loader (torch.utils.data.DataLoader): data loader for test data
        criterion (torch.nn.Module): loss function
        device (torch.device): device to test on

    Returns:
        dict: dictionary of metrics
    """
    model.eval()  # set model to evaluation mode

    # initialize inputs and labels tensors
    inputs: torch.Tensor = torch.empty(0)
    labels: torch.Tensor = torch.empty(0)

    # Acumulate labels and predictions for F1 score
    result_labels: torch.Tensor = torch.empty(0)
    result_preds: torch.Tensor = torch.empty(0)

    # initialize loss and accuracy
    running_loss: float = 0.0
    running_corrects: int = 0

    # loop over data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += int(
            torch.sum((outputs >= config.PRED_THRESHOLD).float() == labels.data)
        )
        result_labels = torch.cat((result_labels, labels.detach().cpu()), dim=0)
        result_preds = torch.cat((result_preds, outputs.detach().cpu()), dim=0)
    # Calculate epoch metrics
    epoch_loss: float = running_loss / float(len(test_loader.dataset))  # type: ignore
    epoch_accuracy = sklearn.metrics.accuracy_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
    )
    epoch_hamming = sklearn.metrics.hamming_loss(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
    )
    epoch_f1 = sklearn.metrics.f1_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        # average="weighted",
        zero_division="warn",
    )
    epoch_roc_auc = sklearn.metrics.roc_auc_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_score=result_preds.cpu().numpy(),
        average="weighted",
        multi_class="ovr",
    )

    if verbose:
        print(
            f"Test Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.4f} "
            f"Hamming: {epoch_hamming:.4f} "
            f"F1: {epoch_f1} "
            f"ROC AUC: {epoch_roc_auc:.4f}"
        )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "hamming_loss": epoch_hamming,
        "f1-score": epoch_f1,
        "roc_auc": epoch_roc_auc,
    }


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Validate a PyTorch model.

    Args:
        model (torch.nn.Module): model to validate
        val_loader (torch.utils.data.DataLoader): data loader for validation data
        criterion (torch.nn.Module): loss function
        device (torch.device): device to validate on

    Returns:
        float: loss
        float: accuracy
        float: f1 score
    """
    model.eval()  # set model to evaluation mode

    # initialize inputs and labels tensors
    inputs: torch.Tensor = torch.empty(0)
    labels: torch.Tensor = torch.empty(0)

    # initialize loss and accuracy
    running_loss: float = 0.0
    running_corrects: int = 0

    # loop over data
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += int(torch.sum(preds == labels.data))

    epoch_loss: float = running_loss / float(len(val_loader.dataset))  # type: ignore
    epoch_acc: float = float(running_corrects) / float(len(val_loader.dataset))  # type: ignore
    epoch_f1: float = sklearn.metrics.f1_score(
        y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), average="weighted"
    )

    if verbose:
        print(
            f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}"
        )

    return epoch_loss, epoch_acc, epoch_f1


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save a PyTorch model.

    Args:
        model (torch.nn.Module): model to save
        path (str): path to save model to

    Returns:
        None
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str) -> None:
    """Load a PyTorch model.

    Args:
        model (torch.nn.Module): model to load
        path (str): path to load model from

    Returns:
        None
    """
    model.load_state_dict(torch.load(path))


# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    pass

# %%
