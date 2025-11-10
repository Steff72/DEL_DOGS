"""Utility helpers for inspecting the dog dataset structure and samples."""

from collections import Counter
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from directory_tree import DisplayTree


def _split_class_folder_name(name: str) -> Tuple[str, str]:
    """
    Return the WordNet ID and description extracted from a folder name.

    Args:
        name (str): Folder name in the form ``<wnid>-<description>`` or
            simply ``<wnid>``.

    Returns:
        tuple[str, str]: ``(wnid, description)`` where ``description`` is an
        empty string when the folder does not include a suffix.
    """
    # Some folders are named ``<wnid>-<description>`` while others contain only
    # the WordNet identifier. Split once to keep both parts for display logic.
    if "-" in name:
        wnid, description = name.split("-", 1)
        return wnid, description
    return name, ""


def resolve_images_root(
    root_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Resolve the root directory that contains the image subfolders.

    The helper prefers a user-supplied path, falls back to the repository's
    ``data/raw/Images`` directory, and guarantees that the returned path
    exists on disk.

    Args:
        root_dir (str | Path | None): Optional custom directory.

    Returns:
        str: Absolute path to the image root directory.

    Raises:
        FileNotFoundError: If the resolved directory does not exist.
    """
    if root_dir is None:
        base_path = Path(__file__).resolve().parent / "data" / "raw" / "Images"
    else:
        base_path = Path(root_dir).expanduser()

    base_path = base_path.resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {base_path}")

    return str(base_path)

def print_data_folder_structure(root_dir: str, max_depth: int = 1) -> None:
    """
    Display the folder and file structure of a given directory.

    Args:
        root_dir (str): Absolute or relative path to the root directory.
        max_depth (int): Maximum depth for the directory traversal.

    Returns:
        None
    """
    # Configure the directory tree visualiser.
    config_tree = {
        "dirPath": root_dir,
        "onlyDirs": False,
        "maxDepth": max_depth,
        "sortBy": 100,
    }
    # Render the tree using the external helper.
    DisplayTree(**config_tree)


def plot_group_overview_grid(
    root_dir: Optional[Union[str, Path]] = None,
    n_rows: int = 20,
    n_cols: int = 6,
    figsize: Tuple[float, float] = (18, 60),
    show: bool = True,
    return_objects: bool = False,
    dataset: Any = None,
) -> Optional[Tuple[plt.Figure, Any]]:
    """
    Plot a grid that shows the first image of each class directory.

    Args:
        root_dir (str | Path | None): Optional dataset root. Falls back to the
            default repository dataset when omitted.
        n_rows (int): Number of rows in the image grid.
        n_cols (int): Number of columns in the image grid.
        figsize (tuple[float, float]): Matplotlib figure size.
        show (bool): Whether to call ``plt.show()`` before returning.
        return_objects (bool): When ``True`` the Matplotlib figure and axes are
            returned for further customization.
        dataset (Any, optional): Dataset exposing ``class_to_idx`` and
            ``get_label_description`` so axes titles can use readable labels.

    Returns:
        Optional[tuple[plt.Figure, Any]]: ``(figure, axes)`` when
        ``return_objects`` is ``True``; otherwise ``None``.
    """
    image_root = Path(resolve_images_root(root_dir))
    class_dirs = sorted(
        item for item in image_root.iterdir() if item.is_dir()
    )

    if not class_dirs:
        raise ValueError(f"No class folders found under {image_root}")

    total_slots = n_rows * n_cols
    selected_dirs = class_dirs[:total_slots]

    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes_grid.flatten()

    for ax, class_dir in zip(axes_flat, selected_dirs):
        image_files = sorted(
            item for item in class_dir.iterdir() if item.is_file()
        )
        if not image_files:
            ax.axis("off")
            continue

        image_path = image_files[0]
        with Image.open(image_path) as image:
            ax.imshow(image.copy())
        ax.axis("off")

        # Start with the raw folder name as a safe fallback.
        title_text = class_dir.name
        if dataset is not None:
            try:
                label_idx = dataset.class_to_idx[class_dir.name]
                title_text = dataset.get_label_description(label_idx)
            except (AttributeError, KeyError):
                wnid, description = _split_class_folder_name(class_dir.name)
                title_text = description or wnid
        else:
            wnid, description = _split_class_folder_name(class_dir.name)
            title_text = description or wnid

        ax.set_title(title_text.replace("_", " "), fontsize=8)

    for ax in axes_flat[len(selected_dirs) :]:
        ax.axis("off")

    fig.tight_layout(pad=0.5)

    if show:
        plt.show()

    if return_objects:
        return fig, axes_grid

    return None


def plot_class_distribution(
    dataset: Any,
    top_n: Optional[int] = None,
    figsize: Tuple[float, float] = (24, 8),
    show: bool = True,
    return_objects: bool = False,
) -> Optional[Tuple[plt.Figure, Axes]]:
    """
    Plot the number of samples available for each label in the dataset.

    Args:
        dataset (Any): Dataset exposing a ``labels`` attribute and optional
            ``get_label_description`` helper.
        top_n (int | None): When provided, limit the plot to the ``top_n``
            most frequent classes.
        figsize (tuple[float, float]): Matplotlib figure size.
        show (bool): Whether to display the plot immediately.
        return_objects (bool): Whether to return the ``(figure, ax)`` pair.

    Returns:
        Optional[tuple[plt.Figure, Axes]]: Matplotlib figure/axes when
        ``return_objects`` is ``True``; otherwise ``None``.
    """
    if not hasattr(dataset, "labels"):
        raise AttributeError("Dataset must expose a 'labels' attribute.")

    labels = getattr(dataset, "labels")
    if labels is None:
        raise ValueError("Dataset labels are not initialized.")

    counts = Counter(labels)
    if not counts:
        raise ValueError("Dataset does not contain any samples.")

    sorted_items = sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    )
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    indices, frequencies = zip(*sorted_items)
    descriptions = []
    for label_idx in indices:
        description = None
        if hasattr(dataset, "get_label_description"):
            try:
                description = dataset.get_label_description(label_idx)
            except Exception:  # pragma: no cover - defensive fallback
                description = None
        if description is None and hasattr(dataset, "idx_to_class"):
            # Fall back to the folder name when no descriptor lives on dataset.
            class_name = dataset.idx_to_class.get(label_idx, str(label_idx))
            _, description = _split_class_folder_name(class_name)
            if not description:
                description = class_name
        if description is None:
            description = str(label_idx)
        descriptions.append(description.replace("_", " "))

    fig, ax = plt.subplots(figsize=figsize)
    positions = range(len(descriptions))
    ax.bar(positions, frequencies, color="tab:blue", align="center")
    # Expand the x-limits so the bars at either edge sit flush with the frame.
    ax.set_xlim(-0.5, len(descriptions) - 0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(descriptions, rotation=90, ha="right")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Dog breed")
    ax.set_title("Samples per class")
    fig.tight_layout()

    if show:
        plt.show()

    if return_objects:
        return fig, ax

    return None


def get_mean_std(
    dataset: Dataset,
    image_size: Tuple[int, int] = (128, 128),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and standard deviation for a dataset.

    Args:
        dataset (Dataset): Dataset yielding PIL images.
        image_size (tuple[int, int]): Size to which images are resized before
            computing statistics.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(mean, std)`` tensors of shape
        ``(3,)`` containing RGB channel statistics.
    """
    preprocess = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    channel_sums = torch.zeros(3)
    channel_sums_sq = torch.zeros(3)
    pixel_count = 0

    for image, _ in dataset:
        # Apply the same preprocessing pipeline the model will see.
        tensor = preprocess(image)
        # Keep track of how many pixels contribute to the statistics.
        pixel_count += tensor.size(1) * tensor.size(2)
        # Accumulate first-order and second-order statistics per channel.
        channel_sums += tensor.sum(dim=(1, 2))
        channel_sums_sq += (tensor ** 2).sum(dim=(1, 2))

    mean = channel_sums / pixel_count
    variance = channel_sums_sq / pixel_count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    return mean, std


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fcn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train ``model`` for a single epoch.

    Args:
        model (nn.Module): Model under training.
        dataloader (DataLoader): Iterable of training batches.
        optimizer (Optimizer): Optimiser used for parameter updates.
        loss_fcn (nn.Module): Loss function applied to each batch.
        device (torch.device): Target device for inputs and model.

    Returns:
        tuple[float, float]: ``(average_loss, accuracy)`` for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    samples = 0

    # Iterate over each mini-batch produced by the dataloader.
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Standard training step: forward, loss, backward, optimizer update.
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fcn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        samples += targets.size(0)

    avg_loss = running_loss / max(samples, 1)
    accuracy = correct / max(samples, 1)
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fcn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate ``model`` on ``dataloader`` without updating weights.

    Args:
        model (nn.Module): Model under evaluation.
        dataloader (DataLoader): Iterable of validation or test batches.
        loss_fcn (nn.Module): Loss function applied to each batch.
        device (torch.device): Target device for inputs and model.

    Returns:
        tuple[float, float]: ``(average_loss, accuracy)`` for the epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    samples = 0

    with torch.no_grad():
        # The validation loop mirrors the training loop minus gradient steps.
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fcn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            samples += targets.size(0)

    avg_loss = running_loss / max(samples, 1)
    accuracy = correct / max(samples, 1)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    n_epochs: int,
    loss_fcn: nn.Module,
    device: torch.device,
    val_dataloader: Optional[DataLoader] = None,
    log_hook: Optional[Callable[[dict, int], None]] = None,
) -> List[dict]:
    """
    Train ``model`` for ``n_epochs`` while optionally logging metrics.

    Args:
        model: The neural network to optimise.
        optimizer: Optimiser instance (e.g., SGD, Adam).
        train_dataloader: Batched training data.
        n_epochs: Number of epochs to run.
        loss_fcn: Criterion used to compute the loss.
        device: Target device (CPU, CUDA, or MPS).
        val_dataloader: Optional validation dataloader.
        log_hook: Callable that receives (metrics, epoch) for logging.

    Returns:
        list[dict]: Sequence of metric dictionaries, one per epoch.
    """
    history: List[dict] = []
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fcn=loss_fcn,
            device=device,
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
        }

        if val_dataloader is not None:
            val_loss, val_acc = validate_epoch(
                model=model,
                dataloader=val_dataloader,
                loss_fcn=loss_fcn,
                device=device,
            )
            metrics["val_loss"] = val_loss
            metrics["val_accuracy"] = val_acc

        history.append(metrics)
        if log_hook is not None:
            # Let callers stream metrics to their preferred logging backend.
            log_hook(metrics, epoch)

    return history
