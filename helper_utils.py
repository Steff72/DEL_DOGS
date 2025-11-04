"""Utility helpers for inspecting the dog dataset structure and samples."""

from collections import Counter
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from directory_tree import DisplayTree


def _split_class_folder_name(name: str) -> Tuple[str, str]:
    """Return the WordNet ID and description extracted from a folder name."""
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
    Displays the folder and file structure of a given directory.

    Args:
        root_dir (str): The path to the root directory to be displayed.
        max_depth (int): The maximum depth to traverse the directory tree.
    """
    # Define the configuration settings for displaying the directory tree.
    config_tree = {
        # Specify the starting path for the directory tree.
        "dirPath": root_dir,
        # Set to False to include both files and directories.
        "onlyDirs": False,
        # Set the maximum depth for the tree traversal.
        "maxDepth": max_depth,
        # Specify a sorting option (100 typically means no specific sort).
        "sortBy": 100,
    }
    # Create and display the tree structure using the unpacked configuration.
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
        root_dir (str | Path | None): Optional custom image root passed to
            ``resolve_images_root``. Defaults to the repository dataset.
        n_rows (int): Number of rows in the grid. Defaults to 20.
        n_cols (int): Number of columns in the grid. Defaults to 6.
        figsize (tuple[float, float]): Matplotlib figure size. Defaults to
            ``(18, 60)``.
        show (bool): Whether to trigger ``plt.show()`` before returning.
        return_objects (bool): If True, return the (figure, axes) tuple so
            callers can further customize the plot.
        dataset (optional): Dataset object exposing ``class_to_idx`` and
            ``get_label_description`` so titles can use friendly labels.

    Returns:
        Optional[tuple[plt.Figure, Any]]: The (figure, axes) tuple when
            ``return_objects`` is True; otherwise ``None``.
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
    """Plot the number of samples available for each label in the dataset."""
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
    """Compute per-channel mean and standard deviation for a dataset."""
    preprocess = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    channel_sums = torch.zeros(3)
    channel_sums_sq = torch.zeros(3)
    pixel_count = 0

    for image, _ in dataset:
        tensor = preprocess(image)
        pixel_count += tensor.size(1) * tensor.size(2)
        channel_sums += tensor.sum(dim=(1, 2))
        channel_sums_sq += (tensor ** 2).sum(dim=(1, 2))

    mean = channel_sums / pixel_count
    variance = channel_sums_sq / pixel_count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    return mean, std
