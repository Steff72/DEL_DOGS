"""Utility helpers for inspecting the dog dataset structure and samples."""

from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from tensorboard.backend.event_processing import event_accumulator

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




class DogDataset(Dataset):
    """Dataset wrapper for the Stanford Dogs folder structure."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None, transform=None):
        self.transform = transform
        self.root_dir = resolve_images_root(root_dir)
        self.image_dir = self.root_dir
        self.labels = self._load_and_correct_labels()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = self._retrieve_image(idx)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def value_counts(self) -> Counter:
        return Counter(self.labels)

    def get_label_description(self, label: int) -> str:
        raw_name = self.idx_to_class[label]
        if '-' in raw_name:
            raw_name = raw_name.split('-', 1)[1]
        return raw_name.lower()

    def _retrieve_image(self, idx: int) -> Image.Image:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert('RGB')
        return image

    def _load_and_correct_labels(self) -> List[int]:
        class_dirs = [
            entry
            for entry in os.listdir(self.image_dir)
            if os.path.isdir(os.path.join(self.image_dir, entry))
        ]
        if not class_dirs:
            raise ValueError(f"No class folders found in {self.image_dir}.")

        class_dirs.sort()
        self.class_names = class_dirs
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        image_paths = []
        labels = []
        valid_extensions = ('.jpg', '.jpeg', '.png')

        for class_name in self.class_names:
            class_path = os.path.join(self.image_dir, class_name)
            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(class_path, file_name))
                    labels.append(self.class_to_idx[class_name])

        if not image_paths:
            raise ValueError(f"No image files found under {self.image_dir}.")

        self.image_paths = image_paths
        return labels

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


class SubsetWithTransform(Dataset):
    """A subset wrapper that lets you apply a specific transform per split."""
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.dataset = subset.dataset 
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return (self.transform(image) if self.transform else image), label



def load_run_scalars(run_name: str, log_root: str = "runs") -> pd.DataFrame:
    """
    Read the latest scalar events for a given TensorBoard run.

    Args:
        run_name (str): Run directory relative to ``log_root``.
        log_root (str): Root directory that contains TensorBoard runs.

    Returns:
        pandas.DataFrame: Columns ``run``, ``epoch``, ``value``, ``split``,
        and ``metric`` capturing every scalar time series.
    """
    run_dir = os.path.join(log_root, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    event_files = []
    for root, _, files in os.walk(run_dir):
        candidates = [
            os.path.join(root, f)
            for f in files
            if f.startswith("events.out.tfevents")
        ]
        if not candidates:
            continue
        event_files.append(max(candidates, key=os.path.getmtime))

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found under: {run_dir}"
        )

    def infer_metric_and_split(tag: str, event_dir: str):
        tag_lower = tag.lower()
        dir_lower = event_dir.lower()

        metric = None
        if "loss" in tag_lower:
            metric = "loss"
        elif "acc" in tag_lower:
            metric = "accuracy"
        elif "loss" in dir_lower:
            metric = "loss"
        elif "acc" in dir_lower:
            metric = "accuracy"

        split = None
        if "train" in tag_lower:
            split = "train"
        elif "val" in tag_lower or "valid" in tag_lower:
            split = "val"
        elif "train" in dir_lower:
            split = "train"
        elif "val" in dir_lower or "valid" in dir_lower:
            split = "val"

        return metric, split

    records = []
    for event_path in sorted(event_files):
        event_dir = os.path.dirname(event_path)
        ea = event_accumulator.EventAccumulator(event_path)
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])
        for tag in scalar_tags:
            metric, split = infer_metric_and_split(tag, event_dir)
            if metric is None:
                continue
            events = ea.Scalars(tag)
            if not events:
                continue
            split_label = split if split is not None else "unspecified"
            for event in events:
                records.append(
                    {
                        "run": run_name,
                        "epoch": event.step,
                        "value": event.value,
                        "split": split_label,
                        "metric": metric,
                    }
                )

    return pd.DataFrame(records, columns=["run", "epoch", "value", "split", "metric"])


def plot_learning_curves(run_name: str, log_root: str = "runs"):
    """
    Load TensorBoard scalars for a given run and plot learning curves.
    """
    run_df = load_run_scalars(run_name, log_root=log_root)

    def select_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        subset = df[df["metric"] == metric]
        if subset.empty:
            return pd.DataFrame(columns=df.columns)
        return subset.copy()

    loss_df = select_metric(run_df, "loss")
    acc_df = select_metric(run_df, "accuracy")

    # Plot with seaborn & viridis
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def plot_metric(ax: plt.Axes, df: pd.DataFrame, ylabel: str, empty_msg: str):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if df.empty:
            ax.text(
                0.5,
                0.5,
                empty_msg,
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            return

        hue_levels = sorted(df["split"].unique())
        palette = sns.color_palette("viridis", max(2, len(hue_levels)))
        sns.lineplot(
            data=df,
            x="epoch",
            y="value",
            hue="split",
            hue_order=hue_levels,
            ax=ax,
            palette=palette,
            errorbar=None
        )
        ax.legend(title="Split")

    axes[0].set_title(f"{run_name} – Loss")
    plot_metric(axes[0], loss_df, "Loss", "No loss scalars found")

    axes[1].set_title(f"{run_name} – Accuracy")
    plot_metric(axes[1], acc_df, "Accuracy", "No accuracy scalars found")

    plt.tight_layout()
    plt.show()


def build_grid_frame(runs):
    rows = []
    for combo in runs:
        label = combo["label"]
        for epoch_metrics in combo["history"]:
            epoch = epoch_metrics["epoch"]
            rows.append(
                {
                    "combo": label,
                    "epoch": epoch,
                    "split": "train",
                    "metric": "loss",
                    "value": epoch_metrics["train_loss"],
                }
            )
            rows.append(
                {
                    "combo": label,
                    "epoch": epoch,
                    "split": "train",
                    "metric": "accuracy",
                    "value": epoch_metrics["train_accuracy"],
                }
            )
            if "val_loss" in epoch_metrics:
                rows.append(
                    {
                        "combo": label,
                        "epoch": epoch,
                        "split": "val",
                        "metric": "loss",
                        "value": epoch_metrics["val_loss"],
                    }
                )
            if "val_accuracy" in epoch_metrics:
                rows.append(
                    {
                        "combo": label,
                        "epoch": epoch,
                        "split": "val",
                        "metric": "accuracy",
                        "value": epoch_metrics["val_accuracy"],
                    }
                )
    return pd.DataFrame(rows)

# Best-epoch helpers

def best_epoch(history, metric="val_accuracy"):
    """
    Return the metrics dict of the best epoch according to `metric`.
    Falls back to `train_accuracy` if val metric is missing.
    """
    key = metric if history and metric in history[0] else "train_accuracy"
    return max(history, key=lambda m: m.get(key, float("-inf")))

def report_best(name, history, metric="val_accuracy"):
    """
    Print a one-liner summary for the best epoch.
    """
    b = best_epoch(history, metric)
    print(
        f"{name}: best@epoch {b['epoch']} | "
        f"val_acc={b.get('val_accuracy', float('nan')):.3%} | "
        f"train_acc={b.get('train_accuracy', float('nan')):.3%} | "
        f"val_loss={b.get('val_loss', float('nan')):.4f} | "
        f"train_loss={b.get('train_loss', float('nan')):.4f}"
    )
    return b

def summarize_runs(named_histories, metric="val_accuracy"):
    """
    named_histories: list of (name, history)
    Prints best-epoch lines sorted by the chosen metric.
    """
    rows = []
    for name, hist in named_histories:
        b = best_epoch(hist, metric)
        rows.append((
            name,
            b["epoch"],
            b.get("val_accuracy", float("nan")),
            b.get("train_accuracy", float("nan")),
            b.get("val_loss", float("nan")),
            b.get("train_loss", float("nan")),
        ))
    # sort by val_accuracy desc
    rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1), reverse=True)  # handle NaN

    print("\n=== Best-epoch summary (sorted by val_acc) ===")
    for name, ep, vacc, tacc, vloss, tloss in rows:
        print(
            f"{name:40s} | epoch {ep:3d} | "
            f"val_acc {vacc:7.3%} | train_acc {tacc:7.3%} | "
            f"val_loss {vloss:8.4f} | train_loss {tloss:8.4f}"
        )
    return rows

