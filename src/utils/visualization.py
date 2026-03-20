from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


def plot_chroma(
    chroma: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    title: str = "Chromagram",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a chromagram using librosa.display.specshow."""
    import librosa.display

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis="time", y_axis="chroma", ax=ax)
    ax.set_title(title)
    return ax


def plot_chord_prediction(
    frame_times: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    title: str = "Chord Prediction",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Two-row color-coded strip showing ground truth (top) and predictions (bottom)."""
    all_labels = sorted(set(list(true_labels) + list(pred_labels)))
    cmap = plt.get_cmap("tab20", len(all_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(all_labels)}

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 2))

    dt = np.diff(frame_times, append=frame_times[-1] + (frame_times[-1] - frame_times[-2]))

    for i, t in enumerate(frame_times):
        w = dt[i]
        ax.barh(1.5, w, left=t, height=0.8, color=label_to_color[true_labels[i]])
        ax.barh(0.5, w, left=t, height=0.8, color=label_to_color[pred_labels[i]])

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Predicted", "Ground Truth"])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    return ax


def plot_training_curves(
    history: dict,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot loss and accuracy training curves."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        axes = ax

    epochs = range(1, len(history["train_losses"]) + 1)

    axes[0].plot(epochs, history["train_losses"], label="Train")
    axes[0].plot(epochs, history["val_losses"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accs"], label="Train")
    axes[1].plot(epochs, history["val_accs"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    return axes


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of a confusion matrix DataFrame."""
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(cm_df, ax=ax, fmt="d", cmap="Blues", linewidths=0.5)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return ax
