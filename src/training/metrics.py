from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.features.vocab import ChordVocab


def frame_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of frames with correct chord prediction."""
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: ChordVocab,
) -> pd.DataFrame:
    """Return a DataFrame with per-chord accuracy and support."""
    rows = []
    for idx in range(vocab.num_classes):
        mask = y_true == idx
        support = int(mask.sum())
        if support > 0:
            acc = float(np.mean(y_pred[mask] == idx))
        else:
            acc = float("nan")
        rows.append({"chord": vocab.decode(np.array([idx]))[0], "accuracy": acc, "support": support})
    return pd.DataFrame(rows)


def confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vocab: ChordVocab,
) -> pd.DataFrame:
    """Return a labeled confusion matrix as a DataFrame."""
    labels = list(range(vocab.num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    chord_names = vocab.decode(np.array(labels))
    return pd.DataFrame(cm, index=chord_names, columns=chord_names)
