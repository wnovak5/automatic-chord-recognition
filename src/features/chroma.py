from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def normalize_chroma(chroma: np.ndarray, method: str = "l2") -> np.ndarray:
    """Normalize each chroma frame. chroma shape: (12, T)."""
    if method == "l2":
        norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return chroma / norms
    elif method == "max":
        maxvals = np.max(np.abs(chroma), axis=0, keepdims=True)
        maxvals = np.where(maxvals == 0, 1.0, maxvals)
        return chroma / maxvals
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def median_filter_chroma(chroma: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter along the time axis. chroma shape: (12, T)."""
    return median_filter(chroma, size=(1, kernel_size))


def build_context_windows(chroma: np.ndarray, context: int = 7) -> np.ndarray:
    """
    Stack neighboring frames into context windows.

    chroma: (12, T)
    returns: (T, 12 * (2*context + 1)) float32
    """
    n_bins, T = chroma.shape
    width = 2 * context + 1
    padded = np.pad(chroma, ((0, 0), (context, context)), mode="constant")
    windows = np.stack(
        [padded[:, t : t + T] for t in range(width)], axis=0
    )  # (width, 12, T)
    windows = windows.transpose(2, 0, 1).reshape(T, width * n_bins)
    return windows.astype(np.float32)


def prepare_track_features(
    chroma: np.ndarray,
    context: int = 7,
    normalize: str = "l2",
) -> np.ndarray:
    """Normalize chroma then build context windows. Returns (T, 12*(2*context+1)) float32."""
    chroma_norm = normalize_chroma(chroma, method=normalize)
    return build_context_windows(chroma_norm, context=context)
