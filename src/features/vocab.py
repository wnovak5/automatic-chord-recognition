from __future__ import annotations

from pathlib import Path

import numpy as np


CHORD_VOCAB: list[str] = [
    "N.C.",
    "Cmaj", "C#maj", "Dmaj", "D#maj", "Emaj", "Fmaj",
    "F#maj", "Gmaj", "G#maj", "Amaj", "A#maj", "Bmaj",
    "Cmin", "C#min", "Dmin", "D#min", "Emin", "Fmin",
    "F#min", "Gmin", "G#min", "Amin", "A#min", "Bmin",
]


class ChordVocab:
    def __init__(self, vocab: list[str]) -> None:
        self._vocab = list(vocab)
        self._label_to_idx: dict[str, int] = {label: i for i, label in enumerate(self._vocab)}

    @classmethod
    def from_default(cls) -> "ChordVocab":
        return cls(CHORD_VOCAB)

    @property
    def num_classes(self) -> int:
        return len(self._vocab)

    def encode(self, labels: np.ndarray) -> np.ndarray:
        """Map string label array to int64 index array. Unknown labels map to 0 (N.C.)."""
        return np.array(
            [self._label_to_idx.get(str(label), 0) for label in labels],
            dtype=np.int64,
        )

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Map int64 index array back to string label array."""
        return np.array([self._vocab[int(i)] for i in indices], dtype=object)

    def __len__(self) -> int:
        return self.num_classes


def build_vocab_from_dataset(root: str | Path | None = None) -> list[str]:
    """Scan all tracks and return the sorted unique chord labels found."""
    from src.data.load_data import AAMDataset, load_training_example

    dataset = AAMDataset(root)
    all_labels: set[str] = set()
    for track_id in dataset.track_ids():
        example = load_training_example(track_id, root=root)
        all_labels.update(example["frame_labels"])
    return sorted(all_labels)
