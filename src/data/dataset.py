from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.load_data import load_training_example
from src.features.chroma import prepare_track_features
from src.features.vocab import ChordVocab


class ChordDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Framewise chord dataset built from one or more AAM tracks."""

    def __init__(
        self,
        track_ids: list[str],
        root: str | Path | None = None,
        vocab: ChordVocab | None = None,
        context: int = 7,
    ) -> None:
        if not track_ids:
            raise ValueError("track_ids must contain at least one track id")

        self.track_ids = list(track_ids)
        self.root = Path(root) if root is not None else None
        self.vocab = vocab or ChordVocab.from_default()
        self.context = context

        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []

        for track_id in self.track_ids:
            example = load_training_example(track_id, root=self.root)
            feature_blocks.append(prepare_track_features(example["chroma"], context=context))
            label_blocks.append(self.vocab.encode(example["frame_labels"]))

        features = np.vstack(feature_blocks).astype(np.float32, copy=False)
        labels = np.concatenate(label_blocks).astype(np.int64, copy=False)

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def create_dataloaders(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    root: str | Path | None = None,
    vocab: ChordVocab | None = None,
    context: int = 7,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders using the notebook's expected API."""

    train_dataset = ChordDataset(train_ids, root=root, vocab=vocab, context=context)
    val_dataset = ChordDataset(val_ids, root=root, vocab=vocab, context=context)
    test_dataset = ChordDataset(test_ids, root=root, vocab=vocab, context=context)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
