from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.load_data import load_training_example
from src.features.chroma import prepare_track_features
from src.features.vocab import ChordVocab

PAD_LABEL = -100


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


class ChordSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Fixed-length chroma sequences for recurrent chord models."""

    def __init__(
        self,
        track_ids: list[str],
        root: str | Path | None = None,
        vocab: ChordVocab | None = None,
        sequence_length: int = 128,
    ) -> None:
        if not track_ids:
            raise ValueError("track_ids must contain at least one track id")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        self.track_ids = list(track_ids)
        self.root = Path(root) if root is not None else None
        self.vocab = vocab or ChordVocab.from_default()
        self.sequence_length = sequence_length
        self.sequences: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.lengths: list[torch.Tensor] = []

        for track_id in self.track_ids:
            example = load_training_example(track_id, root=self.root)
            features = prepare_track_features(example["chroma"], context=0)
            labels = self.vocab.encode(example["frame_labels"])

            for start in range(0, len(labels), sequence_length):
                end = min(start + sequence_length, len(labels))
                self.sequences.append(torch.from_numpy(features[start:end].astype(np.float32, copy=False)))
                self.labels.append(torch.from_numpy(labels[start:end].astype(np.int64, copy=False)))
                self.lengths.append(torch.tensor(end - start, dtype=torch.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index], self.lengths[index]


def collate_sequence_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length chord sequences in one minibatch."""

    features, labels, lengths = zip(*batch)
    max_len = max(int(length.item()) for length in lengths)
    batch_size = len(batch)
    feature_dim = features[0].shape[-1]

    padded_features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    padded_labels = torch.full((batch_size, max_len), PAD_LABEL, dtype=torch.int64)
    length_tensor = torch.tensor([int(length.item()) for length in lengths], dtype=torch.int64)

    for index, (sequence, target, length) in enumerate(zip(features, labels, lengths)):
        valid_steps = int(length.item())
        padded_features[index, :valid_steps] = sequence
        padded_labels[index, :valid_steps] = target

    return padded_features, padded_labels, length_tensor


def create_sequence_dataloaders(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    root: str | Path | None = None,
    vocab: ChordVocab | None = None,
    sequence_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders for recurrent sequence models."""

    train_dataset = ChordSequenceDataset(train_ids, root=root, vocab=vocab, sequence_length=sequence_length)
    val_dataset = ChordSequenceDataset(val_ids, root=root, vocab=vocab, sequence_length=sequence_length)
    test_dataset = ChordSequenceDataset(test_ids, root=root, vocab=vocab, sequence_length=sequence_length)

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_sequence_batch,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sequence_batch,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sequence_batch,
        ),
    )
