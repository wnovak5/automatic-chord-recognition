from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data import (
    AAMDataset,
    beat_annotations_to_intervals,
    extract_chroma_features,
    frame_labels_from_intervals,
    frame_times,
)


NO_CHORD_LABEL = "N"

# Restricting the first-pass label space to major/minor/no-chord keeps the
# classification problem manageable while the end-to-end pipeline is still being built.
DEFAULT_CHORD_VOCABULARY = tuple(
    [NO_CHORD_LABEL]
    + [f"{root}:maj" for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]
    + [f"{root}:min" for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]
)

# Map enharmonic spellings into a single canonical spelling so the model does not
# waste classes on notes that are musically equivalent for this first pass.
ENHARMONIC_ROOTS = {
    "CB": "B",
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
    "E#": "F",
    "B#": "C",
}


@dataclass(frozen=True)
class EncodedLabelSet:
    """A reusable mapping between chord strings and integer class ids."""

    labels: list[str]
    label_to_index: dict[str, int]


@dataclass(frozen=True)
class PreprocessedTrack:
    """Model-ready representation of one track after preprocessing."""

    track_id: str
    features: np.ndarray
    frame_times: np.ndarray
    chord_labels: np.ndarray
    encoded_labels: np.ndarray
    sample_rate: int
    hop_length: int


def canonicalize_root(root: str) -> str:
    """Normalize a chord root into a single uppercase sharp-based spelling."""

    normalized = root.strip().upper()
    return ENHARMONIC_ROOTS.get(normalized, normalized)


def simplify_chord_label(chord_label: str | None, no_chord_label: str = NO_CHORD_LABEL) -> str:
    """
    Collapse raw chord labels into a small major/minor/no-chord vocabulary.

    Examples:
    - `Fmaj` -> `F:maj`
    - `Dmin` -> `D:min`
    - `Bb:maj7` -> `A#:maj`
    - empty labels -> `N`
    """

    if chord_label is None:
        return no_chord_label

    chord = str(chord_label).strip().strip("'").strip('"')
    if not chord or chord.upper() in {"N", "NO_CHORD"}:
        return no_chord_label

    if ":" in chord:
        root, quality = chord.split(":", 1)
        root = canonicalize_root(root)
        quality = quality.lower()

        # Reduce richer chord qualities to the basic triad categories used by the
        # current model label space.
        if quality.startswith("min"):
            return f"{root}:min"
        if quality.startswith("maj"):
            return f"{root}:maj"
        if quality.startswith("dim") or quality.startswith("aug") or quality.startswith("sus"):
            return no_chord_label

    upper = chord.upper()
    if upper.endswith("MIN"):
        root = canonicalize_root(chord[:-3])
        return f"{root}:min"
    if upper.endswith("MAJ"):
        root = canonicalize_root(chord[:-3])
        return f"{root}:maj"

    # If the label has an unfamiliar format, keep the pipeline stable by treating it
    # as no-chord rather than inventing a possibly inconsistent class.
    return no_chord_label


def simplify_chord_sequence(
    chord_labels: np.ndarray | list[str],
    no_chord_label: str = NO_CHORD_LABEL,
) -> np.ndarray:
    """Apply major/minor simplification to every chord label in a sequence."""

    return np.asarray(
        [simplify_chord_label(chord_label, no_chord_label=no_chord_label) for chord_label in chord_labels],
        dtype=object,
    )


def build_label_set(labels: list[str] | tuple[str, ...] = DEFAULT_CHORD_VOCABULARY) -> EncodedLabelSet:
    """Create a stable string-to-index mapping for model targets."""

    ordered_labels = list(labels)
    return EncodedLabelSet(
        labels=ordered_labels,
        label_to_index={label: index for index, label in enumerate(ordered_labels)},
    )


def encode_chord_labels(
    chord_labels: np.ndarray | list[str],
    label_set: EncodedLabelSet | None = None,
    no_chord_label: str = NO_CHORD_LABEL,
) -> np.ndarray:
    """
    Convert chord strings into integer class ids.

    Labels are simplified first so the encoded output matches the chosen model vocabulary.
    """

    encoding = label_set or build_label_set()
    simplified = simplify_chord_sequence(chord_labels, no_chord_label=no_chord_label)

    encoded: list[int] = []
    for label in simplified:
        if label not in encoding.label_to_index:
            raise ValueError(
                f"Chord label {label!r} is not present in the label set. "
                "Update the vocabulary or simplify the label differently."
            )
        encoded.append(encoding.label_to_index[label])

    return np.asarray(encoded, dtype=np.int64)


def preprocess_track(
    track_id: str,
    root: str | Path | None = None,
    sr: int = 22050,
    hop_length: int = 512,
    feature_type: Literal["chroma"] = "chroma",
    label_set: EncodedLabelSet | None = None,
) -> PreprocessedTrack:
    """
    Load one track and turn it into model-ready features and encoded labels.

    This is the per-track preprocessing entry point you can call during experiments,
    dataset building, or debugging.
    """

    dataset = AAMDataset(root)
    track = dataset.load_track(track_id, sr=sr, mono=True)

    if feature_type != "chroma":
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    # Compute frame-based input features from the waveform.
    features = extract_chroma_features(track.audio, track.sample_rate, hop_length=hop_length)
    times = frame_times(features.shape[1], track.sample_rate, hop_length=hop_length)

    # Align beatwise chord annotations to the feature frame grid.
    intervals = beat_annotations_to_intervals(
        track.beatinfo,
        audio_duration=len(track.audio) / track.sample_rate,
    )
    raw_frame_labels = frame_labels_from_intervals(intervals, times, default_label=NO_CHORD_LABEL)
    simplified_labels = simplify_chord_sequence(raw_frame_labels)
    encoded_labels = encode_chord_labels(simplified_labels, label_set=label_set)

    return PreprocessedTrack(
        track_id=track_id,
        features=features,
        frame_times=times,
        chord_labels=simplified_labels,
        encoded_labels=encoded_labels,
        sample_rate=track.sample_rate,
        hop_length=hop_length,
    )


def preprocess_dataset(
    root: str | Path | None = None,
    track_ids: list[str] | None = None,
    sr: int = 22050,
    hop_length: int = 512,
    feature_type: Literal["chroma"] = "chroma",
    label_set: EncodedLabelSet | None = None,
) -> dict[str, object]:
    """
    Preprocess many tracks into a single dataset package.

    The returned dictionary keeps per-track arrays grouped together so you can inspect
    individual songs while still having one object to pass into later training code.
    """

    dataset = AAMDataset(root)
    selected_track_ids = track_ids or dataset.track_ids()
    encoding = label_set or build_label_set()

    track_examples: list[PreprocessedTrack] = []
    metadata_rows: list[dict[str, object]] = []

    for track_id in selected_track_ids:
        example = preprocess_track(
            track_id=track_id,
            root=root,
            sr=sr,
            hop_length=hop_length,
            feature_type=feature_type,
            label_set=encoding,
        )
        track_examples.append(example)
        metadata_rows.append(
            {
                "track_id": example.track_id,
                "num_frames": example.features.shape[1],
                "num_feature_bins": example.features.shape[0],
                "sample_rate": example.sample_rate,
                "hop_length": example.hop_length,
            }
        )

    return {
        "tracks": track_examples,
        "label_set": encoding,
        "metadata": pd.DataFrame(metadata_rows),
    }


def stack_frame_dataset(track_examples: list[PreprocessedTrack]) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate preprocessed tracks into framewise matrices for baseline models.

    Returns:
    - `X`: shape `(num_frames_total, num_feature_bins)`
    - `y`: shape `(num_frames_total,)`
    """

    if not track_examples:
        raise ValueError("track_examples must contain at least one preprocessed track")

    # Features are stored as (feature_bins, time_frames); transpose so each row is a frame.
    feature_rows = [track.features.T for track in track_examples]
    label_rows = [track.encoded_labels for track in track_examples]
    return np.vstack(feature_rows), np.concatenate(label_rows)
