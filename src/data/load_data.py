from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Literal

import arff
import librosa
import numpy as np
import pandas as pd


DEFAULT_AAM_ROOT = Path("data/raw/tinyAAM")


@dataclass(frozen=True)
class AAMTrackPaths:
    """Filesystem locations for one AAM track and its annotations."""

    track_id: str
    mix_audio: Path
    stems: list[Path]
    beatinfo: Path
    onsets: Path
    segments: Path


@dataclass(frozen=True)
class AAMTrackData:
    """Loaded mix audio plus the three annotation tables for one track."""

    track_id: str
    audio: np.ndarray
    sample_rate: int
    beatinfo: pd.DataFrame
    onsets: pd.DataFrame
    segments: pd.DataFrame


def _resolve_root(root: str | Path | None = None) -> Path:
    """Use the caller-provided dataset root or fall back to tinyAAM."""

    dataset_root = Path(root) if root is not None else DEFAULT_AAM_ROOT
    dataset_root = dataset_root.expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"AAM dataset root does not exist: {dataset_root}")
    return dataset_root


def _require_file(path: Path) -> Path:
    """Fail early with a clear message if an expected dataset path is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    return path


def _decode_attribute_name(attribute: object) -> str:
    """Extract the column name from liac-arff's attribute representation."""

    if isinstance(attribute, (list, tuple)) and attribute:
        return str(attribute[0])
    return str(attribute)


def load_arff_table(path: str | Path) -> pd.DataFrame:
    """
    Load an annotation file into a DataFrame.

    AAM mostly uses ARFF-like files, but some files in this dataset do not fully
    follow the ARFF spec, so we fall back to a custom parser when needed.
    """

    arff_path = Path(path)
    try:
        with arff_path.open("r", encoding="utf-8") as handle:
            contents = arff.load(handle)
    except arff.BadLayout:
        return _load_aam_annotation_table(arff_path)

    columns = [_decode_attribute_name(attribute) for attribute in contents["attributes"]]
    return pd.DataFrame(contents["data"], columns=columns)


def _load_aam_annotation_table(path: Path) -> pd.DataFrame:
    """
    Parse the simplified annotation format used by some AAM files.

    In particular, `beatinfo` files can omit the `@DATA` marker that strict ARFF
    parsers expect, but they are still easy to read as CSV rows after the header.
    """

    columns: list[str] = []
    rows: list[list[str]] = []
    in_data = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("@RELATION"):
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                columns.append(_parse_attribute_name(line))
                continue
            if line.upper().startswith("@DATA"):
                in_data = True
                continue
            if line.startswith("@"):
                continue
            if columns and (in_data or not line.startswith("@")):
                rows.append(next(csv.reader([line], quotechar="'", skipinitialspace=True)))

    return pd.DataFrame(rows, columns=columns)


def _parse_attribute_name(line: str) -> str:
    """Extract the human-readable column name from one @ATTRIBUTE line."""

    _, remainder = line.split(" ", 1)
    remainder = remainder.strip()
    if remainder.startswith("'"):
        end = remainder.find("'", 1)
        return remainder[1:end]
    return remainder.split()[0]


def _normalize_chord_name(chord: str | None) -> str:
    """Map empty or missing chord labels to a neutral no-chord token."""

    if chord is None:
        return "N"
    chord_name = str(chord).strip().strip("'").strip('"')
    return chord_name if chord_name else "N"


def _prepare_annotation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize annotation tables after parsing.

    This converts time columns to numeric values and strips quotes from chord/key
    labels so downstream code can treat them as ordinary strings.
    """

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    if "Start time in seconds" in cleaned.columns:
        cleaned["Start time in seconds"] = pd.to_numeric(
            cleaned["Start time in seconds"], errors="coerce"
        )
    if "Onset time in seconds" in cleaned.columns:
        cleaned["Onset time in seconds"] = pd.to_numeric(
            cleaned["Onset time in seconds"], errors="coerce"
        )
    if "Chord name" in cleaned.columns:
        cleaned["Chord name"] = cleaned["Chord name"].map(_normalize_chord_name)
    if "Key" in cleaned.columns:
        cleaned["Key"] = cleaned["Key"].map(_normalize_chord_name)

    return cleaned


class AAMDataset:
    """Convenience wrapper around the tinyAAM directory layout."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = _resolve_root(root)
        self.annotation_dir = _require_file(self.root / "annotations")
        self.mix_dir = _require_file(self.root / "audio-mixes-mp3")
        self.stem_dir = _require_file(self.root / "audio-multitracks-mp3")

    def track_ids(self) -> list[str]:
        """List all track ids that have a mix file available."""

        return sorted(path.name.split("_")[0] for path in self.mix_dir.glob("*_mix.mp3"))

    def _paths_for_track(self, track_id: str) -> AAMTrackPaths:
        """Build the expected file paths for one track id."""

        stem_paths = sorted(self.stem_dir.glob(f"{track_id}_*.mp3"))
        return AAMTrackPaths(
            track_id=track_id,
            mix_audio=_require_file(self.mix_dir / f"{track_id}_mix.mp3"),
            stems=stem_paths,
            beatinfo=_require_file(self.annotation_dir / f"{track_id}_beatinfo.arff"),
            onsets=_require_file(self.annotation_dir / f"{track_id}_onsets.arff"),
            segments=_require_file(self.annotation_dir / f"{track_id}_segments.arff"),
        )

    def list_tracks(self) -> pd.DataFrame:
        """Return a table summarizing the tracks currently available on disk."""

        rows: list[dict[str, object]] = []
        for track_id in self.track_ids():
            paths = self._paths_for_track(track_id)
            rows.append(
                {
                    "track_id": track_id,
                    "mix_audio": str(paths.mix_audio),
                    "num_stems": len(paths.stems),
                    "beatinfo": str(paths.beatinfo),
                    "onsets": str(paths.onsets),
                    "segments": str(paths.segments),
                }
            )
        return pd.DataFrame(rows)

    def load_annotations(self, track_id: str) -> dict[str, pd.DataFrame]:
        """Load beat, onset, and segment annotations for a track."""

        paths = self._paths_for_track(track_id)
        return {
            "beatinfo": _prepare_annotation_frame(load_arff_table(paths.beatinfo)),
            "onsets": _prepare_annotation_frame(load_arff_table(paths.onsets)),
            "segments": _prepare_annotation_frame(load_arff_table(paths.segments)),
        }

    def load_audio(
        self,
        track_id: str,
        source: Literal["mix", "stems"] = "mix",
        sr: int | None = None,
        mono: bool = True,
    ) -> tuple[np.ndarray, int] | list[tuple[str, np.ndarray, int]]:
        """
        Load either the full mix or each individual stem.

        The mix path is what you will usually want for chord-recognition training.
        """

        paths = self._paths_for_track(track_id)

        if source == "mix":
            audio, sample_rate = librosa.load(paths.mix_audio, sr=sr, mono=mono)
            return audio, sample_rate

        stem_audio: list[tuple[str, np.ndarray, int]] = []
        for stem_path in paths.stems:
            audio, sample_rate = librosa.load(stem_path, sr=sr, mono=mono)
            stem_audio.append((stem_path.stem, audio, sample_rate))
        return stem_audio

    def load_track(
        self,
        track_id: str,
        sr: int | None = None,
        mono: bool = True,
    ) -> AAMTrackData:
        """Load the mix audio together with all annotations for one track."""

        audio, sample_rate = self.load_audio(track_id=track_id, source="mix", sr=sr, mono=mono)
        annotations = self.load_annotations(track_id)
        return AAMTrackData(
            track_id=track_id,
            audio=audio,
            sample_rate=sample_rate,
            beatinfo=annotations["beatinfo"],
            onsets=annotations["onsets"],
            segments=annotations["segments"],
        )


def extract_chroma_features(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> np.ndarray:
    """
    Convert raw audio into a 12-bin chroma representation.

    Chroma is a common input feature for chord recognition because it collapses
    pitches into pitch classes (C through B) while largely ignoring octave.
    """

    return librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )


def beat_annotations_to_intervals(
    beatinfo: pd.DataFrame,
    audio_duration: float | None = None,
) -> pd.DataFrame:
    """
    Turn beatwise chord annotations into explicit [start, end) intervals.

    The AAM beatinfo file stores one chord label per beat timestamp; this helper
    converts that into intervals that can be aligned with frame-based features.
    """

    required_columns = {"Start time in seconds", "Chord name"}
    missing = required_columns - set(beatinfo.columns)
    if missing:
        raise ValueError(f"beatinfo is missing required columns: {sorted(missing)}")

    ordered = (
        beatinfo.loc[:, ["Start time in seconds", "Chord name"]]
        .rename(columns={"Start time in seconds": "start", "Chord name": "chord"})
        .sort_values("start")
        .reset_index(drop=True)
    )
    ordered["end"] = ordered["start"].shift(-1)

    if audio_duration is not None:
        ordered.loc[ordered.index[-1], "end"] = audio_duration

    ordered = ordered.dropna(subset=["start", "end"]).copy()
    ordered["start"] = ordered["start"].astype(float)
    ordered["end"] = ordered["end"].astype(float)
    ordered["chord"] = ordered["chord"].map(_normalize_chord_name)
    return ordered.loc[ordered["end"] > ordered["start"]].reset_index(drop=True)


def frame_times(
    num_frames: int,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute the timestamp for each feature frame."""

    return librosa.frames_to_time(np.arange(num_frames), sr=sample_rate, hop_length=hop_length)


def frame_labels_from_intervals(
    intervals: pd.DataFrame,
    times: np.ndarray,
    default_label: str = "N",
) -> np.ndarray:
    """
    Assign a chord label to each feature frame timestamp.

    Frames that do not fall inside any labeled interval keep the default label.
    """

    labels = np.full(times.shape, default_label, dtype=object)
    for row in intervals.itertuples(index=False):
        mask = (times >= row.start) & (times < row.end)
        labels[mask] = row.chord
    return labels


def load_training_example(
    track_id: str,
    root: str | Path | None = None,
    sr: int = 22050,
    hop_length: int = 512,
) -> dict[str, object]:
    """
    End-to-end helper for the common training path.

    This loads one track, computes chroma features, and aligns beat annotations
    with the resulting frame grid so you get inputs and labels together.
    """

    dataset = AAMDataset(root)
    track = dataset.load_track(track_id, sr=sr, mono=True)
    chroma = extract_chroma_features(track.audio, sample_rate=track.sample_rate, hop_length=hop_length)
    times = frame_times(chroma.shape[1], sample_rate=track.sample_rate, hop_length=hop_length)
    intervals = beat_annotations_to_intervals(
        track.beatinfo, audio_duration=len(track.audio) / track.sample_rate
    )
    labels = frame_labels_from_intervals(intervals, times)

    return {
        "track_id": track.track_id,
        "audio": track.audio,
        "sample_rate": track.sample_rate,
        "chroma": chroma,
        "frame_times": times,
        "frame_labels": labels,
        "beatinfo": track.beatinfo,
        "onsets": track.onsets,
        "segments": track.segments,
    }
