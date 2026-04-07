from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch

from src.data.load_data import extract_chroma_features, frame_times
from src.features.chroma import prepare_track_features
from src.features.vocab import ChordVocab
from src.models.rnn import ChordLSTM
from src.training.trainer import load_checkpoint


DEFAULT_CHECKPOINT = Path("pretrained/best_lstm_checkpoint.pt")
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_LENGTH = 512
DEFAULT_CONTEXT = 0
DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_MIN_SEGMENT_SECONDS = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chord inference on an audio file using the committed pretrained LSTM checkpoint."
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to an input audio file such as .mp3 or .wav.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a saved training checkpoint. Defaults to pretrained/best_lstm_checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for inference outputs. Defaults to inference/<audio-stem>/.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device to use. Defaults to auto.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate used before feature extraction. Defaults to 22050 Hz.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop length for chroma extraction and timestamping. Defaults to 512.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT,
        help="Neighboring chroma-frame context on each side. The current pretrained model expects 0.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Odd frame window size for majority-vote smoothing. Set to 1 to disable smoothing.",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SECONDS,
        help="Minimum chord-segment duration after smoothing. Shorter segments are merged into neighbors.",
    )
    args = parser.parse_args()

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    if args.hop_length <= 0:
        parser.error("--hop-length must be positive")
    if args.context < 0:
        parser.error("--context must be non-negative")
    if args.smoothing_window <= 0:
        parser.error("--smoothing-window must be positive")
    if args.smoothing_window % 2 == 0:
        parser.error("--smoothing-window must be odd")
    if args.min_segment_seconds < 0.0:
        parser.error("--min-segment-seconds must be non-negative")

    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_output_dir(audio_path: Path) -> Path:
    return Path("inference") / audio_path.stem


def load_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio


def build_model(checkpoint: dict[str, Any], device: torch.device) -> ChordLSTM:
    model_kwargs = checkpoint.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise KeyError("Checkpoint is missing model_kwargs")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint is missing model_state_dict")

    model = ChordLSTM(**model_kwargs)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_vocab(checkpoint: dict[str, Any]) -> ChordVocab:
    checkpoint_vocab = checkpoint.get("vocab")
    if isinstance(checkpoint_vocab, list):
        return ChordVocab(checkpoint_vocab)
    return ChordVocab.from_default()


def predict_labels(
    model: ChordLSTM,
    vocab: ChordVocab,
    audio: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    context: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    chroma = extract_chroma_features(audio, sample_rate=sample_rate, hop_length=hop_length)
    features = prepare_track_features(chroma, context=context)
    times = frame_times(chroma.shape[1], sample_rate=sample_rate, hop_length=hop_length)

    x = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.int64, device=device)
    prediction_idx = model.predict(x, lengths=lengths).squeeze(0).cpu().numpy()
    prediction_labels = vocab.decode(prediction_idx)
    return times, prediction_labels


def frame_predictions_to_df(
    frame_times_seconds: np.ndarray,
    labels: np.ndarray,
    *,
    audio_duration_seconds: float,
) -> pd.DataFrame:
    if frame_times_seconds.size == 0:
        return pd.DataFrame(columns=["start_time_seconds", "end_time_seconds", "predicted_label"])

    end_times = np.empty_like(frame_times_seconds, dtype=float)
    end_times[:-1] = frame_times_seconds[1:]
    end_times[-1] = audio_duration_seconds

    return pd.DataFrame(
        {
            "start_time_seconds": frame_times_seconds.astype(float),
            "end_time_seconds": end_times,
            "predicted_label": labels.astype(str),
        }
    )


def smooth_label_indices(prediction_idx: np.ndarray, num_classes: int, window_size: int) -> np.ndarray:
    if window_size <= 1 or prediction_idx.size == 0:
        return prediction_idx.copy()

    radius = window_size // 2
    smoothed = np.empty_like(prediction_idx)
    for index in range(prediction_idx.size):
        start = max(0, index - radius)
        end = min(prediction_idx.size, index + radius + 1)
        window = prediction_idx[start:end]
        counts = np.bincount(window, minlength=num_classes)
        smoothed[index] = int(np.argmax(counts))
    return smoothed


def merge_consecutive_frames(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=["start_time_seconds", "end_time_seconds", "predicted_label", "duration_seconds"])

    segments: list[dict[str, Any]] = []
    current_label = str(frame_df.iloc[0]["predicted_label"])
    current_start = float(frame_df.iloc[0]["start_time_seconds"])
    current_end = float(frame_df.iloc[0]["end_time_seconds"])

    for row in frame_df.iloc[1:].itertuples(index=False):
        row_label = str(row.predicted_label)
        row_end = float(row.end_time_seconds)
        if row_label == current_label:
            current_end = row_end
            continue

        segments.append(
            {
                "start_time_seconds": current_start,
                "end_time_seconds": current_end,
                "predicted_label": current_label,
                "duration_seconds": current_end - current_start,
            }
        )
        current_label = row_label
        current_start = float(row.start_time_seconds)
        current_end = row_end

    segments.append(
        {
            "start_time_seconds": current_start,
            "end_time_seconds": current_end,
            "predicted_label": current_label,
            "duration_seconds": current_end - current_start,
        }
    )
    return pd.DataFrame(segments)


def merge_short_segments(segment_df: pd.DataFrame, min_segment_seconds: float) -> pd.DataFrame:
    if segment_df.empty or min_segment_seconds <= 0.0:
        return segment_df.reset_index(drop=True)

    segments = [dict(row._asdict()) for row in segment_df.itertuples(index=False)]
    index = 0
    while index < len(segments):
        segment = segments[index]
        if float(segment["duration_seconds"]) >= min_segment_seconds or len(segments) == 1:
            index += 1
            continue

        if index == 0:
            target_index = 1
        elif index == len(segments) - 1:
            target_index = index - 1
        else:
            left_duration = float(segments[index - 1]["duration_seconds"])
            right_duration = float(segments[index + 1]["duration_seconds"])
            target_index = index - 1 if left_duration >= right_duration else index + 1

        if target_index < index:
            segments[target_index]["end_time_seconds"] = segment["end_time_seconds"]
            segments[target_index]["duration_seconds"] = (
                float(segments[target_index]["end_time_seconds"]) - float(segments[target_index]["start_time_seconds"])
            )
            del segments[index]
            index = max(target_index, 0)
        else:
            segments[target_index]["start_time_seconds"] = segment["start_time_seconds"]
            segments[target_index]["duration_seconds"] = (
                float(segments[target_index]["end_time_seconds"]) - float(segments[target_index]["start_time_seconds"])
            )
            del segments[index]

    return pd.DataFrame(segments)


def save_metadata(
    output_dir: Path,
    *,
    audio_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    sample_rate: int,
    hop_length: int,
    context: int,
    smoothing_window: int,
    min_segment_seconds: float,
    num_frames: int,
    num_segments: int,
) -> None:
    payload = {
        "audio_path": str(audio_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "device": str(device),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "context": context,
        "smoothing_window": smoothing_window,
        "min_segment_seconds": min_segment_seconds,
        "num_frames": num_frames,
        "num_segments": num_segments,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_inference(
    *,
    audio_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_dir: Path | None = None,
    device_name: str = "auto",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    context: int = DEFAULT_CONTEXT,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
) -> dict[str, Any]:
    audio_path = audio_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve() if output_dir is not None else default_output_dir(audio_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    device = resolve_device(device_name)
    model = build_model(checkpoint, device)
    vocab = build_vocab(checkpoint)

    expected_input_dim = int(checkpoint["model_kwargs"]["input_dim"])
    actual_input_dim = 12 * (2 * context + 1)
    if actual_input_dim != expected_input_dim:
        raise ValueError(
            f"Configured context={context} produces input_dim={actual_input_dim}, "
            f"but checkpoint expects input_dim={expected_input_dim}."
        )

    audio = load_audio(audio_path, sample_rate=sample_rate)
    audio_duration_seconds = len(audio) / sample_rate
    frame_times_seconds, labels = predict_labels(
        model,
        vocab,
        audio,
        sample_rate=sample_rate,
        hop_length=hop_length,
        context=context,
        device=device,
    )

    raw_label_idx = vocab.encode(labels)
    smoothed_label_idx = smooth_label_indices(raw_label_idx, num_classes=len(vocab), window_size=smoothing_window)
    smoothed_labels = vocab.decode(smoothed_label_idx)

    raw_frame_df = frame_predictions_to_df(frame_times_seconds, labels, audio_duration_seconds=audio_duration_seconds)
    frame_df = frame_predictions_to_df(
        frame_times_seconds, smoothed_labels, audio_duration_seconds=audio_duration_seconds
    )
    segment_df = merge_short_segments(
        merge_consecutive_frames(frame_df),
        min_segment_seconds=min_segment_seconds,
    )

    raw_frame_path = resolved_output_dir / "raw_frame_predictions.csv"
    frame_path = resolved_output_dir / "frame_predictions.csv"
    segment_path = resolved_output_dir / "chord_segments.csv"

    raw_frame_df.to_csv(raw_frame_path, index=False)
    frame_df.to_csv(frame_path, index=False)
    segment_df.to_csv(segment_path, index=False)
    save_metadata(
        resolved_output_dir,
        audio_path=audio_path,
        checkpoint_path=checkpoint_path,
        device=device,
        sample_rate=sample_rate,
        hop_length=hop_length,
        context=context,
        smoothing_window=smoothing_window,
        min_segment_seconds=min_segment_seconds,
        num_frames=len(frame_df),
        num_segments=len(segment_df),
    )

    return {
        "audio_path": audio_path,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "output_dir": resolved_output_dir,
        "raw_frame_df": raw_frame_df,
        "frame_df": frame_df,
        "segment_df": segment_df,
        "raw_frame_path": raw_frame_path,
        "frame_path": frame_path,
        "segment_path": segment_path,
    }


def main() -> int:
    args = parse_args()
    result = run_inference(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        context=args.context,
        smoothing_window=args.smoothing_window,
        min_segment_seconds=args.min_segment_seconds,
    )
    print(f"Audio: {result['audio_path']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Device: {result['device']}")
    print(f"Saved raw frame predictions to {result['raw_frame_path']}")
    print(f"Saved frame predictions to {result['frame_path']}")
    print(f"Saved merged chord segments to {result['segment_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
