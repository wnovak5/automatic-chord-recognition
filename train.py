from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import create_cached_sequence_dataloaders, load_cached_track
from src.features.vocab import CHORD_VOCAB, ChordVocab
from src.models.rnn import ChordLSTM
from src.training.metrics import confusion_matrix_df, frame_accuracy, per_class_accuracy
from src.training.trainer import evaluate_sequence, load_checkpoint, save_checkpoint, train_sequence
from src.utils.visualization import plot_chord_prediction, plot_confusion_matrix, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the chord-recognition LSTM from cached AAM features and save deployable artifacts."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/processed/aam_lstm_cache"),
        help="Directory produced by prepare_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, plots, and metrics. Defaults to runs/lstm-<timestamp>.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional LSTM layers.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cache-tracks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--prediction-track",
        type=str,
        default=None,
        help="Specific track id to render as a chord-prediction strip. Defaults to the first test track.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to load model weights from before training.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be positive")
    if args.num_layers <= 0:
        parser.error("--num-layers must be positive")
    if args.max_cache_tracks <= 0:
        parser.error("--max-cache-tracks must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")
    if args.val_fraction + args.test_fraction >= 1.0:
        parser.error("--val-fraction + --test-fraction must be less than 1")

    return args


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"lstm-{timestamp}"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_track_ids(
    track_ids: list[str],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(track_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    val_count = max(1, int(round(total * val_fraction)))
    test_count = max(1, int(round(total * test_fraction)))
    if val_count + test_count >= total:
        raise ValueError("Not enough tracks to create non-empty train/val/test splits with the requested fractions")

    test_ids = sorted(shuffled[:test_count])
    val_ids = sorted(shuffled[test_count : test_count + val_count])
    train_ids = sorted(shuffled[test_count + val_count :])
    return train_ids, val_ids, test_ids


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_cache_metadata(cache_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = cache_dir / "manifest.csv"
    metadata_path = cache_dir / "metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cached dataset manifest not found: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cached dataset metadata not found: {metadata_path}")

    manifest = pd.read_csv(manifest_path, dtype={"track_id": str, "cache_path": str})
    manifest["track_id"] = manifest["track_id"].astype(str)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return manifest, metadata


def evaluate_test_predictions(
    model: ChordLSTM,
    test_ids: list[str],
    *,
    cache_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_blocks: list[np.ndarray] = []
    y_pred_blocks: list[np.ndarray] = []

    for track_id in test_ids:
        track_data = load_cached_track(cache_dir, track_id)
        features = track_data["features"]
        labels = track_data["labels"]
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        lengths = torch.tensor([len(features)], dtype=torch.int64, device=device)
        with torch.no_grad():
            pred_idx = model(x, lengths=lengths).argmax(dim=-1).squeeze(0).cpu().numpy()
        y_true_blocks.append(labels)
        y_pred_blocks.append(pred_idx)

    return np.concatenate(y_true_blocks), np.concatenate(y_pred_blocks)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, cache_metadata = load_cache_metadata(cache_dir)
    all_track_ids = sorted(manifest["track_id"].tolist())
    if len(all_track_ids) < 3:
        raise ValueError("Need at least 3 cached tracks to create train/val/test splits")

    train_ids, val_ids, test_ids = split_track_ids(
        all_track_ids,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    prediction_track = args.prediction_track or test_ids[0]
    if prediction_track not in test_ids:
        raise ValueError(f"--prediction-track must be in the test split. Got {prediction_track}, test split is {test_ids}")

    vocab = ChordVocab.from_default()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cache dir: {cache_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Train tracks: {len(train_ids)}")
    print(f"Val tracks: {len(val_ids)}")
    print(f"Test tracks: {len(test_ids)}")

    train_loader, val_loader, test_loader = create_cached_sequence_dataloaders(
        cache_dir=cache_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_cache_tracks=args.max_cache_tracks,
    )

    model_kwargs = {
        "num_classes": vocab.num_classes,
        "input_dim": 12,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": not args.no_bidirectional,
    }
    model = ChordLSTM(**model_kwargs)

    if args.resume_from is not None:
        checkpoint = load_checkpoint(args.resume_from)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state")
        if state_dict is None:
            raise KeyError(f"No model weights found in checkpoint: {args.resume_from}")
        model.load_state_dict(state_dict)

    history = train_sequence(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        patience=args.patience,
    )

    model.load_state_dict(history["best_state_dict"])
    model = model.to(device)

    test_metrics = evaluate_sequence(
        model,
        test_loader,
        criterion=torch.nn.CrossEntropyLoss(ignore_index=-100),
        device=device,
    )
    y_true, y_pred = evaluate_test_predictions(model, test_ids, cache_dir=cache_dir, device=device)
    aggregate_frame_accuracy = frame_accuracy(y_true, y_pred)
    per_class_df = per_class_accuracy(y_true, y_pred, vocab)
    cm_df = confusion_matrix_df(y_true, y_pred, vocab)

    checkpoint_payload = {
        "model_type": "ChordLSTM",
        "model_kwargs": model_kwargs,
        "model_state_dict": history["best_state_dict"],
        "vocab": CHORD_VOCAB,
        "cache_dir": str(cache_dir),
        "cache_metadata": cache_metadata,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "prediction_track": prediction_track,
        "best_epoch": int(history["best_epoch"]),
        "history": {
            "train_losses": [float(x) for x in history["train_losses"]],
            "val_losses": [float(x) for x in history["val_losses"]],
            "train_accs": [float(x) for x in history["train_accs"]],
            "val_accs": [float(x) for x in history["val_accs"]],
        },
        "test_metrics": {
            "sequence_loss": float(test_metrics["loss"]),
            "sequence_accuracy": float(test_metrics["accuracy"]),
            "frame_accuracy": float(aggregate_frame_accuracy),
        },
    }
    checkpoint_path = output_dir / "best_lstm_checkpoint.pt"
    save_checkpoint(checkpoint_payload, checkpoint_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_training_curves(history, ax=axes)
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 12))
    plot_confusion_matrix(cm_df, ax=ax)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    prediction_track_data = load_cached_track(cache_dir, prediction_track)
    prediction_x = torch.from_numpy(prediction_track_data["features"]).unsqueeze(0).to(device)
    prediction_lengths = torch.tensor([len(prediction_track_data["features"])], dtype=torch.int64, device=device)
    with torch.no_grad():
        prediction_idx = model(prediction_x, lengths=prediction_lengths).argmax(dim=-1).squeeze(0).cpu().numpy()
    prediction_labels = vocab.decode(prediction_idx)
    true_labels = prediction_track_data["frame_labels"]
    frame_times = prediction_track_data["frame_times"]

    fig, ax = plt.subplots(figsize=(14, 2))
    plot_chord_prediction(
        frame_times,
        true_labels,
        prediction_labels,
        title=f"Track {prediction_track} - LSTM Predictions",
        ax=ax,
    )
    fig.savefig(output_dir / f"prediction_strip_{prediction_track}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    per_class_df.to_csv(output_dir / "per_class_accuracy.csv", index=False)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    prediction_df = pd.DataFrame(
        {
            "time_seconds": frame_times,
            "true_label": true_labels,
            "predicted_label": prediction_labels,
        }
    )
    prediction_df.to_csv(output_dir / f"prediction_track_{prediction_track}.csv", index=False)

    save_json(
        output_dir / "metrics.json",
        {
            "cache_dir": str(cache_dir),
            "device": str(device),
            "train_tracks": len(train_ids),
            "val_tracks": len(val_ids),
            "test_tracks": len(test_ids),
            "best_epoch": int(history["best_epoch"]),
            "test_sequence_loss": float(test_metrics["loss"]),
            "test_sequence_accuracy": float(test_metrics["accuracy"]),
            "test_frame_accuracy": float(aggregate_frame_accuracy),
            "checkpoint": str(checkpoint_path),
            "prediction_track": prediction_track,
        },
    )
    save_json(
        output_dir / "splits.json",
        {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        },
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved artifacts under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
