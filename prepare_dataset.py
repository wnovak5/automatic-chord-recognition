from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import AAMDataset, FULL_AAM_ROOT, DEFAULT_AAM_ROOT, load_training_example
from src.features.chroma import prepare_track_features
from src.features.vocab import CHORD_VOCAB, ChordVocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute per-track AAM features and labels for memory-efficient training."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Raw dataset root. Defaults to data/raw/AAM when present, otherwise data/raw/tinyAAM.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/processed/aam_lstm_cache"),
        help="Destination directory for cached per-track files.",
    )
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument(
        "--track-ids",
        nargs="*",
        default=None,
        help="Optional subset of track ids to preprocess. Defaults to all tracks found under the dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate cached files even when the output .npz already exists.",
    )
    return parser.parse_args()


def resolve_root(root: Path | None) -> Path:
    if root is not None:
        return root.expanduser().resolve()
    if FULL_AAM_ROOT.exists():
        return FULL_AAM_ROOT.resolve()
    return DEFAULT_AAM_ROOT.resolve()


def main() -> int:
    args = parse_args()
    root = resolve_root(args.root)
    cache_dir = args.cache_dir.expanduser().resolve()
    track_dir = cache_dir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)

    dataset = AAMDataset(root)
    track_ids = args.track_ids or dataset.track_ids()
    vocab = ChordVocab.from_default()

    rows: list[dict[str, object]] = []
    for index, track_id in enumerate(track_ids, start=1):
        cache_path = track_dir / f"{track_id}.npz"
        print(f"[{index}/{len(track_ids)}] {track_id}")

        if cache_path.exists() and not args.overwrite:
            with np.load(cache_path, allow_pickle=False) as cached:
                rows.append(
                    {
                        "track_id": track_id,
                        "cache_path": str(cache_path),
                        "num_frames": int(cached["features"].shape[0]),
                        "feature_dim": int(cached["features"].shape[1]),
                        "sample_rate": int(cached["sample_rate"]),
                        "hop_length": int(cached["hop_length"]),
                    }
                )
            print(f"  reused {cache_path.name}")
            continue

        example = load_training_example(track_id, root=root, sr=args.sr, hop_length=args.hop_length)
        features = prepare_track_features(example["chroma"], context=0).astype(np.float32, copy=False)
        labels = vocab.encode(example["frame_labels"]).astype(np.int64, copy=False)
        frame_times = np.asarray(example["frame_times"], dtype=np.float32)
        frame_labels = np.asarray(example["frame_labels"], dtype=str)

        np.savez_compressed(
            cache_path,
            track_id=np.asarray(track_id),
            features=features,
            labels=labels,
            frame_times=frame_times,
            frame_labels=frame_labels,
            sample_rate=np.asarray(example["sample_rate"], dtype=np.int64),
            hop_length=np.asarray(args.hop_length, dtype=np.int64),
        )
        rows.append(
            {
                "track_id": track_id,
                "cache_path": str(cache_path),
                "num_frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "sample_rate": int(example["sample_rate"]),
                "hop_length": int(args.hop_length),
            }
        )
        print(f"  saved {cache_path.name} ({features.shape[0]} frames)")

    manifest = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    manifest.to_csv(cache_dir / "manifest.csv", index=False)
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_root": str(root),
                "num_tracks": int(len(manifest)),
                "sample_rate": args.sr,
                "hop_length": args.hop_length,
                "vocab": CHORD_VOCAB,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote manifest to {cache_dir / 'manifest.csv'}")
    print(f"Wrote metadata to {cache_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
