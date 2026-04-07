# Automatic Chord Recognition

Contributors: Edward Anderson, Sam Kunitz-Levy, Will Novak

This repository trains a chord-recognition model on the AAM dataset. The current production workflow is:

1. download the raw AAM data
2. preprocess it into a cached feature dataset
3. train an LSTM from that cache
4. save a checkpoint and evaluation artifacts for local use or repository distribution later

## Pipeline

The current data path is:

`AAM mix audio + beat annotations -> chroma features + frame labels -> cached per-track files -> LSTM training -> checkpoint + plots`

More concretely:

1. Raw song mixes live under `data/raw/AAM/audio-mixes-mp3/`
2. Beat-level chord labels live under `data/raw/AAM/annotations/` as `*_beatinfo.arff`
3. `prepare_dataset.py` loads each track, resamples it to mono `22050 Hz`, computes chroma, aligns labels to frames, and saves one cached file per song
4. `train.py` trains a `ChordLSTM` from those cached files through a lazy dataset loader
5. The best checkpoint and evaluation outputs are written to a run directory under `runs/`
6. The committed inference artifact for teammates lives under `pretrained/`

## What Data Is Actually Needed

For the current model, only these AAM components are required:

- `annotations`
- `audio-mixes`

The current code does not require:

- `audio-multitracks`
- `midis`
- `info`

The loader supports mix files with `.mp3`, `.flac`, or `.wav` suffixes.

## Important Files

- `get_data.py`: downloads the minimal AAM subset by default
- `prepare_dataset.py`: builds the cached per-track feature dataset
- `train.py`: trains the LSTM and saves artifacts
- `prepare_dataset.slurm`: Rivanna preprocessing job
- `train_lstm.slurm`: Rivanna training job
- `src/data/load_data.py`: raw AAM loading and label alignment
- `src/data/dataset.py`: lazy cached dataset and dataloaders
- `src/models/rnn.py`: `ChordLSTM`
- `src/training/trainer.py`: training loops and checkpoint helpers
- `src/utils/visualization.py`: plots

## Environment

Dependencies are managed in:

- [pyproject.toml](/home/eca4zm/school/ml3/automatic-chord-recognition/pyproject.toml)
- [uv.lock](/home/eca4zm/school/ml3/automatic-chord-recognition/uv.lock)

Create the environment with:

```bash
uv sync
```

If the repo contains `.venv/`, the SLURM scripts will use it automatically.

## Downloading Raw Data

By default, `get_data.py` downloads only the training-required AAM subset:

```bash
python get_data.py
```

Useful options:

```bash
python get_data.py --download-only --keep-archives
python get_data.py --include-multitracks
python get_data.py --full
```

The downloader supports resumable archive downloads when the server allows ranged requests.

## Building the Cache

Run once before full training:

```bash
python prepare_dataset.py --root data/raw/AAM --cache-dir data/processed/aam_lstm_cache
```

This produces:

- `data/processed/aam_lstm_cache/manifest.csv`
- `data/processed/aam_lstm_cache/metadata.json`
- `data/processed/aam_lstm_cache/tracks/<track_id>.npz`

Each cached track file contains:

- `features`
- `labels`
- `frame_times`
- `frame_labels`
- `sample_rate`
- `hop_length`

`prepare_dataset.py` logs:

- current track id
- progress through the corpus
- elapsed time
- ETA

## Training

Train locally from the cache:

```bash
python train.py --cache-dir data/processed/aam_lstm_cache
```

Common options:

```bash
python train.py \
  --cache-dir data/processed/aam_lstm_cache \
  --epochs 30 \
  --batch-size 32 \
  --sequence-length 128 \
  --hidden-dim 128 \
  --num-layers 2 \
  --log-every-batches 100
```

The training script:

- builds train/val/test splits from cached tracks
- trains a `ChordLSTM`
- saves the best checkpoint
- evaluates on the test split
- saves plots and CSV summaries

## Rivanna Workflow

Use two separate jobs:

1. preprocessing job
2. training job

Submit them in order:

```bash
sbatch prepare_dataset.slurm
sbatch train_lstm.slurm
```

Or chain them:

```bash
PREP_JOB=$(sbatch --parsable prepare_dataset.slurm)
sbatch --dependency=afterok:$PREP_JOB train_lstm.slurm
```

### `prepare_dataset.slurm`

Current resources:

- `--partition=standard`
- `--cpus-per-task=8`
- `--mem=32G`
- `--time=08:00:00`

### `train_lstm.slurm`

Current resources:

- `--partition=gpu`
- `--gres=gpu:1`
- `--cpus-per-task=8`
- `--mem=32G`
- `--time=08:00:00`

The training job expects the cache manifest to already exist and will fail fast if it does not.

You can pass extra options through `sbatch`:

```bash
sbatch train_lstm.slurm --epochs 40 --batch-size 64 --sequence-length 256
```

You can override paths:

```bash
RAW_ROOT=/path/to/AAM \
CACHE_DIR=/scratch/$USER/aam_lstm_cache \
OUTPUT_DIR=/scratch/$USER/lstm_run \
sbatch train_lstm.slurm --epochs 40
```

## Outputs

`train.py` writes a run-specific directory under `runs/` by default:

- `runs/lstm-<jobid>/best_lstm_checkpoint.pt`
- `runs/lstm-<jobid>/training_curves.png`
- `runs/lstm-<jobid>/confusion_matrix.png`
- `runs/lstm-<jobid>/confusion_matrix.csv`
- `runs/lstm-<jobid>/per_class_accuracy.csv`
- `runs/lstm-<jobid>/prediction_strip_<track>.png`
- `runs/lstm-<jobid>/prediction_track_<track>.csv`
- `runs/lstm-<jobid>/metrics.json`
- `runs/lstm-<jobid>/splits.json`

The checkpoint includes:

- `model_state_dict`
- model architecture kwargs
- vocabulary
- split information
- training history
- test metrics
- cache metadata

The selected inference artifact for this repository is committed at `pretrained/best_lstm_checkpoint.pt`.

## Progress Logging

Long jobs emit useful progress information:

- `prepare_dataset.py`
  - track index
  - percentage complete
  - elapsed time
  - ETA
- `train.py` / `trainer.py`
  - batch-level progress during train and validation
  - running loss
  - running accuracy
  - elapsed time
  - ETA

Useful monitoring commands:

```bash
squeue -u $USER
tail -f slurm-prepare-data-<jobid>.out
tail -f slurm-train-lstm-<jobid>.out
```

## Deployment

The intended deployment path is:

1. train on Rivanna
2. promote the selected checkpoint into `pretrained/`
3. rebuild the `ChordLSTM` from the saved metadata
4. load `model_state_dict`
5. run inference on new songs

Teammates should use `pretrained/best_lstm_checkpoint.pt` for shared inference instead of retraining the model.

## Inference

For local one-song-at-a-time inference, you can run the committed checkpoint directly on CPU:

```bash
.venv/bin/python infer.py --audio path/to/song.mp3 --device cpu
```

Supported inputs depend on `librosa`/backend support and typically include formats such as `.mp3`, `.wav`, and `.flac`.

By default, `infer.py`:

- loads audio as mono at `22050 Hz`
- computes the same chroma features used during training
- restores `ChordLSTM` from `pretrained/best_lstm_checkpoint.pt`
- writes outputs under `inference/<audio-stem>/`

Inference outputs:

- `raw_frame_predictions.csv`: unsmoothed frame-by-frame chord labels
- `frame_predictions.csv`: smoothed frame-by-frame chord labels
- `chord_segments.csv`: merged chord spans intended to be easier to play from
- `metadata.json`: checkpoint path and inference settings used

Useful options:

```bash
.venv/bin/python infer.py \
  --audio path/to/song.wav \
  --output-dir inference/song_name \
  --smoothing-window 9 \
  --min-segment-seconds 0.35
```

Postprocessing controls:

- `--smoothing-window`: odd-number majority-vote window over neighboring frames; larger values reduce flicker
- `--min-segment-seconds`: merges very short chord segments into adjacent ones to produce a cleaner chord chart

For now, the most practical output is `chord_segments.csv`. This model predicts only the current 25-class vocabulary of major, minor, and `N.C.` labels.

## Desktop App

There is also a simple local GUI for teammates who do not want to use the command line:

```bash
.venv/bin/python app.py
```

The desktop app:

- lets the user choose an audio file from disk
- runs the same pretrained inference pipeline used by `infer.py`
- displays the merged chord timeline on screen
- saves the CSV outputs under the default `inference/<audio-stem>/` folder

This first version is a local `tkinter` desktop app, so it should be run on a machine with a graphical desktop session. It is intended as a simple front end for iteration, not a production deployment target.

## Notes

- The model vocabulary is a 25-class major/minor/no-chord label space
- Unsupported raw annotation exceptions are normalized to `N.C.`
- Generated outputs such as ad hoc `runs/`, caches, `tmp/`, inference outputs, and SLURM logs should stay out of git history; `pretrained/` is the intentional exception for the shared inference checkpoint
