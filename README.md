# Automatic Chord Recognition

Contributors: Edward Anderson, Sam Kunitz-Levy, Will Novak

This repository contains the production path for the current automatic chord recognition system:

1. download the AAM dataset subset needed for training
2. preprocess each song into cached chroma features plus frame-aligned labels
3. train a bidirectional LSTM on those cached sequences
4. save a deployable checkpoint and evaluation artifacts
5. run inference from the committed checkpoint
6. serve the same inference pipeline through a local Streamlit UI

## End-to-End Pipeline

The shipped workflow is:

`AAM mix audio + beat annotations -> chroma features + frame labels -> cached per-track .npz files -> ChordLSTM training -> checkpoint -> inference smoothing/segment merging -> Streamlit UI`

## Production Architecture

The current production model is a framewise chord classifier built with a recurrent neural network:

- network type: `ChordLSTM`
- implementation: [`src/models/rnn.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/models/rnn.py)
- recurrent block: bidirectional LSTM
- input dimension: `12`
- hidden dimension: `128`
- number of LSTM layers: `2`
- dropout: `0.3`
- bidirectional: `True` by default
- output classes: `25`

The 25-class vocabulary is:

- `N.C.`
- 12 major chords: `Cmaj` through `Bmaj`
- 12 minor chords: `Cmin` through `Bmin`

The default production checkpoint bundled in the repo is:

- [`pretrained/best_lstm_checkpoint.pt`](/home/eca4zm/school/ml3/automatic-chord-recognition/pretrained/best_lstm_checkpoint.pt)

## Data Preparation

The production training path uses only:

- `annotations`
- `audio-mixes`

The current code does not require:

- `audio-multitracks`
- `midis`
- `info`

Raw data layout:

- mix audio: `data/raw/AAM/audio-mixes-mp3/`
- beat annotations: `data/raw/AAM/annotations/*_beatinfo.arff`

Preparation logic lives in:

- [`prepare_dataset.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/prepare_dataset.py)
- [`src/data/load_data.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/data/load_data.py)
- [`src/features/chroma.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/features/chroma.py)

For each track, preprocessing does the following:

1. load the song mix with `librosa`
2. resample to mono at `22050 Hz`
3. compute `12`-bin `chroma_cqt`
4. use hop length `512`
5. compute one timestamp per frame
6. convert beat-level chord annotations into `[start, end)` intervals
7. assign one chord label to each feature frame
8. L2-normalize the chroma
9. build features with context `0`, so each frame stays a `12`-dimensional vector
10. save one cached `.npz` file per track

Each cached track file contains:

- `features`
- `labels`
- `frame_times`
- `frame_labels`
- `sample_rate`
- `hop_length`

Cache outputs:

- `data/processed/aam_lstm_cache/manifest.csv`
- `data/processed/aam_lstm_cache/metadata.json`
- `data/processed/aam_lstm_cache/tracks/<track_id>.npz`

## Training Pipeline

Training is driven by [`train.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/train.py) and uses the cached dataset implementation in [`src/data/dataset.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/data/dataset.py).

### Sequence Setup

Training does not feed full songs as one giant tensor. Instead it:

1. loads cached per-track features lazily from disk
2. slices each track into sequences of length `128`
3. pads variable-length final chunks within a batch
4. uses packed padded sequences inside the LSTM
5. ignores padded labels with `ignore_index=-100`

### Default Hyperparameters

Current training defaults from [`train.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/train.py):

- epochs: `30`
- learning rate: `1e-3`
- early stopping patience: `8`
- batch size: `32`
- sequence length: `128`
- hidden dimension: `128`
- number of LSTM layers: `2`
- dropout: `0.3`
- bidirectional: enabled by default
- workers: `0` locally
- max cached tracks held in memory: `2`
- seed: `42`
- validation fraction: `0.1`
- test fraction: `0.1`

### Optimization

Training logic lives in [`src/training/trainer.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/training/trainer.py).

The training loop:

- uses `Adam`
- computes inverse-frequency class weights from the training labels
- uses weighted `CrossEntropyLoss`
- tracks train and validation loss and accuracy
- keeps the best model state by validation loss
- stops early when validation loss stops improving

### Training Outputs

Each run writes a directory under `runs/` by default:

- `best_lstm_checkpoint.pt`
- `training_curves.png`
- `confusion_matrix.png`
- `confusion_matrix.csv`
- `per_class_accuracy.csv`
- `prediction_strip_<track>.png`
- `prediction_track_<track>.csv`
- `metrics.json`
- `splits.json`

The saved checkpoint includes:

- model architecture kwargs
- model weights
- vocabulary
- cache metadata
- train/val/test splits
- training history
- test metrics

## Downloading Data

Download the minimum dataset needed for the production model:

```bash
python get_data.py
```

Useful options:

```bash
python get_data.py --download-only --keep-archives
python get_data.py --include-multitracks
python get_data.py --full
```

The downloader is implemented in [`get_data.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/get_data.py).

## Building the Cache

Create the cached training dataset:

```bash
python prepare_dataset.py --root data/raw/AAM --cache-dir data/processed/aam_lstm_cache
```

## Training

Train the production LSTM from the cache:

```bash
python train.py --cache-dir data/processed/aam_lstm_cache
```

Example with explicit hyperparameters:

```bash
python train.py \
  --cache-dir data/processed/aam_lstm_cache \
  --epochs 30 \
  --batch-size 32 \
  --sequence-length 128 \
  --hidden-dim 128 \
  --num-layers 2 \
  --dropout 0.3 \
  --lr 1e-3
```

## Rivanna Workflow

The production cluster workflow is:

1. preprocess on CPU
2. train on GPU

Submit the jobs in order:

```bash
sbatch prepare_dataset.slurm
sbatch train_lstm.slurm
```

Or chain them:

```bash
PREP_JOB=$(sbatch --parsable prepare_dataset.slurm)
sbatch --dependency=afterok:$PREP_JOB train_lstm.slurm
```

SLURM entrypoints:

- [`prepare_dataset.slurm`](/home/eca4zm/school/ml3/automatic-chord-recognition/prepare_dataset.slurm)
- [`train_lstm.slurm`](/home/eca4zm/school/ml3/automatic-chord-recognition/train_lstm.slurm)

## Inference Pipeline

Inference is implemented in [`infer.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/infer.py).

Run it with:

```bash
.venv/bin/python infer.py --audio path/to/song.mp3 --device cpu
```

The production inference path does the following:

1. load the audio file as mono at `22050 Hz`
2. compute the same `12`-bin chroma features used in training
3. rebuild `ChordLSTM` from checkpoint metadata
4. predict one chord per frame
5. smooth frame labels with majority-vote windowing
6. merge consecutive equal-label frames into segments
7. merge very short segments into neighbors for a cleaner chart

### Default Inference Hyperparameters

Current defaults from [`infer.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/infer.py):

- checkpoint: `pretrained/best_lstm_checkpoint.pt`
- sample rate: `22050`
- hop length: `512`
- context: `0`
- smoothing window: `9`
- minimum segment length: `0.35` seconds

Inference outputs:

- `raw_frame_predictions.csv`
- `frame_predictions.csv`
- `chord_segments.csv`
- `metadata.json`

The most useful final artifact for a person playing along is usually:

- `chord_segments.csv`

## Final UI

The final user-facing layer in this repo is the Streamlit app in [`app.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/app.py).

Launch it with:

```bash
.venv/bin/streamlit run app.py
```

The UI is a local browser app that:

1. lets the user upload an audio file
2. previews the uploaded audio in the browser
3. runs the same pretrained inference pipeline used by `infer.py`
4. shows a play-along panel with the current chord and next chord
5. displays a merged chord timeline
6. displays a table of merged chord segments
7. lets the user download CSV outputs

The app exposes two main postprocessing controls in the sidebar:

- smoothing window
- minimum segment length

Temporary files written by the UI live under:

- `tmp/streamlit/uploads/`
- `tmp/streamlit/outputs/`

## Environment

Dependencies are declared in:

- [`pyproject.toml`](/home/eca4zm/school/ml3/automatic-chord-recognition/pyproject.toml)
- [`uv.lock`](/home/eca4zm/school/ml3/automatic-chord-recognition/uv.lock)

Create the environment with:

```bash
uv sync
```

If the repo already has `.venv/`, the SLURM scripts will use it automatically.

## Repo Guide

Important production files:

- [`get_data.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/get_data.py)
- [`prepare_dataset.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/prepare_dataset.py)
- [`train.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/train.py)
- [`infer.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/infer.py)
- [`app.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/app.py)
- [`src/data/load_data.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/data/load_data.py)
- [`src/data/dataset.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/data/dataset.py)
- [`src/features/chroma.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/features/chroma.py)
- [`src/models/rnn.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/models/rnn.py)
- [`src/training/trainer.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/training/trainer.py)
- [`src/training/metrics.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/training/metrics.py)
- [`src/utils/visualization.py`](/home/eca4zm/school/ml3/automatic-chord-recognition/src/utils/visualization.py)

## Notes

- The production path in this repo is the cached-feature LSTM pipeline, not the older experimental preprocessing modules.
- The shared inference checkpoint in `pretrained/` is intended for teammates who want to run inference without retraining.
- Generated run directories, caches, local inference outputs, temporary UI outputs, and SLURM logs should stay out of git history unless intentionally promoted.
