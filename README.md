# Automatic Chord Recognition

Contributors: Edward Anderson, Sam Kunitz-Levy, Will Novak

This repository trains a chord-recognition model on the AAM dataset. The current pipeline uses rendered song mixes plus beat-level chord annotations, converts each track into chroma features, caches those features to disk, and trains an LSTM to predict framewise chord labels.

## Overview

The project is organized around a two-stage workflow:

1. Download the AAM data needed for training.
2. Precompute per-track features and labels into a cache.
3. Train an LSTM from the cache.
4. Save a deployable checkpoint plus evaluation artifacts.

The cached-feature step is important for the full 3000-song dataset. Training directly from raw audio would repeatedly decode tens of gigabytes of audio and would not scale well on the cluster.

## Data Flow

The current end-to-end data path is:

1. Raw AAM song mixes live under `data/raw/AAM/audio-mixes-mp3/`.
2. Beat annotations live under `data/raw/AAM/annotations/` as `*_beatinfo.arff`.
3. `prepare_dataset.py` loads each song, resamples it to mono `22050 Hz`, computes chroma features, aligns beatwise chord labels to frames, encodes labels, and saves one cached file per track.
4. `train.py` reads those cached files lazily through `CachedChordSequenceDataset`, slices them into fixed-length sequences, and trains a `ChordLSTM`.
5. The best checkpoint and training plots are written to a per-run output directory.

In short:

`mix audio + beatinfo -> chroma + frame labels -> cached track features -> LSTM training -> checkpoint + plots`

## Dataset

Primary dataset:

- AAM: Artificial Audio Multitracks
- Zenodo record: https://zenodo.org/records/5794629

What the current model actually needs:

- `annotations`
- `audio-mixes`

What is optional for the current pipeline:

- `audio-multitracks`
- `midis`
- `info`

The loader supports mix files with `.mp3`, `.flac`, or `.wav` suffixes. The full AAM release commonly uses `*_mix.flac`.

## Repository Layout

Important files and directories:

- [get_data.py](/home/eca4zm/school/ml3/automatic-chord-recognition/get_data.py): download the minimal AAM subset or optional extras
- [prepare_dataset.py](/home/eca4zm/school/ml3/automatic-chord-recognition/prepare_dataset.py): build the cached feature dataset
- [train.py](/home/eca4zm/school/ml3/automatic-chord-recognition/train.py): train the LSTM from the cache and save artifacts
- [train_lstm.slurm](/home/eca4zm/school/ml3/automatic-chord-recognition/train_lstm.slurm): Rivanna SLURM job for cache prep + training
- [src/data/load_data.py](/home/eca4zm/school/ml3/automatic-chord-recognition/src/data/load_data.py): raw AAM loading and label alignment
- [src/data/dataset.py](/home/eca4zm/school/ml3/automatic-chord-recognition/src/data/dataset.py): dataset classes, including the lazy cached sequence loader
- [src/models/rnn.py](/home/eca4zm/school/ml3/automatic-chord-recognition/src/models/rnn.py): recurrent model definitions
- [src/training/trainer.py](/home/eca4zm/school/ml3/automatic-chord-recognition/src/training/trainer.py): train/eval loops and checkpoint helpers
- [src/utils/visualization.py](/home/eca4zm/school/ml3/automatic-chord-recognition/src/utils/visualization.py): training and prediction plots
- `data/raw/`: raw dataset download location
- `data/processed/`: cached feature datasets
- `runs/`: per-training-run outputs
- `checkpoints/`: legacy folder from earlier experiments

## Environment

Dependencies are managed in [pyproject.toml](/home/eca4zm/school/ml3/automatic-chord-recognition/pyproject.toml) and [uv.lock](/home/eca4zm/school/ml3/automatic-chord-recognition/uv.lock).

Create the environment with `uv`:

```bash
uv sync
```

If you are on a cluster and the repository contains `.venv/`, the SLURM scripts will use that interpreter automatically.

## Downloading the Data

By default, [get_data.py](/home/eca4zm/school/ml3/automatic-chord-recognition/get_data.py) downloads only the data needed for training: annotations plus song mixes.

```bash
python get_data.py
```

Useful options:

```bash
python get_data.py --download-only --keep-archives
python get_data.py --include-multitracks
python get_data.py --full
```

The downloader supports resuming partial archive downloads when the remote server allows ranged requests.

## Precomputing Cached Features

Before full training, build the cache once:

```bash
python prepare_dataset.py --root data/raw/AAM --cache-dir data/processed/aam_lstm_cache
```

Outputs:

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

## Training Locally

Train the LSTM from the cache:

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

- builds train/val/test splits from cached track ids
- trains a `ChordLSTM`
- saves the best checkpoint
- evaluates on the test split
- saves plots and CSV summaries

## Training on Rivanna

Use the provided SLURM file:

```bash
sbatch train_lstm.slurm
```

Current SLURM resources in [train_lstm.slurm](/home/eca4zm/school/ml3/automatic-chord-recognition/train_lstm.slurm):

- `--partition=gpu`
- `--gres=gpu:1`
- `--cpus-per-task=8`
- `--mem=32G`
- `--time=24:00:00`

The SLURM job:

1. uses the repo-local `.venv` if present
2. prepares the cache if `manifest.csv` is missing
3. launches `train.py`

You can pass extra training flags through `sbatch`:

```bash
sbatch train_lstm.slurm --epochs 40 --batch-size 64 --sequence-length 256
```

You can also override paths:

```bash
RAW_ROOT=/path/to/AAM \
CACHE_DIR=/scratch/$USER/aam_lstm_cache \
OUTPUT_DIR=/scratch/$USER/lstm_run \
sbatch train_lstm.slurm --epochs 40
```

## Output Artifacts

`train.py` writes a per-run output directory under `runs/` by default:

- `runs/lstm-<jobid>/best_lstm_checkpoint.pt`
- `runs/lstm-<jobid>/training_curves.png`
- `runs/lstm-<jobid>/confusion_matrix.png`
- `runs/lstm-<jobid>/confusion_matrix.csv`
- `runs/lstm-<jobid>/per_class_accuracy.csv`
- `runs/lstm-<jobid>/prediction_strip_<track>.png`
- `runs/lstm-<jobid>/prediction_track_<track>.csv`
- `runs/lstm-<jobid>/metrics.json`
- `runs/lstm-<jobid>/splits.json`

The checkpoint contains:

- the trained `model_state_dict`
- the model architecture kwargs
- the vocabulary
- split information
- training history
- test metrics
- cache metadata

This checkpoint is the artifact you would copy to your local machine for inference.

## Progress Logging

Long jobs now emit progress information so the SLURM log is interpretable:

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
tail -f slurm-train-lstm-<jobid>.out
```

## Deployment / Inference

The intended deployment path is:

1. train on Rivanna
2. copy `best_lstm_checkpoint.pt` to your local machine
3. rebuild the `ChordLSTM` with the saved `model_kwargs`
4. load `model_state_dict`
5. run inference on new songs

The checkpoint is not automatically pushed to GitHub and generally should not be committed to the repo.

## Notebooks

The notebooks are still useful for:

- exploring the dataset
- inspecting annotations
- checking predictions visually
- debugging features on a small subset

For the full 3000-song training run, use the script + cache + SLURM pipeline instead of the notebook.

Relevant notebooks:

- [notebooks/chord_recognition.ipynb](/home/eca4zm/school/ml3/automatic-chord-recognition/notebooks/chord_recognition.ipynb)
- [notebooks/aam_loader_playground.ipynb](/home/eca4zm/school/ml3/automatic-chord-recognition/notebooks/aam_loader_playground.ipynb)
- [notebooks/aam_audio_playground.ipynb](/home/eca4zm/school/ml3/automatic-chord-recognition/notebooks/aam_audio_playground.ipynb)

## Notes

- The model vocabulary is a 25-class major/minor/no-chord label space.
- Unsupported or exceptional raw annotation labels are normalized into `N.C.` for the current model.
- Generated outputs such as `runs/`, cached features, and checkpoints should generally stay out of git history.

## References

- Ostermann, F., Vatolkin, I. & Ebeling, M. *AAM: a dataset of Artificial Audio Multitracks for diverse music information retrieval tasks.* J AUDIO SPEECH MUSIC PROC. 2023, 13 (2023). https://doi.org/10.1186/s13636-023-00278-7
- Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, *Guitarset: A Dataset for Guitar Transcription*, in 19th International Society for Music Information Retrieval Conference, Paris, France, Sept. 2018.
- Muller, M. (2021). *Fundamentals of Music Processing - Using Python and Jupyter Notebooks.* 2nd edition, Springer Verlag.
