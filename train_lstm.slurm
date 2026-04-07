#!/bin/bash
#SBATCH --job-name=acr-lstm-train
#SBATCH --output=slurm-train-lstm-%j.out
#SBATCH --error=slurm-train-lstm-%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Repo root: $REPO_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RAW_ROOT="${RAW_ROOT:-$REPO_ROOT/data/raw/AAM}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/data/processed/aam_lstm_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/runs/lstm-${SLURM_JOB_ID:-manual}}"

echo "Using Python: $PYTHON_BIN"
echo "Raw dataset root: $RAW_ROOT"
echo "Cache dir: $CACHE_DIR"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

if [[ ! -f "$CACHE_DIR/manifest.csv" ]]; then
  echo "Cached dataset manifest not found at $CACHE_DIR/manifest.csv"
  echo "Run prepare_dataset.slurm first, or set CACHE_DIR to an existing cache."
  exit 1
fi

echo "Found existing cached dataset manifest at $CACHE_DIR/manifest.csv"

echo "Starting LSTM training."
srun "$PYTHON_BIN" train.py \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"

echo "Job finished at: $(date)"
