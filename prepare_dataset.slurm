#!/bin/bash
#SBATCH --job-name=acr-prepare-data
#SBATCH --output=slurm-prepare-data-%j.out
#SBATCH --error=slurm-prepare-data-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Repo root: $REPO_ROOT"

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

echo "Using Python: $PYTHON_BIN"
echo "Raw dataset root: $RAW_ROOT"
echo "Cache dir: $CACHE_DIR"

mkdir -p "$CACHE_DIR"

echo "Starting dataset preprocessing."
srun "$PYTHON_BIN" prepare_dataset.py \
  --root "$RAW_ROOT" \
  --cache-dir "$CACHE_DIR" \
  "$@"

echo "Job finished at: $(date)"
