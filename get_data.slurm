#!/bin/bash
#SBATCH --job-name=get-aam-data
#SBATCH --output=slurm-get-data-%j.out
#SBATCH --error=slurm-get-data-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=standard

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

echo "Job started at: $(date)"
echo "Running from: $REPO_ROOT"
echo "Host: $(hostname)"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

echo "Using Python: $PYTHON_BIN"

# Default behavior downloads only the training-required AAM subset:
# annotations + audio mixes. Archives are kept so interrupted downloads can
# resume cleanly on a later run.
"$PYTHON_BIN" get_data.py --keep-archives "$@"

echo "Job finished at: $(date)"
