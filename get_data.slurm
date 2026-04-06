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

module load miniforge/24.11.3-py3.12
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

echo "Job started at: $(date)"
echo "Running from: $REPO_ROOT"
echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "uv: $(which uv)"

# Build/update the env from pyproject.toml + uv.lock
uv sync

# Run inside the uv-managed environment
uv run python get_data.py --keep-archives

echo "Job finished at: $(date)"