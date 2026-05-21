#!/bin/bash
#SBATCH --output="%j.err"
#SBATCH --job-name="ljb_coth_b02"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p planck-cpu01,planck-cpu02
#SBATCH --array=101-150


set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
fi
conda activate ljb


cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
./run_one.sh
