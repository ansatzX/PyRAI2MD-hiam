#!/bin/bash
#SBATCH --job-name=ljb_coth_qmmm_b04
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=huge-cpu
#SBATCH --array=201-250
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


set -euo pipefail

source /software/envs/bash.profile
source /software/envs/anaconda3.env
conda activate ljb


cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
./run_one.sh
