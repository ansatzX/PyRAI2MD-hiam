#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
sbatch tps_qmmm_batch_00_001_050.slurm
sbatch tps_qmmm_batch_01_051_100.slurm
sbatch tps_qmmm_batch_02_101_150.slurm
sbatch tps_qmmm_batch_03_151_200.slurm
sbatch tps_qmmm_batch_04_201_250.slurm
sbatch tps_qmmm_batch_05_251_300.slurm
sbatch tps_qmmm_batch_06_301_350.slurm
sbatch tps_qmmm_batch_07_351_400.slurm
sbatch tps_qmmm_batch_08_401_450.slurm
sbatch tps_qmmm_batch_09_451_500.slurm
