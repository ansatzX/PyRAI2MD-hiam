# LJB QMMM ensemble Slurm jobs

Generated job staging directory for COTH/TPS/HPS aggregate QMMM trajectories.

- No jobs were submitted by the generator.
- Each system has 10 Slurm job-array scripts.
- Each array contains 50 one-core trajectory tasks.
- Submit one system with `<system>/submit.sh`.
- Submit all systems interleaved with `./submit_interleaved.sh`; it submits batch 00 for coth/tps/hps, then batch 01 for coth/tps/hps, etc.
- Results are copied to `<system>/results/traj_XXXXXX/`.
- Failed tasks keep a copy of their scratch working directory under `failed_workdir/`.

The run uses the original `run_files` from `/home/cuhksz-shuaizhigang/gongcunxi/data/comp/ljb/aie_ljb/aggregate/<system>/run_files` and the matching `NN-<system>.tar.xz`.
Inputs are patched to one CPU core. Each array task writes a deterministic `<title>.velo` from `910000 + SLURM_ARRAY_TASK_ID`, then runs with `randvelo 0`.
Planck variant:

- Slurm header uses `#SBATCH -p planck-cpu01,planck-cpu02`.
- Stdout/stderr goes to `%j.err` as requested.
- Conda activation assumes an environment named `ljb`; create it with `envs/ljb_environment.yml` from the repo root.
