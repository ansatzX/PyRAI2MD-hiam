#!/bin/bash
set -euo pipefail

TRAJ_ID="${SLURM_ARRAY_TASK_ID:-${1:-1}}"
TRAJ_PAD=$(printf "%06d" "$TRAJ_ID")
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="$BASE_DIR/results/traj_${TRAJ_PAD}"
SOURCE_DIR="$BASE_DIR/source"
SEED=$((910000 + TRAJ_ID))

mkdir -p "$RESULT_DIR"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    JOB_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TRAJ_ID}"
else
    JOB_TAG="local_${TRAJ_ID}_$(date +%s)"
fi

SCRATCH_BASE="${SLURM_TMPDIR:-}"
if [ -z "$SCRATCH_BASE" ]; then
    if [ -d "/cache_local/${USER}" ]; then
        SCRATCH_BASE="/cache_local/${USER}"
    elif [ -d "/software/cache/${USER}" ]; then
        SCRATCH_BASE="/software/cache/${USER}"
    else
        SCRATCH_BASE="$BASE_DIR/scratch"
    fi
fi
mkdir -p "$SCRATCH_BASE"
WORK_DIR=$(mktemp -d -p "$SCRATCH_BASE" "coth_traj_${TRAJ_PAD}_${JOB_TAG}_XXXXXX")

cleanup() {
    rc=$?
    echo "return_code $rc" > "$RESULT_DIR/return_code.txt"
    echo "$WORK_DIR" > "$RESULT_DIR/work_dir.txt"
    if [ "$rc" -ne 0 ]; then
        mkdir -p "$RESULT_DIR/failed_workdir"
        cp -a "$WORK_DIR"/. "$RESULT_DIR/failed_workdir/" 2>/dev/null || true
    else
        rm -rf "$WORK_DIR"
    fi
    exit "$rc"
}
trap cleanup EXIT

cp -a "$SOURCE_DIR/run_files"/. "$WORK_DIR/"
cp "$SOURCE_DIR/NN-coth.tar.xz" "$WORK_DIR/"

cd "$WORK_DIR"
tar -xf "NN-coth.tar.xz"
sed "s/__TRAJ_SEED__/${SEED}/g" "$SOURCE_DIR/input.template" > input
python - <<'PY'
import numpy as np
from PyRAI2MD.Molecule.atom import Atom
from PyRAI2MD.Utils.sampling import random_velocity

seed = int("${SEED}")
title = "cothagg"
temp = 298.15
np.random.seed(seed)
with open(title + ".xyz", "r") as inp:
    lines = inp.read().splitlines()
natom = int(lines[0].strip())
atoms = [line.split()[0] for line in lines[2:2 + natom]]
mass = np.array([Atom(atom).get_mass() * 1822.8884871474306 for atom in atoms]).reshape((-1, 1))
velo = random_velocity(mass, temp, [])
np.savetxt(title + ".velo", velo, fmt="%30.16f%30.16f%30.16f")
PY

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="/home/cuhksz-shuaizhigang/gongcunxi/data/code/PyRAI2MD-hiam:${PYTHONPATH:-}"

hostname > host.txt
date '+%F %T' > start_time.txt
/usr/bin/time -p python -m PyRAI2MD.pyrai2md input > stdout.log 2> time.stderr
date '+%F %T' > end_time.txt

cp -a input stdout.log time.stderr start_time.txt end_time.txt host.txt "$RESULT_DIR/"
cp -a cothagg.log cothagg.md.energies cothagg.md.xyz cothagg.md.velo "$RESULT_DIR/" 2>/dev/null || true
cp -a cothagg.sh.energies cothagg.sh.xyz cothagg.sh.velo "$RESULT_DIR/" 2>/dev/null || true
