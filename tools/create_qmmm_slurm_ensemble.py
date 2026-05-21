#!/usr/bin/env python3
"""Create Slurm array jobs for LJB aggregate QMMM trajectory ensembles."""

from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path('/home/cuhksz-shuaizhigang/gongcunxi/data/comp/ljb/aie_ljb')
OUT_ROOT = REPO / 'examples' / 'slurm_qmmm_ensemble_500'
PLANCK_OUT_ROOT = REPO / 'examples' / 'slurm_qmmm_ensemble_500_planck'
SYSTEMS = {
    'coth': {
        'title': 'cothagg',
        'index': 'cothh',
        'model': 'NN-coth',
        'maxenergy': '0.03',
        'minenergy': '0.03',
        'maxgrad': '0.08',
        'mingrad': '0.08',
    },
    'tps': {
        'title': 'tpsagg',
        'index': 'tpss',
        'model': 'NN-tps',
        'maxenergy': '0.06',
        'minenergy': '0.06',
        'maxgrad': '0.18',
        'mingrad': '0.18',
    },
    'hps': {
        'title': 'hpsagg',
        'index': 'hpss',
        'model': 'NN-hps',
        'maxenergy': '0.05',
        'minenergy': '0.05',
        'maxgrad': '0.15',
        'mingrad': '0.15',
    },
}
NTRAJ = 500
BATCH_SIZE = 50


def patch_input(text, system, seed_base=910000):
    cfg = SYSTEMS[system]
    lines = text.splitlines()
    out = []
    in_control = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith('&control'):
            in_control = True
            out.append(line)
            continue
        if stripped.startswith('&') and not lower.startswith('&control'):
            if in_control:
                out.append('gl_seed __TRAJ_SEED__')
                in_control = False
            out.append(line)
            continue

        key = lower.split()[0] if lower.split() else ''
        if key == 'qm':
            out.append('qm nn xtb xtb')
        elif key in ('ml_ncpu', 'qc_ncpu', 'xtb_nproc'):
            out.append(f'{key} 1')
        elif key == 'mem':
            out.append('mem 4000')
        elif key == 'modeldir':
            out.append(f'modeldir {cfg["model"]}')
        elif key == 'train_data':
            out.append('train_data data.json')
        elif key == 'invd_index':
            out.append(f'invd_index {cfg["index"]}')
        elif key == 'randvelo':
            out.append('randvelo 0')
        elif key == 'silent':
            out.append('silent 1')
        elif key == 'verbose':
            out.append('verbose 0')
        else:
            out.append(line)

    if in_control:
        out.append('gl_seed __TRAJ_SEED__')

    return '\n'.join(out).strip() + '\n'


def write_run_one(system_dir, system):
    cfg = SYSTEMS[system]
    script = f"""#!/bin/bash
set -euo pipefail

TRAJ_ID="${{SLURM_ARRAY_TASK_ID:-${{1:-1}}}}"
TRAJ_PAD=$(printf "%06d" "$TRAJ_ID")
BASE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RESULT_DIR="$BASE_DIR/results/traj_${{TRAJ_PAD}}"
SOURCE_DIR="$BASE_DIR/source"
SEED=$((910000 + TRAJ_ID))

mkdir -p "$RESULT_DIR"

if [ -n "${{SLURM_JOB_ID:-}}" ]; then
    JOB_TAG="${{SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}_${{TRAJ_ID}}"
else
    JOB_TAG="local_${{TRAJ_ID}}_$(date +%s)"
fi

SCRATCH_BASE="${{SLURM_TMPDIR:-}}"
if [ -z "$SCRATCH_BASE" ]; then
    if [ -d "/cache_local/${{USER}}" ]; then
        SCRATCH_BASE="/cache_local/${{USER}}"
    elif [ -d "/software/cache/${{USER}}" ]; then
        SCRATCH_BASE="/software/cache/${{USER}}"
    else
        SCRATCH_BASE="$BASE_DIR/scratch"
    fi
fi
mkdir -p "$SCRATCH_BASE"
WORK_DIR=$(mktemp -d -p "$SCRATCH_BASE" "{system}_traj_${{TRAJ_PAD}}_${{JOB_TAG}}_XXXXXX")

cleanup() {{
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
}}
trap cleanup EXIT

cp -a "$SOURCE_DIR/run_files"/. "$WORK_DIR/"
cp "$SOURCE_DIR/{cfg['model']}.tar.xz" "$WORK_DIR/"

cd "$WORK_DIR"
tar -xf "{cfg['model']}.tar.xz"
sed "s/__TRAJ_SEED__/${{SEED}}/g" "$SOURCE_DIR/input.template" > input
python - <<'PY'
import numpy as np
from PyRAI2MD.Molecule.atom import Atom
from PyRAI2MD.Utils.sampling import random_velocity

seed = int("${{SEED}}")
title = "{cfg['title']}"
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
export PYTHONPATH="{REPO}:${{PYTHONPATH:-}}"

hostname > host.txt
date '+%F %T' > start_time.txt
/usr/bin/time -p python -m PyRAI2MD.pyrai2md input > stdout.log 2> time.stderr
date '+%F %T' > end_time.txt

cp -a input stdout.log time.stderr start_time.txt end_time.txt host.txt "$RESULT_DIR/"
cp -a {cfg['title']}.log {cfg['title']}.md.energies {cfg['title']}.md.xyz {cfg['title']}.md.velo "$RESULT_DIR/" 2>/dev/null || true
cp -a {cfg['title']}.sh.energies {cfg['title']}.sh.xyz {cfg['title']}.sh.velo "$RESULT_DIR/" 2>/dev/null || true
"""
    path = system_dir / 'run_one.sh'
    path.write_text(script)
    path.chmod(0o755)


def write_slurm(system_dir, system, batch_id, start, end, cluster='sai'):
    if cluster == 'planck':
        header = f"""#SBATCH --output="%j.err"
#SBATCH --job-name="ljb_{system}_b{batch_id:02d}"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p planck-cpu01,planck-cpu02
#SBATCH --array={start}-{end}
"""
        env_setup = """if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
fi
conda activate ljb
"""
    else:
        header = f"""#SBATCH --job-name=ljb_{system}_qmmm_b{batch_id:02d}
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=huge-cpu
#SBATCH --array={start}-{end}
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
"""
        env_setup = """source /software/envs/bash.profile
source /software/envs/anaconda3.env
conda activate ljb
"""

    slurm = f"""#!/bin/bash
{header}

set -euo pipefail

{env_setup}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
./run_one.sh
"""
    path = system_dir / f'{system}_qmmm_batch_{batch_id:02d}_{start:03d}_{end:03d}.slurm'
    path.write_text(slurm)
    path.chmod(0o755)


def write_submit_helper(system_dir, system):
    commands = '\n'.join(
        f'sbatch {system}_qmmm_batch_{batch_id:02d}_{start:03d}_{end:03d}.slurm'
        for batch_id, start, end in batches()
    )
    helper = f"""#!/bin/bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
{commands}
"""
    path = system_dir / 'submit.sh'
    path.write_text(helper)
    path.chmod(0o755)


def batches():
    batch_id = 0
    for start in range(1, NTRAJ + 1, BATCH_SIZE):
        end = min(start + BATCH_SIZE - 1, NTRAJ)
        yield batch_id, start, end
        batch_id += 1


def write_submit_interleaved(out_root=OUT_ROOT):
    commands = []
    for batch_id, start, end in batches():
        for system in SYSTEMS:
            commands.append(
                f'(cd {system} && sbatch {system}_qmmm_batch_{batch_id:02d}_{start:03d}_{end:03d}.slurm)'
            )
    helper = '#!/bin/bash\nset -euo pipefail\ncd "$(dirname "${BASH_SOURCE[0]}")"\n' + '\n'.join(commands) + '\n'
    path = out_root / 'submit_interleaved.sh'
    path.write_text(helper)
    path.chmod(0o755)


def write_readme(out_root=OUT_ROOT, cluster='sai'):
    if cluster == 'planck':
        cluster_note = """Planck variant:

- Slurm header uses `#SBATCH -p planck-cpu01,planck-cpu02`.
- Stdout/stderr goes to `%j.err` as requested.
- Conda activation assumes an environment named `ljb`; create it with `envs/ljb_environment.yml` from the repo root.
"""
    else:
        cluster_note = """SAI variant:

- Slurm header uses `CPU-MISC` and `huge-cpu`.
"""
    readme = """# LJB QMMM ensemble Slurm jobs

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
""" + cluster_note
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'README.md').write_text(readme)


def generate(out_root=OUT_ROOT, cluster='sai'):
    out_root.mkdir(parents=True, exist_ok=True)
    write_readme(out_root, cluster)
    write_submit_interleaved(out_root)

    for system, cfg in SYSTEMS.items():
        system_dir = out_root / system
        source_dir = system_dir / 'source'
        run_files_dst = source_dir / 'run_files'

        system_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        (system_dir / 'logs').mkdir(exist_ok=True)
        (system_dir / 'results').mkdir(exist_ok=True)
        for old in system_dir.glob(f'{system}_qmmm_*.slurm'):
            old.unlink()

        run_files_src = SOURCE_ROOT / 'aggregate' / system / 'run_files'
        model_src = SOURCE_ROOT / f'{cfg["model"]}.tar.xz'
        if run_files_dst.exists():
            shutil.rmtree(run_files_dst)
        shutil.copytree(run_files_src, run_files_dst, ignore=shutil.ignore_patterns('tmp', 'test'))
        shutil.copy2(model_src, source_dir / model_src.name)

        original_input = (run_files_src / 'input').read_text()
        (source_dir / 'input.template').write_text(patch_input(original_input, system))

        write_run_one(system_dir, system)
        for batch_id, start, end in batches():
            write_slurm(system_dir, system, batch_id, start, end, cluster)
        write_submit_helper(system_dir, system)

    print(out_root)


def main():
    generate(OUT_ROOT, 'sai')
    generate(PLANCK_OUT_ROOT, 'planck')


if __name__ == '__main__':
    main()
