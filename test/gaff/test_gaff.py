######################################################
#
# PyRAI2MD GAFF support unit tests
#
######################################################

import tempfile
import types
import unittest
import shutil
import subprocess
import time
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np


TIMING_FILE = Path('examples/gaff_timing/last_timing.tsv')
RUN_ROOT = Path('examples/gaff_runs')


def setUpModule():
    TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMING_FILE.write_text('method\tntomp\twall_seconds\tsteps\ttimestep_fs\tsimulated_fs\n')
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def record_timing(method, ntomp, wall_seconds, steps, timestep_fs):
    TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TIMING_FILE.open('a') as out:
        out.write('%s\t%d\t%.6f\t%d\t%.3f\t%.3f\n' % (
            method,
            ntomp,
            wall_seconds,
            steps,
            timestep_fs,
            steps * timestep_fs,
        ))


def make_run_dir(method):
    stamp = time.strftime('%Y%m%d-%H%M%S')
    root = RUN_ROOT / ('%s_%s' % (stamp, method))
    idx = 1
    while root.exists():
        root = RUN_ROOT / ('%s_%s_%02d' % (stamp, method, idx))
        idx += 1
    root.mkdir(parents=True)
    return root


def copytree_clean(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare_coth_pyrai2md_run(root, steps=200, timestep_au=20.67, nproc=1):
    example = Path('examples/gaff_pyrai2md_coth')
    required = [
        example / 'run_files' / 'cothagg.xyz',
        example / 'run_files' / 'cothagg.init.xyz',
        example / 'run_files' / 'cothh',
        example / 'run_files' / 'data.json',
        example / 'run_files' / 'input',
        example / 'models' / 'NN-coth' / 'energy_gradient' / 'weights_v0.h5',
        example / 'models' / 'NN-coth' / 'energy_gradient' / 'weights_v1.h5',
        example / 'gaff' / 'aggregate.top',
        example / 'gaff' / 'aggregate.gro',
        example / 'gaff' / 'highlevel.top',
        example / 'gaff' / 'highlevel.gro',
        example / 'gaff' / 'singlepoint.mdp',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise unittest.SkipTest('COTH PyRAI2MD/GAFF example file is missing: %s' % ', '.join(missing))

    for name in ('cothagg.xyz', 'cothagg.init.xyz', 'cothh', 'data.json'):
        shutil.copy2(example / 'run_files' / name, root / name)

    copytree_clean(example / 'models' / 'NN-coth', root / 'NN-coth')
    copytree_clean(example / 'gaff', root / 'gaff')
    shutil.copy2(example / 'run_files' / 'input', root / 'source_xtb_input')

    input_text = """&CONTROL
title cothagg
jobtype md
qm nn gaff
ml_ncpu {nproc}
qc_ncpu {nproc}
maxenergy 1000
minenergy 1000
maxgrad 1000
mingrad 1000

&MOLECULE
ci 3
spin 0
coupling 1 2,2 3
shape ellipsoid
highlevel 1-28
midlevel 29-392
embedding 0
cavity 18.343
factor 40
center 12 13 16 17 4 5 8 9

&GAFF
gmx gmx
mdp gaff/singlepoint.mdp
topol gaff/aggregate.top
gro gaff/aggregate.gro
highlevel_topol gaff/highlevel.top
highlevel_gro gaff/highlevel.gro
gaff_nproc {nproc}
keep_tmp 0

&MD
reset 1
resetstep 4000
randvelo 0
temp 298.15
step {steps}
size {timestep_au}
root 2
thermo nvt
direct 1
buffer 1
sfhp fssh
nactype ktdc
gap 1.0
silent 1
verbose 0
checkpoint 0
record_step 0

&NN
modeldir NN-coth
train_data data.json
nn_eg_type 2
silent 1

&EG
invd_index cothh
depth 3
nn_size 500
batch_size 64
loss_weights 1 1
use_reg_activ l2
use_reg_weight l2
use_reg_bias l2
reg_l2 1e-7
epo 600
epostep 10
learning_rate 1e-3
learning_rate_step 1e-3 1e-4 1e-5
epoch_step_reduction 350 200 50

&EG2
invd_index cothh
depth 3
nn_size 700
batch_size 64
loss_weights 1 1
use_reg_activ l2
use_reg_weight l2
use_reg_bias l2
reg_l2 1e-7
epo 600
epostep 10
learning_rate 1e-3
learning_rate_step 1e-3 1e-4 1e-5
epoch_step_reduction 350 200 50
""".format(nproc=nproc, steps=steps, timestep_au=timestep_au)

    (root / 'input').write_text(input_text)


class TestGaffSupport(unittest.TestCase):

    def test_key_gaff_parses_inputs(self):
        from PyRAI2MD.Keywords.key_gaff import KeyGaff

        keywords = KeyGaff().update([
            '&gaff',
            'gmx /usr/bin/gmx',
            'mdp singlepoint.mdp',
            'topol topol.top',
            'gro conf.gro',
            'highlevel_topol monomer.top',
            'highlevel_gro monomer.gro',
            'midlevel_topol mid.top',
            'midlevel_gro mid.gro',
            'gaff_nproc 8',
            'keep_tmp 0',
        ])

        self.assertEqual(keywords['gmx'], '/usr/bin/gmx')
        self.assertEqual(keywords['mdp'], 'singlepoint.mdp')
        self.assertEqual(keywords['topol'], 'topol.top')
        self.assertEqual(keywords['gro'], 'conf.gro')
        self.assertEqual(keywords['highlevel_topol'], 'monomer.top')
        self.assertEqual(keywords['highlevel_gro'], 'monomer.gro')
        self.assertEqual(keywords['midlevel_topol'], 'mid.top')
        self.assertEqual(keywords['midlevel_gro'], 'mid.gro')
        self.assertEqual(keywords['gaff_nproc'], 8)
        self.assertEqual(keywords['keep_tmp'], 0)

    def test_read_input_includes_gaff_info(self):
        from PyRAI2MD.variables import read_input

        variables, log_info = read_input([
            'control\n'
            'title test\n'
            'jobtype sp\n'
            'qm gaff\n',
            'molecule\n'
            'ci 1\n'
            'spin 0\n',
            'gaff\n'
            'gmx /usr/bin/gmx\n'
            'topol topol.top\n'
            'gro conf.gro\n'
            'mdp singlepoint.mdp\n',
        ])

        self.assertEqual(variables['gaff']['gaff_project'], 'test')
        self.assertEqual(variables['gaff']['gmx'], '/usr/bin/gmx')
        self.assertIn('&gaff', log_info)

    def test_method_registry_routes_gaff_modes(self):
        import PyRAI2MD.methods as methods

        calls = []

        class FakeGaff:
            def __init__(self, **kwargs):
                calls.append(('gaff', kwargs.get('runtype')))

        class FakeNN:
            def __init__(self, **kwargs):
                calls.append(('nn', kwargs.get('runtype')))

        class FakeQMQM2:
            def __init__(self, methods=None, **kwargs):
                calls.append(('qmqm2', methods))

        class FakeNNGaff:
            def __init__(self, **kwargs):
                calls.append(('nn_gaff', kwargs))

        def fake_load_model(module, name):
            if name == 'DNN':
                return FakeNN
            return object

        with mock.patch.object(methods, 'Gaff', FakeGaff), \
                mock.patch.object(methods, '_load_model', fake_load_model), \
                mock.patch.object(methods, 'QMQM2', FakeQMQM2), \
                mock.patch.object(methods, 'NNGaff', FakeNNGaff):

            keywords = {
                'control': {'ms_ncpu': 1},
                'gaff': {},
                'xtb': {},
                'molecule': {},
            }

            methods.QM(['gaff'], keywords=keywords)
            methods.QM(['nn', 'gaff'], keywords=keywords)
            methods.QM(['nn', 'xtb', 'xtb'], keywords=keywords)
            methods.QM(['nn', 'xtb', 'gaff'], keywords=keywords)

        self.assertEqual(calls[0], ('gaff', None))
        self.assertEqual(calls[1][0], 'nn_gaff')
        self.assertEqual(calls[2][0], 'qmqm2')
        self.assertEqual(calls[3][0], 'qmqm2')

    def test_gaff_unit_conversion_and_force_to_gradient(self):
        from PyRAI2MD.Quantum_Chemistry.qc_gaff import (
            KJMOL_TO_HARTREE,
            KJMOL_NM_TO_HARTREE_BOHR,
            force_to_gradient,
        )

        force = np.array([[[1.0, -2.0, 0.5]]])
        gradient = force_to_gradient(force)

        self.assertTrue(np.isclose(KJMOL_TO_HARTREE, 1 / 2625.499638))
        self.assertTrue(np.isclose(KJMOL_NM_TO_HARTREE_BOHR, 0.1 / 2625.499638 * 0.529177210903))
        self.assertTrue(np.allclose(gradient, -force * KJMOL_NM_TO_HARTREE_BOHR))

    def test_nn_gaff_combines_energy_and_gradient(self):
        import PyRAI2MD.Quantum_Chemistry.nn_gaff as nn_gaff

        class FakeNN:
            def __init__(self, **kwargs):
                pass

            def load(self):
                return self

            def evaluate(self, traj):
                out = traj.copy()
                out.energy = np.array([10.0, 11.0])
                out.grad = np.array([
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                ])
                out.nac = np.array([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]])
                out.soc = np.array([5.0])
                out.status = 1
                return out

        class FakeGaff:
            def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):
                self.runtype = runtype

            def load(self):
                return self

            def evaluate(self, traj):
                out = traj.copy()
                if self.runtype == 'qm_high':
                    out.energy = np.array([2.0])
                    out.grad = np.array([[[0.5, 0.0, 0.0], [0.25, 0.0, 0.0]]])
                else:
                    out.energy = np.array([20.0])
                    out.grad = np.array([[
                        [5.0, 0.0, 0.0],
                        [6.0, 0.0, 0.0],
                        [7.0, 0.0, 0.0],
                    ]])
                out.status = 1
                return out

        with mock.patch.object(nn_gaff, 'DNN', FakeNN), mock.patch.object(nn_gaff, 'Gaff', FakeGaff):
            traj = types.SimpleNamespace()
            traj.natom = 3
            traj.nstate = 2
            traj.highlevel = np.array([0, 1])
            traj.energy = np.zeros(0)
            traj.grad = np.zeros(0)
            traj.nac = np.zeros(0)
            traj.soc = np.zeros(0)
            traj.status = 1
            traj.copy = lambda: types.SimpleNamespace(**traj.__dict__)

            model = nn_gaff.NNGaff(keywords={'control': {'ms_ncpu': 1}}, job_id_1=None, job_id_2=None)
            result = model.evaluate(traj)

        self.assertTrue(np.allclose(result.energy, [28.0, 29.0]))
        self.assertEqual(result.grad.shape, (2, 3, 3))
        self.assertTrue(np.allclose(result.grad[0, :, 0], [5.5, 7.75, 7.0]))
        self.assertTrue(np.allclose(result.grad[1, :, 0], [7.5, 9.75, 7.0]))
        self.assertTrue(np.allclose(result.nac[0, :, 0], [0.1, 0.2, 0.0]))
        self.assertTrue(np.allclose(result.soc, [5.0]))
        self.assertEqual(result.status, 1)

    def test_gaff_generator_dry_run(self):
        from tools.gaff_generator import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            monomer = tmp_path / 'coth.pdb'
            aggregate = tmp_path / 'cothagg.xyz'
            outdir = tmp_path / 'gaff_files'

            monomer.write_text('ATOM      1  C1  COT A   1       0.000   0.000   0.000\nEND\n')
            aggregate.write_text('1\ncomment\nC 0.0 0.0 0.0\n')

            rc = main([
                '--monomer', str(monomer),
                '--name', 'COTH',
                '--charge', '0',
                '--multiplicity', '1',
                '--count', '1',
                '--aggregate-xyz', str(aggregate),
                '--out', str(outdir),
                '--dry-run',
            ])

            self.assertEqual(rc, 0)
            self.assertTrue((outdir / 'tleap.in').exists())
            self.assertTrue((outdir / 'topol.top').exists())
            self.assertTrue((outdir / 'cothagg.gro').exists())
            self.assertTrue((outdir / 'singlepoint.mdp').exists())

    @unittest.skipIf(shutil.which('gmx') is None, 'gmx executable is not available')
    def test_gaff_runs_real_gromacs_water_singlepoint(self):
        from PyRAI2MD.Quantum_Chemistry.qc_gaff import Gaff

        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            example = Path('examples/gaff_water_md')
            for name in ('water.gro', 'topol.top', 'singlepoint.mdp'):
                shutil.copy2(example / name, root / name)

            keywords = {
                'gaff': {
                    'keep_tmp': 1,
                    'verbose': 0,
                    'gaff_project': 'water',
                    'gaff_workdir': str(root),
                    'gmx': 'gmx',
                    'mdp': str(root / 'singlepoint.mdp'),
                    'gaff_nproc': 1,
                    'topol': str(root / 'topol.top'),
                    'gro': str(root / 'water.gro'),
                    'highlevel_topol': str(root / 'topol.top'),
                    'highlevel_gro': str(root / 'water.gro'),
                    'midlevel_topol': '',
                    'midlevel_gro': '',
                }
            }
            traj = types.SimpleNamespace()
            traj.atoms = np.array([['O'], ['H'], ['H']])
            traj.coord = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399872, 0.927297, 0.0]])
            traj.natom = 3

            out = Gaff(keywords=keywords, runtype='qm_high_mid_low').evaluate(traj)

            self.assertEqual(out.energy.shape, (1,))
            self.assertEqual(out.grad.shape, (1, 3, 3))
            self.assertEqual(out.status, 1)
            self.assertGreater(float(np.linalg.norm(out.grad)), 0.0)

    @unittest.skipIf(shutil.which('gmx') is None, 'gmx executable is not available')
    def test_gaff_runs_real_gromacs_water_100fs_trajectory(self):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            example = Path('examples/gaff_water_md')
            for name in ('water.gro', 'topol.top', 'md_100fs.mdp'):
                shutil.copy2(example / name, root / name)

            grompp = [
                'gmx', 'grompp',
                '-f', 'md_100fs.mdp',
                '-c', 'water.gro',
                '-p', 'topol.top',
                '-o', 'md_100fs.tpr',
                '-maxwarn', '1',
            ]
            mdrun = ['gmx', 'mdrun', '-deffnm', 'md_100fs', '-ntomp', '1']

            proc = subprocess.run(grompp, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            start = time.perf_counter()
            proc = subprocess.run(mdrun, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wall_seconds = time.perf_counter() - start
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            record_timing('water_gaff_gromacs_md', 1, wall_seconds, 200, 0.5)

            self.assertTrue((root / 'md_100fs.trr').exists())
            self.assertTrue((root / 'md_100fs.edr').exists())
            self.assertTrue((root / 'md_100fs.log').exists())
            log = (root / 'md_100fs.log').read_text(errors='ignore')
            self.assertIn('Finished mdrun', log)

    @unittest.skipIf(shutil.which('gmx') is None, 'gmx executable is not available')
    def test_gaff_runs_real_gromacs_coth_qm_100fs_trajectory(self):
        example = Path('examples/gaff_coth_qm/gaff_files_real/COT.amb2gmx')
        required = ['COT_GMX.gro', 'COT_GMX.top', 'md_100fs.mdp']
        for name in required:
            if not (example / name).exists():
                self.skipTest('COTH GAFF example file is missing: %s' % (example / name))

        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)
            for name in required:
                shutil.copy2(example / name, root / name)

            grompp = [
                'gmx', 'grompp',
                '-f', 'md_100fs.mdp',
                '-c', 'COT_GMX.gro',
                '-p', 'COT_GMX.top',
                '-o', 'md_100fs.tpr',
                '-maxwarn', '1',
            ]
            mdrun = ['gmx', 'mdrun', '-deffnm', 'md_100fs', '-ntomp', '1']

            proc = subprocess.run(grompp, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            start = time.perf_counter()
            proc = subprocess.run(mdrun, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wall_seconds = time.perf_counter() - start
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            record_timing('coth_qm_gaff_gromacs_md', 1, wall_seconds, 200, 0.5)

            self.assertTrue((root / 'md_100fs.trr').exists())
            self.assertTrue((root / 'md_100fs.edr').exists())
            log = (root / 'md_100fs.log').read_text(errors='ignore')
            self.assertIn('nsteps                         = 200', log)
            self.assertIn('Step           Time', log)
            self.assertIn('200        0.10000', log)
            self.assertIn('Finished mdrun', log)

    @unittest.skipIf(shutil.which('gmx') is None, 'gmx executable is not available')
    def test_pyrai2md_runs_coth_nn_gaff_100fs_trajectory(self):
        try:
            import tensorflow  # noqa: F401
        except Exception as exc:
            self.skipTest('TensorFlow is not importable: %s' % exc)

        try:
            from PyRAI2MD.Dynamics.Propagators.surface_hopping import FSSH
        except Exception as exc:
            self.skipTest('FSSH extension is not importable: %s' % exc)
        if FSSH is None:
            self.skipTest('FSSH extension is not compiled; run pyrai2md update')

        root = make_run_dir('pyrai2md_coth_nn_gaff_md')
        steps = 200
        timestep_fs = 0.5
        nproc = 1
        prepare_coth_pyrai2md_run(root, steps=steps, timestep_au=20.67, nproc=nproc)

        env = os.environ.copy()
        repo = str(Path.cwd())
        env['PYTHONPATH'] = repo if env.get('PYTHONPATH', '') == '' else repo + os.pathsep + env['PYTHONPATH']
        env['OMP_NUM_THREADS'] = str(nproc)

        start = time.perf_counter()
        with (root / 'stdout.log').open('w') as out:
            proc = subprocess.run(
                [sys.executable, '-m', 'PyRAI2MD.pyrai2md', 'input'],
                cwd=root,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
            )
        wall_seconds = time.perf_counter() - start
        record_timing('pyrai2md_coth_nn_gaff_md', nproc, wall_seconds, steps, timestep_fs)

        if proc.returncode != 0:
            stdout = (root / 'stdout.log').read_text(errors='ignore')
            self.fail('PyRAI2MD NN/GAFF run failed in %s\n%s' % (root, stdout[-4000:]))

        log = (root / 'cothagg.log').read_text(errors='ignore')
        energies = (root / 'cothagg.md.energies').read_text(errors='ignore').splitlines()

        self.assertIn('Nonadiabatic Molecular Dynamics End:', log)
        self.assertIn('Total:', log)
        self.assertGreaterEqual(len(energies), steps + 1)
        self.assertTrue(energies[-1].lstrip().startswith('4134.00'), energies[-1])
        self.assertTrue((root / 'cothagg.md.xyz').exists())
        self.assertTrue((root / 'cothagg.md.velo').exists())
        self.assertFalse(any(root.glob('tmp_gaff*')))


if __name__ == '__main__':
    unittest.main()
