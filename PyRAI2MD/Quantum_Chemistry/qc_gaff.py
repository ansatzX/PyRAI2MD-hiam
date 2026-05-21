######################################################
#
# PyRAI2MD 2 module for GAFF/GROMACS interface
#
######################################################

import os
import sys
import shutil
import subprocess
import numpy as np


KJMOL_TO_HARTREE = 1 / 2625.499638
KJMOL_NM_TO_HARTREE_BOHR = 0.1 / 2625.499638 * 0.529177210903


def force_to_gradient(force):
    return -np.array(force, dtype=float) * KJMOL_NM_TO_HARTREE_BOHR


def _read_gro_atom_count(gro):
    with open(gro, 'r') as inp:
        lines = inp.read().splitlines()
    if len(lines) < 3:
        sys.exit('\n  FileError\n  GAFF: invalid gro file %s' % gro)
    return int(lines[1].strip())


def _write_gro_from_template(template, output, atoms, coord):
    with open(template, 'r') as inp:
        lines = inp.read().splitlines()

    natom = int(lines[1].strip())
    if natom != len(coord):
        sys.exit('\n  FileError\n  GAFF: gro atom count %s does not match coordinates %s' % (natom, len(coord)))

    out = [lines[0], lines[1]]
    for n, line in enumerate(lines[2: 2 + natom]):
        prefix = line[:20]
        suffix = line[44:] if len(line) > 44 else ''
        x, y, z = np.array(coord[n], dtype=float) / 10.0
        out.append('%s%8.3f%8.3f%8.3f%s' % (prefix, x, y, z, suffix))

    out.extend(lines[2 + natom:])

    with open(output, 'w') as gro:
        gro.write('\n'.join(out) + '\n')


def _parse_xvg_last_value(xvg):
    value = None
    with open(xvg, 'r') as inp:
        for line in inp:
            if line.startswith(('#', '@')) or len(line.split()) < 2:
                continue
            value = float(line.split()[-1])
    if value is None:
        sys.exit('\n  FileError\n  GAFF: cannot parse energy from %s' % xvg)
    return value


def _parse_xvg_forces(xvg, natom):
    rows = []
    with open(xvg, 'r') as inp:
        for line in inp:
            if line.startswith(('#', '@')) or len(line.split()) < 2:
                continue
            rows.append([float(x) for x in line.split()[1:]])

    if len(rows) == 0:
        sys.exit('\n  FileError\n  GAFF: cannot parse forces from %s' % xvg)

    data = np.array(rows[-1], dtype=float)
    if len(data) != natom * 3:
        sys.exit('\n  FileError\n  GAFF: force count %s does not match atom count %s' % (len(data), natom))

    return data.reshape((1, natom, 3))


class Gaff:
    """GAFF/GROMACS single-point interface.

    This class returns one classical energy and gradient. NAC and SOC are not
    available for GAFF.
    """

    def __init__(self, keywords=None, job_id=None, runtype='qm_high_mid_low'):
        self.runtype = runtype
        variables = keywords['gaff']
        self.keep_tmp = variables['keep_tmp']
        self.verbose = variables['verbose']
        self.project = variables['gaff_project']
        self.workdir = variables['gaff_workdir']
        self.gmx = variables['gmx']
        self.mdp = variables['mdp']
        self.nproc = variables['gaff_nproc']
        self.topol, self.gro = self._select_region_files(variables)

        if job_id is not None and job_id != 'Read':
            self.workdir = '%s/tmp_gaff-%s' % (self.workdir, job_id)
        else:
            self.workdir = '%s/tmp_gaff' % self.workdir

        suffix = {
            'qm_high': 'qm_h',
            'qm2_high': 'qm2_h',
            'qm2_high_mid': 'qm2_m',
            'mm_high_mid': 'mm_m',
            'qm_high_mid_low': 'all',
            'mm_high_mid_low': 'mm_l',
        }.get(self.runtype, 'all')
        self.workdir = '%s_%s' % (self.workdir, suffix)

    def _select_region_files(self, variables):
        if self.runtype in ('qm_high', 'qm2_high'):
            topol = variables['highlevel_topol']
            gro = variables['highlevel_gro']
            region = 'highlevel'
        elif self.runtype in ('qm2_high_mid', 'mm_high_mid'):
            topol = variables['midlevel_topol']
            gro = variables['midlevel_gro']
            region = 'high+mid'
        else:
            topol = variables['topol']
            gro = variables['gro']
            region = 'active/full'

        if topol == '' or gro == '':
            sys.exit('\n  FileNotFoundError\n  GAFF: missing %s topology or gro file for runtype %s' % (region, self.runtype))

        return topol, gro

    def _region_xyz(self, traj):
        if self.runtype in ('qm_high', 'qm2_high'):
            traj = traj.apply_qmmm() if hasattr(traj, 'apply_qmmm') else traj
            atoms = traj.qm_atoms if hasattr(traj, 'qm_atoms') and len(traj.qm_atoms) > 0 else traj.atoms[traj.highlevel]
            coord = traj.qm_coord if hasattr(traj, 'qm_coord') and len(traj.qm_coord) > 0 else traj.coord[traj.highlevel]
        elif self.runtype in ('qm2_high_mid', 'mm_high_mid'):
            traj = traj.apply_qmmm() if hasattr(traj, 'apply_qmmm') else traj
            atoms = traj.qmqm2_atoms
            coord = traj.qmqm2_coord
        else:
            atoms = traj.atoms
            coord = traj.coord

        return atoms, coord

    def _setup_gaff(self, atoms, coord):
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        for filename in (self.topol, self.mdp, self.gro):
            if not os.path.exists(filename):
                sys.exit('\n  FileNotFoundError\n  GAFF: looking for file %s' % filename)

        natom = _read_gro_atom_count(self.gro)
        if natom != len(coord):
            sys.exit('\n  FileError\n  GAFF: gro atom count %s does not match requested region %s' % (natom, len(coord)))

        shutil.copy2(self.topol, '%s/topol.top' % self.workdir)
        shutil.copy2(self.mdp, '%s/singlepoint.mdp' % self.workdir)
        _write_gro_from_template(self.gro, '%s/conf.gro' % self.workdir, atoms, coord)

    def _run_gaff(self):
        maindir = os.getcwd()
        os.chdir(self.workdir)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(self.nproc)

        grompp = [self.gmx, 'grompp', '-f', 'singlepoint.mdp', '-c', 'conf.gro', '-p', 'topol.top', '-o', 'sp.tpr', '-maxwarn', '1']
        mdrun = [self.gmx, 'mdrun', '-deffnm', 'sp', '-ntomp', str(self.nproc)]
        energy = [self.gmx, 'energy', '-f', 'sp.edr', '-o', 'energy.xvg']
        traj = [self.gmx, 'traj', '-s', 'sp.tpr', '-f', 'sp.trr', '-of', 'force.xvg']

        try:
            for cmd, stdin in ((grompp, None), (mdrun, None), (energy, b'Potential\n'), (traj, b'0\n')):
                proc = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                if proc.returncode != 0:
                    sys.exit('\n  GROMACSError\n  GAFF: command failed: %s\n%s' % (' '.join(cmd), proc.stderr.decode()))
        finally:
            os.chdir(maindir)

    def _read_data(self, natom):
        energy = np.array([_parse_xvg_last_value('%s/energy.xvg' % self.workdir) * KJMOL_TO_HARTREE])
        force = _parse_xvg_forces('%s/force.xvg' % self.workdir, natom)
        gradient = force_to_gradient(force)
        nac = np.zeros(0)
        soc = np.zeros(0)

        return energy, gradient, nac, soc

    def evaluate(self, traj):
        atoms, coord = self._region_xyz(traj)
        self._setup_gaff(atoms, coord)
        self._run_gaff()
        energy, gradient, nac, soc = self._read_data(len(coord))

        if self.keep_tmp == 0:
            shutil.rmtree(self.workdir)

        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.array(nac)
        traj.soc = np.array(soc)
        traj.err_energy = None
        traj.err_grad = None
        traj.err_nac = None
        traj.err_soc = None
        traj.status = 1

        return traj

    def train(self):
        return self

    def load(self):
        return self

    def appendix(self, _):
        return self

    def read_data(self, natom, ncharge=0):
        energy, gradient, nac, soc = self._read_data(natom)
        coord = np.zeros(0)
        charge = np.zeros(0)
        cell = np.zeros(0)
        pbc = np.zeros(0)
        return coord, charge, cell, pbc, energy, gradient, nac, soc
