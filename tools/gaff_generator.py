#!/usr/bin/env python
######################################################
#
# Helper to prepare GAFF/GROMACS input files
#
######################################################

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _check_commands(commands):
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError('Missing required command(s): %s' % ', '.join(missing))


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError('Command failed: %s\n%s' % (' '.join(cmd), proc.stderr.decode()))


def _read_xyz(path):
    lines = Path(path).read_text().splitlines()
    natom = int(lines[0].strip())
    atoms = []
    coord = []
    for line in lines[2: 2 + natom]:
        e, x, y, z = line.split()[0:4]
        atoms.append(e)
        coord.append((float(x), float(y), float(z)))
    return atoms, coord


def _write_gro(path, title, atoms, coord):
    out = [title, '%5d' % len(atoms)]
    for idx, (atom, xyz) in enumerate(zip(atoms, coord), start=1):
        x, y, z = [v / 10.0 for v in xyz]
        out.append('%5d%-5s%5s%5d%8.3f%8.3f%8.3f' % (1, 'MOL', atom[:5], idx, x, y, z))
    out.append('%10.5f%10.5f%10.5f' % (0.0, 0.0, 0.0))
    Path(path).write_text('\n'.join(out) + '\n')


def _write_tleap(path, name, gaff_version):
    leaprc = 'leaprc.gaff2' if gaff_version == 'gaff2' else 'leaprc.gaff'
    text = """source %s
%s = loadmol2 %s.mol2
loadamberparams %s.frcmod
saveamberparm %s %s.prmtop %s.inpcrd
quit
""" % (leaprc, name, name, name, name, name, name)
    Path(path).write_text(text)


def _write_topol(path, name, count):
    text = """#include "%s_GMX.itp"

[ system ]
%s aggregate

[ molecules ]
%s %d
""" % (name, name, name, count)
    Path(path).write_text(text)


def _write_mdp(path):
    text = """integrator  = md
nsteps      = 0
dt          = 0.001
nstxout     = 1
nstvout     = 1
nstfout     = 1
nstenergy   = 1
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
constraints = none
"""
    Path(path).write_text(text)


def build_parser():
    parser = argparse.ArgumentParser(description='Prepare GAFF/GROMACS helper files for PyRAI2MD.')
    parser.add_argument('--monomer', required=True, help='Monomer PDB/MOL2 input for antechamber.')
    parser.add_argument('--name', required=True, help='Residue/molecule name, e.g. COTH.')
    parser.add_argument('--charge', required=True, type=int, help='Total monomer charge.')
    parser.add_argument('--multiplicity', required=True, type=int, help='Monomer spin multiplicity.')
    parser.add_argument('--count', required=True, type=int, help='Number of monomers in aggregate topology.')
    parser.add_argument('--aggregate-xyz', required=True, help='Aggregate xyz file with final atom order.')
    parser.add_argument('--out', required=True, help='Output directory.')
    parser.add_argument('--gaff', choices=['gaff', 'gaff2'], default='gaff2', help='GAFF version.')
    parser.add_argument('--dry-run', action='store_true', help='Write generated text files without running external tools.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    monomer = Path(args.monomer)
    aggregate = Path(args.aggregate_xyz)
    if not monomer.exists():
        raise FileNotFoundError(args.monomer)
    if not aggregate.exists():
        raise FileNotFoundError(args.aggregate_xyz)

    name = args.name
    monomer_suffix = monomer.suffix.lower().replace('.', '')
    if monomer_suffix == '':
        monomer_suffix = 'pdb'

    _write_tleap(outdir / 'tleap.in', name, args.gaff)
    _write_topol(outdir / 'topol.top', name, args.count)
    _write_mdp(outdir / 'singlepoint.mdp')

    atoms, coord = _read_xyz(aggregate)
    _write_gro(outdir / 'cothagg.gro', '%s aggregate' % name, atoms, coord)

    if args.dry_run:
        return 0

    _check_commands(['antechamber', 'parmchk2', 'tleap', 'acpype'])
    shutil.copy2(monomer, outdir / monomer.name)

    antechamber = [
        'antechamber',
        '-i', monomer.name,
        '-fi', monomer_suffix,
        '-o', '%s.mol2' % name,
        '-fo', 'mol2',
        '-c', 'bcc',
        '-s', '2',
        '-at', args.gaff,
        '-nc', str(args.charge),
        '-m', str(args.multiplicity),
    ]
    parmchk = ['parmchk2', '-i', '%s.mol2' % name, '-f', 'mol2', '-o', '%s.frcmod' % name, '-s', args.gaff]
    tleap = ['tleap', '-f', 'tleap.in']
    acpype = ['acpype', '-p', '%s.prmtop' % name, '-x', '%s.inpcrd' % name]

    for cmd in (antechamber, parmchk, tleap, acpype):
        _run(cmd, outdir)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('GAFF generator error: %s' % exc, file=sys.stderr)
        sys.exit(1)
