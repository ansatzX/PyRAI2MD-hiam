# GAFF Backend Design

## Context

The current COTH aggregate run uses PyRAI2MD's QMQM2 path:

```text
qm nn xtb xtb
highlevel 1-28
midlevel 29-392
```

In this mode xTB is used twice: as the QM2 correction method and as the MM method through GFN-FF. The requested change is to add optional GAFF/GROMACS support without changing existing xTB behavior. If an input still specifies xTB, the current path must continue to run as before.

The COTH `highlevel` region is a complete 28-atom monomer. That makes the first production target, `qm nn gaff`, physically clean because no link atoms or cut covalent boundary correction are required.

## Goal

Add GAFF as a user-selectable method wherever the PyRAI2MD method string allows it and provide helper tooling to prepare GAFF/GROMACS files.

The primary target mode is:

```text
qm nn gaff
```

This mode removes xTB from the runtime calculation. The NN method provides state-specific QM energies, gradients, NACs, and SOCs for the high-level monomer. GAFF/GROMACS provides ground-state classical energies and forces for the full active system and for the high-level region.

GAFF should also be available in the method registry as a classical single-state evaluator. If a user explicitly writes a supported combination such as `qm gaff`, `qm gaff xtb`, or `qm nn xtb gaff`, PyRAI2MD should attempt to run that combination according to the existing method-combination semantics rather than blocking it because GAFF is "only MM". The code should still reject combinations that cannot be evaluated consistently, such as missing topology for the requested region or cut-boundary cases that require unsupported link-atom handling.

## Non-Goals

- Do not silently change existing `qm nn xtb xtb` behavior.
- Do not make GROMACS assign GAFF atom types by itself.
- Do not support cut covalent QM/MM boundaries in the first implementation.
- Do not add electrostatic embedding into the NN model in the first `qm nn gaff` mode.
- Do not require OpenMM or ParmEd at runtime.

## Method Semantics

### Single GAFF

For a single-method calculation:

```text
qm gaff
```

GAFF evaluates the active PyRAI2MD coordinates with the user-provided GROMACS topology and returns one classical energy and gradient:

```text
traj.energy = [E_GAFF(active)]
traj.grad   = [G_GAFF(active)]
traj.nac    = []
traj.soc    = []
```

This is useful for testing and for users who intentionally want classical dynamics through the PyRAI2MD driver.

### Two-Layer NN + GAFF

For:

```text
qm nn gaff
```

Use a subtractive ONIOM-style expression:

```text
E_total(i) = E_GAFF(active) - E_GAFF(highlevel) + E_NN(highlevel, i)
```

For gradients:

```text
G_total(i, all atoms) = G_GAFF(active)
G_total(i, highlevel) += -G_GAFF(highlevel) + G_NN(highlevel, i)
```

This path should be implemented explicitly rather than forced through the current QMQM2 class, because QMQM2 assumes a separate QM2 method.

### QMQM2 With GAFF

For existing three-method QMQM2 expressions:

```text
qm <qm1> <qm2> <mm>
```

GAFF may appear as `<qm2>` or `<mm>` if the user supplies the required topology for that region. Examples:

```text
qm nn xtb gaff
qm nn gaff gaff
qm gaff xtb gaff
```

When GAFF is used as `<qm2>`, it is still a single-state classical method. QMQM2 will repeat that single energy and gradient over electronic states as it already does for xTB-derived single-state corrections. This is a user-controlled modeling choice, not an implicit recommendation.

## Energy, Force, And Unit Conversion

GAFF forces from GROMACS must be converted into PyRAI2MD gradients:

```text
gradient = -force
```

Units:

```text
energy: kJ/mol -> Hartree
force:  kJ/mol/nm -> Hartree/Bohr gradient
```

The resulting trajectory should preserve NN NAC and SOC values on the high-level atoms only for `qm nn gaff`, matching the current NN behavior.

## Input Interface

Add a new `&GAFF` section. A target COTH aggregate input can look like:

```text
&CONTROL
title cothagg
jobtype md
qm nn gaff
ml_ncpu 16
qc_ncpu 16

&MOLECULE
ci 3
spin 0
coupling 1 2,2 3
highlevel 1-28

&GAFF
gmx /opt/apps/gromacs/gmx2025.4_sai2603_classic_cuda12_avx2/bin/gmx
mdp singlepoint.mdp
topol topol.top
gro cothagg.gro
highlevel_topol monomer.top
highlevel_gro coth.gro
gaff_nproc 16
keep_tmp 1
```

Keyword meanings:

- `gmx`: path to the GROMACS executable.
- `mdp`: single-point GROMACS parameter file.
- `topol`: topology for the active/full system.
- `gro`: coordinate template for the active/full system.
- `highlevel_topol`: topology for the high-level region when a subtractive high-level GAFF calculation is needed.
- `highlevel_gro`: coordinate template for the high-level region.
- `midlevel_topol`: optional topology for the high+mid region when GAFF is used inside QMQM2.
- `midlevel_gro`: optional coordinate template for the high+mid region.
- `gaff_nproc`: OpenMP thread count for GROMACS.
- `keep_tmp`: keep or delete temporary GAFF work directories.

The GAFF interface should select the topology by `runtype`:

```text
qm_high or qm2_high:          highlevel_topol/highlevel_gro
qm2_high_mid or mm_high_mid:  midlevel_topol/midlevel_gro if supplied, otherwise highlevel+midlevel cannot use GAFF
qm_high_mid_low or mm_high_mid_low or single gaff: topol/gro
```

If the requested GAFF region has no matching topology and coordinate template, PyRAI2MD should stop with a clear error.

## GAFF Preparation Helper

Add a helper script under `tools/`, for example:

```text
tools/gaff_generator.py
```

The helper should automate the standard AmberTools-to-GROMACS workflow but keep parameterization outside the MD runtime. It should be a command-line utility, not part of each MD step.

Recommended command shape:

```bash
python tools/gaff_generator.py \
  --monomer coth.pdb \
  --name COTH \
  --charge 0 \
  --multiplicity 1 \
  --count 33 \
  --aggregate-xyz cothagg.xyz \
  --out gaff_files
```

The helper should:

1. Run `antechamber` to create a GAFF or GAFF2 mol2 file with charges.
2. Run `parmchk2` to create missing parameters.
3. Write and run a `tleap` input to create Amber `prmtop/inpcrd`.
4. Convert Amber files to GROMACS using `acpype` when available.
5. Write a simple aggregate `topol.top` that includes the monomer `.itp` and the requested molecule count.
6. Convert or write a `.gro` coordinate file from the aggregate xyz while preserving atom order.
7. Write a conservative `singlepoint.mdp` suitable for force-only evaluation.

The helper should check required commands and fail early if `antechamber`, `parmchk2`, `tleap`, or `acpype` are missing. If ACPYPE is unavailable, it should stop with instructions rather than generating partial unsupported files.

GROMACS itself should not be treated as a GAFF parameter generator. `gmx pdb2gmx` is not the default path for arbitrary organic GAFF systems.

## Architecture

### Keyword Reader

Add `PyRAI2MD/Keywords/key_gaff.py` and register it in `PyRAI2MD/variables.py`. It should parse `&GAFF`, provide defaults, attach the project title and verbosity, and include a summary block in the startup log when `gaff` appears in `qm`.

### GROMACS GAFF Interface

Add `PyRAI2MD/Quantum_Chemistry/qc_gaff.py`.

Responsibilities:

- Prepare temporary folders by requested `runtype`.
- Select the topology and coordinate template for the requested region.
- Rewrite `.gro` coordinates from the PyRAI2MD trajectory at every MD step while preserving topology atom order.
- Run `gmx grompp` and `gmx mdrun` for a single-point force calculation.
- Extract potential energy and forces.
- Convert units and return a PyRAI2MD-compatible trajectory with `energy`, `grad`, and `status`.

### NN + GAFF Combiner

Add `PyRAI2MD/Quantum_Chemistry/nn_gaff.py`.

Responsibilities:

- Instantiate the existing NN method for the high-level region.
- Instantiate GAFF evaluators for the active/full system and the high-level region.
- Evaluate the GAFF active system, GAFF high-level region, and NN high-level region.
- Combine energies and gradients with the subtractive formula.
- Place NN NACs on high-level atoms and zero elsewhere.
- Store diagnostic fields:
  - `energy_qm`
  - `energy_mm1` for GAFF high-level
  - `energy_mm2` for GAFF active/full

### Method Registry

Update `PyRAI2MD/methods.py`:

- Add `gaff` to the single-method `qm_list`.
- Add `gaff` to `qm1_list`, `qm2_list`, and `mm_list` so user-specified QMQM2 combinations are possible.
- Special-case `qm nn gaff` to use the new NN+GAFF combiner.
- Leave `qm nn xtb xtb` and other existing combinations unchanged.

## Error Handling

The implementation should fail early with clear messages when:

- `gmx` is missing or not executable.
- Required GAFF files are missing for the requested region.
- The selected `.gro` atom count differs from the requested PyRAI2MD region.
- GROMACS exits nonzero.
- Energy or force extraction fails.
- Link atoms or automatic QM/MM boundaries are detected in a GAFF region.
- A requested QMQM2 GAFF region needs high+mid topology but `midlevel_topol` or `midlevel_gro` is missing.
- The GAFF preparation helper is asked to continue after a required AmberTools command is missing.

## Testing

Add focused tests that do not require a full production GAFF setup:

1. Keyword parsing for `&GAFF`.
2. Method routing:
   - `qm nn xtb xtb` still creates the existing QMQM2 path.
   - `qm nn gaff` creates the new NN+GAFF path.
   - `qm gaff` creates the GAFF single-method path.
   - `qm nn xtb gaff` still routes through QMQM2.
3. Unit conversion for energy and force.
4. Gradient combination shape:
   - Active/full gradient has shape `(nstate, natom, 3)`.
   - NN replacement only affects `highlevel` atoms in `qm nn gaff`.
5. Error handling for missing files and inconsistent atom counts.
6. GAFF helper command-generation tests with subprocess calls mocked.

If a small GAFF fixture is available, add an integration test that runs GROMACS on a tiny molecule with `nsteps = 0`.

## Runtime Requirements

Before production MD, the user must provide or generate:

- Region-specific GROMACS topology and coordinate files needed by the chosen `qm` method string.
- A single-point `.mdp`.
- Confirmation that every `.gro` atom order matches the corresponding PyRAI2MD atom order.

The helper script can generate these files for simple repeated-monomer aggregates, but PyRAI2MD should still validate them at runtime before starting MD.
