# GAFF Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional GAFF/GROMACS support plus a GAFF preparation helper while preserving existing xTB behavior.

**Architecture:** Add `gaff` as a classical single-state method with a focused GROMACS interface, then add a dedicated NN+GAFF subtractive combiner for `qm nn gaff`. Register a new `&GAFF` keyword block and provide `tools/gaff_generator.py` to prepare AmberTools/ACPYPE-derived GROMACS inputs.

**Tech Stack:** Python, PyRAI2MD method registry, GROMACS command-line tools, AmberTools command-line tools, pytest-style unit tests where possible.

---

### Task 1: Keyword Parsing And Routing Tests

**Files:**
- Create: `test/gaff/test_gaff.py`
- Modify later: `PyRAI2MD/Keywords/key_gaff.py`
- Modify later: `PyRAI2MD/variables.py`
- Modify later: `PyRAI2MD/methods.py`

- [ ] **Step 1: Write failing tests**

Create tests that verify `&GAFF` parsing, method info generation, and method routing for `qm gaff`, `qm nn gaff`, and legacy `qm nn xtb xtb`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest test/gaff/test_gaff.py -q`
Expected: FAIL because `key_gaff`, `Gaff`, and `NNGaff` do not exist.

### Task 2: GAFF Keyword Reader

**Files:**
- Create: `PyRAI2MD/Keywords/key_gaff.py`
- Modify: `PyRAI2MD/variables.py`

- [ ] **Step 1: Implement `KeyGaff`**

Add defaults and `update/info` methods following `KeyXtb` style.

- [ ] **Step 2: Register `gaff` in `variables.py`**

Import `KeyGaff`, add it to `keywords_list`, set `gaff_project` and `verbose`, and add `gaff` to `method_info_dict`.

- [ ] **Step 3: Run keyword tests**

Run: `python -m pytest test/gaff/test_gaff.py -q`
Expected: routing tests still fail until method classes exist.

### Task 3: GROMACS GAFF Interface

**Files:**
- Create: `PyRAI2MD/Quantum_Chemistry/qc_gaff.py`
- Test: `test/gaff/test_gaff.py`

- [ ] **Step 1: Add unit conversion and `.gro` helpers**

Implement conversion constants, region selection, coordinate writing, energy parsing, and force parsing.

- [ ] **Step 2: Implement `Gaff.evaluate()`**

Return a PyRAI2MD-compatible trajectory with one energy, one gradient, empty NAC/SOC, and completion status.

- [ ] **Step 3: Run GAFF unit tests**

Run: `python -m pytest test/gaff/test_gaff.py -q`
Expected: method routing still fails until registry is updated.

### Task 4: NN + GAFF Combiner

**Files:**
- Create: `PyRAI2MD/Quantum_Chemistry/nn_gaff.py`
- Modify: `PyRAI2MD/methods.py`
- Test: `test/gaff/test_gaff.py`

- [ ] **Step 1: Implement `NNGaff`**

Instantiate NN, GAFF active, and GAFF high-level evaluators. Combine energies and gradients with subtractive ONIOM semantics.

- [ ] **Step 2: Register GAFF in `methods.py`**

Add `gaff` to `qm_list`, `qm1_list`, `qm2_list`, and `mm_list`. Route exactly `qm nn gaff` to `NNGaff`; keep `qm nn xtb xtb` on QMQM2.

- [ ] **Step 3: Run routing and combiner tests**

Run: `python -m pytest test/gaff/test_gaff.py -q`
Expected: PASS for unit tests that do not invoke real GROMACS.

### Task 5: GAFF Preparation Helper

**Files:**
- Create: `tools/gaff_generator.py`
- Test: `test/gaff/test_gaff.py`

- [ ] **Step 1: Implement command-line parser and command checks**

Support monomer input, molecule name, charge, multiplicity, molecule count, aggregate xyz, output directory, and GAFF version.

- [ ] **Step 2: Implement file generation**

Write `tleap.in`, `topol.top`, aggregate `.gro`, and `singlepoint.mdp`; run AmberTools/ACPYPE commands unless `--dry-run` is selected.

- [ ] **Step 3: Run helper tests**

Run: `python -m pytest test/gaff/test_gaff.py -q`
Expected: PASS with subprocess calls mocked or dry-run paths.

### Task 6: Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest test/gaff/test_gaff.py -q`
Expected: PASS.

- [ ] **Step 2: Run import smoke checks**

Run: `python - <<'PY'
from PyRAI2MD.Keywords.key_gaff import KeyGaff
from PyRAI2MD.Quantum_Chemistry.qc_gaff import Gaff
from PyRAI2MD.Quantum_Chemistry.nn_gaff import NNGaff
print(KeyGaff.__name__, Gaff.__name__, NNGaff.__name__)
PY`
Expected: prints `KeyGaff Gaff NNGaff`.

- [ ] **Step 3: Check git status without committing**

Run: `git status --short`
Expected: changed files are visible and no commit is created.
