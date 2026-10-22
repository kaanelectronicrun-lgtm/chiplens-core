# ChipLens Core

[![Core numerical checks](https://github.com/kaanelectronicrun-lgtm/chiplens-core/actions/workflows/tests.yml/badge.svg)](https://github.com/kaanelectronicrun-lgtm/chiplens-core/actions/workflows/tests.yml)

Generate **estimated capacitor and inductor models** from JSON parameters or
digitized CSV curves. Export SPICE RLC subcircuits, two-port Touchstone `.s2p`,
impedance CSV, and a Markdown engineering report. NumPy is the only runtime
dependency. The package includes a separate nodal solver that checks emitted
netlist text against the model.

This is a standalone extraction of the numerical core of
[ChipLens](https://github.com/kaanelectronicrun-lgtm/s2p-model-generator).
The desktop interface, PDF/datasheet extraction, third-party PDFs, and the
optional scikit-rf backend are outside this package. The scikit-rf backend
has a known passive-fit failure in the desktop project's recorded environment;
this extraction neither fixes nor validates that separate backend.

## Install and run

Use Python 3.10 or later. From this directory:

```sh
python -m venv .venv
# Activate .venv with the command for your shell, then:
python -m pip install .
chiplens-core examples/capacitor.json -o outputs/cap
chiplens-core examples/inductor.json --topology shunt -o outputs/ind
python -m unittest discover -s tests -v
```

On Windows, activation in PowerShell is `.venv\Scripts\Activate.ps1`.
You can instead call `.venv\Scripts\python.exe` and
`.venv\Scripts\chiplens-core.exe` directly. On Linux/macOS activate with
`. .venv/bin/activate`.

The model-generator CLI accepts `--z0`, `--fstart`, `--fstop` and
`--topology series|shunt`. Examples are synthetic; their values are not
manufacturer specifications. SPICE exports describe the component between
two terminals; the Touchstone topology places that component in series or
shunt relative to the reference line.

The package namespace is `s2p_tool`, inherited from ChipLens. Install this core
and the desktop project in separate environments to avoid namespace conflicts.

## What is checked

- Known rational functions and noisy synthetic resonance fits.
- Passive RLC synthesis and a separate solver for exported netlists.
- Series/shunt closed-form comparisons and DC-bias behavior.
- Two-port S-matrix passivity at sampled frequencies, including active
  common-mode and reverse-excitation regressions.
- CLI exports against a hand-derived RLC fixture.

Tests check software behavior, not agreement with laboratory measurements.
Graph fitting can produce a circuit different from the fitted impedance when
nonphysical elements are clamped. The CLI now refuses such inconsistent
exports before writing model files (relative discrepancy limit 1e-4 on the
export grid). `examples/graphfit.json` is a known rejection case, not a
successful example. The underlying synthesis limitation is not repaired.
Sampled passivity is not an all-frequency rational-model proof. Finite samples
do not prove causality, and nonnegative real impedance alone does not prove
pole stability. Source-confidence percentages in inherited reports are
heuristic scores, not statistically calibrated accuracy or probability.

## Maintenance and provenance

Kaan AKCAN maintains ChipLens. `PROVENANCE.json` records the source repository,
baseline commit, file hashes, extracted functions, and retained upstream tests.
The clean Windows check ran **19 tests, all passed**. See
[CONTRIBUTING.md](CONTRIBUTING.md) for bug reports and
[VALIDATION.md](VALIDATION.md) for the check record. All four GitHub Actions
jobs passed on Windows/Linux with Python 3.12/3.13:
[verified run](https://github.com/kaanelectronicrun-lgtm/chiplens-core/actions/runs/39826023309).

## License

This core package is available under the [MIT License](LICENSE).
Desktop/PDF code and its dependencies are outside this license grant.
NumPy retains its own BSD license. The package has not been published to PyPI.
