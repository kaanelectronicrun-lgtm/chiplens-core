# Local validation — 2026-10-22

Windows, Python 3.12.10, NumPy 2.5.4. A newly created virtual environment
contained only `chiplens-core`, `numpy`, and `pip` after installation.
PyQt, PyMuPDF, scikit-rf, and the desktop package were not installed.

`python -m pip install .` built and installed the package successfully.
`python -m unittest discover -s tests -v` ran **19 tests, all passed**.
The installed `chiplens-core` command exported `.cir`, `.s2p`, `_Zf.csv`,
and `_report.md` for two synthetic examples:

```sh
chiplens-core examples/capacitor.json -o outputs/cap
chiplens-core examples/inductor.json --topology shunt -o outputs/ind
```

Ten numerical tests were retained from the desktop project. Five regressions
exercise the sampled S-matrix passivity checker; four tests exercise the CLI
and exported artifacts. The checker regressions were run against the prior
implementation first: it incorrectly accepted active common-mode and reverse
excitation cases, and errored on an empty sweep. The corrected implementation
passed all five cases. The new CLI tests failed before its implementation
and passed afterwards.

The fourth CLI regression checks a real inconsistency found in
`examples/graphfit.json`. Before the guard, its exported netlist differed from
its exported impedance by a maximum pointwise relative error of about 2.028.
A negative parallel resistance was clamped near zero, changing the circuit
substantially. The core now rejects this export before writing models when
the circuit/response discrepancy exceeds 1e-4 on the output grid. The new
regression failed before the guard and passed after it. This prevents an
inconsistent export; it does not repair the underlying fitting/synthesis.

The common-mode regression has S11=S12=S21=S22=0.6. Each column power is 0.72,
but the largest singular value is 1.2. The old single-column check accepted
it; the full S-matrix check correctly rejects it.

The tests include a hand-derived RLC reference and an independent nodal solve
of exported netlist text. This is software validation on synthetic data, not
laboratory or manufacturer validation. GitHub Actions is configured but has
not run. Other operating systems and Python versions are not claimed tested.

The full desktop project was also rechecked after the shared passivity fix:
31 checks passed and one optional scikit-rf passive-fit regression still failed.
That backend is excluded from this core's API and dependency set; it has not
been repaired. Do not infer that all desktop tests pass from this core result.
