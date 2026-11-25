# Contributing

Install the package in a fresh virtual environment and run:

```sh
python -m unittest discover -s tests -v
```

For numerical issues, provide a minimal synthetic JSON/CSV, the CLI command,
Python and NumPy versions, reference impedance, topology, frequency interval,
and an independently derived expected result. Report absolute and relative
errors where the reference approaches zero.

Useful work includes independent model comparisons, export round-trip
regressions, input validation, and documenting numerical limits. Do not
contribute confidential designs or vendor assets without redistribution rights.
Do not describe estimated models as measured data or validation checks as
hardware acceptance. The former blanket MIT policy for new code submissions no longer applies.
Please open an issue before proposing code so permissions can be agreed
explicitly. Existing contributions and their previously granted licenses are
unchanged; see [LICENSE](LICENSE).

Planned maintenance priorities are stronger model-input validation,
frequency-qualified graph-fit errors, and reproducible cross-platform checks.
These are plans, not implemented capabilities or promised release dates.
