"""Exercise exported netlists and Touchstone output, not implementation text."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        return subprocess.run([sys.executable, "-m", "s2p_tool", *args],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def test_capacitor_netlist_and_touchstone_follow_known_rlc(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("examples/capacitor.json", "-o", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            from s2p_tool.spicesim import impedance
            netlist = next(Path(directory).glob("*.cir")).read_text()
            f = np.array([1e5, 1e6, 1e7])
            expected = 0.05 + 1j * (2 * np.pi * f * 1e-9 - 1 / (2 * np.pi * f * 1e-7))
            actual = impedance(netlist, f)
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
            rows = [list(map(float, line.split())) for line in
                    next(Path(directory).glob("*.s2p")).read_text().splitlines()
                    if line and line[0].isdigit()]
            row = rows[len(rows) // 2]
            z = .05 + 1j * (2*np.pi*row[0]*1e-9 - 1/(2*np.pi*row[0]*1e-7))
            np.testing.assert_allclose(row[3] + 1j*row[4], 100/(z+100), rtol=2e-6)

    def test_inductor_and_shunt_modes_export_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("examples/inductor.json", "--topology", "shunt", "-o", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(p.suffix for p in Path(directory).iterdir()),
                             [".cir", ".csv", ".md", ".s2p"])

    def test_invalid_sweep_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("examples/capacitor.json", "--fstart", "nan", "-o", directory)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_inconsistent_graphfit_is_refused_before_writing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("examples/graphfit.json", "-o", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("inconsistent", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
