"""Regression: single-port power does not establish multiport passivity."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2p_tool.validation import passivity


class PassivityTests(unittest.TestCase):
    def test_rejects_active_common_mode_despite_column_power_below_one(self):
        # Each column power is .72, but the common-mode amplitude gain is 1.2.
        s = {key: np.array([0.6 + 0j]) for key in ("S11", "S21", "S12", "S22")}
        self.assertFalse(passivity(s).passed)

    def test_checks_reverse_excitation(self):
        s = {key: np.zeros(1, complex) for key in ("S11", "S21", "S12", "S22")}
        s["S12"][:] = 2
        self.assertFalse(passivity(s).passed)

    def test_accepts_passive_attenuator(self):
        s = {key: np.zeros(2, complex) for key in ("S11", "S21", "S12", "S22")}
        s["S12"][:] = s["S21"][:] = 0.8
        self.assertTrue(passivity(s).passed)

    def test_rejects_nonfinite_data(self):
        s = {key: np.array([complex("nan")]) for key in ("S11", "S21", "S12", "S22")}
        self.assertFalse(passivity(s).passed)

    def test_rejects_empty_sweep(self):
        s = {key: np.array([], complex) for key in ("S11", "S21", "S12", "S22")}
        self.assertFalse(passivity(s).passed)


if __name__ == "__main__":
    unittest.main()
