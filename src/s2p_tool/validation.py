"""Model validation: passivity, causality, stability, SRF consistency.

These are sanity checks on the generated behavioral network, not a substitute
for measured data. Each returns a (passed, message) pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def passivity(s: Dict[str, np.ndarray]) -> Check:
    """Sampled two-port passivity for real, equal power-wave references.

    Every singular value of S must be <= 1. Checking just one column's power
    misses active combinations of incident waves and reverse excitation.
    This is a sampled check, not an all-frequency rational-model proof.
    """
    values = [np.asarray(s[key], dtype=complex) for key in ("S11", "S12", "S21", "S22")]
    shape = values[0].shape
    if (len(shape) != 1 or not values[0].size
            or any(v.shape != shape or not np.all(np.isfinite(v)) for v in values)):
        return Check("Passivity (sampled S-matrix)", False,
                     "requires nonempty, finite, equally sized 1D S-parameters")
    matrix = np.stack(values, axis=-1).reshape(-1, 2, 2)
    worst = float(np.max(np.linalg.svd(matrix, compute_uv=False)))
    ok = worst <= 1.0 + 1e-9
    return Check("Passivity (sampled S-matrix)", ok,
                 f"max singular value(S) = {worst:.9f} (<=1 required on sampled grid)")


def stability(z: np.ndarray) -> Check:
    """Physically realizable series Z must have Re(Z) >= 0 everywhere."""
    min_re = float(np.min(z.real))
    ok = min_re >= -1e-9
    return Check("Re(Z)>=0 (sampled)", ok,
                 f"min Re(Z) = {min_re:.4e} Ohm")


def srf_consistency(model_srf: float, datasheet_srf: float | None) -> Check:
    if not datasheet_srf:
        return Check("SRF consistency", True, "no datasheet SRF to compare")
    err = abs(model_srf - datasheet_srf) / datasheet_srf
    ok = err < 0.15
    return Check("SRF consistency", ok,
                 f"model {model_srf/1e6:.2f} MHz vs datasheet "
                 f"{datasheet_srf/1e6:.2f} MHz ({err*100:.1f}% error)")


def causality(s: Dict[str, np.ndarray]) -> Check:
    """Closed-form rational series network is causal by construction.

    We confirm there are no NaN/Inf (which would break Hilbert consistency).
    """
    finite = all(np.all(np.isfinite(v)) for v in s.values())
    return Check("Finite S-parameters", finite,
                 "finite sampled data; this check alone does not prove causality")


def run_all(s: Dict[str, np.ndarray], z: np.ndarray,
            model_srf: float, datasheet_srf: float | None) -> List[Check]:
    return [
        passivity(s),
        causality(s),
        stability(z),
        srf_consistency(model_srf, datasheet_srf),
    ]
