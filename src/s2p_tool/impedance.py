"""Frequency-dependent impedance models and the log-spaced sweep.

Returns complex Z(f) arrays. The sweep densifies points around the SRF so the
resonance notch/peak is well sampled, per the FREQUENCY MODELING spec.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .models import Capacitor, Inductor

TWO_PI = 2.0 * math.pi


def build_sweep(f_start: float = 1e4, f_stop: float = 1e10,
                points: int = 401, srf_hz: Optional[float] = None,
                extra_around_srf: int = 120) -> np.ndarray:
    """Log-spaced frequency vector, extra density in +/- one decade of SRF."""
    base = np.logspace(math.log10(f_start), math.log10(f_stop), points)
    if srf_hz and f_start < srf_hz < f_stop:
        lo = max(f_start, srf_hz / 10.0)
        hi = min(f_stop, srf_hz * 10.0)
        dense = np.logspace(math.log10(lo), math.log10(hi), extra_around_srf)
        base = np.unique(np.concatenate([base, dense]))
    return base


def capacitor_impedance(cap: Capacitor, f: np.ndarray) -> np.ndarray:
    """Series RLC: Z = ESR + j(omega*ESL - 1/(omega*C))."""
    w = TWO_PI * f
    return cap.esr_ohm + 1j * (w * cap.esl_h - 1.0 / (w * cap.capacitance_f))


def inductor_impedance(ind: Inductor, f: np.ndarray) -> np.ndarray:
    """(DCR + jwL) in parallel with Cp, the parallel branch shunted by Rp.

    Z_series = DCR + jwL
    Y_total  = 1/Z_series + jw*Cp + 1/Rp
    Z        = 1 / Y_total
    """
    w = TWO_PI * f
    z_series = ind.dcr_ohm + 1j * w * ind.inductance_h
    y_total = 1.0 / z_series + 1j * w * ind.cp_f + 1.0 / ind.rp_ohm
    return 1.0 / y_total


def srf_check(z: np.ndarray, f: np.ndarray, kind: str) -> float:
    """Return the empirical SRF from the model (|Z| min for cap, max for ind)."""
    mag = np.abs(z)
    idx = int(np.argmin(mag)) if kind == "capacitor" else int(np.argmax(mag))
    return float(f[idx])
