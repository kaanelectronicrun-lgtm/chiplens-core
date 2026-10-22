"""Vendor Touchstone (.s2p) importer — MODEL SOURCE HIERARCHY level 1/2.

Reads a measured 2-port Touchstone (series-mode DUT, e.g. Murata SimSurfing),
converts S -> Z, extracts the real component parameters, and feeds the same
vector-fit + RLC-synthesis path used elsewhere. Output is flagged MEASURED so the
report never confuses it with an estimate.

Series-through 2-port:  Z = 2*Z0*(1 - S21)/S21.
Only RI/MA, 2-port, R 50 Touchstone v1 is handled (what SimSurfing exports).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Measured:
    freq: np.ndarray
    z: np.ndarray          # complex series impedance
    z0: float
    part_number: str


def load_touchstone(path: str) -> Measured:
    fmt = "ri"
    z0 = 50.0
    part = ""
    f, cols = [], []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("!"):
                if "Part Number" in line:
                    part = line.split(":", 1)[-1].strip()
                continue
            if line.startswith("#"):
                toks = line.lower().split()
                if "ma" in toks:
                    fmt = "ma"
                elif "db" in toks:
                    fmt = "db"
                if "r" in toks:
                    try:
                        z0 = float(toks[toks.index("r") + 1])
                    except (ValueError, IndexError):
                        pass
                continue
            v = [float(x) for x in line.replace("E", "e").split()]
            f.append(v[0])
            cols.append(v[1:])
    freq = np.array(f)
    data = np.array(cols)

    def pair(a, b):
        if fmt == "ri":
            return a + 1j * b
        mag = 10 ** (a / 20) if fmt == "db" else a
        return mag * np.exp(1j * np.deg2rad(b))

    s21 = pair(data[:, 2], data[:, 3])      # S21 columns
    z = 2 * z0 * (1 - s21) / s21
    return Measured(freq, z, z0, part or "imported_part")


def effective_capacitance(freq: np.ndarray, z: np.ndarray) -> float:
    """Effective (low-signal) capacitance from frequency-domain data.

    For an MLCC the nominal value is NOT the value that governs the impedance:
    Class-II dielectrics (X5R/X7R) lose capacitance with DC bias, AC level and
    age. We read C from the capacitive band (well below SRF, where Im(Z)<0 and
    Z ~= 1/(jwC)) of the MEASURED data:  C = 1/(2*pi*f*|Im Z|).
    """
    mag = np.abs(z)
    srf = float(freq[int(np.argmin(mag))])
    band = (freq < srf / 5) & (z.imag < 0)
    if not band.any():
        band = z.imag < 0
    return float(np.median(1.0 / (2 * np.pi * freq[band] * np.abs(z[band].imag))))


def extract_capacitor(m: Measured) -> Dict[str, float]:
    mag = np.abs(m.z)
    i = int(np.argmin(mag))
    srf = float(m.freq[i])
    esr = float(m.z[i].real)
    c = effective_capacitance(m.freq, m.z)   # effective, not nominal
    # ESL from the inductive band (above SRF, reactance positive)
    bandl = (m.freq > srf * 5) & (m.z.imag > 0)
    esl = (float(np.median(np.abs(m.z[bandl].imag) / (2 * np.pi * m.freq[bandl])))
           if bandl.any() else 1.0 / ((2 * np.pi * srf) ** 2 * c))
    return {"capacitance_f": c, "esr_ohm": esr, "esl_h": esl, "srf_hz": srf}


def extract_inductor(m: Measured) -> Dict[str, float]:
    mag = np.abs(m.z)
    i = int(np.argmax(mag))                 # parallel resonance peak
    srf = float(m.freq[i])
    # DCR from the lowest-frequency real part
    dcr = float(m.z[:3].real.mean())
    # L from the inductive band (below SRF, reactance positive)
    band = (m.freq < srf / 5) & (m.z.imag > 0)
    l = float(np.median(np.abs(m.z[band].imag) / (2 * np.pi * m.freq[band])))
    cp = 1.0 / ((2 * np.pi * srf) ** 2 * l)

    # Core-loss Rp from the resonance peak (parallel model). At SRF the series
    # reactance and Cp cancel, so Z_peak is real and set by the parallel losses:
    #   1/Z_peak = Re(1/(DCR+jwL)) + 1/Rp   ->   Rp = 1 / (1/Z_peak - g_series)
    # This is the tank Q expressed as a resistance:  Q_tank = Rp / (w_srf * L).
    w_srf = 2 * np.pi * srf
    z_peak = float(mag[i])
    g_series = dcr / (dcr ** 2 + (w_srf * l) ** 2)
    g_rp = 1.0 / z_peak - g_series
    rp = 1.0 / g_rp if g_rp > 1e-12 else 1e6
    return {"inductance_h": l, "dcr_ohm": dcr, "cp_f": cp, "srf_hz": srf,
            "rp_ohm": rp, "q_tank": rp / (w_srf * l)}
