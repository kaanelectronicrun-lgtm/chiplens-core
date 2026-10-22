"""Operating-condition derating of ceramic-capacitor capacitance.

Class II ceramics (X7R, X5R, X6S, X7S, X7T, X8R, Y5V, Z5U, ...) lose capacitance
under DC bias and at temperature extremes; Class I (C0G / NP0 / NPO) is
essentially stable. Inductors are not derated here.

Two accuracy tiers:

1. **Vendor curve (preferred).** If the JSON carries a digitized curve
   (``dc_bias_curve`` = ``[[Vdc, dC_%], ...]`` and/or ``tcc_curve`` =
   ``[[T_C, dC_%], ...]`` — exactly what Murata SimSurfing / a datasheet graph
   gives) the factor is interpolated from it. This is a real datapoint, not a
   guess, so the model keeps its source level.

2. **Behavioral estimate (fallback).** With no curve, a documented,
   dielectric-class-aware behavioral model is used. It is a rough class-typical
   estimate — NOT part-specific — and every use is logged so the report flags it.

Effective capacitance:  C_eff = C_nominal * f_dc * f_temp * f_ac  (each >= a floor).
"""
from __future__ import annotations

import csv
import math
from typing import List, Optional, Tuple


def load_curve_csv(path: str):
    """Read a derating curve CSV -> [[x, dC_%], ...] (sorted by x).

    Tolerant of headers, blank lines, and comma/semicolon/whitespace/tab
    delimiters — matches Murata SimSurfing 'Save as CSV' exports whose columns
    are e.g. ``DC Bias[V], Cap. Change Rate[%]`` or ``Temperature[degC], ...``.
    The first two numeric columns of each data row are taken.
    """
    pts = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh, skipinitialspace=True):
            if not row:
                continue
            cells = row if len(row) >= 2 else row[0].replace(";", " ").split()
            try:
                x, d = float(cells[0]), float(cells[1])
            except (ValueError, IndexError):
                continue  # header / label / comment row
            pts.append([x, d])
    if len(pts) < 2:
        raise ValueError(f"{path}: need >=2 numeric (x, dC_%) rows")
    pts.sort(key=lambda p: p[0])
    return pts

# EIA Class I dielectrics: temperature-compensating, negligible bias/temp derate.
_CLASS_I = {"C0G", "COG", "NP0", "NPO", "CH", "CG"}

# Behavioral-model coefficients (class-typical X7R/X5R, mid case size).
_DC_DMAX = 0.55   # max fractional loss as Vdc -> rated
_DC_N = 1.6       # Hill exponent (curve steepness)
_DC_K = 0.75      # normalized half-loss bias (u at which loss ~ Dmax/2)
_T_DROOP = 0.15   # Class-II parabolic droop: -15% near +/-100 degC from 25
_CI_TCC = 30e-6   # Class-I ~ +/-30 ppm/degC
_FLOOR = 0.02     # never derate below 2% of nominal (numerical safety)


def dielectric_class(dielectric: Optional[str]) -> str:
    """Return 'I' (stable) or 'II' (derating) from an EIA dielectric code."""
    if not dielectric:
        return "II"  # unknown ceramic -> assume Class II (conservative)
    code = dielectric.upper().replace("/", " ").replace("-", " ")
    for tok in code.split():
        if tok in _CLASS_I:
            return "I"
    # A leading C0G/NP0 without spaces (e.g. "C0G"):
    if any(code.startswith(c) for c in _CLASS_I):
        return "I"
    return "II"


def _interp_curve(curve, x: float) -> float:
    """Linear-interpolate dC_% at x from [[x0,d0],[x1,d1],...] (clamped ends)."""
    pts = sorted((float(a), float(b)) for a, b in curve)
    xs = [p[0] for p in pts]
    ds = [p[1] for p in pts]
    if x <= xs[0]:
        return ds[0]
    if x >= xs[-1]:
        return ds[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ds[i - 1] + t * (ds[i] - ds[i - 1])
    return ds[-1]


def _f_dc_behavioral(u: float) -> float:
    """Class-II DC-bias capacitance factor from normalized bias u = Vdc/Vrated."""
    u = max(0.0, u)
    hill = u ** _DC_N / (_DC_K ** _DC_N + u ** _DC_N)
    return 1.0 - _DC_DMAX * hill


def compute_factor(
    dielectric: Optional[str],
    voltage_rating_v: Optional[float],
    dc_bias_v: Optional[float] = None,
    temp_c: Optional[float] = None,
    ac_vrms: Optional[float] = None,
    dc_bias_curve=None,
    tcc_curve=None,
) -> Tuple[float, List[str], bool]:
    """Return (capacitance_multiplier, log_lines, used_estimate).

    ``used_estimate`` is True if any behavioral (non-curve) factor was applied,
    so the caller can downgrade the model's source level and flag the report.
    """
    log: List[str] = []
    cls = dielectric_class(dielectric)
    f_total = 1.0
    used_estimate = False

    # --- DC bias -----------------------------------------------------------
    if dc_bias_v:
        if dc_bias_curve:
            d = _interp_curve(dc_bias_curve, dc_bias_v)
            f = max(_FLOOR, 1.0 + d / 100.0)
            f_total *= f
            log.append(f"DC bias {dc_bias_v:g} V: {d:+.1f}% "
                       f"(vendor curve) -> x{f:.3f}")
        elif cls == "I":
            log.append(f"DC bias {dc_bias_v:g} V: Class I (C0G/NP0) -> negligible")
        elif voltage_rating_v:
            u = dc_bias_v / voltage_rating_v
            f = max(_FLOOR, _f_dc_behavioral(u))
            f_total *= f
            used_estimate = True
            log.append(f"DC bias {dc_bias_v:g} V ({u*100:.0f}% of {voltage_rating_v:g} V "
                       f"rated): {(f-1)*100:+.1f}% [Class-II behavioral ESTIMATE] -> x{f:.3f}")
        else:
            log.append(f"DC bias {dc_bias_v:g} V given but no voltage rating -> "
                       "cannot derate (skipped)")

    # --- Temperature -------------------------------------------------------
    if temp_c is not None and temp_c != 25.0:
        if tcc_curve:
            d = _interp_curve(tcc_curve, temp_c)
            f = max(_FLOOR, 1.0 + d / 100.0)
            f_total *= f
            log.append(f"Temp {temp_c:g} degC: {d:+.1f}% (vendor curve) -> x{f:.3f}")
        elif cls == "I":
            d = _CI_TCC * (temp_c - 25.0) * 100.0  # percent
            f = 1.0 + _CI_TCC * (temp_c - 25.0)
            f_total *= f
            log.append(f"Temp {temp_c:g} degC: {d:+.3f}% (Class I ~30 ppm/degC) "
                       f"-> x{f:.4f}")
        else:
            f = max(_FLOOR, 1.0 - _T_DROOP * ((temp_c - 25.0) / 100.0) ** 2)
            f_total *= f
            used_estimate = True
            log.append(f"Temp {temp_c:g} degC: {(f-1)*100:+.1f}% "
                       f"[Class-II behavioral ESTIMATE] -> x{f:.3f}")

    # --- AC drive (Class II only, second-order) ----------------------------
    if ac_vrms and cls == "II":
        # Mild rise then fall; small effect at signal levels. Behavioral only.
        f = 1.0 + 0.06 * ac_vrms * math.exp(-ac_vrms)
        f_total *= f
        used_estimate = True
        log.append(f"AC drive {ac_vrms:g} Vrms: {(f-1)*100:+.1f}% "
                   f"[Class-II behavioral ESTIMATE] -> x{f:.3f}")

    return f_total, log, used_estimate
