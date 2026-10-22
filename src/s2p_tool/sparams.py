"""Z(f) -> S-parameters (series or shunt 2-port) and Touchstone .s2p export.

A 2-terminal component can be placed two ways in a Z0 line:

SERIES (in-line, e.g. DC-block / matching):
    S11 = S22 = Z / (Z + 2*Z0)
    S21 = S12 = 2*Z0 / (Z + 2*Z0)
  Low-Z passes (S21->1), high-Z blocks (S21->0).

SHUNT (to ground, e.g. bypass / decoupling):
    S11 = S22 = -Z0 / (2*Z + Z0)
    S21 = S12 = 2*Z  / (2*Z + Z0)
  Low-Z shorts the line (S21->0), high-Z passes (S21->1).

Both are reciprocal (S21=S12) and symmetric (S11=S22) by construction, so the
generated network is passive and causal as long as Re(Z) >= 0.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import numpy as np

TOPOLOGIES = ("series", "shunt")


def series_z_to_s(z: np.ndarray, z0: float = 50.0) -> Dict[str, np.ndarray]:
    denom = z + 2.0 * z0
    s11 = z / denom
    s21 = (2.0 * z0) / denom
    return {"S11": s11, "S21": s21, "S12": s21.copy(), "S22": s11.copy()}


def shunt_z_to_s(z: np.ndarray, z0: float = 50.0) -> Dict[str, np.ndarray]:
    denom = 2.0 * z + z0
    s11 = (-z0) / denom
    s21 = (2.0 * z) / denom
    return {"S11": s11, "S21": s21, "S12": s21.copy(), "S22": s11.copy()}


def z_to_s(z: np.ndarray, z0: float = 50.0,
           topology: str = "series") -> Dict[str, np.ndarray]:
    """Convert Z(f) to 2-port S per topology ('series' or 'shunt')."""
    topo = (topology or "series").lower()
    if topo == "shunt":
        return shunt_z_to_s(z, z0)
    if topo == "series":
        return series_z_to_s(z, z0)
    raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")


def write_touchstone(path: str, f: np.ndarray, s: Dict[str, np.ndarray],
                     z0: float = 50.0, part: str = "", comment: str = "",
                     topology: str = "series") -> None:
    """Write a Touchstone v1 .s2p file:  # Hz S RI R 50  (real/imag).

    Column order per Touchstone 2-port: S11 S21 S12 S22.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    topo = (topology or "series").lower()
    topo_desc = ("shunt-to-ground (bypass/decoupling)" if topo == "shunt"
                 else "series-through (in-line)")
    lines = [
        f"! Touchstone 2-port S-parameters  ({topo_desc} model)",
        f"! Part: {part}",
        f"! Generated: {stamp}  by s2p tool",
        f"! Reference impedance: {z0:.1f} Ohm",
    ]
    if comment:
        for cl in comment.splitlines():
            lines.append(f"! {cl}")
    lines.append("! ESTIMATED / behavioral model - NOT manufacturer-measured data.")
    lines.append(f"# Hz S RI R {z0:g}")
    lines.append("! freq        ReS11      ImS11      ReS21      ImS21"
                 "      ReS12      ImS12      ReS22      ImS22")

    order = ["S11", "S21", "S12", "S22"]
    for i, freq in enumerate(f):
        vals = [f"{freq:.6e}"]
        for key in order:
            vals.append(f"{s[key][i].real:+.6e}")
            vals.append(f"{s[key][i].imag:+.6e}")
        lines.append(" ".join(vals))

    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\n".join(lines) + "\n")
