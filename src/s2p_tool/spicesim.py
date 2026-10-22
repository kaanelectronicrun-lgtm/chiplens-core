"""Independent nodal-analysis AC simulator — SPICE round-trip validation.

Parses a `.cir` subcircuit body (R, L, C between named nodes) as a black box and
solves the nodal admittance system at each frequency. This is deliberately a
SEPARATE code path from the synthesis math, so agreement between this simulator
and the model proves the emitted netlist text actually realises the claimed Z(f)
(catches wrong nodes, swapped values, format bugs) — a real round-trip.

Driving convention for a 2-terminal series subckt `.subckt NAME 1 2 ... .ends`:
inject 1 A into port "1", treat port "2" as ground reference => V(1) = Z(f).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

_ELEM = re.compile(r"^([RLC])\w*\s+(\S+)\s+(\S+)\s+([\d.eE+\-]+)", re.I)


def parse_subckt(cir_text: str) -> Tuple[List[Tuple[str, str, str, float]], str, str]:
    """Return ([(kind, na, nb, value), ...], port1, port2) from a .subckt body."""
    elems: List[Tuple[str, str, str, float]] = []
    p1 = p2 = None
    in_sub = False
    for line in cir_text.splitlines():
        s = line.strip()
        if s.lower().startswith(".subckt"):
            toks = s.split()
            p1, p2 = toks[2], toks[3]
            in_sub = True
            continue
        if s.lower().startswith(".ends"):
            in_sub = False
            continue
        if not in_sub or not s or s.startswith("*"):
            continue
        m = _ELEM.match(s)
        if m:
            elems.append((m.group(1).upper(), m.group(2), m.group(3), float(m.group(4))))
    if p1 is None:
        raise ValueError("no .subckt found in netlist")
    return elems, p1, p2


def impedance(cir_text: str, freq: np.ndarray) -> np.ndarray:
    """Z(f) of a 2-terminal series subckt by independent nodal analysis."""
    elems, p1, p2 = parse_subckt(cir_text)
    # Node indexing; ground = port2 (reference), excluded from the matrix.
    nodes: Dict[str, int] = {}
    for _, na, nb, _v in elems:
        for n in (na, nb):
            if n != p2 and n not in nodes:
                nodes[n] = len(nodes)
    n = len(nodes)
    z = np.empty(len(freq), dtype=complex)

    def adm(kind: str, val: float, w: float) -> complex:
        if kind == "R":
            return 1.0 / val
        if kind == "C":
            return 1j * w * val
        return 1.0 / (1j * w * val)             # L

    for k, f in enumerate(freq):
        w = 2 * np.pi * f
        Y = np.zeros((n, n), dtype=complex)
        for kind, na, nb, val in elems:
            y = adm(kind, val, w)
            ia = nodes.get(na, -1)
            ib = nodes.get(nb, -1)
            if ia >= 0:
                Y[ia, ia] += y
            if ib >= 0:
                Y[ib, ib] += y
            if ia >= 0 and ib >= 0:
                Y[ia, ib] -= y
                Y[ib, ia] -= y
        inj = np.zeros(n, dtype=complex)
        inj[nodes[p1]] = 1.0                    # 1 A into port1
        v = np.linalg.solve(Y, inj)
        z[k] = v[nodes[p1]]                      # V(port1) = Z (port2 = gnd)
    return z


def roundtrip(cir_text: str, freq: np.ndarray, z_model: np.ndarray,
              z_meas: np.ndarray | None = None) -> dict:
    """Round-trip: simulate the .cir text with the independent solver and compare.

    `z_model` = the impedance the synthesis claims; agreement proves the emitted
    netlist realises it. `z_meas` (optional) = measured reference for accuracy.
    Returns relative errors (netlist vs claim, and sim vs measured).
    """
    z_sim = impedance(cir_text, freq)
    denom = np.max(np.abs(z_model)) + 1e-30
    netlist_err = float(np.max(np.abs(z_sim - z_model)) / denom)
    out = {"z_sim": z_sim, "netlist_rel_err": netlist_err,
           "netlist_ok": netlist_err < 1e-4}
    if z_meas is not None:
        ref = np.sqrt(np.mean(np.abs(z_meas) ** 2)) + 1e-30
        out["sim_vs_meas_rms"] = float(
            np.sqrt(np.mean(np.abs(z_sim - z_meas) ** 2)) / ref)
    return out
