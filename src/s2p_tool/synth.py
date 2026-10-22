"""RLC-ladder SPICE synthesis from a vector-fit pole/residue model.

Realises  Z(s) = d + s*e + sum_n r_n/(s - p_n)  as a Foster-I network: series
branches between chained nodes. Exact agreement with the vector-fit model
requires a realizable fit without material element clamps. Check warnings and
compare impedances; the export pipeline rejects inconsistent graph models.

Branch realisation:
    constant d   -> series resistor R0 = d
    linear  e    -> series inductor  L0 = e
    real pole    r/(s+a)               -> parallel R-C   (C=1/r, R=r/a)
    cplx pair    (k1 s + k0)/(s^2+a1 s+a0)
                 -> parallel [C=1/k1] || [R=1/G] || [series R-L]
                    with G=(a1-k0/k1)/k1, Rrl_const=a0-G*k0,
                    L=k1/Rrl_const, Rrl=k0/Rrl_const

Non-positive elements (from a fit that is not strictly positive-real) are clamped
to a tiny positive value and recorded as warnings — never emitted as negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .vectorfit import VFModel, vector_fit

_TINY = 1e-15


@dataclass
class Branch:
    kind: str                 # 'RC' or 'RLC'
    values: dict              # element name -> value (F, H, Ohm)

    def impedance(self, s: np.ndarray) -> np.ndarray:
        v = self.values
        if self.kind == "RC":
            return 1.0 / (s * v["C"] + 1.0 / v["R"])
        y = s * v["Cp"] + 1.0 / v["Rp"] + 1.0 / (s * v["L"] + v["Rs"])
        return 1.0 / y


@dataclass
class Synth:
    r0: float                 # series resistor (constant term)
    l0: float                 # series inductor (linear term)
    branches: List[Branch]
    warnings: List[str] = field(default_factory=list)

    def impedance(self, s: np.ndarray) -> np.ndarray:
        z = np.full(s.shape, self.r0, dtype=complex) + s * self.l0
        for b in self.branches:
            z = z + b.impedance(s)
        return z


def _pos(x: float, name: str, warn: List[str]) -> float:
    if x <= 0 or not np.isfinite(x):
        warn.append(f"{name} would be {x:.3e} (non-physical) -> clamped to {_TINY:g}")
        return _TINY
    return x


def synthesize(vf: VFModel) -> Synth:
    warn: List[str] = []
    r0 = _pos(vf.d, "R0 (constant term)", warn) if vf.d != 0 else _TINY
    l0 = _pos(vf.e, "L0 (linear term)", warn) if vf.e != 0 else 0.0

    branches: List[Branch] = []
    used = np.zeros(len(vf.poles), dtype=bool)
    for i, p in enumerate(vf.poles):
        if used[i]:
            continue
        r = vf.residues[i]
        if abs(p.imag) <= 1e-6 * (abs(p) + 1e-30):      # real pole
            used[i] = True
            a = -p.real if abs(p.real) > _TINY else _TINY  # p = -a, a>0 stable
            rr = r.real if abs(r.real) > _TINY else _TINY
            C = _pos(1.0 / rr, f"C(real pole {i})", warn)
            R = _pos(rr / a, f"R(real pole {i})", warn)
            branches.append(Branch("RC", {"C": C, "R": R}))
        else:                                            # complex conjugate pair
            # find and consume the conjugate partner
            for j in range(i + 1, len(vf.poles)):
                if not used[j] and abs(vf.poles[j] - np.conj(p)) < 1e-6 * (abs(p) + 1e-30):
                    used[j] = True
                    break
            used[i] = True
            alpha, beta = -p.real, p.imag
            a1, a0 = 2 * alpha, alpha ** 2 + beta ** 2
            k1 = 2 * r.real
            k0 = 2 * (r.real * alpha - r.imag * beta)
            k1s = k1 if abs(k1) > _TINY else _TINY
            G = (a1 - k0 / k1s) / k1s
            rrl_const = a0 - G * k0
            rrl_const = rrl_const if abs(rrl_const) > _TINY else _TINY
            Cp = _pos(1.0 / k1s, f"Cp(pair {i})", warn)
            Rp = _pos(1.0 / G, f"Rp(pair {i})", warn)
            L = _pos(k1 / rrl_const, f"L(pair {i})", warn)
            Rs = _pos(k0 / rrl_const, f"Rs(pair {i})", warn)
            branches.append(Branch("RLC", {"Cp": Cp, "Rp": Rp, "L": L, "Rs": Rs}))
    return Synth(r0, l0, branches, warn)


def fit_and_synthesize(freq: np.ndarray, z: np.ndarray, max_pairs: int = 8,
                       n_iter: int = 10, weighting: str = "none"):
    """Adaptive order: pick the highest VF order that still synthesizes to a
    fully positive-real (passive, all-positive-element) network.

    Higher order = lower fit error but eventually overfits into non-physical
    (negative) elements. We prefer the most accurate model that stays clean;
    if none is perfectly clean we fall back to the fewest-warning candidate.

    weighting: passed to vector_fit ("rel" minimises relative error so deep
    impedance nulls fit as well as peaks). Passivity is still enforced here, so
    a weighted fit cannot produce a non-physical (negative-element) netlist.

    Returns (vf, synth, meta) where meta records the chosen order and cleanliness.
    """
    best = None  # (clean, n_warn, rms, n_pairs, vf, synth)
    for npair in range(1, max_pairs + 1):
        vf = vector_fit(freq, z, n_pairs=npair, n_iter=n_iter, weighting=weighting)
        synth = synthesize(vf)
        nw = len(synth.warnings)
        cand = (nw == 0, -nw, -vf.rms_rel, npair, vf, synth)
        if best is None:
            best = cand
            continue
        # Prefer clean; among clean prefer lower rms (i.e. higher -rms); else fewer warns.
        if (cand[0], cand[1], cand[2]) > (best[0], best[1], best[2]):
            best = cand
    clean, neg_w, neg_rms, npair, vf, synth = best
    meta = {"n_pairs": npair, "passive_synth": clean, "n_warnings": -neg_w}
    return vf, synth, meta


def synth_spice(synth: Synth, part_number: str) -> str:
    name = "".join(ch if ch.isalnum() else "_" for ch in part_number) or "PART"
    lines = [
        "* ===========================================================",
        f"* Broadband RLC-ladder model (Foster-I) for {part_number}",
        f"* Synthesized from vector-fit: {len(synth.branches)} branches",
        "* Compare against the fitted response; element clamps can change it.",
        "* NOTE: behavioral / estimated model - not measured.",
        "* ===========================================================",
        f".subckt {name} 1 2",
    ]
    if synth.warnings:
        for w in synth.warnings:
            lines.append(f"* WARNING: {w}")
    prev = "1"
    nidx = 0
    last = len(synth.branches) - 1 if synth.branches else -1

    lines.append(f"Rser {prev} n{nidx} {synth.r0:.9e}")
    prev = f"n{nidx}"; nidx += 1
    if synth.l0 > 0:
        lines.append(f"Lser {prev} n{nidx} {synth.l0:.9e}")
        prev = f"n{nidx}"; nidx += 1

    for bi, b in enumerate(synth.branches):
        nb = "2" if bi == last else f"n{nidx}"
        if bi != last:
            nidx += 1
        if b.kind == "RC":
            lines.append(f"C{bi} {prev} {nb} {b.values['C']:.9e}")
            lines.append(f"R{bi} {prev} {nb} {b.values['R']:.9e}")
        else:
            mid = f"m{bi}"
            lines.append(f"Cp{bi} {prev} {nb} {b.values['Cp']:.9e}")
            lines.append(f"Rp{bi} {prev} {nb} {b.values['Rp']:.9e}")
            lines.append(f"L{bi} {prev} {mid} {b.values['L']:.9e}")
            lines.append(f"Rs{bi} {mid} {nb} {b.values['Rs']:.9e}")
        prev = nb
    if not synth.branches:
        lines.append(f"R_short {prev} 2 {_TINY:.9e}")
    lines.append(f".ends {name}")
    return "\n".join(lines) + "\n"
