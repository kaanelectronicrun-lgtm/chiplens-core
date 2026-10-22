"""Parasitic estimation — fill missing values from physics + SRF.

Every estimate downgrades the model's effective source level to PHYSICS_ESTIMATE
for the affected branch, and records a human-readable note so the engineering
review can flag it as ESTIMATED (never measured).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Tuple

from . import geometry
from .models import Capacitor, Inductor, SourceLevel

TWO_PI = 2.0 * math.pi


def _esl_from_srf(c_f: float, srf_hz: float) -> float:
    """SRF = 1/(2*pi*sqrt(L*C))  ->  L = 1/((2*pi*SRF)^2 * C)."""
    return 1.0 / ((TWO_PI * srf_hz) ** 2 * c_f)


def _cp_from_srf(l_h: float, srf_hz: float) -> float:
    return 1.0 / ((TWO_PI * srf_hz) ** 2 * l_h)


def resolve_capacitor(cap: Capacitor) -> Tuple[Capacitor, List[str]]:
    """Return a Capacitor with ESL / ESR guaranteed populated, plus a log."""
    log: List[str] = []
    esl = cap.esl_h
    esr = cap.esr_ohm
    srf = cap.srf_hz

    if esl is None:
        if srf:
            esl = _esl_from_srf(cap.capacitance_f, srf)
            log.append(f"ESL estimated from SRF: {esl*1e9:.3f} nH "
                       f"(SRF={srf/1e6:.1f} MHz) [PHYSICS ESTIMATE]")
        elif cap.case or (cap.length_mm and cap.width_mm):
            # Package geometry -> ESL, then COMPUTE SRF (datasheet gives the case).
            if cap.case and geometry.esl_from_case(cap.case):
                esl = geometry.esl_from_case(cap.case)
                src_txt = f"case {cap.case}"
            else:
                esl, code = geometry.esl_from_dimensions(cap.length_mm, cap.width_mm)
                src_txt = f"dimensions {code}"
            srf = geometry.srf_from_lc(esl, cap.capacitance_f)
            log.append(f"ESL computed from {src_txt}: {esl*1e9:.3f} nH; "
                       f"SRF computed = {srf/1e6:.2f} MHz [GEOMETRY ESTIMATE]")
        else:
            esl = 0.5e-9
            log.append("ESL not derivable (no SRF, no case) -> default 0.5 nH "
                       "[PHYSICS ESTIMATE, low confidence]")

    if esr is None:
        if cap.dissipation_factor and cap.esr_ref_hz:
            # ESR = DF / (2*pi*f*C)
            esr = cap.dissipation_factor / (TWO_PI * cap.esr_ref_hz * cap.capacitance_f)
            log.append(f"ESR derived from DF={cap.dissipation_factor} @ "
                       f"{cap.esr_ref_hz/1e3:.0f} kHz: {esr*1e3:.1f} mOhm [DATASHEET FIT]")
        else:
            esr = 0.01
            log.append("ESR unknown -> default 10 mOhm [PHYSICS ESTIMATE, low confidence]")

    src = cap.source if cap.esl_h and cap.esr_ohm else SourceLevel.PHYSICS_ESTIMATE
    return replace(cap, esl_h=esl, esr_ohm=esr, srf_hz=srf, source=src), log


def resolve_inductor(ind: Inductor) -> Tuple[Inductor, List[str]]:
    """Return an Inductor with DCR / Cp / Rp populated, plus a log."""
    log: List[str] = []
    dcr = ind.dcr_ohm
    cp = ind.cp_f
    rp = ind.rp_ohm

    if cp is None:
        if ind.srf_hz:
            cp = _cp_from_srf(ind.inductance_h, ind.srf_hz)
            log.append(f"Cp estimated from SRF: {cp*1e12:.3f} pF "
                       f"(SRF={ind.srf_hz/1e6:.1f} MHz) [PHYSICS ESTIMATE]")
        else:
            cp = 0.1e-12
            log.append("Cp not derivable (no SRF) -> default 0.1 pF "
                       "[PHYSICS ESTIMATE, low confidence]")

    if dcr is None:
        dcr = 0.05
        log.append("DCR unknown -> default 50 mOhm [PHYSICS ESTIMATE, low confidence]")

    if rp is None:
        # Core-loss resistor sized from Q at its quoted frequency, if available:
        #   Q = Rp / (omega*L)  (parallel model)  ->  Rp = Q * omega * L
        if ind.q_factor and ind.q_ref_hz:
            rp = ind.q_factor * TWO_PI * ind.q_ref_hz * ind.inductance_h
            log.append(f"Rp (core loss) from Q={ind.q_factor} @ "
                       f"{ind.q_ref_hz/1e6:.1f} MHz: {rp/1e3:.1f} kOhm [DATASHEET FIT]")
        else:
            rp = 1e6  # effectively lossless parallel branch
            log.append("Rp unknown -> 1 MOhm (low core loss assumption) [PHYSICS ESTIMATE]")

    src = ind.source if (ind.dcr_ohm and ind.cp_f) else SourceLevel.PHYSICS_ESTIMATE
    return replace(ind, dcr_ohm=dcr, cp_f=cp, rp_ohm=rp, source=src), log
