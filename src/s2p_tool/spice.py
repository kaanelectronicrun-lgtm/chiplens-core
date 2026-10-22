"""SPICE (.cir) equivalent-circuit netlist generation.

Targets LTspice / ANSYS SIwave / HFSS Circuit / ADS subcircuit syntax.
The subckt has two external nodes (1, 2) so it drops straight into a series path.
"""
from __future__ import annotations

from .models import Capacitor, Inductor


def capacitor_spice(cap: Capacitor) -> str:
    """Series ESL - ESR - C between port 1 and port 2."""
    name = _san(cap.part_number)
    return f"""* ===========================================================
* Equivalent circuit: CAPACITOR {cap.part_number}
* Series RLC model:  port1 -- ESL -- ESR -- C -- port2
* C   = {cap.capacitance_f:.6e} F
* ESR = {cap.esr_ohm:.6e} Ohm
* ESL = {cap.esl_h:.6e} H
* SRF = {(cap.srf_hz or 0)/1e6:.3f} MHz
* NOTE: behavioral / estimated model — not measured.
* ===========================================================
.subckt {name} 1 2
Lesl 1 n1 {cap.esl_h:.6e}
Resr n1 n2 {cap.esr_ohm:.6e}
Ccap n2 2 {cap.capacitance_f:.6e}
.ends {name}
"""


def inductor_spice(ind: Inductor) -> str:
    """Series DCR-L, shunted by parasitic Cp and core-loss Rp across the ports."""
    name = _san(ind.part_number)
    return f"""* ===========================================================
* Equivalent circuit: INDUCTOR {ind.part_number}
* (DCR + L) series, with Cp and core-loss Rp in parallel across ports
* L   = {ind.inductance_h:.6e} H
* DCR = {ind.dcr_ohm:.6e} Ohm
* Cp  = {ind.cp_f:.6e} F
* Rp  = {ind.rp_ohm:.6e} Ohm  (core loss)
* SRF = {(ind.srf_hz or 0)/1e6:.3f} MHz
* NOTE: behavioral / estimated model — not measured.
* ===========================================================
.subckt {name} 1 2
Rdcr 1 n1 {ind.dcr_ohm:.6e}
Lind n1 2 {ind.inductance_h:.6e}
Cpar 1 2 {ind.cp_f:.6e}
Rpar 1 2 {ind.rp_ohm:.6e}
.ends {name}
"""


def _san(s: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in s)
    return out or "PART"
