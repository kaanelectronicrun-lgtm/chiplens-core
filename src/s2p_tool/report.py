"""Accuracy report + engineering review (Markdown).

Honesty rule from the spec: ESTIMATED data is never presented as measured. The
report separates MEASURED / VENDOR / ESTIMATED explicitly and prints the source
level (stars) and a confidence band.
"""
from __future__ import annotations

import os
from typing import List, Union

import numpy as np

from . import derate
from .models import Capacitor, Inductor, SourceLevel
from .validation import Check

Component = Union[Capacitor, Inductor]

# Confidence band per source level — see TODO below to tune for your lab's policy.
_CONFIDENCE = {
    SourceLevel.MEASURED_SPARAM: (95, 99),
    SourceLevel.VENDOR_TOUCHSTONE: (90, 97),
    SourceLevel.VENDOR_SPICE: (85, 95),
    SourceLevel.GRAPH_FIT: (70, 88),
    SourceLevel.DATASHEET_TABLE: (60, 80),
    SourceLevel.PHYSICS_ESTIMATE: (35, 60),
}


def confidence_band(level: SourceLevel, n_estimates: int) -> str:
    """Map source level (+ how many branches were guessed) to a % band.

    TODO(user): this is the one genuinely judgement-based knob in the tool.
    Each estimated parasitic erodes confidence by 5% here. If your SI sign-off
    policy is stricter (e.g. estimates above 1 GHz are untrustworthy), adjust
    the penalty or make it frequency-aware.
    """
    lo, hi = _CONFIDENCE[level]
    penalty = min(5 * n_estimates, hi - lo)
    return f"{lo - penalty}–{hi - penalty}%"


def build_report(comp: Component, est_log: List[str], checks: List[Check],
                 model_srf: float, z0: float = 50.0,
                 vf_info: dict | None = None,
                 measured_src: str | None = None,
                 topology: str = "series") -> str:
    n_est = 0 if measured_src else len(est_log)
    band = confidence_band(comp.source, n_est)
    is_cap = comp.kind == "capacitor"
    topo = (topology or "series").lower()
    topo_label = "Shunt-to-ground (bypass)" if topo == "shunt" else "Series-through (in-line)"

    out: List[str] = []
    out.append(f"# {comp.part_number} — Simulation Model Report\n")
    out.append(f"**Component class:** {comp.kind.title()}  ")
    out.append(f"**Reference impedance:** {z0:.0f} Ω  ")
    out.append(f"**Topology:** {topo_label}  ")
    out.append(f"**Source level:** {comp.source.stars} {comp.source.label}\n")

    # 1. Component Summary
    out.append("## 1. Component Summary\n")
    if is_cap:
        if comp.nominal_capacitance_f:
            delta = (comp.capacitance_f - comp.nominal_capacitance_f) / \
                comp.nominal_capacitance_f * 100
            _derived = ("condition-derated" if (getattr(comp, "dc_bias_v", None)
                        or getattr(comp, "temp_c", None) is not None
                        or getattr(comp, "ac_vrms", None)) else "freq-domain derived")
            out.append(f"- Capacitance (**effective**, {_derived}): "
                       f"**{comp.capacitance_f*1e9:.3f} nF** "
                       f"(nominal {comp.nominal_capacitance_f*1e9:.1f} nF, {delta:+.1f}%)")
        elif measured_src:
            out.append(f"- Capacitance (**effective**, measured): "
                       f"**{comp.capacitance_f*1e9:.3f} nF** "
                       "(low-signal, from frequency-domain data — not nominal)")
        else:
            out.append(f"- Capacitance (nominal — no freq-domain data): "
                       f"**{comp.capacitance_f*1e9:.3f} nF** "
                       f"({comp.capacitance_f*1e6:.4g} µF)")
        out.append(f"- Dielectric: {comp.dielectric or 'n/a'}")
        out.append(f"- Voltage rating: {comp.voltage_rating_v or 'n/a'} V")
        out.append(f"- ESR: {comp.esr_ohm*1e3:.2f} mΩ")
        out.append(f"- ESL: {comp.esl_h*1e9:.3f} nH")
        conds = []
        if getattr(comp, "dc_bias_v", None):
            conds.append(f"DC bias {comp.dc_bias_v:g} V")
        if getattr(comp, "temp_c", None) is not None:
            conds.append(f"{comp.temp_c:g} °C")
        if getattr(comp, "ac_vrms", None):
            conds.append(f"AC {comp.ac_vrms:g} Vrms")
        if conds:
            out.append(f"- **Operating conditions:** {', '.join(conds)} "
                       "(capacitance derated — see Accuracy Report)")
    else:
        out.append(f"- Inductance: **{comp.inductance_h*1e9:.3f} nH** "
                   f"({comp.inductance_h*1e6:.4g} µH)")
        out.append(f"- DCR: {comp.dcr_ohm*1e3:.2f} mΩ")
        out.append(f"- Parasitic Cp: {comp.cp_f*1e12:.3f} pF")
        out.append(f"- Core-loss Rp: {comp.rp_ohm/1e3:.1f} kΩ")
        out.append(f"- Core material: {comp.core_material or 'n/a'}")
        out.append(f"- Irms / Isat: {comp.irms_a or 'n/a'} A / {comp.isat_a or 'n/a'} A")
    out.append(f"- SRF (datasheet): "
               f"{comp.srf_hz/1e6:.2f} MHz" if comp.srf_hz else "- SRF (datasheet): n/a")
    out.append(f"- SRF (model): {model_srf/1e6:.2f} MHz\n")

    # 2. Equivalent Circuit
    out.append("## 2. Equivalent Circuit\n")
    if is_cap:
        core = "[ ESL ]──[ ESR ]──[ C ]"
    else:
        core = "[ DCR ]──[ L ]  (∥ Cp ∥ Rp)"
    if topo == "shunt":
        out.append(f"```\nport1 ─────┬───── port2   (through line)\n"
                   f"           │\n"
                   f"        {core}\n"
                   f"           │\n"
                   f"          GND\n```\n")
    elif is_cap:
        out.append("```\nport1 ──[ ESL ]──[ ESR ]──[ C ]── port2\n```\n")
    else:
        out.append("```\nport1 ──[ DCR ]──[ L ]── port2\n"
                   "  └────────[ Cp ]────────┘\n"
                   "  └────────[ Rp ]────────┘   (core loss)\n```\n")

    # 2b. Curve fit (graph-fit path only)
    if vf_info:
        out.append("## 2b. Curve Fit (Vector Fitting)\n")
        out.append(f"- Method: **vector fitting** (Gustavsen pole-residue), "
                   f"{vf_info['n_poles']} poles")
        out.append(f"- Fitted to digitized graph(s): "
                   f"{', '.join(os.path.basename(g) for g in vf_info['graphs'])}")
        out.append(f"- Graph SRF: {vf_info['graph_srf']/1e6:.2f} MHz")
        out.append(f"- Relative RMS fit error: **{vf_info['rms_rel']*100:.2f}%**")
        out.append("- Stability: all poles in left half-plane — stable & causal")
        if vf_info.get("engine") == "scikit-rf":
            out.append(f"- **SPICE**: scikit-rf vector-fit S-parameter subcircuit, "
                       f"{vf_info['n_poles']} poles, "
                       f"passive={vf_info['synth_passive']} (passivity-tested). "
                       "Null-accurate (no Foster positivity constraint).\n")
        else:
            out.append(f"- Passivity: Re(Z) floored at "
                       f"{vf_info['passivity_floor']*1e3:.1f} mΩ (ESR min); "
                       f"{vf_info['perturbed_points']} point(s) perturbed")
            synth_state = ("all elements positive (passive)" if vf_info["synth_passive"]
                           else f"{vf_info['synth_warnings']} element(s) perturbed")
            out.append(f"- **SPICE**: synthesized RLC ladder, "
                       f"{vf_info['synth_branches']} Foster branches — {synth_state}. "
                       "Netlist impedance equals the fit by construction.\n")

    # 3. Validation
    out.append("## 3. Model Validation\n")
    out.append("| Check | Result | Detail |")
    out.append("|-------|--------|--------|")
    for c in checks:
        out.append(f"| {c.name} | {'✅ PASS' if c.passed else '❌ FAIL'} | {c.detail} |")
    out.append("")

    # 4. Accuracy Report
    out.append("## 4. Accuracy Report\n")
    out.append(f"- **Data source:** {comp.source.stars} {comp.source.label}")
    out.append(f"- **Heuristic source score:** {band} (not calibrated accuracy or probability)")
    out.append(f"- **Estimated branches:** {n_est}\n")

    # 5. Data provenance separation (spec-mandated)
    out.append("## 5. Data Provenance — kept separate\n")
    if measured_src:
        out.append(f"**MEASURED DATA** (vendor Touchstone — authoritative): "
                   f"`{measured_src}`. Parameters below are extracted from it.")
        for line in est_log:
            out.append(f"- {line}")
        out.append("\n**ESTIMATED DATA:** none — model derived from measurement.")
        out.append("**VENDOR DATA:** measured S-parameters (level ★★★★★).\n")
    else:
        out.append("**ESTIMATED DATA** (physics / SRF back-solve — *not measured*):")
        if est_log:
            for line in est_log:
                out.append(f"- {line}")
        else:
            out.append("- (none — all parasitics supplied directly)")
        out.append("\n**MEASURED DATA:** none supplied.")
        out.append("**VENDOR DATA:** none supplied.\n")

    # 6. Engineering Review
    out.append("## 6. Engineering Review (Senior SI/PI)\n")
    out.append(_engineering_review(comp, n_est, is_cap, bool(measured_src), topo))

    out.append("\n---")
    if measured_src:
        out.append("> ✅ Parameters and the `.s2p` are derived from "
                   "**manufacturer-measured** S-parameters (vendor Touchstone). The "
                   "synthesized `.cir` is a vector-fit of that measurement. Valid at "
                   "the measured bias/temperature condition only.")
    else:
        out.append("> ⚠️ This is a **behavioral / estimated** model. Do not present the "
                   "generated S-parameters as manufacturer-measured data.")
    return "\n".join(out)


def _engineering_review(comp: Component, n_est: int, is_cap: bool,
                        measured: bool = False, topology: str = "series") -> str:
    pts: List[str] = []
    derated = bool(getattr(comp, "dc_bias_v", None)
                   or getattr(comp, "temp_c", None) is not None
                   or getattr(comp, "ac_vrms", None))
    topo_txt = ("shunt-to-ground 2-port" if (topology or "series").lower() == "shunt"
                else "series-through 2-port")
    if measured:
        pts.append(f"**Assumptions:** linear, time-invariant; {topo_txt} at "
                   "50 Ω. Model reflects the measured bias/temperature condition of the "
                   "vendor file — re-import other DC-bias/temperature curves for those "
                   "operating points.")
    else:
        derate_txt = ("effective C derated for the stated operating conditions"
                      if derated else
                      "no DC-bias / temperature derating applied to the nominal value")
        pts.append("**Assumptions:** lumped, linear, time-invariant; single dominant "
                   f"resonance; {topo_txt} at 50 Ω; {derate_txt}.")
    if is_cap:
        pts.append("**High-frequency risk:** above SRF the part is inductive; mounting "
                   "inductance dominates and is layout-dependent — the ESL here is a "
                   "single lumped value, real boards vary ±50%.")
        if comp.dielectric and derate.dielectric_class(comp.dielectric) == "II":
            if derated:
                pts.append("**DC-bias / temperature:** Class-II dielectric "
                           f"({comp.dielectric}) — capacitance derated for the stated "
                           "conditions. Behavioral estimates are class-typical, not "
                           "part-exact; supply a vendor dc_bias_curve / tcc_curve for "
                           "datasheet accuracy.")
            else:
                pts.append("**DC-bias / temperature:** Class-II dielectric "
                           f"({comp.dielectric}) loses significant capacitance under DC "
                           "bias and temperature — set dc_bias_v / temp_c to derate.")
    else:
        pts.append("**High-frequency risk:** single-pole Cp captures the first SRF "
                   "only; secondary resonances and frequency-dependent core loss are "
                   "not modelled. Above ~SRF the behaviour is approximate.")
        pts.append("**Saturation:** Isat/Irms derating is not in the linear model — "
                   "large-signal behaviour will differ.")
    if measured:
        pts.append("**Limitations:** model is only as broadband as the measured file "
                   "(check its frequency span) and fixed to its bias/temperature point; "
                   "mounting inductance on your board adds to the measured ESL.")
    else:
        pts.append(f"**Limitations:** {n_est} parameter(s) were back-solved from physics. "
                   "Confidence degrades accordingly; validate against measured "
                   "S-parameters before sign-off on controlled-impedance or PI work.")
    return "\n\n".join(pts)
