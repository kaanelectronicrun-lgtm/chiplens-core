"""End-to-end pipeline: load JSON -> resolve parasitics -> Z(f) -> S -> export.

Outputs per part: <part>.s2p, <part>.cir, <part>_report.md, <part>_Zf.csv
"""


from __future__ import annotations


import json


import os


from dataclasses import replace


from typing import Optional, Tuple


import numpy as np


from . import graphs as graphmod


from . import derate, impedance, importer, report, sparams, spice, validation


from .models import Capacitor, Inductor, SourceLevel


from .parasitics import resolve_capacitor, resolve_inductor


from .synth import fit_and_synthesize, synth_spice


def load_component(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    kind = data.pop("kind", None) or data.pop("type", None)
    if "source" in data and isinstance(data["source"], (int, str)):
        data["source"] = SourceLevel(int(data["source"]))
    # Derating curves may be given as a CSV path (relative to the JSON) instead
    # of an inline [[x, dC_%], ...] list -> load them (Murata SimSurfing CSV).
    _base = os.path.dirname(os.path.abspath(path))
    for _key in ("dc_bias_curve", "tcc_curve"):
        _v = data.get(_key)
        if isinstance(_v, str):
            _p = _v if os.path.isabs(_v) else os.path.join(_base, _v)
            data[_key] = derate.load_curve_csv(_p)
    if kind == "capacitor":
        return Capacitor(**data)
    if kind == "inductor":
        return Inductor(**data)
    raise ValueError(f"Unsupported or missing 'kind' in {path}: {kind!r}. "
                     "Only 'capacitor' or 'inductor' are supported.")


def _apply_conditions(cap: Capacitor):
    """Derate effective capacitance for DC bias / temperature / AC drive.

    Returns (updated_cap, log_lines). If any behavioral (non-curve) estimate is
    applied, the model source level is downgraded to PHYSICS_ESTIMATE so the
    report stays honest. No-op when no operating conditions are set.
    """
    if not (cap.dc_bias_v or cap.temp_c is not None or cap.ac_vrms):
        return cap, []
    factor, log, used_estimate = derate.compute_factor(
        cap.dielectric, cap.voltage_rating_v,
        dc_bias_v=cap.dc_bias_v, temp_c=cap.temp_c, ac_vrms=cap.ac_vrms,
        dc_bias_curve=cap.dc_bias_curve, tcc_curve=cap.tcc_curve)
    if factor == 1.0 or not log:
        return cap, log
    c_eff = cap.capacitance_f * factor
    src = SourceLevel.PHYSICS_ESTIMATE if used_estimate else cap.source
    updated = replace(cap, nominal_capacitance_f=cap.capacitance_f,
                      capacitance_f=c_eff, source=src)
    tag = "behavioral ESTIMATE" if used_estimate else "vendor curve"
    log.append(f"Effective C under operating conditions: {c_eff*1e9:.3f} nF "
               f"(nominal {cap.capacitance_f*1e9:.1f} nF, "
               f"{(factor-1)*100:+.1f}%) [{tag}]")
    return updated, log


def _graph_fit(comp, json_path: str, f_start: float, f_stop: float):
    """Graph-fit path: reconstruct Z from digitized curves and vector-fit it.

    Returns (freq, z, vf_info) where vf_info feeds the report. Falls back to the
    lumped model (vf_info=None) only if no graphs are attached.
    """
    base = os.path.dirname(os.path.abspath(json_path))
    gpaths = {k: (v if os.path.isabs(v) else os.path.join(base, v))
              for k, v in comp.graphs.items()}
    gf = graphmod.reconstruct(gpaths, comp.kind, comp.srf_hz)
    eff_c = (importer.effective_capacitance(gf.freq, gf.z)
             if comp.kind == "capacitor" else None)
    # Adaptive order: most accurate VF that still synthesizes to a passive RLC ladder.
    vf, synth, meta = fit_and_synthesize(gf.freq, gf.z, max_pairs=8)
    # Stay within the digitized range — VF extrapolation outside data is unsafe.
    lo, hi = max(f_start, float(gf.freq.min())), min(f_stop, float(gf.freq.max()))
    f = impedance.build_sweep(lo, hi, srf_hz=gf.srf_hz)
    z = vf.eval(2j * np.pi * f)
    # Passivity enforcement: a series Z is passive iff Re(Z) >= 0. Vector fitting
    # is unconstrained, so floor Re(Z) at the measured ESR minimum (perturbation).
    floor = max(1e-3, float(np.min(gf.z.real)))
    perturbed = int(np.count_nonzero(z.real < floor))
    z = np.maximum(z.real, floor) + 1j * z.imag
    # A positive-element clamp can change the fitted circuit substantially.
    # Refuse exports whose SPICE circuit would disagree with the Touchstone Z.
    z_circuit = synth.impedance(2j * np.pi * f)
    mismatch = np.abs(z_circuit - z) / np.maximum(np.abs(z), 1e-12)
    if not np.all(np.isfinite(mismatch)) or np.max(mismatch) > 1e-4:
        raise ValueError("inconsistent graph-fit SPICE/Touchstone model; export refused")
    vf_info = {"rms_rel": vf.rms_rel, "n_poles": len(vf.poles),
               "graph_srf": gf.srf_hz, "graphs": gf.source_graphs,
               "passivity_floor": floor, "perturbed_points": perturbed,
               "synth_branches": len(synth.branches),
               "synth_passive": meta["passive_synth"],
               "synth_warnings": meta["n_warnings"],
               "effective_c": eff_c}
    return f, z, vf_info, synth


def process(path: str, out_dir: str, z0: float = 50.0,
            f_start: float = 1e4, f_stop: float = 1e10,
            topology: str = "series",
            dc_bias_v: Optional[float] = None,
            temp_c: Optional[float] = None,
            ac_vrms: Optional[float] = None,
            dc_bias_curve=None,
            tcc_curve=None) -> Tuple[str, str, str]:
    comp = load_component(path)
    os.makedirs(out_dir, exist_ok=True)
    is_cap = comp.kind == "capacitor"
    graph_mode = bool(comp.graphs)

    # CLI/GUI operating-condition overrides take precedence over JSON values.
    if is_cap:
        ov = {}
        if dc_bias_v is not None:
            ov["dc_bias_v"] = dc_bias_v
        if temp_c is not None:
            ov["temp_c"] = temp_c
        if ac_vrms is not None:
            ov["ac_vrms"] = ac_vrms
        if dc_bias_curve is not None:
            ov["dc_bias_curve"] = dc_bias_curve
        if tcc_curve is not None:
            ov["tcc_curve"] = tcc_curve
        if ov:
            comp = replace(comp, **ov)

    return _run(comp, out_dir, z0, f_start, f_stop, topology, graph_path=path)


def _run(comp, out_dir: str, z0: float, f_start: float, f_stop: float,
         topology: str, graph_path: Optional[str] = None,
         extra_log: Optional[list] = None) -> Tuple[str, str, str]:
    """Core: resolve -> (derate) -> Z(f) -> S -> write .s2p/.cir/report/.csv."""
    os.makedirs(out_dir, exist_ok=True)
    is_cap = comp.kind == "capacitor"
    graph_mode = bool(comp.graphs)

    if is_cap:
        comp, est_log = resolve_capacitor(comp)
        if not graph_mode:
            comp, dlog = _apply_conditions(comp)
            est_log += dlog
        spice_text = spice.capacitor_spice(comp)
    else:
        comp, est_log = resolve_inductor(comp)
        spice_text = spice.inductor_spice(comp)
    if extra_log:
        est_log = list(extra_log) + est_log

    vf_info: Optional[dict] = None
    if graph_mode:
        f, z, vf_info, synth = _graph_fit(comp, graph_path, f_start, f_stop)
        comp.source = SourceLevel.GRAPH_FIT  # measured-curve fit outranks physics
        # Effective (low-signal) capacitance from the impedance curve replaces
        # nominal for the model; nominal is retained only for the delta report.
        if is_cap and vf_info.get("effective_c"):
            comp = replace(comp, nominal_capacitance_f=comp.capacitance_f,
                           capacitance_f=vf_info["effective_c"])
            est_log.append(
                f"Effective C from impedance curve: {vf_info['effective_c']*1e9:.3f} nF "
                f"(nominal {comp.nominal_capacitance_f*1e9:.1f} nF) "
                "[freq-domain derived, preferred over nominal]")
        spice_text = synth_spice(synth, comp.part_number)  # RLC-ladder, not lumped
        est_log.append(
            f"Z(f) vector-fit to digitized graph(s): {vf_info['n_poles']} poles, "
            f"rel. RMS error {vf_info['rms_rel']*100:.2f}% [GRAPH FIT ***]")
        est_log.append(
            f"SPICE = synthesized RLC ladder ({vf_info['synth_branches']} branches, "
            f"{'passive' if vf_info['synth_passive'] else 'perturbed'})")
    else:
        f = impedance.build_sweep(f_start, f_stop, srf_hz=comp.srf_hz)
        z = (impedance.capacitor_impedance(comp, f) if is_cap
             else impedance.inductor_impedance(comp, f))

    s = sparams.z_to_s(z, z0, topology)
    model_srf = impedance.srf_check(z, f, comp.kind)
    checks = validation.run_all(s, z, model_srf, comp.srf_hz)

    topo = (topology or "series").lower()
    base = os.path.join(out_dir, _safe(comp.part_number) + f"_{topo}")
    s2p_path = base + ".s2p"
    cir_path = base + ".cir"
    rep_path = base + "_report.md"
    csv_path = base + "_Zf.csv"

    sparams.write_touchstone(s2p_path, f, s, z0, comp.part_number,
                             comment="\n".join(est_log), topology=topo)
    with open(cir_path, "w", encoding="utf-8") as fh:
        fh.write(spice_text)
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write(report.build_report(comp, est_log, checks, model_srf, z0,
                                     vf_info, topology=topo))
    _write_zf_csv(csv_path, f, z)
    return s2p_path, cir_path, rep_path


def _write_zf_csv(path: str, f: np.ndarray, z: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as fh:
        fh.write("freq_Hz,Re_Z_ohm,Im_Z_ohm,mag_Z_ohm,phase_deg\n")
        mag = np.abs(z)
        ph = np.angle(z, deg=True)
        for i in range(len(f)):
            fh.write(f"{f[i]:.6e},{z[i].real:.6e},{z[i].imag:.6e},"
                     f"{mag[i]:.6e},{ph[i]:.4f}\n")


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s) or "part"
