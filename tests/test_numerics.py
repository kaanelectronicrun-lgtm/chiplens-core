import os, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2p_tool.vectorfit import vector_fit
from s2p_tool.synth import fit_and_synthesize, synth_spice
def test_recovers_known_rational():
    """A 2-pair rational function is fitted to machine precision and stays stable."""
    f = np.geomspace(1e4, 1e10, 400)
    s = 2j * np.pi * f
    p1, p2 = complex(-1e7, 8e7), complex(-5e7, 1.2e9)
    r1, r2 = complex(3e7, -1e7), complex(-2e8, 5e7)
    resp = (r1 / (s - p1) + np.conj(r1) / (s - np.conj(p1))
            + r2 / (s - p2) + np.conj(r2) / (s - np.conj(p2)) + 0.3 + 2e-12 * s)
    m = vector_fit(f, resp, n_pairs=4, n_iter=8)
    assert m.rms_rel < 1e-6, f"rms too high: {m.rms_rel}"
    assert all(p.real <= 1e-6 for p in m.poles), "unstable pole present"

def test_noisy_resonance_fit_under_2pct():
    """A noisy single-resonance |Z| curve fits within a few percent, stays stable."""
    f = np.geomspace(1e4, 1e9, 80)
    s = 2j * np.pi * f
    C, ESL = 1e-7, 0.9e-9
    z = 0.012 + 1j * (2 * np.pi * f * ESL - 1 / (2 * np.pi * f * C))
    rng = np.random.default_rng(1)
    z = z * (1 + rng.normal(0, 0.02, len(f)))
    m = vector_fit(f, z, n_pairs=6, n_iter=8)
    assert m.rms_rel < 0.05, f"rms too high: {m.rms_rel}"
    assert all(p.real <= 1e-6 for p in m.poles), "unstable pole present"

def test_synthesis_matches_fit_and_is_passive():
    """Adaptive synthesis: netlist Z equals the VF Z, all elements positive,
    SPICE element names unique."""
    f = np.geomspace(1e4, 1e9, 80)
    s = 2j * np.pi * f
    C, ESL = 1e-7, 0.9e-9
    z = 0.012 + 1j * (2 * np.pi * f * ESL - 1 / (2 * np.pi * f * C))
    rng = np.random.default_rng(1)
    z = z * (1 + rng.normal(0, 0.02, len(f)))

    vf, synth, meta = fit_and_synthesize(f, z, max_pairs=8)
    err = np.max(np.abs(synth.impedance(s) - vf.eval(s)) / (np.abs(vf.eval(s)) + 1e-30))
    assert err < 1e-9, f"synth impedance != fit: {err}"
    assert meta["passive_synth"], "chosen model is not passive"

    # All emitted element values strictly positive, names unique.
    names, vals = [], []
    for line in synth_spice(synth, "PART").splitlines():
        if line and line[0] in "RLC" and " " in line:
            tok = line.split()
            names.append(tok[0])
            vals.append(float(tok[-1]))
    assert len(names) == len(set(names)), f"duplicate element name: {names}"
    assert all(v > 0 for v in vals), f"non-positive element: {vals}"

def test_spice_roundtrip_netlist_realises_model():
    """The emitted .cir, re-simulated by the independent nodal solver, reproduces
    the synthesis impedance (proves the netlist text is correct)."""
    from s2p_tool import spicesim
    from s2p_tool.synth import fit_and_synthesize, synth_spice
    f = np.geomspace(1e4, 1e9, 200)
    w = 2 * np.pi * f
    z = 0.02 + 1j * (w * 0.6e-9 - 1.0 / (w * 1.0e-7))
    vf, synth, meta = fit_and_synthesize(f, z, max_pairs=8)
    cir = synth_spice(synth, "DUT")
    r = spicesim.roundtrip(cir, f, synth.impedance(2j * np.pi * f))
    assert r["netlist_ok"], f"netlist round-trip error too high: {r['netlist_rel_err']}"

def test_series_shunt_closed_form():
    """Series and shunt topologies match their closed-form S-params and are
    reciprocal + symmetric. Uses a plain resistive Z so the algebra is exact."""
    from s2p_tool import sparams
    z0 = 50.0
    z = np.array([10.0, 50.0, 200.0], dtype=complex)
    ss = sparams.series_z_to_s(z, z0)
    sh = sparams.shunt_z_to_s(z, z0)
    # series closed form
    assert np.allclose(ss["S11"], z / (z + 2 * z0))
    assert np.allclose(ss["S21"], (2 * z0) / (z + 2 * z0))
    # shunt closed form
    assert np.allclose(sh["S11"], (-z0) / (2 * z + z0))
    assert np.allclose(sh["S21"], (2 * z) / (2 * z + z0))
    # reciprocity + symmetry for both
    for s in (ss, sh):
        assert np.allclose(s["S21"], s["S12"])
        assert np.allclose(s["S11"], s["S22"])
    # dispatcher agrees
    assert np.allclose(sparams.z_to_s(z, z0, "series")["S21"], ss["S21"])
    assert np.allclose(sparams.z_to_s(z, z0, "shunt")["S21"], sh["S21"])

def test_series_shunt_srf_signature():
    """At SRF (Z->min) a series cap passes (|S21|->1) while a shunt cap blocks
    (|S21|->0) — the physical fingerprint that distinguishes the two models."""
    import tempfile
    from s2p_tool.pipeline import process

    out = tempfile.mkdtemp()
    j = os.path.join(os.path.dirname(__file__), "..", "components", "example_cap.json")

    def s21_mags(path):
        rows = [ln.split() for ln in Path(path).read_text(encoding="ascii").splitlines() if ln[:1].isdigit()]
        return [(float(r[3]) ** 2 + float(r[4]) ** 2) ** 0.5 for r in rows]

    s_series = process(j, out, 50.0, 1e4, 1e10, "series")[0]
    s_shunt = process(j, out, 50.0, 1e4, 1e10, "shunt")[0]
    assert s_series.endswith("_series.s2p")
    assert s_shunt.endswith("_shunt.s2p")
    assert max(s21_mags(s_series)) > 0.95, "series should pass near SRF"
    assert min(s21_mags(s_shunt)) < 0.05, "shunt should block near SRF"

def test_derate_class_and_dc_bias():
    """Class II derates under DC bias (monotonic loss); Class I stays put."""
    from s2p_tool import derate
    assert derate.dielectric_class("C0G") == "I"
    assert derate.dielectric_class("NP0") == "I"
    assert derate.dielectric_class("X7R") == "II"
    assert derate.dielectric_class(None) == "II"
    # X7R, 16 V rated: more bias -> less capacitance, monotonic, bounded.
    f0, _, e0 = derate.compute_factor("X7R", 16.0, dc_bias_v=0.0)
    f_half, _, _ = derate.compute_factor("X7R", 16.0, dc_bias_v=8.0)
    f_full, _, est = derate.compute_factor("X7R", 16.0, dc_bias_v=16.0)
    assert f0 == 1.0
    assert 0.0 < f_full < f_half < 1.0, (f_full, f_half)
    assert est is True, "behavioral estimate must flag"
    # C0G at the same bias: negligible change.
    f_c0g, _, _ = derate.compute_factor("C0G", 16.0, dc_bias_v=16.0)
    assert abs(f_c0g - 1.0) < 1e-9

def test_derate_vendor_curve_is_exact():
    """A digitized dc_bias_curve is interpolated exactly and is NOT flagged as
    a behavioral estimate (it is real vendor data)."""
    from s2p_tool import derate
    curve = [[0, 0.0], [10, -40.0]]  # -40% at 10 V, linear
    f, log, est = derate.compute_factor("X7R", 16.0, dc_bias_v=5.0,
                                        dc_bias_curve=curve)
    assert abs(f - 0.80) < 1e-6, f  # -20% at 5 V (midpoint)
    assert est is False
    assert any("vendor curve" in ln for ln in log)

def test_dc_bias_shifts_srf_up():
    """Derating lowers effective C, so the model SRF moves UP (1/2π√(LC))."""
    import json
    import tempfile
    from s2p_tool.pipeline import process

    out = tempfile.mkdtemp()
    base = json.loads((Path(__file__).resolve().parents[1] / "components" / "example_cap.json").read_text(encoding="utf-8"))

    def srf_of(path):
        mags = []
        for ln in Path(path).read_text(encoding="ascii").splitlines():
            if ln[:1].isdigit():
                v = ln.split()
                # |S21| max == series resonance (pass peak) for a series cap
                mags.append((float(v[0]),
                             (float(v[3]) ** 2 + float(v[4]) ** 2) ** 0.5))
        return max(mags, key=lambda x: x[1])[0]

    p0 = os.path.join(out, "n.json"); Path(p0).write_text(json.dumps(base), encoding="utf-8")
    s0 = process(p0, out, 50.0, 1e4, 1e10, "series")[0]
    d = dict(base); d["part_number"] = "biased"; d["dc_bias_v"] = 16.0
    p1 = os.path.join(out, "b.json"); Path(p1).write_text(json.dumps(d), encoding="utf-8")
    s1 = process(p1, out, 50.0, 1e4, 1e10, "series")[0]
    assert srf_of(s1) > srf_of(s0), (srf_of(s1), srf_of(s0))

def test_curve_csv_loading_and_json_path():
    """A SimSurfing-style CSV (with header) loads, and a JSON that references it
    by path drives exact (non-estimate) derating end to end."""
    import json
    import tempfile
    from s2p_tool import derate
    from s2p_tool.pipeline import load_component, process

    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "dcbias.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("DC Bias[V],Cap. Change Rate[%]\n0,0\n5,-25\n10,-60\n")
    curve = derate.load_curve_csv(csv_path)
    assert curve[0] == [0.0, 0.0] and curve[-1] == [10.0, -60.0]

    # JSON references the CSV by relative path -> load_component resolves it.
    cj = {"kind": "capacitor", "part_number": "CURVE", "capacitance_f": 1e-7,
          "voltage_rating_v": 16, "dielectric": "X7R", "srf_hz": 1.6e7,
          "dc_bias_v": 5.0, "dc_bias_curve": "dcbias.csv"}
    jp = os.path.join(d, "c.json"); Path(jp).write_text(json.dumps(cj), encoding="utf-8")
    comp = load_component(jp)
    assert comp.dc_bias_curve == curve  # string path resolved to list

    # -25% at 5 V (curve point) -> effective 75 nF, exact, source kept (not estimate).
    rep = process(jp, d, 50.0, 1e4, 1e10, "series")[2]
    txt = Path(rep).read_text(encoding="utf-8")
    assert "condition-derated" in txt
    assert "75.000 nF" in txt, txt[:600]
    assert "vendor curve" in txt

def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(globals()[name]) for name in ['test_recovers_known_rational', 'test_noisy_resonance_fit_under_2pct', 'test_synthesis_matches_fit_and_is_passive', 'test_spice_roundtrip_netlist_realises_model', 'test_series_shunt_closed_form', 'test_series_shunt_srf_signature', 'test_derate_class_and_dc_bias', 'test_derate_vendor_curve_is_exact', 'test_dc_bias_shifts_srf_up', 'test_curve_csv_loading_and_json_path'])
