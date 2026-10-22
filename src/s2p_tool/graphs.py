"""Datasheet graph ingestion + complex-impedance reconstruction.

Datasheets plot magnitude curves (|Z| vs f, ESR vs f). To build a model we need
the *complex* Z(f). We reconstruct it as:

    Re(Z) = ESR(f)               (from the ESR graph, else the resonance floor)
    |Im(Z)| = sqrt(|Z|^2 - Re^2)
    sign(Im) from the SRF        (capacitor: -below/+above ; inductor: +below/-above)

CSV format (header optional, comma or whitespace):  freq_Hz, value
Digitize datasheet curves with e.g. WebPlotDigitizer and export the points here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class GraphFit:
    freq: np.ndarray      # reconstruction grid [Hz]
    z: np.ndarray         # complex impedance on that grid
    srf_hz: float         # SRF read off the magnitude curve
    source_graphs: Tuple[str, ...]


def read_graph_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read (freq, value) points; tolerant of headers and delimiters."""
    xs, ys = [], []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh, skipinitialspace=True):
            if not row:
                continue
            cells = row if len(row) >= 2 else row[0].split()
            try:
                x, y = float(cells[0]), float(cells[1])
            except (ValueError, IndexError):
                continue  # header / comment line
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        raise ValueError(f"{path}: need >=3 numeric (freq,value) rows")
    f = np.asarray(xs)
    v = np.asarray(ys)
    order = np.argsort(f)
    return f[order], v[order]


def _loginterp(f_new: np.ndarray, f: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Interpolate in log-log space (datasheet curves are near power-laws)."""
    lf, lfn = np.log10(f), np.log10(f_new)
    lv = np.log10(np.clip(v, 1e-30, None))
    return 10.0 ** np.interp(lfn, lf, lv)


def reconstruct(graphs: Dict[str, str], kind: str,
                srf_hz: Optional[float] = None, points: int = 400) -> GraphFit:
    if "impedance" not in graphs:
        raise ValueError("graph-fit needs an 'impedance' curve (|Z| vs f)")
    fz, mag = read_graph_csv(graphs["impedance"])
    grid = np.geomspace(fz.min(), fz.max(), points)
    magg = _loginterp(grid, fz, mag)

    # Re(Z): ESR graph if present, else the magnitude floor (resonance minimum).
    if "esr" in graphs:
        fe, esr = read_graph_csv(graphs["esr"])
        re = _loginterp(grid, fe, esr)
    else:
        re = np.full_like(grid, float(np.min(magg)))
    re = np.minimum(re, magg)  # Re can't exceed |Z|

    # SRF from the magnitude curve: dip for capacitor, peak for inductor.
    if srf_hz is None:
        idx = int(np.argmin(magg)) if kind == "capacitor" else int(np.argmax(magg))
        srf_hz = float(grid[idx])

    im_abs = np.sqrt(np.clip(magg ** 2 - re ** 2, 0.0, None))
    if kind == "capacitor":
        sign = np.where(grid < srf_hz, -1.0, 1.0)
    else:  # inductor: inductive below SRF, capacitive above
        sign = np.where(grid < srf_hz, 1.0, -1.0)
    z = re + 1j * sign * im_abs

    used = tuple(graphs[k] for k in ("impedance", "esr") if k in graphs)
    return GraphFit(grid, z, srf_hz, used)
