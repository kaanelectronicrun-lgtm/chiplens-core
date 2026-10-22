"""Vector fitting (Gustavsen) — rational pole/residue approximation.

Fits a complex frequency response f(s) over s = j*omega with:

    f(s) ~= sum_n  r_n/(s - p_n)  +  d  +  s*e

Real-valued formulation: complex poles are kept as conjugate pairs and the
unknown residues (rr, ri) are real, so the resulting model has real coefficients
(physically realisable). Unstable poles (Re(p) > 0) are reflected into the left
half-plane every iteration, so the fit is always stable & causal by construction.

Reference: B. Gustavsen & A. Semlyen, "Rational approximation of frequency
domain responses by vector fitting", IEEE Trans. Power Delivery, 1999.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_TOL_REAL = 1e-6  # |Im(p)|/|p| below this -> treat pole as real


@dataclass
class VFModel:
    poles: np.ndarray      # complex poles (full set, conjugates included)
    residues: np.ndarray   # complex residues aligned with poles
    d: float               # constant term
    e: float               # linear (s) term
    rms_rel: float         # relative RMS fit error over the sample set

    def eval(self, s: np.ndarray) -> np.ndarray:
        out = np.full(s.shape, self.d, dtype=complex) + s * self.e
        for p, r in zip(self.poles, self.residues):
            out += r / (s - p)
        return out


def _is_real(p: complex) -> bool:
    return abs(p.imag) <= _TOL_REAL * (abs(p) + 1e-30)


def _basis(s: np.ndarray, poles: List[complex]):
    """Build the real-unknown partial-fraction basis.

    Returns (cols, meta). `cols` is a list of complex columns; `meta` records
    how to turn the solved real coefficients back into complex residues.
    Real pole  -> 1 column  (real residue).
    Cplx pair  -> 2 columns: [1/(s-p)+1/(s-p*)],  [j*(1/(s-p)-1/(s-p*))].
    """
    cols = []
    meta = []  # ('real', p) or ('cplx', p)
    for p in poles:
        if _is_real(p):
            cols.append(1.0 / (s - p.real))
            meta.append(("real", complex(p.real)))
        else:
            d1 = 1.0 / (s - p)
            d2 = 1.0 / (s - np.conj(p))
            cols.append(d1 + d2)
            cols.append(1j * (d1 - d2))
            meta.append(("cplx", p))
    return cols, meta


def _starting_poles(w: np.ndarray, n_pairs: int) -> List[complex]:
    lo = max(w.min(), 1e-12)
    beta = np.geomspace(lo, w.max(), n_pairs)
    return [complex(-b / 100.0, b) for b in beta]


def _solve_real(A_cplx: np.ndarray, rhs: np.ndarray,
                weight: np.ndarray | None = None) -> np.ndarray:
    """Least squares of a complex system with REAL unknowns: stack Re/Im rows.

    Columns span many decades (partial fractions vs. the s*e term), so each is
    normalised to unit norm before solving and the solution rescaled after —
    without this the small sigma residues get truncated and the poles never move.

    `weight` (per-sample) lets the caller minimise *relative* error: weighting by
    1/|response| makes deep impedance nulls (the decoupling minimum) fit as well
    as the high-impedance regions, instead of being ignored by the LS.
    """
    if weight is not None:
        A_cplx = A_cplx * weight[:, None]
        rhs = rhs * weight
    scale = np.sqrt((A_cplx.real ** 2 + A_cplx.imag ** 2).sum(axis=0))
    scale[scale == 0] = 1.0
    An = A_cplx / scale
    A = np.vstack([An.real, An.imag])
    b = np.concatenate([rhs.real, rhs.imag])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x / scale


def _relocate(poles: List[complex], c: np.ndarray) -> List[complex]:
    """New poles = eigenvalues of (A - b c^T) in the real state-space form."""
    blocks = []
    bvec = []
    ci = 0
    for p in poles:
        if _is_real(p):
            blocks.append(np.array([[p.real]]))
            bvec.append(1.0)
            ci += 1
        else:
            pr, pi = p.real, p.imag
            blocks.append(np.array([[pr, pi], [-pi, pr]]))
            bvec.extend([2.0, 0.0])
            ci += 2
    n = sum(b.shape[0] for b in blocks)
    A = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        A[i:i + k, i:i + k] = b
        i += k
    H = A - np.outer(np.array(bvec), c[:n])
    ev = np.linalg.eigvals(H)
    # Enforce stability and collapse conjugate pairs to Im>=0 representatives.
    new: List[complex] = []
    used = np.zeros(len(ev), dtype=bool)
    for idx, e in enumerate(ev):
        if used[idx]:
            continue
        pr = -abs(e.real) if e.real > 0 else e.real  # reflect unstable
        if abs(e.imag) <= _TOL_REAL * (abs(e) + 1e-30):
            new.append(complex(pr, 0.0))
            used[idx] = True
        else:
            new.append(complex(pr, abs(e.imag)))
            # mark the conjugate as consumed
            for j in range(len(ev)):
                if not used[j] and abs(ev[j] - np.conj(e)) < 1e-9 * (abs(e) + 1e-30):
                    used[j] = True
                    break
            used[idx] = True
    return new


def _expand(poles: List[complex], coeffs, meta, d: float, e: float) -> VFModel:
    full_poles: List[complex] = []
    full_res: List[complex] = []
    ci = 0
    for kind, p in meta:
        if kind == "real":
            full_poles.append(complex(p.real))
            full_res.append(complex(coeffs[ci]))
            ci += 1
        else:
            rr, ri = coeffs[ci], coeffs[ci + 1]
            r = complex(rr, ri)
            full_poles.extend([p, np.conj(p)])
            full_res.extend([r, np.conj(r)])
            ci += 2
    return VFModel(np.array(full_poles), np.array(full_res), d, e, 0.0)


def vector_fit(freq: np.ndarray, response: np.ndarray, n_pairs: int = 6,
               n_iter: int = 6, fit_linear: bool = True,
               weighting: str = "none") -> VFModel:
    """Fit response(freq) with n_pairs complex-conjugate pole pairs.

    freq: Hz (real, >0).  response: complex.  Returns a stable VFModel.
    weighting: "none" (minimise absolute error) or "rel" (minimise relative
    error, weight 1/|response| — fits deep nulls as well as peaks).
    """
    s = 2j * np.pi * freq
    w = np.abs(s.imag)
    poles = _starting_poles(w, n_pairs)
    f = response.astype(complex)
    wt = 1.0 / (np.abs(f) + 1e-30) if weighting == "rel" else None

    for _ in range(n_iter):
        cols, meta = _basis(s, poles)
        n_aug = len(cols)                      # residue cols for f
        sigma_cols = [-f * col for col in cols]  # sigma residue cols
        extra = [np.ones_like(s)]              # d
        if fit_linear:
            extra.append(s)                    # s*e
        A = np.column_stack(cols + extra + sigma_cols)
        x = _solve_real(A, f, wt)
        c = x[n_aug + len(extra):]             # sigma residues (real coeffs)
        poles = _relocate(poles, np.asarray(c, dtype=float))

    # Final residue identification with the converged poles.
    cols, meta = _basis(s, poles)
    n_aug = len(cols)
    extra = [np.ones_like(s)]
    if fit_linear:
        extra.append(s)
    A = np.column_stack(cols + extra)
    x = _solve_real(A, f, wt)
    coeffs = x[:n_aug]
    d = float(x[n_aug])
    e = float(x[n_aug + 1]) if fit_linear else 0.0

    model = _expand(poles, coeffs, meta, d, e)
    fit = model.eval(s)
    denom = np.sqrt(np.mean(np.abs(f) ** 2)) + 1e-30
    model.rms_rel = float(np.sqrt(np.mean(np.abs(fit - f) ** 2)) / denom)
    return model
