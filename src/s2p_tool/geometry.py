"""Package-geometry -> ESL estimation (and therefore SRF).

ESL is NOT derivable from capacitance, but it IS a function of the component's
physical current loop, which scales with case size. The datasheet always gives
case dimensions, so we can estimate ESL from the package and then COMPUTE the
self-resonant frequency  SRF = 1/(2*pi*sqrt(L*C)).

The case->ESL table below holds typical mounted ESL values for standard 2-terminal
MLCCs (EIA case codes), consistent with vendor S-parameter data across the
industry. Reverse-geometry / low-inductance (LICC) parts run lower; verify with
measured S-parameters for sign-off.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

# EIA imperial case code -> typical mounted ESL [H].
_CASE_ESL_H = {
    "0201": 0.25e-9,
    "0402": 0.45e-9,
    "0603": 0.70e-9,
    "0805": 0.90e-9,
    "1206": 1.20e-9,
    "1210": 1.30e-9,
    "1812": 1.60e-9,
}

# Metric (mm) L x W -> EIA code, for datasheets that quote metric.
_METRIC_TO_EIA = {
    (1.0, 0.5): "0402",
    (1.6, 0.8): "0603",
    (2.0, 1.25): "0805",
    (3.2, 1.6): "1206",
    (3.2, 2.5): "1210",
    (4.5, 3.2): "1812",
}


def normalize_case(case: str) -> Optional[str]:
    c = case.strip().lower().replace("m", "").replace(" ", "")
    # accept "0603", "0603 (1608m)", "1608" (metric)
    digits = "".join(ch for ch in c if ch.isdigit())
    if digits[:4] in _CASE_ESL_H:
        return digits[:4]
    metric = {"1005": "0402", "1608": "0603", "2012": "0805",
              "3216": "1206", "3225": "1210", "4532": "1812"}
    return metric.get(digits[:4])


def esl_from_case(case: str) -> Optional[float]:
    code = normalize_case(case)
    return _CASE_ESL_H.get(code) if code else None


def esl_from_dimensions(length_mm: float, width_mm: float) -> Tuple[float, str]:
    """Fallback when no case code: nearest standard size, else length scaling."""
    for (l, w), code in _METRIC_TO_EIA.items():
        if abs(length_mm - l) < 0.25 and abs(width_mm - w) < 0.25:
            return _CASE_ESL_H[code], code
    # Empirical: ESL ~ 0.44 nH per mm of body length for 2-terminal MLCCs.
    return 0.44e-9 * length_mm, f"~{length_mm}x{width_mm}mm"


def srf_from_lc(l_h: float, c_f: float) -> float:
    return 1.0 / (2.0 * math.pi * math.sqrt(l_h * c_f))
