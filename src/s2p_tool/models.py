"""Component parameter data models.

Two supported component classes only: Capacitor and Inductor.
All values SI base units unless the field name says otherwise:
    F, H, Ohm, Hz.

A `source_level` (1..6) per the spec's MODEL SOURCE HIERARCHY is carried so the
accuracy report can be honest about where each number came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class SourceLevel(IntEnum):
    """MODEL SOURCE HIERARCHY — lower number = higher trust."""
    MEASURED_SPARAM = 1   # ★★★★★
    VENDOR_TOUCHSTONE = 2  # ★★★★
    VENDOR_SPICE = 3       # ★★★★
    GRAPH_FIT = 4          # ★★★
    DATASHEET_TABLE = 5    # ★★
    PHYSICS_ESTIMATE = 6   # ★

    @property
    def stars(self) -> str:
        return {1: "★★★★★", 2: "★★★★", 3: "★★★★",
                4: "★★★", 5: "★★", 6: "★"}[int(self)]

    @property
    def label(self) -> str:
        return {1: "Measured", 2: "Vendor Model", 3: "Vendor Model",
                4: "Graph Fit", 5: "Datasheet Fit", 6: "Physics Estimate"}[int(self)]


@dataclass
class Capacitor:
    part_number: str
    capacitance_f: float                 # C used in the model [F] (effective if freq-data exists)
    nominal_capacitance_f: Optional[float] = None  # datasheet nominal, when C above is effective
    voltage_rating_v: Optional[float] = None
    esr_ohm: Optional[float] = None      # ESR at the reference frequency
    esl_h: Optional[float] = None        # ESL [H]; if None -> estimated from SRF
    srf_hz: Optional[float] = None       # self-resonant frequency [Hz]
    dissipation_factor: Optional[float] = None  # DF (tan delta) at esr_ref_hz
    esr_ref_hz: Optional[float] = None   # frequency at which ESR / DF is quoted
    dielectric: Optional[str] = None     # e.g. "X7R", "C0G/NP0"
    case: Optional[str] = None           # EIA/metric case, e.g. "0603" -> ESL calc
    length_mm: Optional[float] = None    # body length (datasheet) -> ESL fallback
    width_mm: Optional[float] = None
    graphs: Optional[dict] = None        # {"impedance": csv, "esr": csv} -> graph fit
    # Operating conditions (Murata-SimSurfing style) -> capacitance derating:
    dc_bias_v: Optional[float] = None    # applied DC bias [V]
    temp_c: Optional[float] = None       # operating temperature [degC]
    ac_vrms: Optional[float] = None      # AC drive level [Vrms]
    dc_bias_curve: Optional[list] = None # digitized [[Vdc, dC_%], ...] (vendor)
    tcc_curve: Optional[list] = None     # digitized [[T_degC, dC_%], ...] (vendor)
    source: SourceLevel = SourceLevel.DATASHEET_TABLE
    notes: str = ""

    kind: str = field(default="capacitor", init=False)


@dataclass
class Inductor:
    part_number: str
    inductance_h: float                  # nominal L [H]
    dcr_ohm: Optional[float] = None      # DC resistance [Ohm]
    q_factor: Optional[float] = None     # Q at q_ref_hz
    q_ref_hz: Optional[float] = None
    srf_hz: Optional[float] = None
    cp_f: Optional[float] = None         # parasitic shunt capacitance [F]
    rp_ohm: Optional[float] = None       # core-loss / parallel loss resistance
    irms_a: Optional[float] = None
    isat_a: Optional[float] = None
    core_material: Optional[str] = None  # e.g. "Ferrite", "Iron Powder", "Air"
    graphs: Optional[dict] = None        # {"impedance": csv, "esr": csv} -> graph fit
    source: SourceLevel = SourceLevel.DATASHEET_TABLE
    notes: str = ""

    kind: str = field(default="inductor", init=False)
