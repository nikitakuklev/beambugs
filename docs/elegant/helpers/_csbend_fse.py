"""Shared case for the CSBEND FSE_CORRECTION reproducers: an asymmetric pure dipole with soft edges.

`track` runs one on-axis electron through the bend with optional twiss_output (which computes the
element matrix) and returns its exit coordinates, physical path and the FSE value that Elegant
printed when it fitted the correction (None when it did not fit).
"""

from __future__ import annotations

import re
import sys

import numpy as np
from _elegant import particles, run

C = 299792458.0  # speed of light (m/s)
P0 = 6000.0  # reference momentum in m_e c
LENGTH = 0.75
# Unequal edge angles make the edge-induced exit slope nonzero, so the correction has work to do.
BEND = (
    f"CSBEND,L={LENGTH!r},ANGLE=0.09,E1=0.02,E2=-0.015,HGAP=0.025,FINT=0.5,EDGE_ORDER=1,"
    "N_SLICES=80,INTEGRATION_ORDER=6,SYNCH_RAD=0,ISR=0"
)
N = 12  # identical particles, so that every MPI worker tracks some


def track(mode: str, model: int, params: dict, twiss: bool) -> tuple[dict, float | None]:
    """Exit coordinates (x, xp, y, yp, path) of the on-axis electron and the fitted FSE, if any."""
    extra = "".join(f",{k}={v!r}" for k, v in params.items())
    lattice = f"B: {BEND},EDGE1_EFFECTS={model},EDGE2_EFFECTS={model}{extra}\nEND: MARK\nBL: LINE=(B,END)\n"
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
        + ("&twiss_output filename=out.twi, matched=0, beta_x=2, beta_y=3 &end\n" if twiss else "")
        + "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": {"p": np.full(N, P0)}}
    )
    out = particles(result["out.sdds"])
    if len(out["p"]) != N or any(np.ptp(out[c]) > 0 for c in ("x", "xp", "y", "yp", "t", "p")):
        sys.exit(f"{mode}: particles lost or not identical: invalid")
    p = out["p"][0]
    coords = {c: float(out[c][0]) for c in ("x", "xp", "y", "yp")}
    coords["path"] = float(out["t"][0] * C * p / np.sqrt(1 + p * p))
    fits = re.findall(r"FSE optimized to (\S+)", result["log"])
    return coords, (float(fits[-1]) if fits else None)
