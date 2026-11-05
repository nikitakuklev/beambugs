"""Transverse transfer of a static-field RFCA: tracked 2x2 map and Elegant's analytic matrix.

The cavity is RFCA with FREQ=0 and PHASE=90, so every particle gains exactly VOLT and the field is a
uniform accelerating gradient VOLT/L between hard edges, with no body focusing (the default model).
Transverse momentum is then conserved inside the cavity, and only the END1_FOCUS and END2_FOCUS
edge lenses focus. Energies are kinetic, momenta beta*gamma in units of m_e c.
"""

from __future__ import annotations

import sys

import numpy as np
from _elegant import particles, run

# SDDS constants.h, as Elegant uses them.
ME_MEV = 0.51099906
LENGTH, VOLT = 0.7, 12e6  # m, V: a 17.14 MV/m gradient
EPS = 1e-7  # m and rad: offsets and slopes that measure the linear map


def momentum(kinetic_mev: float) -> float:
    gamma = 1 + kinetic_mev / ME_MEV
    return float(np.sqrt(gamma**2 - 1))


def transfer(mode: str, kinetic_mev: float, kicks: int, end1: int, end2: int) -> dict:
    """Tracked map R (2x2, horizontal) and the matrix_output map M of the same element.

    Particles at x = +-EPS and at x' = +-EPS give the columns of the tracked map by central
    differences; tracking is linear in them, so the differences are exact up to rounding.
    """
    p0 = momentum(kinetic_mev)
    lattice = (
        f"R: RFCA,L={LENGTH!r},VOLT={VOLT!r},FREQ=0,PHASE=90,N_KICKS={kicks},"
        f"END1_FOCUS={end1},END2_FOCUS={end2},CHANGE_P0=0,T_REFERENCE=0\n"
        "BL: LINE=(R)\n"
    )
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
        "output=out.sdds, default_order=1 &end\n"
        "&matrix_output SDDS_output=matrix.sdds, individual_matrices=1 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    beam = {
        "x": np.array([EPS, -EPS, 0.0, 0.0]),
        "xp": np.array([0.0, 0.0, EPS, -EPS]),
        "p": np.full(4, p0),
        "particleID": np.arange(1, 5),
    }
    result = run(
        {"lattice.lte": lattice, "run.ele": commands},
        mode,
        {"beam.sdds": beam},
        outputs=("out.sdds", "matrix.sdds"),
    )
    out = particles(result["out.sdds"])
    if len(out["x"]) != 4:
        sys.exit(f"{mode}: particles were lost")
    x, xp = out["x"], out["xp"]
    tracked = np.array(
        [
            [(x[0] - x[1]) / (2 * EPS), (x[2] - x[3]) / (2 * EPS)],
            [(xp[0] - xp[1]) / (2 * EPS), (xp[2] - xp[3]) / (2 * EPS)],
        ]
    )
    matrix = result["matrix.sdds"]
    names = [str(n) for n in matrix[0, "ElementName"]]
    row = names.index("R")
    analytic = np.array([[float(matrix[0, f"R{i}{j}"][row]) for j in (1, 2)] for i in (1, 2)])
    gamma_in = 1 + kinetic_mev / ME_MEV
    gamma_out = float(np.hypot(out["p"][0], 1))
    if abs(gamma_out - gamma_in - VOLT / (1e6 * ME_MEV)) > 1e-12 * gamma_out:
        sys.exit(f"{mode}: the energy gain is not VOLT")
    return {"tracked": tracked, "matrix": analytic, "gamma_in": gamma_in, "gamma_out": gamma_out}
