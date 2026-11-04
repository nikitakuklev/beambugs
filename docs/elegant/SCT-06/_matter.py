"""Shared setup for the MATTER reproducers: a thin helium target tracked by Elegant's single-scattering model.

One MATTER element of length 0.01 m filled with helium (Z = 2, A = 4, density 0.02813 kg/m^3),
traversed by electrons of momentum 1000 m_e c. PLIMIT bounds the scattering probability of one
section, and the element is divided into the number of sections that keeps it below the bound.
"""

from __future__ import annotations

import numpy as np
from _elegant import particles, run

P0 = 1000.0  # reference and beam momentum, units of m_e c
LENGTH = 0.01  # m
RHO, Z, A = 0.0281327461321535, 2, 4  # kg/m^3 (dilute helium), atomic number, mass number

# SDDS constants.h and the MATTER source, as Elegant uses them.
RE, AMU = 2.81794092e-15, 1.6605e-27  # classical electron radius (m), atomic mass unit (kg)
BS_Y0 = 1e-8  # lower cutoff of the recoil and bremsstrahlung energy-transfer samplers


# Mean nuclear-bremsstrahlung events per electron: 4 L/(3 X0) [-ln y0 - (1 - y0) + 3/8 (1 - y0^2)],
# with the helium radiation length X0 = 94.32 g/cm^2 (PDG atomic and nuclear properties of helium
# gas) divided by the density, converted from cm to m. Bremsstrahlung is rare in this thin target.
X0 = 94.32 / (RHO * 1e-3) * 1e-2  # m
BREMS_PER_ELECTRON = 4 * LENGTH / (3 * X0) * (-np.log(BS_Y0) - (1 - BS_Y0) + 3 / 8 * (1 - BS_Y0**2))


def track(mode, element, seed, columns):
    """Track `columns` (beam arrays with p in m_e c) through `MATTER` + element text; return columns.

    `element` holds the MATTER parameters after the fixed Z, A and RHO. Rows are returned ordered by
    particleID, which the beam must carry (MPI output order varies).
    """
    lattice = f"M: MATTER,Z={Z},A={A},RHO={RHO!r},{element}\nBL: LINE=(M)\n"
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
        f"output=out.sdds, random_number_seed={seed} &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
    # particles() rejects nonfinite coordinates; the output must also hold exactly the input particles.
    out = particles(result["out.sdds"])
    if len(out["p"]) != len(columns["p"]):
        raise SystemExit(f"{mode}: {len(columns['p']) - len(out['p'])} particles lost")
    if not np.array_equal(out["particleID"], np.sort(columns["particleID"])):
        raise SystemExit(f"{mode}: the output particleIDs differ from the input ones")
    return out


def beam(count, xp=0.0, yp=0.0):
    """`count` electrons on axis at momentum P0 with the given slopes, numbered from 1."""
    return {
        "x": np.zeros(count),
        "xp": np.full(count, xp),
        "y": np.zeros(count),
        "yp": np.zeros(count) + yp,
        "t": np.zeros(count),
        "p": np.full(count, P0),
        "particleID": np.arange(1, count + 1),
    }


def gamma(p):
    """Lorentz factor of momentum p in m_e c."""
    return np.sqrt(1 + np.asarray(p) ** 2)
