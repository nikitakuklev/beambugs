"""Shared setup for the smooth-IBS reproducers: a closed ring of identical sector bends with one IBSCATTER.

The ring has 16 identical SBENs (radius 10 m, K1 = -0.003 m^-2) and an IBSCATTER at its end. An
SBEN of radius R focuses with k_x = 1/R^2 + K1 and k_y = -K1, so a ring of identical bends has
constant matched beta = 1/sqrt(k), alpha = 0, and horizontal dispersion eta = (1/R)/k_x, eta' = 0.
The beam is 256 particles with exactly the requested second moments, matched to this optics.
"""

from __future__ import annotations

import numpy as np
from _elegant import particles, run

C = 299792458.0  # SDDS constants.h
P0 = 800.0  # reference momentum, units of m_e c
RADIUS, K1 = 10.0, -0.003  # m, 1/m^2
KX, KY = 1 / RADIUS**2 + K1, -K1
BETA_X, BETA_Y = 1 / np.sqrt(KX), 1 / np.sqrt(KY)  # m
ETA_X = (1 / RADIUS) / KX  # m
EX, EY, SIGMA_DELTA, SIGMA_Z = 2e-9, 3e-10, 1e-5, 0.004  # m, m, 1, m
COUNT = 256
# Largest applied fractional change of any plane's variance or spread is 2% at this FACTOR, which
# keeps the square-root scale factors of the smooth update near 1.
FACTOR = 0.09465332535958268


def beam(mean_delta):
    """Matched ring beam with mean momentum offset `mean_delta`; dispersive offset eta*delta in x.

    Orthonormalized normal samples, scaled by sqrt(COUNT), have zero mean and unit covariance, so the
    sample emittances, momentum spread and length equal the requested ones.
    """
    rng = np.random.default_rng(832)
    s = rng.normal(size=(COUNT, 6))
    s -= s.mean(axis=0)
    s = np.linalg.qr(s)[0] * np.sqrt(COUNT)
    delta = mean_delta + SIGMA_DELTA * s[:, 5]
    return {
        "x": np.sqrt(EX * BETA_X) * s[:, 0] + ETA_X * delta,
        "xp": np.sqrt(EX / BETA_X) * s[:, 1],
        "y": np.sqrt(EY * BETA_Y) * s[:, 2],
        "yp": np.sqrt(EY / BETA_Y) * s[:, 3],
        "t": SIGMA_Z / C * s[:, 4],
        "p": P0 * (1 + delta),
    }


def track(mode, columns, factor, planes=(1, 1, 1)):
    """Track `columns` once around the ring; `planes` are the DO_X, DO_Y, DO_Z switches."""
    lattice = (
        "Q: CHARGE,TOTAL=1e-9\n"
        f"B: SBEN,L={2 * np.pi * RADIUS / 16!r},ANGLE={2 * np.pi / 16!r},K1={K1!r},E1=0,E2=0\n"
        "I: IBSCATTER,ISRING=1,BUNCHED_BEAM_MODE=0,SMOOTH=1,FORCE_MATCHED_TWISS=1,"
        f"FACTOR={factor!r},DO_X={planes[0]},DO_Y={planes[1]},DO_Z={planes[2]}\n"
        "BL: LINE=(Q,16*B,I)\n"
    )
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
        "output=out.sdds, default_order=2 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output matched=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
    out = particles(result["out.sdds"])
    if len(out["p"]) != COUNT:
        raise SystemExit(f"{mode}: {COUNT - len(out['p'])} particles lost")
    return out


def spreads(out):
    """Variances of x, xp, y, yp and rms momentum deviation of the tracked beam."""
    return np.array(
        [np.var(out["x"]), np.var(out["xp"]), np.var(out["y"]), np.var(out["yp"])]
        + [np.std(out["p"])]
    )
