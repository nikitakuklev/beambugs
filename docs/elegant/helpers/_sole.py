"""Shared runner for the SOLE spin-tracking reproducers: one solenoid, one pass, spin tracking on.

`track` sends identical copies of one particle (so that each MPI worker tracks some) through a
lattice holding a single element and returns that particle's final coordinates and spin. Spin
components (spx, spy, spz) are in the fixed straight element frame; p is βγ. Constants are those of
SDDS `constants.h` and of the species table of Elegant's `change_particle` command, so that a
different choice of constants cannot pass for an algorithm defect.
"""

from __future__ import annotations

import sys

import numpy as np
from _elegant import particles, run

C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31  # SDDS constants.h
A_ELECTRON = 0.00115965218128  # electron and positron anomalous magnetic moment in Elegant
A_PROTON = 1.792847
A_MUON = 1.165921e-3
COPIES = 6
COORDINATES = ("x", "xp", "y", "yp", "t", "p")
SPIN = ("spx", "spy", "spz")


def track(
    mode: str,
    element: str,
    p0: float,
    start: dict[str, float],
    before: str = "",
    setup: str = "",
    after_setup: str = "",
    outputs: tuple[str, ...] = ("out.sdds",),
) -> tuple[dict[str, float], dict]:
    """Track COPIES copies of one particle through `S: <element>` and return its exit state.

    start: initial x, xp, y, yp, spx, spy, spz (missing values are zero; p is p0). before: commands
    ahead of run_setup (change_particle); setup: extra run_setup fields; after_setup: commands
    between run_control and the beam (matrix_output). Returns ({name: value}, run result).
    """
    lattice = f"S: {element}\nBL: LINE=(S)\n"
    commands = (
        before
        + f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
        f"spin_tracking=1, default_order=2{setup} &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        + after_setup
        + "&sdds_beam input=beam.sdds &end\n&track &end\n"
    )
    beam = {n: np.full(COPIES, float(start.get(n, 0.0))) for n in ("x", "xp", "y", "yp", *SPIN)}
    beam["p"] = np.full(COPIES, p0)
    result = run(
        {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam}, outputs=outputs
    )
    out = particles(result["out.sdds"])
    if len(out["p"]) != COPIES:
        sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
    state = np.column_stack([out[n] for n in (*COORDINATES, *SPIN)])
    if not np.all(np.isfinite(state)):
        sys.exit(f"{mode}: nonfinite orbit or spin")
    if np.ptp(state, axis=0).max() != 0:
        sys.exit(f"{mode}: identical particles ended in different states")
    if abs(np.linalg.norm(state[0, 6:]) - 1) > 1e-14:
        sys.exit(f"{mode}: the spin norm is not 1")
    return {n: float(v) for n, v in zip((*COORDINATES, *SPIN), state[0], strict=True)}, result


def axial_rotation(theta: float, spin=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Spin rotated by the angle theta about +z (counterclockwise seen from +z)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * spin[0] - s * spin[1], s * spin[0] + c * spin[1], spin[2]])


def on_axis_angle(ks: float, length: float, anomaly: float) -> float:
    """Spin rotation angle about +z on the axis of a solenoid of strength KS, on momentum.

    KS = q B/(m c p0) for the signed physical charge q (Elegant's "-Bs/(B*Rho)" with its legacy
    charge sign). With the velocity along B, magnetic Thomas-BMT reduces to
    dS/dt = -(q/m)(1 + a) B/gamma z-hat x S; dt = gamma dz/(beta gamma c) turns it into
    dtheta/dz = -q (1 + a) B/(m c p) = -KS (1 + a).
    """
    return -ks * length * (1 + anomaly)


def fmt(values) -> str:
    return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


def compare(label: str, observed, expected, atol: float, rtol: float = 0.0) -> bool:
    """Print one case; return True when observed and expected agree within atol + rtol*|expected|."""
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bool(np.allclose(observed, expected, rtol=rtol, atol=atol))
