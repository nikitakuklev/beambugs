"""TRK-04: KQUAD fringes ignore the fractional strength error FSE that scales the body

KQUAD scales its body's normalized gradient by 1 + FSE, where FSE is the fractional strength error,
but both nonlinear fringe kicks keep the unscaled gradient K₁. A uniform field error leaves the
fringe profile normalized to the central gradient unchanged, so K₁ = 5 m⁻² with FSE = 0.2 must track
like K₁ = 6 m⁻² with FSE = 0, and FSE = −1 must be a plain drift. For a 0.4 m magnet and electron
momentum 2000 mₑc, the exit slopes differ by up to 419.4 nrad, and at FSE = −1 the exit position is
displaced by up to 54.35 nm.

The script tracks twelve particles through a KQUAD with cubic fringes at both ends, specified by an
explicit K1 so that the pole-tip conversion does not enter. It compares K1 = 5 with FSE = 0.2
against K1 = 6, and K1 = 5 with FSE = -1 against the drift x + L xp, y + L yp. Controls compare
the finite-strength pair with both fringes off, and K1 = 0 with the analytic drift.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

P0 = 2000.0  # electron momentum, units of m_e c
LENGTH = 0.4
# Twelve particles inside the bore: (x, xp, y, yp) scaled by 0.5 ... 1.
SCALE = np.linspace(0.5, 1.0, 12)
BEAM = {
    "x": 0.006 * SCALE,
    "xp": 0.001 * SCALE,
    "y": -0.004 * SCALE,
    "yp": -0.0005 * SCALE,
    "p": np.full(12, P0),
    "particleID": np.arange(1, 13),
}
ATOL = 1e-14  # the controls agree to 2e-15


def track(mode, k1, fse, edges):
    """Exit (x, xp, y, yp) of the twelve particles through a KQUAD with the given K1 and FSE."""
    flags = f"EDGE1_EFFECTS={edges},EDGE2_EFFECTS={edges},EDGE1_LINEAR=0,EDGE2_LINEAR=0"
    lattice = f"E: KQUAD,L={LENGTH!r},N_SLICES=64,K1={k1!r},FSE={fse!r},{flags}\nBL: LINE=(E)\n"
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
        "output=out.sdds, default_order=2 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": BEAM})
    out = particles(result["out.sdds"])
    if len(out["x"]) != 12:
        sys.exit(f"{mode}: {12 - len(out['x'])} particles lost")
    return np.column_stack([out["x"], out["xp"], out["y"], out["yp"]])


def drift():
    # Without a field a particle moves on a straight line and keeps its slopes.
    return np.column_stack(
        [
            BEAM["x"] + LENGTH * BEAM["xp"],
            BEAM["xp"],
            BEAM["y"] + LENGTH * BEAM["yp"],
            BEAM["yp"],
        ]
    )


def compare(label, observed, expected):
    """Print one case (last particle, largest difference per column); True when they disagree."""
    bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
    last = lambda a: "[" + ", ".join(f"{v:.16g}" for v in a[-1]) + "]"
    print(
        f"{label}: observed={last(observed)} expected={last(expected)} "
        f"max|observed-expected|=[{', '.join(f'{v:.4g}' for v in np.abs(observed - expected).max(axis=0))}]"
    )
    return bad


defect = False
print("columns: x (m), xp, y (m), yp of the last particle")
for mode in modes():
    # Controls: without fringes the two descriptions agree, and K1 = 0 is a drift.
    if compare(
        f"{mode} control, no fringes, K1=5 FSE=0.2 vs K1=6",
        track(mode, 5.0, 0.2, 0),
        track(mode, 6.0, 0.0, 0),
    ):
        sys.exit("The bodies of equivalent descriptions differ: the comparison is not valid.")
    if compare(f"{mode} control, K1=0 vs drift", track(mode, 0.0, 0.0, 2), drift()):
        sys.exit("A zero-gradient magnet is not a drift: the comparison is not valid.")
    # The field of (K1, FSE) = (5, 0.2) is that of (6, 0), with the same normalised fringe profile.
    defect |= compare(
        f"{mode} K1=5 FSE=0.2 vs K1=6 FSE=0", track(mode, 5.0, 0.2, 2), track(mode, 6.0, 0.0, 2)
    )
    # FSE = -1 removes the whole field, fringes included.
    defect |= compare(f"{mode} K1=5 FSE=-1 vs drift", track(mode, 5.0, -1.0, 2), drift())
print(f"comparison: atol={ATOL:g}")
verdict(defect)
