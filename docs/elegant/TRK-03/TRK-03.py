"""TRK-03: KQUAD given by pole-tip field and bore tracks its fringes with zero strength until a matrix or Twiss request

KQUAD can be specified by pole-tip field B and bore radius BORE instead of normalized gradient K₁.
With nonlinear fringes enabled and K₁ left at its default zero, the body uses the field-derived
strength but both fringes use zero strength until a matrix or Twiss output request overwrites K₁.
The same beam then leaves the same magnet differently depending on whether optics output was
requested; after the request it matches the equivalent explicit K₁. For a 0.4 m magnet with K₁ = 5
m⁻² and electron momentum 2000 mₑc, exit slopes change by up to 1.749 µrad and positions by up to
95.29 nm. Pole-tip quadrupoles with enabled fringes are affected.

The script tracks twelve particles through the magnet described by B and BORE, with and without a
matrix or Twiss request, and compares each result with the same magnet described by the equivalent
explicit K1, which no request changes. Controls: the explicit-K1 magnet with a matrix request, and
the pole-tip magnet with the fringes disabled, must be unaffected by the request.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

# SDDS constants.h, as Elegant uses them.
C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31
P0 = 2000.0  # electron momentum, units of m_e c
LENGTH, BORE, K1 = 0.4, 0.02, 5.0
# A pole-tip field B at radius BORE is the gradient B/BORE, and K1 = gradient * e/(m c p0).
# Inverting: the B equivalent to K1 = 5 m^-2.
B = K1 * BORE * ME * C * P0 / QE
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
REQUESTS = {
    "matrix": "&matrix_output SDDS_output=matrix.sdds &end\n",
    "twiss": "&twiss_output filename=twiss.sdds, matched=0, beta_x=3, beta_y=2 &end\n",
}


def track(mode, strength, edges, request=""):
    """Exit (x, xp, y, yp) of the twelve particles; strength is the KQUAD strength attributes."""
    flags = f"EDGE1_EFFECTS={edges},EDGE2_EFFECTS={edges},EDGE1_LINEAR=0,EDGE2_LINEAR=0"
    lattice = f"E: KQUAD,L={LENGTH!r},N_SLICES=64,{strength},FSE=0,{flags}\nBL: LINE=(E)\n"
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
        "output=out.sdds, default_order=2 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        f"{REQUESTS.get(request, '')}"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": BEAM})
    out = particles(result["out.sdds"])
    if len(out["x"]) != 12:
        sys.exit(f"{mode}: {12 - len(out['x'])} particles lost")
    return np.column_stack([out["x"], out["xp"], out["y"], out["yp"]])


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
print("columns: x (m), xp, y (m), yp of the last particle; expected from the explicit-K1 magnet")
for mode in modes():
    explicit = f"K1={K1!r}"
    pole_tip = f"K1=0,BORE={BORE!r},B={B!r}"
    # The field is identical in both descriptions, so every pole-tip result must equal this one.
    expected = track(mode, explicit, 2)
    # Controls: a request changes neither the explicit-K1 magnet nor a magnet without fringes.
    if compare(
        f"{mode} control, explicit K1 with matrix request",
        track(mode, explicit, 2, "matrix"),
        expected,
    ):
        sys.exit("The request changes an explicit-K1 magnet: the comparison is not valid.")
    off = track(mode, pole_tip, 0)
    if compare(
        f"{mode} control, no fringes, pole-tip with matrix request",
        track(mode, pole_tip, 0, "matrix"),
        off,
    ):
        sys.exit("The request changes a magnet without fringes: the comparison is not valid.")
    if compare(
        f"{mode} control, no fringes, pole-tip vs explicit K1", off, track(mode, explicit, 0)
    ):
        sys.exit(
            "The pole-tip body differs from the explicit-K1 body: the comparison is not valid."
        )
    # Defect: the pole-tip magnet differs from the explicit one until a request has been made.
    defect |= compare(f"{mode} pole-tip, no request", track(mode, pole_tip, 2), expected)
    # Both requests must restore the explicit-K1 track; otherwise "until a request" is not the cause.
    for request in REQUESTS:
        if compare(
            f"{mode} pole-tip, {request} request", track(mode, pole_tip, 2, request), expected
        ):
            sys.exit("The pole-tip request control differs from explicit K1: comparison not valid.")
print(f"B = {B!r} T at BORE = {BORE!r} m; comparison: atol={ATOL:g}")
verdict(defect)
