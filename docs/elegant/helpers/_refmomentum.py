"""Fixed-ray tracking and mass-shell/rigidity checks for reference-momentum probes."""

import math

import _elegant as native
import numpy as np
from _diagnostics import run

C = 299792458.0
# Electron SI constants from the SDDS pin; use the executable's constant set.
MASS = 9.1093897e-31
CHARGE = 1.60217733e-19
RAYS = np.array([1e-6, -1e-6, 0.5e-6, -0.5e-6, 0.0, 0.0])
native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


def close(observed, expected, *, rtol=4e-14, atol=0.0):
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)


def track(lattice, p0, *, rays=None, mode="serial", twiss=False, watch=False, setup=""):
    """Fresh native process with six fixed particles, checked survival and ID ordering."""
    x = np.zeros(6) if rays is None else rays
    beam = {"x": x, "p": np.full(6, p0), "particleID": np.arange(1, 7)}
    commands = (
        "&run_setup lattice=lattice.lte,use_beamline=BL,"
        f"p_central={p0:.17g},output=out.sdds,default_order=1,"
        f"random_number_seed=271828{setup} &end\n"
        "&run_control n_passes=1 &end\n"
    )
    outputs = ["out.sdds"]
    if twiss:
        commands += "&twiss_output filename=optics.sdds,matched=0,beta_x=1,beta_y=1 &end\n"
        outputs.append("optics.sdds")
    if watch:
        outputs.append("reference.sdds")
    commands += "&sdds_beam input=initial.sdds &end\n&track soft_failure=0 &end\n"
    result = run(
        {"lattice.lte": lattice + "\nUSE,BL\n", "run.ele": commands},
        mode,
        {"initial.sdds": beam},
        tuple(outputs),
    )
    data = native.particles(result["out.sdds"])
    if not np.array_equal(data["particleID"], beam["particleID"]):
        raise RuntimeError("Particles were lost or particle IDs changed")
    result["particles"] = data
    return result


def response(data, *, order=1):
    """Linear slope response or coefficient of x squared, extrapolated to zero amplitude."""
    xp, h = data["xp"], RAYS[0]
    if order == 1:
        coarse = (xp[0] - xp[1]) / (2 * h)
        fine = (xp[2] - xp[3]) / h
    else:
        coarse = (xp[0] + xp[1] - 2 * xp[4]) / (2 * h**2)
        fine = (xp[2] + xp[3] - 2 * xp[4]) / (2 * (h / 2) ** 2)
    # Symmetry isolates the derivative; Richardson extrapolation removes its O(h^2) error.
    return float((4 * fine - coarse) / 3)


def quadrupole_slope(k, length):
    # Linear Lorentz force: x'' + k*x = 0, with k = K1*P0/p for an off-reference ray.
    if k > 0:
        root = math.sqrt(k)
        return -root * math.sin(root * length)
    if k < 0:
        root = math.sqrt(-k)
        return root * math.sinh(root * length)
    return 0.0
