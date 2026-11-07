"""Shared fixed-particle CWIGGLER fixtures with checked survival and physical constants."""

import _elegant as native
import numpy as np
from _diagnostics import run

C = 299792458.0
# CWIGGLER uses gwig.h's electron constants, which differ from SDDS's set.
FIELD_RIGIDITY = 1.602176462e-19 / (9.10938188e-31 * C)
native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


def close(observed, expected, atol, label):
    """Refuse nonfinite values and discrepancies beyond a fixed absolute tolerance."""
    error = float(np.max(np.abs(np.asarray(observed) - expected)))
    if not np.isfinite(error) or error > atol:
        raise RuntimeError(f"{label}: error={error:.17g}, tolerance={atol:g}")


def track(lattice, p0, beam):
    """Track explicit (x, xp, y, yp, t, p) rows in one fresh serial process."""
    rays = np.asarray(beam, dtype=float)
    if rays.ndim != 2 or rays.shape[1] != 6 or not np.isfinite(rays).all():
        raise ValueError("Expected finite six-coordinate particle rows")
    ids = np.arange(1, len(rays) + 1)
    columns = {name: rays[:, i] for i, name in enumerate(("x", "xp", "y", "yp", "t", "p"))}
    columns["particleID"] = ids
    commands = (
        "&run_setup lattice=lattice.lte,use_beamline=BL,"
        f"p_central={p0:.17g},default_order=2,output=out.sdds,"
        "random_number_seed=271828 &end\n"
        "&run_control n_steps=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands},
        "serial",
        {"beam.sdds": columns},
        ("out.sdds",),
    )
    data = native.particles(result["out.sdds"])
    if not np.array_equal(data["particleID"], ids):
        raise RuntimeError("Particles were lost or their IDs changed")
    return data
