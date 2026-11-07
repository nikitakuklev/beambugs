"""Fresh native matrix/particle fixtures for finite-speed momentum conversions."""

import _elegant as native
import numpy as np
from _diagnostics import run

C = 299792458.0
native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


def close(actual, expected, tolerance, label):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f"{label}: error={error:.17g}, tolerance={tolerance:.17g}")
    return error


def discrepancy(actual, expected, faulty, tolerance, label):
    """Accept the physical result or the identified faulty law; refuse other output."""
    if abs(actual - expected) <= tolerance:
        return False
    close(actual, faulty, tolerance, label)
    return True


def track(lattice, p0, delta, *, s=None, order=2, twiss=False):
    """Compare particle and cumulative/individual matrix output from one fresh process."""
    momentum = p0 * (1 + np.asarray(delta, dtype=float))
    beta = momentum / np.hypot(momentum, 1)
    clock = np.zeros(len(momentum)) if s is None else np.asarray(s)
    beam = {"p": momentum, "t": clock / (C * beta), "particleID": np.arange(1, len(momentum) + 1)}
    commands = (
        "&run_setup lattice=lattice.lte,use_beamline=BL,"
        f"p_central={p0:.17g},default_order={order},output=particles.sdds,"
        "random_number_seed=271828 &end\n"
        "&run_control n_steps=1 &end\n"
        "&matrix_output SDDS_output=cumulative.sdds,SDDS_output_order=1,output_at_each_step=1 &end\n"
        "&matrix_output SDDS_output=individual.sdds,SDDS_output_order=1,individual_matrices=1,output_at_each_step=1 &end\n"
    )
    outputs = ["particles.sdds", "cumulative.sdds", "individual.sdds"]
    if twiss:
        commands += (
            "&twiss_output filename=twiss.sdds,matched=0,beta_x=1,beta_y=1,"
            "concat_order=2,local_dispersion=0,output_at_each_step=1 &end\n"
        )
        outputs.append("twiss.sdds")
    commands += "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    result = run(
        {"lattice.lte": lattice, "run.ele": commands}, "serial", {"beam.sdds": beam}, tuple(outputs)
    )
    data = native.particles(result["particles.sdds"])
    if not np.array_equal(data["particleID"], beam["particleID"]):
        raise RuntimeError("Particles were lost or particle IDs changed")
    result["particles"] = data
    result["s"] = C * data["t"] * data["p"] / np.hypot(data["p"], 1)
    return result


def value(result, name, *, output="cumulative.sdds", element=None):
    data = result[output]
    if data.n_pages != 1:
        raise RuntimeError("Expected exactly one SDDS page")
    index = -1
    if element is not None:
        names = [str(x) for x in data[0, "ElementName"]]
        if names.count(element) != 1:
            raise RuntimeError(f"Expected exactly one element named {element}")
        index = names.index(element)
    scalar = float(data[0, name][index])
    if not np.isfinite(scalar):
        raise RuntimeError(f"Nonfinite {name} in {output}")
    return scalar
