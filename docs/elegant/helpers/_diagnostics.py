"""Shared execution and finite-value checks for HISTOGRAM and WATCH output probes."""

import math
import os

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC",
):
    os.environ[_name] = "1"

from _elegant import run as native_run


def run(files, mode, beams, outputs):
    """Keep serial execution on one inherited core; MPI inherits the launcher's affinity."""
    affinity = os.sched_getaffinity(0)
    try:
        if mode == "serial":
            os.sched_setaffinity(0, {min(affinity)})
        return native_run(files, mode, beams, outputs=outputs)
    finally:
        os.sched_setaffinity(0, affinity)


def scalar(data, name, *, column=False):
    """Read exactly one finite value, refusing malformed or missing native output."""
    if data.n_pages != 1:
        raise RuntimeError("Expected exactly one SDDS page")
    if column and len(data[0, name]) != 1:
        raise RuntimeError(f"Expected one {name} row")
    value = float(data[0, name][0] if column else data[name][0])
    if not math.isfinite(value):
        raise RuntimeError(f"Expected one finite {name} value")
    return value


def conversion_defect(observed, expected, faulty, *, rtol):
    """Accept the physical result or the specific faulty conversion; reject other results."""
    if math.isclose(observed, expected, rel_tol=rtol, abs_tol=0):
        return False
    if not math.isclose(observed, faulty, rel_tol=rtol, abs_tol=0):
        raise RuntimeError(f"Unexpected conversion: {observed}, expected {expected} or {faulty}")
    return True
