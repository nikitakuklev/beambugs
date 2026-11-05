"""Shared parsing and analytic quadrupole maps for the optics/matrix/radiation cases."""

import re

import numpy as np
from _bmad import compare


def rows(log: str) -> dict[str, np.ndarray]:
    """Require unique finite RESULT records and reject upstream error diagnostics."""
    if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error", log):
        raise RuntimeError(log)
    result = {}
    for line in log.splitlines():
        fields = line.split()
        if not fields or fields[0] != "RESULT":
            continue
        name = fields[1]
        if name in result or len(fields) < 3:
            raise RuntimeError(f"Invalid result record: {line}")
        values = np.array([float(s.replace("D", "E")) for s in fields[2:]])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite result: {line}")
        result[name] = values
    if not result:
        raise RuntimeError(f"No result records:\n{log}")
    return result


def control(label, observed, expected, *, atol=2e-14, rtol=2e-13):
    """A failed prerequisite is a failed run, never a defect verdict."""
    if compare(label, observed, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"Control failed: {label}")


def quad(k, length):
    """Thick-lens solution of x'=p, p'=-k*x in slope coordinates."""
    root = np.sqrt(abs(k))
    phase = root * length
    if k > 0:
        c, s = np.cos(phase), np.sin(phase)
        return np.array([[c, s / root], [-root * s, c]])
    c, s = np.cosh(phase), np.sinh(phase)
    return np.array([[c, s / root], [root * s, c]])
