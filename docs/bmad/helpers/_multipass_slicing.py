"""Input and output helpers for the multipass and slicing reproducers."""

import re

from _bmad import compare

HEADER = """no_digested
parameter[particle]=electron
parameter[p0c]=1e9
parameter[geometry]=open
beginning[beta_a]=10
beginning[beta_b]=10
"""


def rows(log: str) -> dict[str, list[float]]:
    """Read unique RESULT records, rejecting Bmad diagnostics and malformed output."""
    if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|ERROR STOP|Fortran runtime error:", log):
        raise RuntimeError(log)
    result = {}
    for line in log.splitlines():
        if not line.startswith("RESULT "):
            continue
        _, name, *values = line.split()
        if name in result or not values:
            raise ValueError(f"Duplicate or empty RESULT {name}: {log}")
        result[name] = [float(v.replace("D", "E")) for v in values]
    if not result:
        raise ValueError(f"Missing RESULT records: {log}")
    return result


def control(label, observed, expected, *, atol=0.0, rtol=0.0):
    """A failed prerequisite is an execution failure, never a defect verdict."""
    if compare(label, observed, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"Control failed: {label}")
