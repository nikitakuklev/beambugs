"""Record parsing and checked-source execution for orbital-matrix/radiation cases."""

from pathlib import Path

import numpy as np
from _bmad import TIMEOUT, Result, _case, _execute, _recipe, _scratch


def records(log: str) -> dict[str, np.ndarray]:
    """Require unique, finite labelled records from a completed native probe."""
    rows = {}
    for line in log.splitlines():
        if not line.startswith("RESULT "):
            continue
        _, key, *values = line.split()
        if key in rows or not values:
            raise RuntimeError(f"Malformed or duplicate record: {line}")
        row = np.array([float(value) for value in values])
        if not np.isfinite(row).all():
            raise RuntimeError(f"Nonfinite record: {line}")
        rows[key] = row
    if not rows:
        raise RuntimeError(f"Missing numerical records:\n{log}")
    return rows


def checked_fortran(program: str, lattice: str, args: tuple[str, ...] = ()) -> Result:
    """Compile an inline source/driver with bounds checking and the pin's recipe."""
    compile_argv, link_argv = _recipe(False)
    with _scratch() as tmp:
        work = Path(tmp)
        (work / "probe.f90").write_text(program)
        _execute([*compile_argv, "-O0", "-g", "-fcheck=all"], work, "", TIMEOUT)
        _execute(link_argv, work, "", TIMEOUT)
        return _case(
            [str(work / "probe"), *args], work / "run", {"lat.bmad": lattice}, (), "", TIMEOUT
        )
