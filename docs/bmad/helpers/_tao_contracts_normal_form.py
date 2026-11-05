"""Inputs and strict Tao record parsing for the contracts and normal-form cases."""

import re
from collections.abc import Mapping, Sequence

import numpy as np
from _bmad import Result, run

ERROR = re.compile(
    r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Program received signal|Fortran runtime error"
    r"|^\s*(?:ERROR(?: STOP| IN\b|:|;|\s*$)|FATAL\b|Backtrace for this error:)",
    re.IGNORECASE | re.MULTILINE,
)
LATTICE = """parameter[geometry] = open
parameter[particle] = electron
parameter[p0c] = 1e6
beginning[beta_a] = 3
beginning[alpha_a] = 0.2
beginning[beta_b] = 4
particle_start[x] = 0.001
particle_start[y] = -0.002
particle_start[z] = 0.03
e: drift, l=1.7
lat: line=(e)
use, lat
"""
FODO = """parameter[geometry] = closed
parameter[particle] = electron
parameter[p0c] = 1e9
qf: quadrupole, l=0.2, k1=1, fringe_type=none
qd: quadrupole, l=0.2, k1=-0.96, fringe_type=none
d: drift, l=0.8
lat: line=(qf,d,qd,d)
use, lat
"""
THIN_RING = """parameter[geometry] = closed
parameter[particle] = electron
parameter[p0c] = 1e9
parameter[taylor_order] = 4
sx: multipole, k2l={strength}
qf: multipole, k1l=0.8
qd: multipole, k1l=-0.7
d: drift, l=1
lat: line=(sx,qf,d,qd,d)
use, lat
"""


def tao(
    lattice: str,
    commands: Sequence[str],
    *,
    files: Mapping[str, str] | None = None,
    outputs: Sequence[str] = (),
    options: Sequence[str] | None = None,
    check: bool = True,
) -> Result:
    """Run an isolated command file; require clean completion for ordinary cases."""
    args = ["-noplot", *(options if options is not None else ["-lat", "lat.bmad", "-noinit"])]
    result = run(
        "tao",
        [*args, "-startup", "run.tao"],
        files={"lat.bmad": lattice, "run.tao": "\n".join([*commands, "quit", ""]), **(files or {})},
        outputs=outputs,
    )
    if check and (ERROR.search(result.log) or result.log.count("Tao: quit\n") != 1):
        raise RuntimeError(result.log)
    return result


def block(log: str, command: str) -> str:
    marker = f"Tao: {command}\n"
    if log.count(marker) != 1:
        raise ValueError(f"Expected exactly one command {command!r}:\n{log}")
    return log.split(marker, 1)[1].split("\nTao:", 1)[0].strip()


def record(text: str, name: str) -> list[str]:
    rows = [line.split(";") for line in text.splitlines() if line.startswith(name + ";")]
    if len(rows) != 1:
        raise ValueError(f"Expected one {name} record:\n{text}")
    return rows[0]


def scalar(text: str, name: str) -> float:
    row = record(text, name)
    if len(row) != 4 or row[1] != "REAL":
        raise ValueError(f"Expected scalar REAL: {row}")
    value = float(row[3])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {name}: {value}")
    return value


def evaluated(log: str, expression: str) -> float:
    rows = block(log, "pipe evaluate " + expression).splitlines()
    if len(rows) != 1:
        raise ValueError(f"Expected one evaluated value: {rows}")
    fields = rows[0].split(";")
    if len(fields) != 2 or fields[0] != "1":
        raise ValueError(f"Expected valid evaluation: {fields}")
    value = float(fields[1])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite evaluation: {fields}")
    return value


def thin_beta() -> float:
    """Periodic horizontal beta at SX from independent thin-lens matrices."""
    drift = np.array([[1.0, 1.0], [0.0, 1.0]])
    matrix = (
        drift @ np.array([[1.0, 0.0], [0.7, 1.0]]) @ drift @ np.array([[1.0, 0.0], [-0.8, 1.0]])
    )
    sine = np.sqrt(1 - (np.trace(matrix) / 2) ** 2)
    return float(matrix[0, 1] / sine)


def fodo_matrix(delta: float = 0.0) -> np.ndarray:
    """Independent thick-quadrupole transverse matrix in normalized momenta."""
    momentum = 1 + delta
    drift = np.array([[1.0, 0.8 / momentum], [0.0, 1.0]])

    def quad(k: float) -> np.ndarray:
        root = np.sqrt(abs(k) / momentum)
        phase = root * 0.2
        c, s = (np.cos(phase), np.sin(phase)) if k > 0 else (np.cosh(phase), np.sinh(phase))
        return np.array([[c, s / (momentum * root)], [-np.sign(k) * momentum * root * s, c]])

    matrix = np.zeros((4, 4))
    for start, sign in ((0, 1), (2, -1)):
        matrix[start : start + 2, start : start + 2] = (
            drift @ quad(-0.96 * sign) @ drift @ quad(sign)
        )
    return matrix
