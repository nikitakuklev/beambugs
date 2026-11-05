"""Shared parsing and inline cases for the tracking contracts and materials entries."""

from __future__ import annotations

import re

import numpy as np
from _bmad import fortran, run


def rows(log: str) -> dict[str, np.ndarray]:
    """Read unique finite numeric RESULT records; reject native error diagnostics."""
    if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error|ERROR STOP", log):
        raise RuntimeError(log)
    result = {}
    for line in log.splitlines():
        if line.startswith("RESULT "):
            _, name, *values = line.split()
            if name in result or not values:
                raise ValueError(f"Duplicate or empty record: {line}")
            result[name] = np.array([float(x.replace("D", "E")) for x in values])
            if not np.isfinite(result[name]).all():
                raise ValueError(f"Nonfinite record: {line}")
    if not result:
        raise ValueError(f"No result records:\n{log}")
    return result


def constants() -> dict[str, np.ndarray]:
    return rows(
        fortran("""
program constants_probe
use bmad
implicit none
write(*, '(a,6es26.17e3)') 'RESULT constants ', &
 m_electron, m_proton, r_e, anomalous_mag_moment_electron, e_charge, c_light
end program
""").log
    )


def lattice(element: str, *, start: str = "", p0c: float = 1e9, line: str = "e") -> str:
    return (
        f"no_digested\nparameter[geometry]=open\nparameter[particle]=electron\n"
        f"parameter[p0c]={p0c:.17g}\nbeginning[beta_a]=10\nbeginning[beta_b]=10\n"
        f"{start}\n{element}\nbl: line=({line})\nuse, bl\n"
    )


def tao(lat: str, commands: list[str]) -> dict[str, str]:
    log = run(
        "tao",
        ["-noplot", "-noinit", "-lat", "lat.bmad", "-startup", "run.tao"],
        files={"lat.bmad": lat, "run.tao": "\n".join([*commands, "quit", ""])},
    ).log
    if "Tao: quit" not in log or re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b", log):
        raise RuntimeError(log)
    blocks = {}
    for command in commands:
        marker = f"Tao: {command}\n"
        if log.count(marker) != 1:
            raise ValueError(log)
        blocks[command] = log.split(marker)[1].split("\nTao:", 1)[0]
    return blocks


def pipe_values(block: str, name: str) -> np.ndarray:
    matches = [line for line in block.splitlines() if line.startswith(name + ";")]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} record:\n{block}")
    fields = matches[0].split(";")[3:]
    values = np.array([float(s) for s in fields if s.strip()])
    if not values.size or not np.isfinite(values).all():
        raise ValueError(matches[0])
    return values


def orbit(block: str) -> np.ndarray:
    if "state;STR;F;Alive" not in block:
        raise ValueError(f"Particle is not alive:\n{block}")
    return np.array([pipe_values(block, k)[0] for k in ("x", "px", "y", "py", "z", "pz")])
