"""Shared setup for the WAKE, TRWAKE and CORGPIPE reproducers: an ASCII SDDS wake table, one serial
run of an electron beam through a lattice that reads it, and the energy change of each particle.

Times are in s, momenta p = beta*gamma in units of m_e c and energies in eV. A positive wake loss
is energy removed from an electron.
"""

from __future__ import annotations

import numpy as np
from _elegant import RunFailure, particles, run

ME_EV = 0.51099906e6  # SDDS constants.h, as Elegant uses it


def table(columns: dict[str, np.ndarray], units: dict[str, str]) -> str:
    """ASCII SDDS page with the given double columns (all of one length)."""
    names = list(columns)
    head = "SDDS1\n" + "".join(
        f"&column name={name}, units={units[name]}, type=double &end\n" for name in names
    )
    rows = "".join(
        " ".join(repr(float(columns[name][i])) for name in names) + "\n"
        for i in range(len(columns[names[0]]))
    )
    return head + "&data mode=ascii, no_row_counts=1 &end\n" + rows


def beam(times: np.ndarray, p: float, x: float = 0.0) -> dict[str, np.ndarray]:
    """Particles at the given arrival times, all at momentum p and offset x, zero slopes."""
    n = len(times)
    return {
        "x": np.full(n, x),
        "t": np.asarray(times, dtype=float),
        "p": np.full(n, p),
        "particleID": np.arange(1, n + 1),
    }


def track(
    elements: str, line: str, particles_in: dict, p0: float, files: dict[str, str] | None = None
) -> tuple[dict[str, np.ndarray], str]:
    """Serial run through BL: LINE=(line); (final coordinates by particle ID, run log).

    `files` holds extra inputs such as wake tables. A nonzero exit raises RunFailure.
    """
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    inputs = {"lattice.lte": elements.rstrip("\n") + f"\nBL: LINE=({line})\n", "run.ele": commands}
    result = run({**inputs, **(files or {})}, beams={"beam.sdds": particles_in})
    out = particles(result["out.sdds"])
    if len(out["p"]) != len(particles_in["p"]):
        raise SystemExit("particles were lost")
    return out, result["log"]


def loss(out: dict[str, np.ndarray], p0: float) -> np.ndarray:
    """Energy lost by each particle (eV), from its final momentum and the initial p0."""
    return ME_EV * (np.sqrt(1 + p0**2) - np.sqrt(1 + out["p"] ** 2))


__all__ = ["ME_EV", "RunFailure", "beam", "loss", "table", "track"]
