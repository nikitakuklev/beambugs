"""Shared setup for the TRWAKE reproducers: one TRWAKE element tracking 18 electron macroparticles,
and the transverse wake kick of every particle computed as a direct causal pair sum, independently
of Elegant's binning and convolution.

The wake table has spacing `spacing` and starts at W(0) = 0, and every particle arrival time lies
on a multiple of `spacing`, so each pair delay is a grid point and no wake interpolation enters.
"""

from __future__ import annotations

import numpy as np
from _elegant import particles, run

# SDDS constants.h: electron rest energy in eV.
REST_EV = 510999.06
N = 18  # particles
RTOL, ATOL = 4e-13, 1e-19  # comparison of slopes


def wake(delay: np.ndarray) -> np.ndarray:
    """Wx(t) in V/C/m for t >= 0 behind the source: 1.7e15 sin(2 pi 6 GHz t) exp(-t/70 ps)."""
    t = np.maximum(delay, 0)
    return np.where(delay >= 0, 1.7e15 * np.sin(2 * np.pi * 6e9 * t) * np.exp(-t / 70e-12), 0.0)


def wake_table(spacing: float) -> str:
    """ASCII SDDS table of Wx and Wy = -Wx on 257 points of the given spacing."""
    grid = np.arange(257) * spacing
    rows = "".join(
        f"{float(t)!r} {float(w)!r} {float(-w)!r}\n" for t, w in zip(grid, wake(grid), strict=True)
    )
    return (
        "SDDS1\n"
        "&column name=time, units=s, type=double &end\n"
        "&column name=wx, units=V/C/m, type=double &end\n"
        "&column name=wy, units=V/C/m, type=double &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n" + rows
    )


def beam(times, x, y, momentum) -> dict[str, np.ndarray]:
    """Particles with zero slopes at the given arrival times, positions and momenta (m_e c)."""
    return {
        "x": np.broadcast_to(x, N).astype(float),
        "y": np.broadcast_to(y, N).astype(float),
        "t": np.asarray(times, dtype=float),
        "p": np.broadcast_to(momentum, N).astype(float),
        "particleID": np.arange(1, N + 1),
    }


def track(
    mode: str,
    columns: dict[str, np.ndarray],
    *,
    shift: tuple[float, float] = (0.0, 0.0),
    drive: int = 1,
    n_bins: int = 0,
    charge: float = 0.6e-9,
    drift: float = 0.0,
    spacing: float = 2e-12,
) -> tuple[np.ndarray, str]:
    """Slope kicks (xp, yp), shape (N, 2), of every particle through TRWAKE, and the native log.

    drive=1 is the ordinary dipole wake (kick from the leading particles' offsets, probe
    exponent 0); drive=0 is the quadrupole wake (kick proportional to the particle's own offset).
    shift is the structure's misalignment (DX, DY); drift is a drift before the structure.
    """
    lattice = (
        f"CH: CHARGE,TOTAL={charge!r}\n"
        f"D0: DRIF,L={drift!r}\n"
        'R: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="time",WXCOLUMN="wx",WYCOLUMN="wy",'
        f"DX={shift[0]!r},DY={shift[1]!r},N_BINS={n_bins},INTERPOLATE=0,SMOOTHING=0,"
        f"X_DRIVE_EXPONENT={drive},Y_DRIVE_EXPONENT={drive},"
        f"X_PROBE_EXPONENT={1 - drive},Y_PROBE_EXPONENT={1 - drive}\n"
        "BL: LINE=(CH,D0,R)\n"
    )
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={float(np.mean(columns['p']))!r},"
        "output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": wake_table(spacing)},
        mode,
        {"beam.sdds": columns},
    )
    out = particles(result["out.sdds"])
    if not np.array_equal(out["particleID"], columns["particleID"]):
        raise RuntimeError(f"{mode}: particles lost")
    # The wake changes only the slopes (the drift acts on s, not on x or y).
    if not (np.array_equal(out["x"], columns["x"]) and np.array_equal(out["y"], columns["y"])):
        raise RuntimeError(f"{mode}: transverse positions changed")
    return np.column_stack([out["xp"], out["yp"]]), result["log"]


def pair_sum(
    columns: dict[str, np.ndarray],
    *,
    shift: tuple[float, float] = (0.0, 0.0),
    drive: int = 1,
    charge: float = 0.6e-9,
    spacing: float = 2e-12,
    probe_offset: tuple[float, float] | None = None,
    binned: np.ndarray | None = None,
) -> np.ndarray:
    """Physical slope kicks (xp, yp), shape (N, 2), from the causal pair sum.

    With structure offset d, drive exponent a and probe exponent b (a + b = 1) and zero incoming
    slopes, particle i receives
      dx'_i = q (x_i - d)^b sum_j [(x_j - d)^a Wx(t_i - t_j)] / (E0 p_i),
    summed over the particles j that are not later than i, with q = charge / N the macroparticle
    charge and E0 the electron rest energy in eV; y is the same with Wy = -Wx. Translating beam
    and structure together leaves every (x - d) unchanged. Giving `probe_offset` replaces d in the
    probe factor by that value, which evaluates the probe at the global coordinate (d = 0).
    `binned` (boolean) restricts both sources and probes to the particles Elegant bins; the
    others receive no kick.
    """
    t, p = columns["t"], columns["p"]
    delays = t[:, None] - t[None, :]
    index = np.rint(delays / spacing).astype(int)
    if not np.allclose(delays / spacing, index, rtol=0, atol=3e-12):
        raise ValueError("pair delays are not multiples of the wake spacing")
    # Causal matrix W(t_i - t_j) taken on the table grid; the coincident pairs have W(0) = 0.
    matrix = np.where(index >= 0, wake(np.maximum(index, 0) * spacing), 0.0)
    if binned is None:
        binned = np.ones(N, dtype=bool)
    matrix = matrix * binned[None, :]
    kicks = np.zeros((N, 2))
    for plane, (name, sign) in enumerate((("x", 1.0), ("y", -1.0))):
        rel = columns[name] - shift[plane]
        probe = rel if probe_offset is None else columns[name] - probe_offset[plane]
        source = rel if drive else np.ones(N)
        voltage = sign * charge / N * (matrix @ source)
        kicks[:, plane] = voltage * (1.0 if drive else probe) / (REST_EV * p)
    kicks[~binned] = 0.0
    return kicks


def fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"


def compare(label: str, observed: np.ndarray, expected: np.ndarray) -> bool:
    """Print the last particle's (xp, yp); return True when any particle disagrees."""
    print(f"{label}: observed={fmt(observed[-1])} expected={fmt(expected[-1])}")
    return not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
