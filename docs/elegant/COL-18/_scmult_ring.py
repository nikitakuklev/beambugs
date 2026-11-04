"""Shared setup for the SCMULT bunch-train reproducers: a closed four-sector ring, bunch trains in
bunched mode and the beam-moment helpers.

The ring has four 0.8 m sector bends of angle pi/2 with K1 = -h^2/2 (h = pi/(2 * 0.8 m)), which
focus equally in both planes, separated by 0.6 m straights; it is stable and closes geometrically.
`insert_sceffects` places one SCMULT after a drift DS of length `length` at the start of the first
straight, and WATCH elements record every particle before DS (entrance) and after the SCMULT
(exit) on each pass. The SCMULT changes only the slopes.
"""

from __future__ import annotations

import itertools
import re
import sys

import numpy as np
from _elegant import particles, run

# SDDS constants.h, as Elegant uses them.
C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
P = 17.0  # electron momentum, units of m_e c
BETA = P / np.hypot(1, P)
SX, SY = 0.7e-3, 1.8e-3  # m, rms sizes of the unscaled bunch
STRAIGHT, BEND_LENGTH = 0.6, 0.8
H = np.pi / (2 * BEND_LENGTH)  # curvature of a 90-degree sector bend
SPACING = 2.3e-9  # s between bunches


def optics(length: float) -> tuple[float, float, float]:
    """Matched (beta, alpha) at the start of the straight and beta at distance `length` into it.

    One cell is a straight followed by a sector bend with K1 = -h^2/2, which focuses equally in
    both planes with k = h/sqrt(2). The periodic Courant-Snyder parameters follow from the
    one-cell matrix M: sin(mu) = sqrt(1 - (tr M/2)^2), beta0 = M12/sin(mu),
    alpha0 = (M11 - M22)/(2 sin(mu)), gamma0 = -M21/sin(mu); a drift of length s transports
    beta to beta0 - 2 alpha0 s + gamma0 s^2.
    """
    k = H / np.sqrt(2)
    phase = k * BEND_LENGTH
    bend = np.array([[np.cos(phase), np.sin(phase) / k], [-k * np.sin(phase), np.cos(phase)]])
    cell = bend @ np.array([[1, STRAIGHT], [0, 1]])
    sine = np.sqrt(1 - (np.trace(cell) / 2) ** 2)
    beta0 = cell[0, 1] / sine
    alpha0 = (cell[0, 0] - cell[1, 1]) / (2 * sine)
    gamma0 = -cell[1, 0] / sine
    return beta0, alpha0, beta0 - 2 * alpha0 * length + gamma0 * length**2


def bunch(matched: bool, times: np.ndarray) -> np.ndarray:
    """Particles (x, xp, y, yp, t) of one unscaled bunch, shape (n, 5).

    Every combination of the four transverse phases and the arrival times. matched=True places
    the phases 0, 90, 180, 270 degrees of a matched ellipse of rms sizes SX, SY at the start of
    the straight (exact matched second moments, including the within-plane position-slope
    covariance, and no coupling between the transverse planes); otherwise the particles are
    cold, at x = +-SX, y = +-SY with zero slopes.
    """
    if matched:
        a, b, t = np.array(list(itertools.product(np.arange(4), np.arange(4), times))).T
        beta0, alpha0, _ = optics(STRAIGHT)
        x = np.sqrt(2) * SX * np.cos(a * np.pi / 2)
        y = np.sqrt(2) * SY * np.cos(b * np.pi / 2)
        xp = (-alpha0 * x + np.sqrt(2) * SX * np.sin(a * np.pi / 2)) / beta0
        yp = (-alpha0 * y + np.sqrt(2) * SY * np.sin(b * np.pi / 2)) / beta0
    else:
        x, y, t = np.array(list(itertools.product([-SX, SX], [-SY, SY], times))).T
        xp, yp = np.zeros_like(x), np.zeros_like(y)
    return np.column_stack([x, xp, y, yp, t])


def train(
    specs: tuple[tuple[float, float], ...],
    times: np.ndarray,
    *,
    matched: bool = False,
    shift: tuple[float, float] = (0.0, 0.0),
    reverse: bool = False,
) -> dict[str, np.ndarray | int]:
    """Bunch train as SDDS columns; specs lists (transverse scale, duration scale) per bunch.

    Bunch b holds particle IDs b*n+1 ... (b+1)*n (`IDSlotsPerBunch` = n) and arrives b * SPACING
    after the first. Rows are interleaved so that every MPI worker holds part of each bunch;
    reverse=True reverses the row order, so that bunch order does not follow file order.
    """
    base = bunch(matched, times)
    n = len(base)
    blocks = []
    for index, (scale, zscale) in enumerate(specs):
        x, xp, y, yp, t = base.T
        blocks.append(
            np.column_stack(
                [
                    scale * x + shift[0],
                    scale * xp,
                    scale * y + shift[1],
                    scale * yp,
                    zscale * t + index * SPACING,
                ]
            )
        )
    values = np.concatenate(blocks)
    ids = np.arange(1, len(values) + 1)
    values = values.reshape(len(specs), n, 5).transpose(1, 0, 2).reshape(-1, 5)
    ids = ids.reshape(len(specs), n).T.reshape(-1)
    if reverse:
        values, ids = values[::-1], ids[::-1]
    columns: dict[str, np.ndarray | int] = {
        key: values[:, i] for i, key in enumerate(("x", "xp", "y", "yp", "t"))
    }
    columns["p"] = np.full(len(values), P)
    columns["particleID"] = ids
    columns["IDSlotsPerBunch"] = n
    return columns


def track(
    mode: str,
    columns: dict,
    bunches: int,
    *,
    length: float = STRAIGHT,
    charge: float = 2e-10,
    passes: int = 1,
    nonlinear: int = 0,
    factor: float = 1.0,
    duration: float = 0.0,
    interpolate: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run the ring for `passes` turns; return per-pass coordinates (entrance, exit) by particle ID.

    `charge` is the charge of each bunch. The SCMULT acts after DS, a drift of `length`.
    """
    lattice = (
        f"Q: CHARGE,TOTAL={bunches * charge!r}\n"
        'W0: WATCH,FILENAME="entrance.sdds",MODE="coordinates"\n'
        f"DS: DRIF,L={length!r}\n"
        'W1: WATCH,FILENAME="exit.sdds",MODE="coordinates"\n'
        f"DT: DRIF,L={STRAIGHT - length!r}\n"
        f"DA: DRIF,L={STRAIGHT!r}\n"
        f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1={-(H**2) / 2!r}\n"
        "BL: LINE=(Q,W0,DS,W1,DT,B,DA,B,DA,B,DA,B)\n"
    )
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        '&insert_sceffects name="DS",skip=1,horizontal=1,vertical=1,'
        f"nonlinear={nonlinear},averaging_factor={factor!r},slice_duration={duration!r},"
        f"slice_interpolation={interpolate} &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=2,"
        "output=out.sdds &end\n"
        f"&run_control n_steps=1,n_passes={passes} &end\n"
        "&twiss_output matched=1 &end\n"
        "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
        "&track soft_failure=0 &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands},
        mode,
        {"beam.sdds": columns},
        outputs=("entrance.sdds", "exit.sdds"),
    )
    total = len(columns["particleID"])
    pages = []
    for name in ("entrance.sdds", "exit.sdds"):
        sdds = result[name]
        if sdds.n_pages != passes:
            sys.exit(f"{mode}: {name} has {sdds.n_pages} pages for {passes} passes")
        pages.append([particles(sdds, k) for k in range(passes)])
    for page in pages[0] + pages[1]:
        if not np.array_equal(page["particleID"], np.arange(1, total + 1)):
            sys.exit(f"{mode}: particles lost")
        if not all(np.isfinite(page[key]).all() for key in ("x", "xp", "y", "yp", "t")):
            sys.exit(f"{mode}: non-finite coordinates")
    if re.search(r"\b(error|fatal|nan)\b", result["log"], re.IGNORECASE):
        sys.exit(f"{mode}: diagnostics in the native log")
    return pages[0], pages[1]


def rms(page: dict, rows: slice = slice(None)) -> np.ndarray:
    """Rms sizes (x, y, z) in m of a bunch; z = beta c times the arrival-time spread."""
    return np.array(
        [np.std(page["x"][rows]), np.std(page["y"][rows]), BETA * C * np.std(page["t"][rows])]
    )


def fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"
