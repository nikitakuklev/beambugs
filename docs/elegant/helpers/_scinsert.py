"""Shared setup for the insert_sceffects command reproducers: a one-segment ring, a bunch with exact
second moments, the manual's linear space-charge kick and a runner that reports the SCMULT
insertions.

The ring is a 0.8 nC electron bunch (p = 10 mₑc) followed by R, a 1 m symplectic EMATRIX quarter
turn matched at beta = 1 m in both planes. With one linear SCMULT after R, every particle's slope
change over one turn is the manual's linear kick evaluated with the rms sizes after R.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
from _elegant import RunFailure, particles, run

# SDDS constants.h, as Elegant uses them.
C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
P = 10.0  # electron momentum, units of m_e c
GAMMA = np.hypot(1, P)
BETA_C = C * P / GAMMA
CHARGE = 0.8e-9  # C
SX, SY, SZ = 1.5e-3, 0.8e-3, 0.025  # m
LENGTH = 1.0  # m

# R: (x, xp) -> (xp, -x) and (y, yp) -> (yp, -y), a quarter turn at beta = 1 m, over 1 m.
LATTICE = (
    f"Q: CHARGE,TOTAL={CHARGE!r}\n"
    f"R: EMATRIX,L={LENGTH!r},ORDER=1,R12=1,R21=-1,R34=1,R43=-1,R55=1,R66=1\n"
    "RING: LINE=(Q,R)\n"
)
TAIL = (
    f"&run_setup lattice=lattice.lte,use_beamline=RING,p_central={P!r},default_order=1,"
    "output=out.sdds &end\n"
    '&twiss_output matched=1,filename="twiss.sdds" &end\n'
    "&run_control n_steps=1,n_passes=1 &end\n"
    "&sdds_beam input=beam.sdds &end\n"
    "&track &end\n"
)

# Every combination of -sqrt(1.5), 0, +sqrt(1.5) in (x, xp, y, yp, z): exact unit variances,
# zero means and no correlations, so the bunch has rms sizes SX, SY, SZ and is matched at 1 m.
_UNIT = np.array(list(itertools.product([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], repeat=5)))
N = len(_UNIT)
_x, _xp, _y, _yp, _z = _UNIT.T
BEAM = {
    "x": SX * _x,
    "xp": SX * _xp,
    "y": SY * _y,
    "yp": SY * _yp,
    "t": SZ * _z / BETA_C,
    "p": np.full(N, P),
    "particleID": np.arange(1, N + 1),
}
# Coordinates (x, xp, y, yp) after R without space charge.
IDEAL = np.column_stack([BEAM["xp"], -BEAM["x"], BEAM["yp"], -BEAM["y"]])


def linear_kick() -> np.ndarray:
    """The manual's linear kick K L lambda(z) x_i/[sigma_i (sigma_x + sigma_y)] for one SCMULT after R.

    K = 2 N r_e/(beta^2 gamma^3) and lambda = exp(-z^2/(2 sigma_z^2))/(sqrt(2 pi) sigma_z); the
    sizes are those after R (the quarter turn exchanges positions and slopes, whose spreads are
    equal at beta = 1 m). Returns shape (N, 2).
    """
    k = 2 * CHARGE / QE * RE / (P * P * GAMMA)
    sigma = np.array([IDEAL[:, 0].std(), IDEAL[:, 2].std()])
    s = SZ * _z
    density = np.exp(-0.5 * ((s - s.mean()) / s.std()) ** 2) / (np.sqrt(2 * np.pi) * s.std())
    return k * LENGTH * density[:, None] * IDEAL[:, [0, 2]] / (sigma * sigma.sum())


def spec(**options) -> str:
    """One insert_sceffects command with both transverse planes on and the given options."""
    options = {"horizontal": 1, "vertical": 1, **options}
    text = ",".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in options.items())
    return f"&insert_sceffects {text} &end\n"


def track(commands: str, mode: str = "serial") -> dict:
    """Run the ring with the given insert_sceffects commands.

    Returns {"status", "log"} and, when the run completes, "insertions" (names of the elements
    that precede each SCMULT in the Twiss output) and "kick" (slope change per particle, shape
    (N, 2)). A run that exits nonzero (an abort or a crash) returns its status and log.
    """
    files = {
        "lattice.lte": LATTICE,
        "run.ele": "&global_settings echo_namelists=0 &end\n" + commands + TAIL,
    }
    try:
        result = run(files, mode, {"beam.sdds": BEAM}, outputs=("out.sdds", "twiss.sdds"))
    except RunFailure as failure:
        return {"status": failure.status, "log": failure.log}
    twiss = result["twiss.sdds"]
    names = [str(n) for n in twiss[0, "ElementName"]]
    types = [str(t) for t in twiss[0, "ElementType"]]
    insertions = [names[i - 1] for i, t in enumerate(types) if t == "SCMULT"]
    out = particles(result["out.sdds"])
    kick = np.column_stack([out["xp"] - IDEAL[:, 1], out["yp"] - IDEAL[:, 3]])
    return {"status": 0, "log": result["log"], "insertions": insertions, "kick": kick}


def failure_line(log: str) -> str:
    """The last line of a failed run that names the failure, for printing."""
    lines = [
        ln.strip()
        for ln in log.splitlines()
        if re.search(r"error|signal|SIG|abort|fault", ln, re.IGNORECASE)
    ]
    return lines[-1] if lines else log.strip().splitlines()[-1]
