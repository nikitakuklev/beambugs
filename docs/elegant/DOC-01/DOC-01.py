"""DOC-01: Manual gives the third-order momentum compaction term as δ² instead of δ³

The Elegant manual gives the path length as s = s₀ + αc L δ + αc2 L δ² + αc3 L δ², with δ the
relative momentum offset, for the Twiss outputs `alphac`, `alphac2` and `alphac3`. The last power
should be 3: `alphac3` is the coefficient of δ³. For a prescribed one-turn map of length 10 m with
path 10 m + 2 m·δ² + 7 m·δ³, Elegant writes αc = 0, αc2 = 0.2000 and αc3 = 0.7000, and tracking
gives 10.01 m at δ = −0.1 and 10.03 m at δ = +0.1; the printed equation predicts 10.09 m at both.

Setup: `ELEGANT_SOURCE` set to a clone of rtsoliday/elegant; the script reads the manual's LaTeX
source from it.

The script reads the compaction equation from the checkout's `doc/elegant.tex` and exits if it is
missing or unrecognized. It prescribes the one-turn map with zero transverse orbit and dispersion,
reads `alphac`, `alphac2` and `alphac3` from `twiss_output`, and tracks particles at six momentum
offsets to obtain the path length from their arrival times. It compares the tracked path with the
manual's equation and with the equation using δ³. A map without the cubic term, for which the two
equations coincide, is the control.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
from _elegant import modes, particles, run, verdict
from _matr import matrix_text, rotations

C = 299792458.0  # m/s, c_mks of SDDS
P0 = 1000.0  # reference momentum, units of m_e c
LENGTH = 10.0  # m
T566 = 2.0  # m: path = LENGTH + T566 delta^2 + Q5666 delta^3
DELTAS = np.tile([-0.1, -0.05, -0.02, 0.02, 0.05, 0.1], 2)  # twice, so each MPI worker tracks some
ATOL = 2e-13  # m, rounding of the tracked path and of the written coefficients

# The manual's path-length equation, s = s_o + alphac L delta + alphac2 L delta^2 + alphac3 L delta^N;
# the exponents of the alphac2 and alphac3 terms are captured.
MANUAL_EQUATION = re.compile(
    r"s\s*=\s*s_o\s*\+\s*\\alpha_c\s+L\s*\\delta\s*"
    r"\+\s*\\alpha_\{c2\}\s+L\s*\\delta\^\{?(\d+)\}?\s*"
    r"\+\s*\\alpha_\{c3\}\s+L\s*\\delta\^\{?(\d+)\}?"
)

COMMANDS = """\
&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3,
 output=out.sdds &end
&run_control n_steps=1, n_passes=1 &end
&twiss_output filename=twiss.sdds, concat_order=3 &end
&sdds_beam input=beam.sdds &end
&track soft_failure=0 &end
"""


def manual_exponent():
    """Power of delta that the manual's path-length equation gives the alphac3 term."""
    source = os.environ.get("ELEGANT_SOURCE")
    if not source:
        sys.exit("ELEGANT_SOURCE must name the Elegant source checkout.")
    manual = Path(source) / "doc" / "elegant.tex"
    if not manual.is_file():
        sys.exit(f"{manual} does not exist.")
    found = MANUAL_EQUATION.findall(manual.read_text())
    if len(found) != 1 or found[0][0] != "2" or found[0][1] not in ("2", "3"):
        sys.exit(f"The manual's compaction equation is missing or unrecognized: {found}")
    return int(found[0][1])


def case(mode, a3):
    """Written (alphac, alphac2, alphac3) and the tracked path of each particle."""
    # Pure longitudinal map: s_out = s_in + 10 + 2 delta^2 + 10 a3 delta^3, transverse rotations.
    t = np.zeros((6, 6, 6))
    t[4, 5, 5] = T566
    q = np.zeros((6, 6, 6, 6))
    q[4, 5, 5, 5] = LENGTH * a3
    constants = np.array([0, 0, 0, 0, LENGTH, 0.0])
    beam = {
        "p": P0 * (1 + DELTAS),
        "particleID": np.arange(1, len(DELTAS) + 1),
    }
    result = run(
        {
            "lattice.lte": "E: MATR,L=10,FILENAME=MAP.TXT,ORDER=3\nBL: LINE=(E)\n",
            "run.ele": COMMANDS,
            "MAP.TXT": matrix_text(constants, rotations(), t, q),
        },
        mode,
        {"beam.sdds": beam},
        outputs=("twiss.sdds", "out.sdds"),
    )
    twiss = result["twiss.sdds"]
    out = particles(result["out.sdds"])
    # Path length from the arrival time and the output velocity: s = beta c t.
    path = out["p"] / np.sqrt(1 + out["p"] ** 2) * C * out["t"]
    return np.array([float(twiss[name][0]) for name in ("alphac", "alphac2", "alphac3")]), path


def fmt(values):
    return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


def compare(label, observed, expected):
    """Print one case; return True when observed and expected disagree."""
    bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bad


manual_power = manual_exponent()
print(f"manual: the alphac3 term of the path length is alphac3 L delta^{manual_power}")
defect = False
print("columns: path length at delta = -0.1, +0.1 (m); expected from the printed equation")
for mode in modes():
    for a3 in (0.0, 0.7):
        alphac, path = case(mode, a3)
        # The map is exactly s = L + T566 delta^2 + L a3 delta^3, so the coefficients of
        # (s - L)/L in delta, delta^2 and delta^3 are 0, T566/L and a3.
        if compare(f"{mode} a3={a3:g} alphac, alphac2, alphac3", alphac, [0, T566 / LENGTH, a3]):
            sys.exit("The written coefficients differ from the prescribed map: invalid comparison.")
        exact = LENGTH + T566 * DELTAS**2 + LENGTH * a3 * DELTAS**3
        if compare(f"{mode} a3={a3:g} tracked path, max error", np.max(np.abs(path - exact)), 0.0):
            sys.exit("The tracked path differs from the prescribed map: invalid comparison.")
        # The manual's equation, s = L (1 + alphac delta + alphac2 delta^2 + alphac3 delta^N)
        # with the power N read from the manual, and the same with the cubic power.
        printed = LENGTH * (
            1 + alphac[0] * DELTAS + alphac[1] * DELTAS**2 + alphac[2] * DELTAS**manual_power
        )
        cubic = LENGTH * (1 + alphac[0] * DELTAS + alphac[1] * DELTAS**2 + alphac[2] * DELTAS**3)
        ends = [0, 5]  # delta = -0.1 and +0.1
        if compare(
            f"{mode} a3={a3:g} delta^3 equation, max deviation", np.max(abs(path - cubic)), 0
        ):
            sys.exit("The cubic equation disagrees with tracking: invalid comparison.")
        bad = compare(f"{mode} a3={a3:g} path at delta=-0.1,+0.1", path[ends], printed[ends])
        if a3 == 0.0 and bad:
            sys.exit("The printed equation fails without a cubic term: invalid comparison.")
        defect |= bad
        print(
            f"{mode} a3={a3:g} largest deviation of the manual equation: {max(abs(path - printed)):.4g} m"
        )
print(f"comparison: absolute tolerance {ATOL:g} m")
verdict(defect)
