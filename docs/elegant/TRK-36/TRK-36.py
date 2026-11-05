"""TRK-36: Identical CCBENDs that share a tracking-based matrix lose the path-length reference correction

When a matrix command such as twiss_output runs before tracking, identical CCBENDs reuse the first
one's tracking-based matrix, the default, but not its stored path-length correction, which the
default REFERENCE_CORRECTION=1 subtracts in tracking. Each later copy of an untilted CCBEND then
adds its uncorrected path excess. In a line of three CCBENDs with L = 0.85 m, ANGLE = 0.16 rad and
K₁ = 0.35 m⁻², the reference electron travels 1.534×10⁻⁵ m more than the nominal length, twice the
7.672×10⁻⁶ m excess of one uncorrected magnet; without twiss_output or without sharing the path is
nominal. A design TILT makes each copy recompute its correction, so tilted CCBENDs are not affected.

The script tracks a reference electron through three identical CCBENDs, each followed by a short
drift, with and without a preceding twiss_output and with matrix sharing on and off. The expected
path is the nominal length, which REFERENCE_CORRECTION=1 gives in the runs without sharing; the
size of the error is predicted independently from one magnet tracked with REFERENCE_CORRECTION=0.
A design TILT, which forces each copy to recompute its reference, is shown as the case where the
defect does not occur.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

C = 299792458.0  # speed of light (m/s)
P0 = 6000.0  # reference momentum in m_e c
BEND = "CCBEND,L=0.85,ANGLE=0.16,K1=0.35,N_SLICES=64"
COPIES, DRIFT = 3, 0.03
NOMINAL = COPIES * (0.85 + DRIFT)
TOL = 1e-12  # the corrected runs return the nominal path to rounding (observed below 5e-16 m)
N = 12  # identical particles, so that every MPI worker tracks some


def path_length(mode, lattice, twiss, share):
    """Path of the reference electron through the line, from its arrival time."""
    commands = (
        f"&global_settings share_tracking_based_matrices={int(share)} &end\n"
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
        + ("&twiss_output filename=out.twi, matched=0, beta_x=2, beta_y=3 &end\n" if twiss else "")
        + "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    beam = {"p": np.full(N, P0)}
    out = particles(
        run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
    )
    if len(out["p"]) != N or np.ptp(out["t"]) > 0:
        sys.exit(f"{mode}: particles lost or not identical: invalid")
    p = out["p"][0]
    return out["t"][0] * C * p / np.sqrt(1 + p * p)


def line(tilt=0.0):
    cells = ",".join(["B,D"] * COPIES)
    return f"B: {BEND},TILT={tilt!r}\nD: DRIF,L={DRIFT!r}\nEND: MARK\nBL: LINE=({cells},END)\n"


defect = []
for mode in modes():
    # Independent size: the uncorrected path excess of one magnet, which the correction subtracts.
    single = (
        path_length(mode, f"B: {BEND},REFERENCE_CORRECTION=0\nBL: LINE=(B)\n", False, True) - 0.85
    )
    predicted = (COPIES - 1) * single  # every copy after the first misses its subtraction
    print(
        f"{mode} one uncorrected magnet: path excess {single:.10g} m;"
        f" predicted excess with sharing {predicted:.10g} m"
    )
    for twiss, share in ((False, False), (False, True), (True, False), (True, True)):
        excess = path_length(mode, line(), twiss, share) - NOMINAL
        label = f"{mode} twiss_output={'yes' if twiss else 'no'} sharing={'on' if share else 'off'}"
        print(f"{label}: path - nominal observed={excess:.10g} m expected=0")
        if not (twiss and share):
            if abs(excess) > TOL:
                sys.exit(f"{label}: the control does not return the nominal path: invalid")
            continue
        if abs(excess - predicted) > TOL:
            sys.exit(f"{label}: the excess is not the missing corrections: the cause differs")
        defect.append(abs(excess) > TOL)
    excess = path_length(mode, line(0.13), True, True) - NOMINAL
    print(
        f"{mode} TILT=0.13 twiss_output=yes sharing=on (copies recompute their reference):"
        f" path - nominal observed={excess:.10g} m expected=0"
    )
    if abs(excess) > TOL:
        sys.exit("The tilted case also loses the correction: the scope differs.")
print(f"comparison: abs tolerance {TOL:g} m")
verdict(bool(defect) and all(defect))
