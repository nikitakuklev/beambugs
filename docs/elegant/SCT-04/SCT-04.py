"""SCT-04: MATTER collision slices omit incoming transverse drift

When MATTER uses single scattering in a finite-length target, each slice containing a collision
omits the transverse drift from the slope entering that slice. Correct transport must include that
slope times the slice length, in addition to the displacement from the scattering kick. In a
one-slice, 0.01 m helium target, shifting the incoming horizontal and vertical slopes by 0.002 and
−0.003 leaves all scattered electrons' exit positions unchanged, instead of shifting them by 20.00
µm and −30.00 µm; unscattered electrons drift correctly. Earlier kicks can also expose the error in later slices.

The script tracks 2400 electrons through the target twice with the same random seed, once on axis
and once with the incoming slopes shifted, so that both runs draw identical collisions and kicks. It
compares the shift of the exit positions with the geometric requirement L·(a, b), separately for the
electrons that scattered and for those that did not; the unscattered electrons are the control and
must drift correctly.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _matter import LENGTH, beam, track

COUNT, SEED = 2400, 56731
SHIFT = np.array([0.002, -0.003])  # incoming slope shift (xp, yp)
# One slice (PLIMIT above the mean 0.3 collisions per electron); no energy loss.
ELEMENT = f"L={LENGTH!r},PLIMIT=0.5,ENERGY_DECAY=0,NUCLEAR_BREMSSTRAHLUNG=0,ELECTRON_RECOIL=0"
ATOL = 1e-18  # m; positions are about 1e-5 m, so this is about 1e-13 relative


def fmt(values):
    return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


def compare(label, observed, expected):
    bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bad


defect = False
print(
    "exit-position shift in m, as (x min, x max, y min, y max) over electrons, for shifted slopes"
)
for mode in modes():
    zero = track(mode, ELEMENT, SEED, beam(COUNT))
    sloped = track(mode, ELEMENT, SEED, beam(COUNT, *SHIFT))
    # The same seed gives the same collisions and kicks in both runs, so the outgoing slopes differ
    # by exactly the incoming shift. Otherwise the electrons cannot be paired.
    kick = np.column_stack([sloped["xp"] - SHIFT[0], sloped["yp"] - SHIFT[1]])
    if not np.allclose(kick, np.column_stack([zero["xp"], zero["yp"]]), rtol=0, atol=1e-17):
        sys.exit(f"{mode}: the paired runs drew different kicks")
    scattered = (zero["xp"] != 0) | (zero["yp"] != 0)
    if not 0 < scattered.sum() < COUNT:
        sys.exit(f"{mode}: need both scattered and unscattered electrons")
    shift = np.column_stack([sloped["x"] - zero["x"], sloped["y"] - zero["y"]])
    # A kick at depth z of a slice of length L gives x_out = x_in + z a + (L - z)(a + k)
    # = x_in + L a + (L - z) k, so the incoming slope a adds L a whatever z and k are.
    expected = LENGTH * SHIFT
    for name, rows in (("unscattered control", ~scattered), ("scattered", scattered)):
        label = f"{mode} {name} ({rows.sum()} electrons)"
        observed = np.ravel([shift[rows].min(axis=0), shift[rows].max(axis=0)], order="F")
        bad = compare(label, observed, np.repeat(expected, 2))
        if name == "unscattered control" and bad:
            sys.exit(f"{mode}: the unscattered control disagrees: the comparison is not valid")
        defect |= bad
print(f"comparison: atol={ATOL:g} m")
verdict(defect)
