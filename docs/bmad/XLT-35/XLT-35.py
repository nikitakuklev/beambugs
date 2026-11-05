"""XLT-35: SAD import of a zero-length BEND drops its kick or writes invalid attributes

Importing a zero-length SAD BEND produces a missing dipole kick or invalid Bmad attributes. A dipole strength K0 = 1 mrad becomes a multipole without the required K0L status; Bmad emits a fatal diagnostic and resets its strength to zero. ROTATE and DROTATE become attributes that are invalid for a multipole, while a nonzero reference angle ANGLE becomes an invalid zero-length sbend. Handwritten translations that distinguish the dipole kick from the reference bend agree with SAD’s thin-map equations.

The script runs the importer selected by BMAD_SOURCE on dipole, rotated quadrupole and reference-bend cases. It disables exit_on_error to inspect the K0L reset after its fatal diagnostic, identifies the parser failures, and checks valid translations against SAD’s thin maps. Controls require successful parsing and tracking without error diagnostics."""

import numpy as np
from _bmad import compare, verdict
from _s3_24_translations import lattice, sad_deck, sad_kick, translate_sad
from _s4_17_translations import diagnostic_track, track

bad = False
cases = [
    ("K0=.001", 0, 0.001, 0.0, "ab_multipole,b0=.001", None),
    ("K0=.001 ROTATE=.17", 0, 0.001, 0.17, "ab_multipole,b0=.001,tilt=-.17", "BAD ATTRIBUTE NAME"),
    ("K1=.5 ROTATE=.17", 1, 0.5, 0.17, "multipole,k1l=.5,tilt=-.17", "BAD ATTRIBUTE NAME"),
    ("K1=.5 DROTATE=.17", 1, 0.5, 0.17, "multipole,k1l=.5,tilt=-.17", "BAD ATTRIBUTE NAME"),
    ("ANGLE=.01", 0, 0.0, 0.0, "multipole,k0l=.01,k0l_status=bends_reference", "ZERO LENGTH"),
]
for attrs, n, k, angle, replacement, diagnostic in cases:
    text = translate_sad(sad_deck(f"BEND M=({attrs});"))
    print(attrs + ": " + next(line for line in text.splitlines() if line.startswith("m:")))
    expected = sad_kick(n, k, angle)
    # Set up the valid control before interpreting a failing translated lattice.
    control, _ = track(lattice(replacement))
    if compare(
        attrs + " valid translation control", control["FIRST"][[1, 3]], expected, atol=3e-16
    ):
        raise RuntimeError("Valid translation control failed")
    result, log = diagnostic_track(
        text, expected_error=diagnostic or "CANNOT HAVE A FINITE K0L VALUE UNLESS K0L_STATUS"
    )
    if diagnostic:
        if result:
            bad |= compare(
                attrs + " accepted translated kick", result["FIRST"][[1, 3]], expected, atol=3e-16
            )
        elif diagnostic in log.upper():
            print(f"{attrs}: observed parser rejection {diagnostic}; expected accepted lattice")
            bad = True
        else:
            raise RuntimeError(log)
    else:
        if not result:
            raise RuntimeError(log)
        if "CANNOT HAVE A FINITE K0L VALUE UNLESS K0L_STATUS" in log:
            print(
                "K0 import: bookkeeping rejects missing K0L_status and resets K0L; diagnostic execution continues"
            )
        elif "[FATAL" in log or "[ERROR" in log or "[ABORT" in log:
            raise RuntimeError(log)
        bad |= compare("K0 imported kick", result["FIRST"][[1, 3]], expected, atol=3e-16)
# SAD tbthin uses internal px/(1+delta): canonical dpx=ANGLE*delta-K0,
# dz=-ANGLE*xb. TENT/TEXIT rotation is the negative of Bmad tilt.
start = np.array([0.001, 0.0002, 0.002, -0.0001, 0.0003, 0.001])
theta = 0.17
c, s = np.cos(theta), np.sin(theta)
for k0 in (0.0, 0.001):
    xb = c * start[0] - s * start[2]
    expected = start.copy()
    kick = 0.01 * start[5] - k0
    expected[1] += c * kick
    expected[3] -= s * kick
    expected[4] -= 0.01 * xb
    tail = f"k:ab_multipole,b0={k0},tilt=-.17"
    lat = lattice("multipole,k0l=.01,k0l_status=bends_reference,tilt=-.17", tail, "m,k")
    result, _ = track(lat, start=start)
    if compare(
        f"ANGLE=.01 K0={k0} off-momentum thin-map control", result["ORBIT"], expected, atol=3e-16
    ):
        raise RuntimeError("Thin reference-bend control failed")
verdict(bad)
