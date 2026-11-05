"""COL-30: insert_sceffects with skip = 0 inserts no SCMULT when name or type matches nothing

With the default skip = 0, insert_sceffects adds its single end-of-beamline SCMULT only if the name
and type selectors match at least one element: a name absent from the lattice, or a type no element
has, leaves the ring without space charge, and the run completes without a warning. The manual
states that skip = 0 inserts one SCMULT at the end of the beamline regardless of whether name or
type is given, and the manual's linear kick for that element is the expected value. In a
one-segment ring with a 0.8000 nC bunch at momentum 10.00 mₑc, the expected slope kick of
2.379×10⁻⁴ becomes exactly zero, while name = "*" gives it.

The script runs a one-segment ring (a 0.8 nC bunch of 243 particles and a 1 m EMATRIX quarter
turn) with insert_sceffects at the default skip = 0 and selectors that match no element: a name
absent from the ring, and an existing element type absent from the ring. The manual requires one
SCMULT at the end of the beamline in every case, so the expected kick is the manual's linear formula
for one SCMULT after the segment. A matching selector, name = "*", is the control.
"""

import numpy as np
from _elegant import modes, verdict
from _scinsert import failure_line, linear_kick, spec, track

EXPECTED = linear_kick()
PROBE = int(np.argmax(EXPECTED[:, 0]))  # the reported particle: largest horizontal kick
ATOL = 1e-18
CASES = (
    ('name="*" (control)', spec(name="*"), True),
    ('name="ABSENT"', spec(name="ABSENT"), False),
    ('name="*",type="QUAD" (no QUAD in the ring)', spec(name="*", type="QUAD"), False),
)

defect = False
for mode in modes():
    for label, commands, control in CASES:
        result = track(commands, mode)
        if result["status"]:
            raise SystemExit(
                f"{mode} {label}: exit {result['status']} ({failure_line(result['log'])})"
            )
        kick = result["kick"]
        print(
            f"{mode} skip=0, {label}: observed={len(result['insertions'])} SCMULT, "
            f"kick={kick[PROBE, 0]:.16g} expected=1 SCMULT, kick={EXPECTED[PROBE, 0]:.16g} "
            f"(particle {PROBE + 1}, exit status 0)"
        )
        ok = len(result["insertions"]) == 1 and bool(np.allclose(kick, EXPECTED, rtol=0, atol=ATOL))
        if control and not ok:
            raise SystemExit(f"{mode} {label}: the control fails")
        defect |= not ok
print(f"comparison: kicks atol={ATOL:g} against the manual's linear kick for one SCMULT")
verdict(defect)
