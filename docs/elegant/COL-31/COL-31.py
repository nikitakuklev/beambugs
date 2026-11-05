"""COL-31: insert_sceffects aborts without a name although the manual's default insertion needs none

insert_sceffects without a name stops with "no name given", even at the default skip = 0, for which
the manual states that one SCMULT is inserted at the end of the beamline regardless of whether name
or type is given; giving only a type stops the same way. The documented defaults leave name and
type unset, so the minimal command the manual describes cannot run. Adding name = "*" inserts the
SCMULT, and its slope kick of 2.379×10⁻⁴ in a one-segment ring with a 0.8000 nC bunch agrees with
the manual's linear formula.

The script runs a one-segment ring (a 0.8 nC bunch of 243 particles and a 1 m EMATRIX quarter
turn) with insert_sceffects at the default skip = 0, once without name and type and once with only
type = "EMATRIX". The manual requires one SCMULT at the end of the beamline whether or not name or
type is given, so the expected kick is the manual's linear formula for one SCMULT after the
segment. The same command with name = "*" is the control.
"""

import numpy as np
from _elegant import modes, verdict
from _scinsert import failure_line, linear_kick, spec, track

EXPECTED = linear_kick()
PROBE = int(np.argmax(EXPECTED[:, 0]))  # the reported particle: largest horizontal kick
ATOL = 1e-18
CASES = (
    ('name="*" (control)', spec(name="*"), True),
    ("no name or type", spec(), False),
    ('type="EMATRIX" only', spec(type="EMATRIX"), False),
)

defect = False
for mode in modes():
    for label, commands, control in CASES:
        result = track(commands, mode)
        expected = f"exit 0, 1 SCMULT, kick={EXPECTED[PROBE, 0]:.16g}"
        if result["status"]:
            observed = f"exit {result['status']} ({failure_line(result['log'])})"
            ok = False
        else:
            kick = result["kick"]
            observed = f"exit 0, {len(result['insertions'])} SCMULT, kick={kick[PROBE, 0]:.16g}"
            ok = len(result["insertions"]) == 1 and bool(
                np.allclose(kick, EXPECTED, rtol=0, atol=ATOL)
            )
        print(
            f"{mode} skip=0, {label}: observed={observed} expected={expected} (particle {PROBE + 1})"
        )
        if control and not ok:
            raise SystemExit(f"{mode} {label}: the control fails")
        defect |= not ok
print(f"comparison: kicks atol={ATOL:g} against the manual's linear kick for one SCMULT")
verdict(defect)
