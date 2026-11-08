"""API-10: Native fatal errors exit with status 0

Many native error paths print a fatal-error message and call the no-message finalizer, which ends the interpreter with exit status 0. Requesting a missing bunch attribute with `Bunch.bunchAttrDouble("absent")` prints "Fatal error!" and exits with 0, as do `Boundary2D(4, 4)`, `Grid1D(0, -1.0, 1.0)` and `Grid2D(2, 2)`, whose constructors reject their sizes. Shell scripts, batch schedulers and test harnesses that check the exit status record the failed run as successful. Errors that pass a message to the finalizer exit with status 1; the no-message call is correct only for the deliberate `orbit_mpi.finalize()` stop.

The script runs each failing call in a child interpreter, followed by a line that prints `continued`, and reports the exit status, whether the script continued, and the first line of error output. The control `orbit_mpi.finalize("msg")` takes the message path and must exit with status 1.
"""

import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def run(code):
    """Exit status, whether the script continued, and the first stderr line of `code` in a child."""
    p = subprocess.run(
        [sys.executable, "-I", "-c", code + "\nprint('continued')"],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    err = [ln.strip() for ln in p.stderr.splitlines() if ln.strip() and not ln.startswith(" ")]
    return p.returncode, "continued" in p.stdout, err[0] if err else "<none>"


SC = "from orbit.core.spacecharge import Boundary2D, Grid1D, Grid2D\n"
cases = [
    (
        "Bunch().bunchAttrDouble('absent')",
        "from orbit.core.bunch import Bunch\nBunch().bunchAttrDouble('absent')",
    ),
    ("Boundary2D(4, 4) (needs nPoints > 2*nModes + 1)", SC + "Boundary2D(4, 4)"),
    ("Grid1D(0, -1.0, 1.0)", SC + "Grid1D(0, -1.0, 1.0)"),
    ("Grid2D(2, 2)", SC + "Grid2D(2, 2)"),
]
reproduced = False
for label, code in cases:
    rc, cont, err = run(code)
    print(
        f"{label}: observed exit {rc}, continued {cont}, first stderr line {err!r}; expected a nonzero exit after the fatal error"
    )
    reproduced |= rc == 0 and not cont
rc, cont, err = run("from orbit.core import orbit_mpi\norbit_mpi.finalize('msg')")
print(
    f"control orbit_mpi.finalize('msg'): observed exit {rc}, continued {cont}, first stderr line {err!r}; expected exit 1"
)

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
