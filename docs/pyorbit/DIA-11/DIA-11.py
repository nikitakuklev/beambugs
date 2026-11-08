"""DIA-11: `profiles` terminates the interpreter for its documented `xp` and `yp` selectors

`orbit.diagnostics.profiles(bunch, coord, histogram, …)` documents its coordinates as "x[m] xp[rad]
y[m] yp[rad] z[m] dE[GeV]" and its error message repeats "x,xp,y,yp,z, or dE", but the code accepts
only `x`, `px`, `y`, `py`, `z` and `dE`. Passing `xp` or `yp`, or any other unrecognized name, calls
the fatal MPI finalizer, which prints the message and exits the Python process with status 1 instead
of returning a histogram or raising an exception. For a five-particle bunch, `xp` and `yp` end the
process without writing the histogram file, while `px` and `py` return a grid with sum 5.

The script calls `profiles` on a five-particle bunch in child interpreters with the documented
selectors `xp` and `yp` and, as the control, with `px` and `py`, which address the same stored
coordinates, and reports each exit status, whether the histogram file was written and the grid
sum.
"""

import os
import subprocess
import sys
import tempfile

CHILD = """
import sys
from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
from orbit.diagnostics import profiles
b = Bunch()
b.macroSize(1.0)
for i in range(5):
    b.addParticle(0.0, 1e-4 * i, 0.0, 2e-4 * i, 0.0, 0.0)
g = profiles(b, sys.argv[1], sys.argv[2], steps=10)
print("grid sum", g.getSum())
"""
tmp = tempfile.mkdtemp()
res = {}
for coord in ("xp", "yp", "px", "py"):
    hist = os.path.join(tmp, f"{coord}.txt")
    r = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, coord, hist],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    res[coord] = (r.returncode, os.path.exists(hist), r.stdout.strip())
    message = r.stderr.strip().splitlines()[-2] if r.returncode else ""
    # Expected for every selector: the five slopes are binned with macro size 1 each, grid sum 5.
    print(
        f"coord {coord!r}: exit status {r.returncode}, histogram written {os.path.exists(hist)}, "
        f"stdout {r.stdout.strip()!r}{', stderr ' + repr(message) if message else ''}; "
        "expected exit status 0, histogram written, grid sum 5.0"
    )

bad = all(res[c][0] == 1 and not res[c][1] for c in ("xp", "yp"))
good = all(res[c] == (0, True, "grid sum 5.0") for c in ("px", "py"))
print("DEFECT REPRODUCED" if bad and good else "NOT REPRODUCED")
