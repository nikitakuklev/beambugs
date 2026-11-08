"""COL-16: `SC1D_AccNode` annotates its integer arguments as `float`, and float values terminate the interpreter

`SC1D_AccNode.__init__` annotates `nmacros_min`, `use_sc` and `nbins` as `float` (`BetFreqDep_SC1D_AccNode` does the same for `nmacros_min`), but passes them to the native `LSpaceChargeCalc`, whose argument parser requires integers. A call that follows the annotations, such as `SC1D_AccNode(3.0, 248.0, 1.0, 1.0, 128.0)`, fails argument parsing, and the native error handler ends the process with exit status 1; the error cannot be caught. Any float in one of the three slots, including a NumPy float, does the same, while the same call with integers constructs the node.

The script prints the annotations and then constructs the node inside `try`/`except BaseException` in a child process, once with the annotated float types, once with a single NumPy float bin count, and once with integers as the control. It reports each child's exit status and last line of output.
"""

import inspect
import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.space_charge.sc1d import SC1D_AccNode

params = inspect.signature(SC1D_AccNode.__init__).parameters
ann = {
    k: v.annotation.__name__ for k, v in params.items() if k in ("nmacros_min", "use_sc", "nbins")
}
print(f"annotations {ann}")
CHILD = """
import numpy as np
from orbit.space_charge.sc1d import SC1D_AccNode
try:
    SC1D_AccNode(3.0, 248.0, {args})
    print("constructed")
except BaseException as e:
    print("caught", type(e).__name__)
"""
res = {}
for label, args in (
    ("float arguments (as annotated)", "1.0, 1.0, 128.0"),
    ("NumPy float bin count", "1, 1, np.float64(128)"),
    ("integer arguments (control)", "1, 1, 128"),
):
    p = subprocess.run(
        [sys.executable, "-I", "-c", CHILD.format(args=args)],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    last = (p.stdout.strip().splitlines() or ["<no stdout>"])[-1]
    res[label] = (p.returncode, last)
    print(
        f"{label}: exit status observed {p.returncode}, stdout '{last}'; expected 0, 'constructed' (or a caught exception)"
    )
bad = all(res[k][0] != 0 and "caught" not in res[k][1] for k in list(res)[:2])
ctl_ok = res["integer arguments (control)"] == (0, "constructed")
print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")
