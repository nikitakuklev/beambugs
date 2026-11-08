"""LAT-07: `RF_AxisFieldAnalysis` cannot analyse a two-gap cavity field and misplaces a single gap

`RF_AxisFieldAnalysis`, which builds per-gap transit-time-factor tables from a cavity's axis field, takes the cavity center as the mean of the interior gap centers and divides their sum by the number of gaps minus two. For a two-gap field the default mode raises `ZeroDivisionError`, and the `zeroIsCenter=True` mode exits for any field with more than one gap, so two-gap cavities cannot be analysed. A one-gap field in the default mode divides by −1 and gets cavity center 0 instead of the gap center, so the gap position relative to the cavity center is 0.06000 m instead of 0 for a gap on [0, 0.12] m. Fields of three to five gaps get correct centers.

The script analyses π-mode fields sin(πz/0.12 m) with one to five gaps of 0.12 m each, in the default mode, and with one and two gaps in the single-gap `zeroIsCenter=True` mode. In the default mode it compares the cavity center and the gap positions relative to it with the mean of the gap centers and the gap centers minus that mean. The three- to five-gap fields are the control.
"""

import contextlib
import io
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core.orbit_utils import Function, SplineCH
from orbit.py_linac.rf_field_readers import RF_AxisFieldAnalysis

CELL = 0.12  # m, gap (half-period) length


def field(ngap):
    """Alternating-sign half waves on [0, ngap*CELL], zeros at the gap borders, 600 samples per gap."""
    fn = Function()
    n = 600 * ngap + 1
    length = CELL * ngap
    for i in range(n):
        z = length * i / (n - 1)
        fn.add(z, math.sin(math.pi * z / CELL))
    spline = SplineCH()
    spline.compile(fn)
    return spline


def analyse(ngap, zero_is_center):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            an = RF_AxisFieldAnalysis(field(ngap), zero_is_center)
        except ZeroDivisionError as err:
            return f"ZeroDivisionError ({err})", None, None
        except SystemExit as err:
            return f"SystemExit({err.code})", None, None
    return "ok", an.rf_center, [g[1] for g in an.gap_slpline_arr]


def fmt(values):
    # Gap centers come from a root search with tolerance 1e-6 m; round to 1e-5 m before printing.
    return "[" + ", ".join(f"{round(v, 5) + 0.0:.4g}" for v in values) + "]"


reproduced = False
# Default mode: cavity center = mean of the gap centers CELL*(i + 1/2); gap positions relative to it.
for ngap in (1, 2, 3, 4, 5):
    centers = [CELL * (i + 0.5) for i in range(ngap)]
    exp_center = sum(centers) / ngap
    exp_positions = [c - exp_center for c in centers]
    status, center, positions = analyse(ngap, False)
    label = f"{ngap} gap(s), default mode" + (" (control)" if ngap >= 3 else "")
    if status != "ok":
        print(
            f"{label}: {status}; expected cavity center {exp_center:.4g} m, gap positions {fmt(exp_positions)} m"
        )
        reproduced = True
        continue
    print(
        f"{label}: cavity center {round(center, 5) + 0.0:.4g} m, expected {exp_center:.4g} m; "
        f"gap positions {fmt(positions)} m, expected {fmt(exp_positions)} m"
    )
    wrong = abs(center - exp_center) > 1e-5 or any(
        abs(a - b) > 1e-5 for a, b in zip(positions, exp_positions)
    )
    if ngap >= 3 and wrong:
        raise RuntimeError(
            "control case failed: a field of three or more gaps is analysed incorrectly"
        )
    reproduced |= wrong

# zeroIsCenter=True is the single-gap route: it takes z = 0 as the gap center and requires exactly one gap,
# so it offers no alternative for a two-gap field. Printed for completeness; the verdict rests on the default mode.
for ngap in (1, 2):
    status, _, _ = analyse(ngap, True)
    print(f"{ngap} gap(s), zeroIsCenter=True (single-gap mode): {status}")
print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
