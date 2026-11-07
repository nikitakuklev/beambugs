"""COL-58: Gaussian table smoothing omits the squared index

Applying Gaussian smoothing to a tabulated short-range wake uses an asymmetric exponential when the grid index is not squared. A symmetric impulse must produce equal kicks at equally spaced witnesses on either side, but the faulty kernel gives an outer-kick ratio of 54.60 for a smoothing width of half a grid spacing. A normalized Gaussian convolution and an unsmoothed self-bin control establish the expected symmetry and magnitude.

The script parses a symmetric impulse table with and without Gaussian smoothing, then compares three tracked kicks against a normalized Gaussian. The self-bin is halved before smoothing, as required by beam loading.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _wake_tables_and_bbu import TABLE_WAKE_SOURCE, require, rows

table = ", ".join(f"{i * 0.01:.2f} {1e18 if i == 0 else 0:.1e}" for i in range(-4, 5))
lattice = (
    "parameter[particle]=electron\nparameter[p0c]=1e12\n"
    "d: lcavity,l=0.6,sr_wake={scale_with_length=F,z_long={smoothing_sigma=0.005,w={"
    + table
    + ", }}}\nl: line=(d)\nuse,l\n"
)
control = rows(
    fortran(
        TABLE_WAKE_SOURCE,
        files={"lat.bmad": lattice.replace("smoothing_sigma=0.005", "smoothing_sigma=0")},
    ).log
)
require("unsmoothed delta; half self-bin", control["table"], [0, -1e-6, 0], rtol=2e-13, atol=1e-22)
actual = rows(fortran(TABLE_WAKE_SOURCE, files={"lat.bmad": lattice}).log)
require("raw impulse (V/C)", actual["raw"], [0, 0, 0, 0, 1e18, 0, 0, 0, 0])
# sigma/dz=0.5; nint(3*sigma/dz)=2. Half the impulse enters the convolution.
kernel = np.exp(-0.5 * (np.arange(-2, 3) / 0.5) ** 2)
expected = -1e-6 * kernel[1:4] / kernel.sum()
defect = compare(
    "smoothed delta; normalized Gaussian of half self-bin",
    actual["table"],
    expected,
    rtol=2e-13,
    atol=1e-22,
)
defect |= compare(
    "outer-kick ratio; Gaussian symmetry",
    actual["table"][2] / actual["table"][0],
    1,
    rtol=2e-13,
    atol=0,
)
verdict(defect)
