"""COL-61: Tolerated table outliers receive wrapped or extrapolated kicks

A tabulated short-range wake can accept an outlier under its one-percent tolerance, skip that particle during charge deposition, and then give it a kick interpolated from wrapped grid nodes. With one outlier among 101 particles, all particles remain alive and the core receives the expected normalized momentum kick of −1.000 × 10⁻⁶. The outlier instead receives −5.720 × 10⁻⁶ where a zero-continued table gives zero. Signed interpolation weights of 1.500 and −0.500 explain this result.

The script tracks 100 core particles and one outlier that lies outside the table’s deposition window. It checks the live count and all core kicks, then compares the outlier with a zero-continued table sum.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _wake_tables_and_bbu import WAKE_SUPPORT_SOURCE, require, rows, table_kicks, table_lattice

lattice, table = table_lattice()
actual = rows(fortran(WAKE_SUPPORT_SOURCE, ["3"], files={"lat.bmad": lattice}).log)
require("live count", actual["live"], [101])
require(
    "core delta range; half self-bin",
    [np.min(actual["core"]), np.max(actual["core"])],
    [-1e-6, -1e-6],
    rtol=3e-14,
    atol=1e-21,
)
expected = table_kicks([0, -0.065], [2e-12, 0], table, 0)[1]
require("zero-continued outlier oracle", expected, 0)
# rz_rel=-0.5: Fortran MOD gives -0.5, so the gather weights are 1.5 and -0.5.
wrapped = -2e-12 / 1e12 * (1.5 * table[9] - 0.5 * table[10])
print(f"signed wrapped-gather prediction delta={wrapped:.17g}")
verdict(compare("outlier delta; zero-continued table", actual["outlier"], [expected], atol=1e-21))
