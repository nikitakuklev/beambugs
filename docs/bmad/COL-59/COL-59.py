"""COL-59: Tabulated-wake origin depends on particle storage order

Reordering the stored particles of an otherwise identical bunch changes its tabulated short-range wake kicks. The dispatcher selects the table origin by averaging the storage indices of the head and tail, rather than their longitudinal coordinates. In a three-particle case, the leading particle’s normalized momentum kick changes from −1.194 × 10⁻⁶ to −1.000 × 10⁻⁶ under a storage permutation. A direct calculation with a fixed physical center agrees with an independent charge-deposition and interpolation sum.

The script tracks a fixed three-particle set in two storage orders and remaps both results to physical order. A direct call with a fixed midpoint is checked against an independent cloud-in-cell sum. BMAD_DIST selects the build.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _wake_tables_and_bbu import WAKE_SUPPORT_SOURCE, require, rows, table_kicks, table_lattice

lattice, table = table_lattice()
actual = rows(fortran(WAKE_SUPPORT_SOURCE, ["1"], files={"lat.bmad": lattice}).log)
z = np.array([0.013, 0.002, -0.014])
charge = np.array([2e-12, 0, 0])
require(
    "fixed midpoint direct delta; independent CIC sum",
    actual["permutation0"],
    table_kicks(z, charge, table, -0.0005),
    rtol=3e-14,
    atol=1e-21,
)
print(
    f"physical-order delta A={actual['permutation1'].tolist()} B={actual['permutation2'].tolist()}"
)
verdict(
    compare(
        "storage permutation delta difference; invariant physical bunch",
        actual["permutation1"] - actual["permutation2"],
        np.zeros(3),
        atol=2e-21,
    )
)
