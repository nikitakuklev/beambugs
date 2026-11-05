"""COL-60: A table-support error marks only the final array particle lost

When a bunch exceeds a tabulated short-range wake’s support, the error path marks only the final stored particle lost. With the first particle outside the table and the last inside it, the outlier remains alive while the in-window particle is lost. Both kicks remain zero, although the documented unrecoverable support error requires tracking of the whole bunch to stop.

The script puts the first particle outside the wake window and the final particle inside it. It requires the support-error diagnostic and zero kicks, then compares both loss flags with rejection of the whole bunch. BMAD_DIST selects the build.
"""

import re

from _bmad import compare, fortran, verdict
from _wake_tables_and_bbu import WAKE_SUPPORT_SOURCE, require, rows, table_lattice

lattice, _ = table_lattice()
result = fortran(
    WAKE_SUPPORT_SOURCE, ["2"], files={"lat.bmad": lattice}, outputs=["diagnostic.log"]
)
diagnostic = " ".join(result.outputs["diagnostic.log"].decode().split())
if not re.fullmatch(
    r"\[ERROR(?: \| [^]]+)?\] sr_z_long_wake: The bunch is longer than the sr-z wake can handle for element: W",
    diagnostic,
):
    raise RuntimeError(diagnostic)
print("diagnostic: " + diagnostic)
actual = rows(result.log)
require("support failure delta", actual["kicks"], [0, 0])
verdict(compare("loss flags; whole bunch must stop", actual["lost"], [1, 1], atol=0))
