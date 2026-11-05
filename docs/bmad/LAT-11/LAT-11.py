"""LAT-11: An omitted HDF5 momentum policy selects inconsistent defaults

Reading an HDF5 beam at a different reference momentum without specifying conserve_momentum combines incompatible defaults. Changing the reference from 1 to 1.7 MeV/c keeps the relative total-momentum deviation δ = 0.1 but changes transverse momenta divided by the reference momentum from (0.04, −0.02) to (0.02353, −0.01176). The documented default is false, which retains all three normalized momentum coordinates. Explicit false preserves those coordinates; the source’s warning branches describe the opposite momentum policy.

The script writes one electron at a 1 MeV/c reference and reads it at 1.7 MeV/c with omitted, false and true policies in fresh processes. It compares omission with explicit false for the three normalized momentum components and validates explicit true against conservation of physical momentum.
"""

import numpy as np
from _bmad import compare, verdict
from _s3_11_particle_io import policy

results = {mode: policy(mode) for mode in ("true", "false", "omitted")}
for mode, result in results.items():
    if compare(
        f"{mode} input control: x, px, y, py, z, delta, reference",
        result["input"][:7],
        [0.001, 0.04, -0.002, -0.02, 0.01, 0.1, 1e6],
        atol=2e-15,
    ):
        raise RuntimeError("Input control failed")
    if compare(
        f"{mode} position/reference/time control",
        result["output"][[0, 2, 6, 8]],
        [0.001, -0.002, 1.7e6, 4.1e-9],
        atol=2e-18,
        rtol=3e-15,
    ):
        raise RuntimeError("Read control failed")
expected = np.array([0.04, -0.02, 0.1])
if compare(
    "explicit false normalized momenta; documented policy",
    results["false"]["output"][[1, 3, 5]],
    expected,
    atol=2e-15,
):
    raise RuntimeError("Explicit-false control failed")
true_expected = [0.04 / 1.7, -0.02 / 1.7, 1.1 / 1.7 - 1]
if compare(
    "explicit true normalized momenta; physical momentum conserved",
    results["true"]["output"][[1, 3, 5]],
    true_expected,
    atol=2e-15,
):
    raise RuntimeError("Explicit-true control failed")
verdict(
    compare(
        "omitted normalized momenta; documented false default",
        results["omitted"]["output"][[1, 3, 5]],
        expected,
        atol=2e-15,
    )
)
