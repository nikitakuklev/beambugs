"""XLT-04: Converter polarization interpolation leaves its output uninitialized

Interpolating a converter distribution between incoming momenta leaves the outgoing longitudinal polarization unassigned. Two tables with polarization transfers 0.25 and 0.75 require a transfer of 0.5 at their momentum midpoint. Diagnostic initialization exposes an undefined value in all 128 samples; adding the missing weighted interpolation gives exactly 0.5 in every sample. The ordinary production library returns zero for this fixture.

The script runs the production converter and compiles a diagnostic copy of the source selected by BMAD_SOURCE with quiet-NaN initialization. A second copy adds only the missing interpolation assignment. Both diagnostic copies link against BMAD_DIST and run wrapper and direct-call controls.
"""

import numpy as np
from _bmad import compare, verdict
from _s3_24_converter import samples

production, _, _ = samples()
print(
    f"production polarization range: {[float(production[:, 2].min()), float(production[:, 2].max())]} expected=0.5"
)
bad = False
for mode in ("converter", "direct"):
    observed, _, _ = samples(mode=mode, diagnostic=True)
    assigned, _, _ = samples(mode=mode, diagnostic=True, assigned=True)
    # Equal incoming-momentum distances give equal weights: (0.25+0.75)/2=0.5.
    if compare(
        f"{mode} assigned interpolation maximum residual",
        np.max(np.abs(assigned[:, 2] - 0.5)),
        0,
        atol=2e-15,
    ):
        raise RuntimeError("Assigned interpolation control failed")
    count = int(np.count_nonzero(np.isnan(observed[:, 2])))
    print(f"{mode} diagnostic undefined polarization: observed_nan_count={count}/128 expected=0")
    if count not in (0, 128):
        raise RuntimeError("Diagnostic spin samples are inconsistent")
    if not count:
        bad |= compare(
            f"{mode} diagnostic polarization", observed[:, 2], np.full(128, 0.5), atol=2e-15
        )
    bad |= count > 0
verdict(bad)
