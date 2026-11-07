"""OPT-22: RF-off equilibrium leaves the no-opening-angle longitudinal emittance positive

Calling emit_6d with RF off and photon opening angle disabled returns a positive
longitudinal emittance from an artificial RF kick. The routine marks the ordinary
longitudinal emittance invalid with −1, but leaves emittance_no_vert positive even though it
comes from the same artificial focusing. A ring with zero damped-map M₆₅ returns −1 in the
ordinary field and 5.578 × 10⁻⁶ m in emittance_no_vert. The invalid marker should apply to
the emittance field selected by the opening-angle option.

The script checks zero longitudinal focusing in the actual RF-off damped map, then executes both
opening-angle branches of emit_6d. It checks the longitudinal emittance markers and runs an RF-on
control with positive emittances.
"""

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import RADIATION, execute, radiation_ring, require

rows = execute(RADIATION, {"lat.bmad": radiation_ring()})
require(rows["damped65"], [0], atol=0, rtol=0)
with_angle = rows["native1"]
without_angle = rows["native0"]
require([with_angle[0], without_angle[0]], [-1, -1], atol=0, rtol=0)
if min(without_angle[[1, 2, 3, 6, 7]]) <= 0 or not np.isfinite(without_angle).all():
    raise RuntimeError("RF-off regularization failed to produce finite positive covariance")
# emit_6d deliberately marks the artificial longitudinal emittance -1; both branches need that marker.
defect = compare(
    "RF off, emittance_no_vert (m), same invalid marker as emittance",
    without_angle[1],
    -1.0,
    atol=0,
)
print(
    f"RF off, artificial sig_z={without_angle[2]:.17g} m sigE_E={without_angle[3]:.17g}; covariance55={without_angle[6]:.17g} covariance66={without_angle[7]:.17g}"
)
# An RF-on ring is a control for valid longitudinal output in both branches.
control = execute(RADIATION, {"lat.bmad": radiation_ring(rf=True)})
if np.any(control["native0"][[1, 2, 3]] <= 0) or np.any(control["native1"][[0, 2, 3]] <= 0):
    raise RuntimeError("RF-on longitudinal equilibrium control failed")
print("RF-on control: longitudinal emittances and spreads positive in both opening-angle branches")
verdict(defect)
