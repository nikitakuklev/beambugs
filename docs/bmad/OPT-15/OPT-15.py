"""OPT-15: Periodic dispersion mixes normalized and raw matrix coordinates

Enabling normalize_twiss in the periodic Twiss initializer at nonzero fractional momentum
offset δ changes the physical dispersion by mixing normalized and ordinary coordinates. For
the transverse one-turn matrix M₄ and momentum-source column c, the closed-orbit derivative
D must solve (I − M₄)D = c in one coordinate system. A map at δ = 0.2 with D = (0.8, 0.13,
0.4, −0.05) returns (0.8951, 0.1170, 0.4223, −0.04347) when normalization is enabled, while
disabling it recovers D.

The script builds a stable periodic map whose fixed-point derivative is prescribed
analytically. It executes the compiled initializer with normalization off and on, validates the
first as a control, and compares the second with a raw-coordinate linear solve. BMAD_DIST
selects the build.
"""

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import periodic, require

baseline, _, eta, _ = periodic(False)
require(baseline[:4], eta)
values, _, _, mat = periodic(True)
# The dispersion is the derivative of the closed orbit fixed point.
expected = np.linalg.solve(np.eye(4) - mat[:4, :4], mat[:4, 5])
require(expected, eta)
verdict(
    compare(
        "normalized periodic (eta_x, etap_x, eta_y, etap_y), fixed-point derivative",
        values[:4],
        expected,
        atol=1e-15,
        rtol=3e-14,
    )
)
