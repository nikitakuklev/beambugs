"""OPT-14: Periodic vertical and b-mode dispersion slopes have a denominator typo

Calling the periodic Twiss initializer with vertical reference-orbit momentum gives incorrect
vertical and b-mode dispersion slopes. The documented paraxial derivative is η′/q − pᵧ/q²,
where q = 1 + δ, δ is fractional momentum offset, pᵧ is vertical momentum divided by reference
momentum, and η′ = ∂pᵧ/∂δ. At δ = 0.2, pᵧ = −0.02 and η′ = −0.05, both slopes are −0.008333
instead of −0.02778; horizontal and a-mode slopes agree with the derivative.

The script constructs a stable periodic map with a prescribed fixed-point derivative and calls the
compiled Twiss initializer. It checks the ordinary dispersion and horizontal slopes before comparing
the vertical and b-mode slopes with the paraxial derivative.
"""

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import periodic, require

values, orbit, eta, _ = periodic(False)
require(values[:4], eta)
# Differentiate the paraxial slope p_y/(1+delta) with respect to delta.
expected = eta[[1, 3]] / (1 + orbit[5]) - orbit[[1, 3]] / (1 + orbit[5]) ** 2
require(values[[4, 6]], np.repeat(expected[0], 2))
verdict(
    compare(
        "vertical and b-mode deta/ds, from d[p_y/(1+delta)]/d(delta)",
        values[[5, 7]],
        np.repeat(expected[1], 2),
        atol=1e-15,
        rtol=3e-14,
    )
)
