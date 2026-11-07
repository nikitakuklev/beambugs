"""OPT-17: Projected dispersion slopes use a momentum ratio as momentum

Propagating an off-momentum lattice can return projected dispersion slopes that disagree with
the modal slopes even without transverse coupling. The projected conversion uses the
exit-to-entrance momentum ratio where it requires the actual exit relative momentum q = 1 + δ,
with δ the fractional momentum offset. For δ = 0.2, normalized transverse momenta (0.03,
−0.02), and zero momentum dispersion, the projected slopes are (−0.03, 0.02) instead of the
documented paraxial derivatives (−0.02083, 0.01389).

The script tracks an off-momentum drift and checks unchanged momenta and zero momentum dispersion.
It compares projected slopes with the paraxial quotient-rule derivative and validates modal slopes
as a control.
"""

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import OPEN_LINE, execute, lattice, require

initial = [0, 0.03, 0, -0.02, 0, 0.2]
values = execute(OPEN_LINE, {"lat.bmad": lattice("e: drift, l=1.7", start=initial)})
require(values["orbit"][[1, 3, 5]], np.array(initial)[[1, 3, 5]])
require(values["exit"][[1, 3]], [0, 0])
# The paraxial slope px/q has derivative etap/q - px/q^2, q=1+delta.
expected = -np.array([0.03, -0.02]) / 1.2**2
require(values["slopes"][2:], expected)
verdict(
    compare(
        "projected deta/ds, derivative of px/(1+delta) and py/(1+delta)",
        values["slopes"][:2],
        expected,
        atol=1e-15,
        rtol=3e-14,
    )
)
