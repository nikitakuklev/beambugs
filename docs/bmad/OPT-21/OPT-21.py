"""OPT-21: Integral bunch length uses summed cavity M65 instead of one-turn M65

The radiation-integral bunch-length calculation sums individual RF-cavity M₆₅ values, while
the manual defines its denominator as the composed one-turn M₆₅. These coincide to first
order in RF strength and differ at second order, within the documented weak-synchrotron
approximation. In a ring with two 10 kV cavities, the implemented denominator is 1.708 ×
10⁻⁴ m⁻¹ and the one-turn value is 1.707 × 10⁻⁴ m⁻¹, giving 0.03239 m versus 0.03240 m in
the stated formula. The denominator description should identify the cavity sum used by the
approximation.

The script executes radiation_integrals on a ring with two separated RF cavities. It validates the
returned bunch length against the cavity sum, compares it with the composed one-turn formula, and
reads the manual’s denominator definition through BMAD_SOURCE.
"""

import numpy as np
from _bmad import compare, source, verdict
from _dispersion_and_moments import RADIATION, execute, radiation_ring, require

rows = execute(RADIATION, {"lat.bmad": radiation_ring(rf=True)})
cavity_sum, one_turn = rows["m65"]
i1 = rows["integrals"][1]
sig_z, sig_delta, _ = rows["spread"]
if min(i1 / cavity_sum, i1 / one_turn) <= 0:
    raise RuntimeError("Expected stable longitudinal focusing")
# The implemented weak-synchrotron approximation uses the sum of the two cavity kicks.
require(sig_z, np.sqrt(i1 / cavity_sum) * sig_delta, atol=0, rtol=3e-14)
manual = (source() / "bmad/doc/synchrotron-radiation.tex").read_text()
if "element for the 1--turn transfer matrix" not in manual:
    raise RuntimeError("The documented denominator changed; inspect its definition")
defect = compare(
    "bunch-length denominator M65 (1/m), documented one-turn value",
    cavity_sum,
    one_turn,
    atol=1e-15,
    rtol=3e-14,
)
defect |= compare(
    "sig_z (m), using the documented one-turn denominator",
    sig_z,
    np.sqrt(i1 / one_turn) * sig_delta,
    atol=0,
    rtol=3e-14,
)
verdict(defect)
