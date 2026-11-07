"""OPT-26: srdt returns NaN for zero-length non-multipole elements with multipoles

Calling srdt on a zero-length ab_multipole or sad_mult with a nonzero sextupole component returns NaN driving terms. Only the multipole element class takes the thin-element path; the other classes divide integrated strength by zero length and then multiply by a zero slice length. Equivalent thin elements should return the same finite terms. In a FODO ring, both affected representations return NaN where the equivalent multipole gives h₂₁₀₀₀ = −1.865 and h₁₀₁₁₀ = 0.9849.

The script places equivalent multipole, ab_multipole and sad_mult sextupoles at zero phase in the same FODO ring. It validates the finite multipole control against the analytic normal-sextupole coefficients before checking the two other representations for NaNs.
"""

import numpy as np
from _bmad import compare, verdict
from _srdt_wiggler_maps import srdt_case

twiss, control = srdt_case("multipole, k2l=1")
ba, bb = twiss[:2]
expected = np.array([[-(ba**1.5) / 16, 0], [ba**0.5 * bb / 8, 0]])
if compare("thin normal control: h21000,h10110", control[:2], expected, atol=2e-12):
    raise AssertionError("Thin multipole control failed")
defect = False
for spec in ("ab_multipole, b2=.5", "sad_mult, b2=.5"):
    _, actual = srdt_case(spec)
    if np.isnan(actual).any():
        print(
            f"{spec}: observed={actual[:2].tolist()} expected={expected.tolist()} from the identical thin normal potential"
        )
        defect = True
    else:
        defect |= compare(spec, actual[:2], expected, atol=2e-12)
verdict(defect)
