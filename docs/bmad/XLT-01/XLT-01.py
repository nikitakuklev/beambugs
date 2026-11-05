"""XLT-01: Converter angular cuts increase yield through an inverted acceptance factor

Applying an angular cut to a converter increases its outgoing physical yield instead of reducing it. For a normalized synthetic distribution and angle_out_max = 0.15, the accepted fraction is 0.06149, but the returned weight is 16.26. The weight is the reciprocal of the required fraction, so even a restrictive cut produces more weight than the unrestricted converter. Integrating the supplied angular density analytically establishes the required normalization.

The script tracks a fixed electron through supplied converter tables with and without the angular cut. It compares the weight with an analytic accepted fraction and checks that the unrestricted weight is one.
"""

import math

from _bmad import compare, verdict
from _s3_24_converter import samples

base, _, _ = samples()
if compare("unrestricted weight, unit-integral table", base[0, 1], 1, atol=2e-14):
    raise RuntimeError("Unrestricted normalization control failed")
cut, _, _ = samples("cv[angle_out_max]=0.15\n")
# The manual and implementation use h=atan(angle_out_max). The x density is
# constant; the y integral of 1/(1+4*y*y) is atan(2*y)/2.
h = math.atan(0.15)
fraction = 2 * h * math.atan(2 * h) / (1.6 * math.atan(1.2))
verdict(
    compare("angular-cut weight, integral of accepted density", cut[0, 1], fraction, atol=2e-13)
)
