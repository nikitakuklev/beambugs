"""COL-03: wakeconvolution.w_t_rf has the wrong functional form.

Bane, SLAC-PUB-9663 (2003): W_perp(s) = 4 Z0 c s0 / (pi a^4) * [1 - (1 + u) exp(-u)], u = sqrt(s/s0),
s0 = 0.169 a^1.79 g^0.38 / L^1.17. Its short-range limit is 2 Z0 c s / (pi a^4) (linear in s).
"""
import numpy as np
from impactx import wakeconvolution

Z0 = 377.0  # value used by ImpactX
c = 299792458.0
a, g, L = 4.92e-3, 6.89e-3, 8.75e-3  # NLC-like cell, inside the fit's range
s0 = 0.169 * a**1.79 * g**0.38 / L**1.17
bad = False
for x in (0.01, 1.0):
    s = x * s0
    u = np.sqrt(x)
    expected = 4 * Z0 * c * s0 / (np.pi * a**4) * (1 - (1 + u) * np.exp(-u))
    observed = wakeconvolution.w_t_rf(s, a, g, L)
    print(f"s/s0 = {x:g}: observed {observed:.4g}, expected {expected:.4g}, ratio {observed / expected:.4g}")
    bad |= abs(observed / expected - 1) > 1e-6
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
