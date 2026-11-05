"""COL-03: `wakeconvolution.w_t_rf` has the wrong functional form

The Python helper `impactx.wakeconvolution.w_t_rf`, the short-range dipole wake W⊥(s) of a periodic
accelerating structure at distance s behind the source, uses the bracket u·e⁻ᵘ with u = √(s/s₀) (s₀
a fitted length), where the cited Bane model (SLAC-PUB-9663, 2003, Eqs. 15–16) has 1 − (1 + u)·e⁻ᵘ.
Bane's form rises linearly in s at short range; the implemented one starts as √s. For a cell
resembling the Next Linear Collider design (iris radius 4.92 mm, gap 6.89 mm, period 8.75 mm) the
wake is 19.34 times too large at s/s₀ = 0.01 and 1.392 times at s/s₀ = 1. Its documentation also
calls it a resistive-wall wake and states units of V/m/pC/mm although it returns SI. Built-in
tracking does not call it.

The script evaluates `w_t_rf` for that cell at s/s₀ = 0.01 and 1, with s₀ from the same fit the code
uses. It compares the result with Bane's formula evaluated with the free-space impedance Z₀ = 377 Ω
that ImpactX uses.
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
    # Bane, SLAC-PUB-9663 (2003), Eqs. 15-16; short-range limit 2 Z0 c s / (pi a^4), linear in s.
    expected = 4 * Z0 * c * s0 / (np.pi * a**4) * (1 - (1 + u) * np.exp(-u))
    observed = wakeconvolution.w_t_rf(s, a, g, L)
    print(f"s/s0 = {x:g}: observed {observed:.4g}, expected {expected:.4g}, ratio {observed / expected:.4g}")
    bad |= abs(observed / expected - 1) > 1e-6
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
