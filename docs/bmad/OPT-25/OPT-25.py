"""OPT-25: srdt counts skew and rotated multipoles as normal ones

When srdt evaluates skew or rotated multipoles, it discards the multipole rotation and uses the returned strength as a normal component. Normal resonance-driving terms require the real part of the rotated multipole coefficient. In a FODO ring, a pure skew sextupole produces the normal sextupole driving coefficient h₂₁₀₀₀ = −1.865 instead of zero, and rotating a normal thin multipole by π/3 leaves −1.865 instead of reversing it to +1.865. A mixed sextupole with integrated normal and skew strengths 0.5 and 0.2 m⁻² is overcounted by 7.703%.

The script puts a thin element at zero betatron phase in a 16-bend FODO ring at 1 GeV total electron energy. It compares normal, skew, tilted, pole-rotated, reversed and mixed sextupoles with an independent expansion of their kick potential.
"""

import numpy as np
from _bmad import compare, verdict
from _srdt_wiggler_maps import srdt_case

cases = [
    ("normal", "multipole, k2l=1", 0.5),
    ("skew", "multipole, k2sl=1", 0),
    ("tilted", "multipole, k2l=1, tilt=pi/6", 0),
    ("pole rotation", "multipole, k2l=1, t2=-pi/6", 0),
    ("reversed", "multipole, k2l=1, tilt=pi/3", -0.5),
    ("mixed", "multipole, k2l=.5, k2sl=.2", 0.25),
]
_, base = srdt_case("multipole, k2l=0")
defect = False
for name, spec, b2 in cases:
    twiss, actual = srdt_case(spec)
    ba, bb, eta, pa, pb = twiss
    np.testing.assert_allclose([pa, pb], 0, atol=1e-14)
    # The normal sextupole potential is b2*(x^3/3-x*y^2); a skew potential
    # contains different monomials. Expanding x=sqrt(beta_x)*(h+ + h-)/2
    # gives these first-order coefficients at zero betatron phase.
    expected = np.column_stack(
        (
            [-b2 * ba**1.5 / 8, b2 * ba**0.5 * bb / 4, -b2 * eta * ba / 2, b2 * eta * bb / 2],
            np.zeros(4),
        )
    )
    actual[2:] -= base[2:]
    bad = compare(
        f"{name}: h21000,h10110,delta_h11001,delta_h00111 (real,imag)", actual, expected, atol=2e-12
    )
    if name == "normal" and bad:
        raise AssertionError("Normal sextupole control failed")
    defect |= bad
verdict(defect)
