"""PAR-05: MAD-X evaluates the same expressions

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script defines the quadrupoles of the main reproducer in MAD-X and prints the k1 MAD-X evaluates for each expression beside Python's `math` value, or for a D-exponent literal the same number written with `e`. They agree for every expression.
"""

import math

from cpymad.madx import Madx

CASES = (
    ("", "asin(0.5)", math.asin(0.5)),
    ("", "acos(0.5)", math.acos(0.5)),
    ("", "atan(0.5)", math.atan(0.5)),
    ("kpip = 0.2;", "kpip", 0.2),
    ("", "1.5d-3", 1.5e-3),
    ("", "1d-3", 1e-3),
    ("", "sin(0.5)", math.sin(0.5)),
    ("", "2*pi/10", 2 * math.pi / 10),
)
for defs, expr, expected in CASES:
    m = Madx(stdout=False)
    m.input(defs + f"qa: quadrupole, l=1, k1={expr};")
    print(f"MAD-X k1={expr}: {m.elements['qa'].k1:.9f}, math {expected:.9f}")
    m.quit()
