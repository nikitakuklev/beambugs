"""PAR-08: MAD-X reads the zero-angle bends and a bare TILT

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script loads 2 m MAD-X sequences holding a 1 m sector bend with ANGLE = 0 and k1, a sector bend without ANGLE, a rectangular bend with ANGLE = 0 and a quadrupole with a bare TILT, and prints the one-pass R₁₁ that MAD-X computes and the TILT it stores. MAD-X reads all four; it stores a bare TILT as 0.
"""

import os
import tempfile

from cpymad.madx import Madx

ELEMENTS = (
    "q: sbend, l=1.0, angle=0.0, k1=0.3;",
    "q: sbend, l=1.0, k1=0.3;",
    "q: rbend, l=1.0, angle=0.0;",
    "q: quadrupole, l=1.0, k1=0.5, tilt;",
)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "seq.madx")
    for elem in ELEMENTS:
        with open(path, "w") as f:
            f.write(elem + "\nring: sequence, l=2;\nq, at=1.5;\nendsequence;\n")
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
        m.call(path)
        m.input("use, sequence=ring;")
        m.input("twiss, betx=1, bety=1, rmatrix;")
        print(
            f"MAD-X [{elem}] read ok, R11 {m.table.twiss.re11[-1]:.6f}, tilt {m.elements['q'].tilt}"
        )
        m.quit()
