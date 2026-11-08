"""PAR-04: MAD-X applies both arrays of the same multipoles

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script loads the sequences of the main reproducer into MAD-X, tracks one proton at x = 1 mm and prints the kicks. MAD-X gives Δpx = −1.000×10⁻⁴ and Δpy = 5.000×10⁻⁵ in either order of the arrays.
"""

import os
import tempfile

from cpymad.madx import Madx

with tempfile.TemporaryDirectory() as d:
    for arrays in ("knl={0, 0.1}, ksl={0, 0.05}", "ksl={0, 0.05}, knl={0, 0.1}", "knl={0, 0.1}"):
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(f"mp: multipole, {arrays};\nring: sequence, l=10;\nmp, at=2;\nendsequence;\n")
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
        m.call(path)
        m.input("use, sequence=ring;")
        m.input("track, onepass=true, onetable=true; start, x=1e-3; run, turns=1; endtrack;")
        t = m.table.trackone
        print(f"MAD-X [{arrays}] px {float(t.px[-1]):.6e}, py {float(t.py[-1]):.6e}")
        m.quit()
