"""PAR-06: MAD-X drifts the beam through the same instrument

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script loads the instrument sequence of the main reproducer and a drift control into MAD-X, tracks one proton with px = 1×10⁻³ and prints x at the end. MAD-X gives 1.000001×10⁻² m for both, its exact drift L·px/√(1 − px²); the 5×10⁻⁷ relative difference from L·px is the drift's square-root term.
"""

import os
import tempfile

from cpymad.madx import Madx

with tempfile.TemporaryDirectory() as d:
    for typ in ("instrument", "drift"):
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(f"el: {typ}, l=2.0;\nring: sequence, l=10;\nel, at=5;\nendsequence;\n")
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
        m.call(path)
        m.input("use, sequence=ring;")
        m.input("track, onepass=true, onetable=true; start, px=1e-3; run, turns=1; endtrack;")
        print(f"MAD-X [{typ}] x at the end {float(m.table.trackone.x[-1]):.6e} m")
        m.quit()
