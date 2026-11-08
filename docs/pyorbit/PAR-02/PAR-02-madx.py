"""PAR-02: MAD-X tracks the same cavity with an energy gain of VOLT·sin(2π·LAG)

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script loads the sequence of the main reproducer into MAD-X, tracks one on-axis proton at the reference time through one pass and prints its energy change pt·p₀c. MAD-X gives 1.000×10⁻⁵ GeV, VOLT·sin(2π·LAG) with VOLT = 0.01 MV.
"""

import os
import tempfile

from cpymad.madx import Madx

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "seq.madx")
    with open(path, "w") as f:
        f.write(
            "rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1;\nring: sequence, l=10;\nrf, at=2;\nendsequence;\n"
        )
    m = Madx(stdout=False)
    m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
    m.call(path)
    m.input("use, sequence=ring;")
    m.input("track, onepass=true, onetable=true; start; run, turns=1; endtrack;")
    pc = m.sequence.ring.beam.pc
    # MAD-X pt = dE/(p0 c), so the energy change is pt * p0c in GeV.
    dE = float(m.table.trackone.pt[-1]) * pc
    print(f"MAD-X energy change {dE:.6e} GeV, VOLT*sin(2*pi*LAG) = 1.000000e-05 GeV")
    m.quit()
