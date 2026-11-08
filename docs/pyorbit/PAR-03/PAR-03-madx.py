"""PAR-03: MAD-X applies KICK on an HKICKER and a VKICKER and saves correctors with KICK

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script loads the MAD-X sequence of the main reproducer, tracks one on-axis proton and prints px and py, then saves the sequence and prints the corrector definitions MAD-X writes. MAD-X gives px = 1.000×10⁻³ and py = 2.000×10⁻³ and writes the correctors with `kick`.
"""

import os
import tempfile

from cpymad.madx import Madx

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "seq.madx")
    saved = os.path.join(d, "saved.seq")
    with open(path, "w") as f:
        f.write(
            "hk: hkicker, l=0, kick=0.001;\nvk: vkicker, l=0, kick=0.002;\nring: sequence, l=10;\nhk, at=4;\nvk, at=5;\nendsequence;\n"
        )
    m = Madx(stdout=False)
    m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
    m.call(path)
    m.input("use, sequence=ring;")
    m.input("track, onepass=true, onetable=true; start; run, turns=1; endtrack;")
    t = m.table.trackone
    print(f"MAD-X px {float(t.px[-1]):.6e} (KICK 1e-3), py {float(t.py[-1]):.6e} (KICK 2e-3)")
    m.input(f"save, sequence=ring, file='{saved}';")
    m.quit()
    with open(saved) as f:
        print("MAD-X save:", [line.strip() for line in f if "kicker" in line])
