"""PAR-09: MAD-X reads a CALL statement with an unquoted file name

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script writes the two MAD-X files of the main reproducer's unquoted CALL case, loads the main file into MAD-X and prints the expanded sequence. MAD-X loads the quadrupole from the called file and places it at 2 m.
"""

import os
import tempfile

from cpymad.madx import Madx

with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "sub.madx"), "w") as f:
        f.write("qa: quadrupole, l=1, k1=0.1;\n")
    with open(os.path.join(d, "main.madx"), "w") as f:
        f.write("call, file=sub.madx;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n")
    m = Madx(stdout=False, cwd=d)
    m.input("beam, particle=proton, energy=2;")
    m.call("main.madx")
    m.input("use, sequence=ring;")
    print("MAD-X sequence:", [(e.name, e.at) for e in m.sequence.ring.expanded_elements])
    m.quit()
