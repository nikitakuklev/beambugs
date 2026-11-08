"""PAR-01: MAD-X reads the same files as one statement per ';'

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script writes the four files of the main reproducer, loads each into MAD-X with CALL, expands the sequence and prints its elements and length; for the multipole file it tracks one proton at x = 1 mm and prints px. MAD-X places each element once, gives px = −1.000×10⁻⁵ and reads the blank first line and the two-line definition.
"""

import os
import tempfile

from cpymad.madx import Madx

FILES = {
    "blank line after thin multipole": "mp: multipole, knl={0, 0.01};\nring: sequence, l=10;\nmp, at=2;\n\nendsequence;\n",
    "indented comment after quadrupole": "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=10;\nqa, at=2;\n  ! comment\nm1, at=6;\nendsequence;\n",
    "blank first line": "\nqa: quadrupole, l=1, k1=0.1;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n",
    "two-line definition": "m0: marker;\nqa: quadrupole, l=1,\n    k1=0.1;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n",
}
ok = True
with tempfile.TemporaryDirectory() as d:
    for tag, text in FILES.items():
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(text)
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
        m.call(path)
        m.input("use, sequence=ring;")
        names = [
            e.name
            for e in m.sequence.ring.expanded_elements
            if "$" not in e.name and not e.name.startswith("drift")
        ]
        line = f"[{tag}] elements {names}, sequence length {m.sequence.ring.length:.6f} m"
        ok = ok and len(names) == len(set(names))
        if tag.startswith("blank line after"):
            m.input("track, onepass=true, onetable=true; start, x=1e-3; run, turns=1; endtrack;")
            px = float(m.table.trackone.px[-1])
            line += f"; px after tracking {px:.6e} (-knl1*x = -1.000000e-05)"
            ok = ok and abs(px + 1e-5) < 1e-15
        print(line)
        m.quit()
print("MAD-X places every element once" if ok else "MAD-X result differs from the expectation")
