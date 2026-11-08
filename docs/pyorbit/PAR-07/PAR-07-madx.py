"""PAR-07: MAD-X positions of the same markers, and its verdict on the upstream repeated-name test file

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script expands the sequences of the main reproducer in MAD-X and prints the marker position; MAD-X puts it at 5 m for every REFER value and at 13 m in the repeated-name case. It then loads the upstream test input `repeated_element.lat` (a BPM at 1 m and 2 m, a probe at `AT=0.5, FROM=bpm`, which the upstream test expects at 2.5 m), which MAD-X refuses with a negative drift because it measures from the first BPM.
"""

import os
import tempfile

from cpymad.madx import Madx

DEF = "qa: quadrupole, l=2, k1=0.1;\nm1: marker;\n"
FROM = "m1, at=3, from=qa;\nendsequence;\n"
FILES = {
    "refer=entry": DEF + "ring: sequence, l=10, refer=entry;\nqa, at=1;\n" + FROM,
    "refer=exit": DEF + "ring: sequence, l=10, refer=exit;\nqa, at=3;\n" + FROM,
    "refer=centre": DEF + "ring: sequence, l=10;\nqa, at=2;\n" + FROM,
    "repeated name": "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=30;\nqa, at=2;\nqa, at=12;\nm1, at=11, from=qa;\nendsequence;\n",
}
UPSTREAM_TEST = "bpm: monitor;\nprobe: marker;\nring: sequence, l=3.0;\nbpm, at=1.0;\nbpm, at=2.0;\nprobe, at=0.5, from=bpm;\nendsequence;\n"

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "seq.madx")
    for tag, text in FILES.items():
        with open(path, "w") as f:
            f.write(text)
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=2;")
        m.call(path)
        m.input("use, sequence=ring;")
        pos = [e.position for e in m.sequence.ring.expanded_elements if e.name.startswith("m1")]
        print(f"MAD-X [{tag}] marker position {pos[0]:.6f} m")
        m.quit()
    with open(path, "w") as f:
        f.write(UPSTREAM_TEST)
    log = os.path.join(d, "madx.log")
    with open(log, "w") as out:
        m = Madx(stdout=out, stderr=out)
        m.input("beam, particle=proton, energy=2;")
        m.call(path)
        try:
            m.input("use, sequence=ring;")
            print(
                "MAD-X [repeated_element.lat]",
                [(e.name, e.position) for e in m.sequence.ring.expanded_elements],
            )
        except RuntimeError:
            pass
        m.quit()
    with open(log) as f:
        print(
            "MAD-X [repeated_element.lat]", [line.strip() for line in f if "negative drift" in line]
        )
