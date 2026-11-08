"""PAR-07: MAD-X FROM= is measured from the reference point of the named element, not its centre

In a MAD-X sequence, `AT=a, FROM=name` places an element at the distance a from the centre of the element `name`. `readMADX` adds a to the stored placement position of `name`, which is its entry or its exit when the sequence uses REFER=ENTRY or REFER=EXIT, so an element placed with FROM lands half the named element's length too early or too late. A marker at `AT=3, FROM=qa` after a 2 m quadrupole lands at 4 m instead of 5 m with REFER=ENTRY and at 6 m with REFER=EXIT. With the default REFER=CENTRE the position is right. For a name placed several times the reader measures from the latest earlier placement, where MAD-X uses the first.

The script reads short MAD-X sequences with `readMADX` and prints the marker's position, the sum of the node lengths before it. In the REFER cases the quadrupole's centre is at 2 m, so the expected marker position is 2 + 3 = 5 m; the REFER=CENTRE sequence is the control. The last case places a 1 m quadrupole at 2 m and 12 m and a marker at `AT=11, FROM=qa`, which MAD-X puts at 2 + 11 = 13 m.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.teapot import TEAPOT_Lattice

# A 2 m quadrupole whose centre is at 2 m: AT=1 is its entry, AT=2 its centre, AT=3 its exit.
CASES = (
    ("refer=entry", "ring: sequence, l=10, refer=entry;\nqa, at=1;\n", 5.0, "clean"),
    ("refer=exit", "ring: sequence, l=10, refer=exit;\nqa, at=3;\n", 5.0, "clean"),
    ("control, refer=centre", "ring: sequence, l=10;\nqa, at=2;\n", 5.0, "control"),
)
DEF = "qa: quadrupole, l=2, k1=0.1;\nm1: marker;\n"
FROM = "m1, at=3, from=qa;\nendsequence;\n"
REPEATED = "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=30;\nqa, at=2;\nqa, at=12;\nm1, at=11, from=qa;\nendsequence;\n"


def marker_position(text):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(text)
        lat = TEAPOT_Lattice()
        lat.readMADX(path, "ring")
    s = 0.0
    for node in lat.getNodes():
        if node.getName() == "m1":
            return s
        s += node.getLength()
    raise RuntimeError("marker m1 not in the lattice")


bad = False
for tag, seq, expected, kind in CASES:
    s = marker_position(DEF + seq + FROM)
    print(
        f"[{tag}] marker position observed {s:.6f} m, expected {expected:.6f} m (centre of qa at 2 m + AT 3 m)"
    )
    if kind == "clean":
        bad = bad or abs(s - expected) > 1e-12
s = marker_position(REPEATED)
print(
    f"[repeated name] marker position observed {s:.6f} m, MAD-X 13.000000 m (first qa centred at 2 m + AT 11 m)"
)
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
