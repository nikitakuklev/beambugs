"""PAR-06: A MAD-X INSTRUMENT with length does not drift the beam

`readMADX` imports an INSTRUMENT as a marker, which the TEAPOT element factory builds as a node without a tracking map. An instrument of nonzero length keeps its length, so element positions and the lattice length are right, but particles cross it without moving. In MAD-X an INSTRUMENT acts on the beam as a drift of its length. In a 10 m sequence holding a 2 m instrument, a particle with px = 1×10⁻³ leaves at x = 8.000×10⁻³ m instead of 1.000×10⁻² m. Monitors and collimators with length are drifted correctly.

The script reads 10 m MAD-X sequences holding one 2 m element with `readMADX`, prints the element's node type and the lattice length, and tracks one proton with px = 1×10⁻³ through the sequence. The expected x is L·px for the full 10 m, the result of a straight sequence of drifts at zero energy offset. The controls replace the instrument with a drift, a monitor and a collimator of the same length.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.core.bunch import Bunch
from orbit.teapot import TEAPOT_Lattice

EXPECTED = 10.0 * 1e-3  # field-free 10 m: x = L * px at dE = 0
bad = False
for typ in ("instrument", "drift", "monitor", "collimator"):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(f"el: {typ}, l=2.0;\nring: sequence, l=10;\nel, at=5;\nendsequence;\n")
        lat = TEAPOT_Lattice()
        lat.readMADX(path, "ring")
    types = [n.getType() for n in lat.getNodes() if n.getName() == "el"]
    b = Bunch()
    b.mass(0.938272)
    b.charge(1)
    b.getSyncParticle().kinEnergy(1.0)
    b.addParticle(0.0, 1e-3, 0.0, 0.0, 0.0, 0.0)
    lat.trackBunch(b)
    label = typ if typ == "instrument" else f"control, {typ}"
    print(
        f"[{label}] node type {types}, lattice length {lat.getLength():.3f} m; "
        f"x observed {b.x(0):.6e} m, expected {EXPECTED:.6e} m (10 m drift with px = 1e-3)"
    )
    if typ == "instrument":
        bad = abs(b.x(0) - EXPECTED) > 1e-12
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
