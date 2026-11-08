"""PAR-02: `readMADX` turns every RFCAVITY into a drift

`TEAPOT_Lattice.readMADX` changes the type of every RFCAVITY to a drift before the TEAPOT element factory sees it, so the imported lattice has no RF node: VOLT, LAG and HARMON are dropped with no message, and the ring has no longitudinal focusing or acceleration. MAD-X gives a particle crossing an RFCAVITY at the reference time the energy kick VOLT·sin(2π·LAG), with VOLT in MV. For `rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1` an on-axis proton gains 0 instead of 1.000×10⁻⁵ GeV, and the cavity is a plain 1 m drift.

The script writes a 10 m MAD-X sequence holding the cavity to a temporary directory, reads it with `readMADX`, prints the cavity node's type and the number of `RingRFTEAPOT` nodes, and tracks an on-axis proton through one pass. The expected gain is the MAD-X RFCAVITY kick at the reference time. The control reads the same cavity from a MAD-8 file with `readMAD`, where the factory's RFCAVITY branch builds one `RingRFTEAPOT` node.
"""

import contextlib
import io
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.core.bunch import Bunch
from orbit.teapot import RingRFTEAPOT, TEAPOT_Lattice


def read(text, reader, name):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(text)
        lat = TEAPOT_Lattice()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            getattr(lat, reader)(path, "ring")
    return lat, out.getvalue()


lat, printed = read(
    "rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1;\nring: sequence, l=10;\nrf, at=2;\nendsequence;\n",
    "readMADX",
    "seq.madx",
)
rf_types = [n.getType() for n in lat.getNodes() if n.getName() == "rf"]
n_rf = sum(1 for n in lat.getNodes() if isinstance(n, RingRFTEAPOT))
print(
    f"[readMADX] cavity node types {rf_types}, RingRFTEAPOT nodes {n_rf}, expected >= 1; reader messages: {bool(printed.strip())}"
)

b = Bunch()
b.mass(0.938272)
b.charge(1)
b.getSyncParticle().kinEnergy(1.0)
b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
K0 = b.getSyncParticle().kinEnergy()
lat.trackBunch(b)
gain = (
    b.getSyncParticle().kinEnergy() - K0 + b.dE(0)
)  # reference energy change plus the particle's dE
# MAD-X RFCAVITY: dE = VOLT * sin(2*pi*(LAG - h*f0*t)); at t = 0, VOLT = 0.01 MV = 1e-5 GeV, LAG = 0.25 gives sin = 1.
expected = 0.01e-3 * math.sin(2 * math.pi * 0.25)
print(
    f"[readMADX] on-axis energy gain observed {gain:.6e} GeV, expected {expected:.6e} GeV (VOLT*sin(2*pi*LAG))"
)

lat8, _ = read(
    "rf: rfcavity, l=1.0, volt=0.01, lag=0.25, harmon=1\nd: drift, l=4.5\nring: line=(d, rf, d)\n",
    "readMAD",
    "line.mad",
)
n_rf8 = sum(1 for n in lat8.getNodes() if isinstance(n, RingRFTEAPOT))
print(f"[control, readMAD] RingRFTEAPOT nodes {n_rf8}, expected 1")
print("DEFECT REPRODUCED" if n_rf == 0 and abs(gain - expected) > 1e-9 else "NOT REPRODUCED")
