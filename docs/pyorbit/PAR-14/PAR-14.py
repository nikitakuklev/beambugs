"""PAR-14: `readMADX` imports a thin MULTIPOLE's dipole strength KNL[0] as an orbit kick on a straight reference

In MAD-X a MULTIPOLE's dipole strength K0L = KNL[0] bends the reference trajectory like a thin bend of angle K0L: the design orbit is unchanged, and a particle with relative momentum deviation δ gets the dispersive kick K0L·δ. `readMADX` builds a TEAPOT multipole whose dipole term kicks every particle by Δx′ = −K0L on a straight reference, deflecting the design particle and omitting the dispersive term. Behind a 1 mrad thin multipole and 1 m of drift, a proton on the design orbit is at x = −1.000 mm, where MAD-X gives 0. In a thin-lens ring whose bends are such multipoles, as MAD-X MAKETHIN writes them, the design particle is 14.73 m off axis after one turn.

The script reads MAD-X sequences with `readMADX` and tracks protons through them: a 1 m drift, a thin multipole with KNL = {0.001} and another 1 m drift, at the design momentum and with dE = 1 MeV, and an 80 m ring of eight thin-lens FODO cells of 60° phase advance whose sixteen dipoles are multipoles with KNL[0] = 2π/16. The expected values are those of a thin reference bend, derived beside the code. Controls are a KICKER, which both codes treat as an orbit kick, and a thin quadrupole term, which PyORBIT imports correctly.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.core.bunch import Bunch
from orbit.teapot import TEAPOT_Lattice

K0L = 1e-3
EK, MASS = 1.0, 0.93827208816  # kinetic energy and proton mass, GeV
E = EK + MASS
P0 = math.sqrt(E * E - MASS * MASS)
DE = 1e-3  # energy offset of the dispersive case, GeV
DELTA = math.sqrt((E + DE) ** 2 - MASS * MASS) / P0 - 1.0  # exact relative momentum deviation
LINE = "d: drift, l=1.0;\nseq: sequence, l=2.0, refer=entry;\nel, at=1.0;\nendsequence;\n"


def read(text, seq):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "deck.madx")
        with open(path, "w") as f:
            f.write(text)
        lat = TEAPOT_Lattice()
        lat.readMADX(path, seq)
    return lat


def track(lat, x0, dE):
    b = Bunch()
    b.mass(MASS)
    b.charge(1.0)
    b.getSyncParticle().kinEnergy(EK)
    b.addParticle(x0, 0.0, 0.0, 0.0, 0.0, dE)
    lat.trackBunch(b)
    return b.x(0)


bad, controls_ok = True, True
# A thin reference bend of angle K0L leaves the design particle on the (bent) reference: x = 0.
# A particle with deviation delta is bent by K0L/(1 + delta), K0L*delta/(1 + delta) less than the
# reference, i.e. px = K0L*delta in momentum units of p0; after 1 m, x = px/(1 + delta) * 1 m.
lat = read(f"el: multipole, knl={{{K0L}}};\n" + LINE, "seq")
for label, dE, expected in (
    ("multipole knl={0.001}, design momentum", 0.0, 0.0),
    ("multipole knl={0.001}, dE = 1 MeV", DE, K0L * DELTA / (1.0 + DELTA)),
):
    x = track(lat, 0.0, dE)
    print(
        f"[{label}] x at 2 m observed {x:+.6e} m, expected {expected:+.6e} m (thin reference bend, delta = {DELTA if dE else 0.0:.6e})"
    )
    bad &= abs(x - expected) > 1e-6

# Thin-lens ring, 8 cells of 10 m: QF at 0, dipole at 2.5, QD at 5, dipole at 7.5 (refer=entry);
# thin quadrupoles of focal length 5 m give 60 degrees of phase advance per cell.
# The sixteen dipoles bend the reference by 2*pi in total; by definition of the reference orbit
# the design particle stays at x = px = 0 all round the ring.
th = 2.0 * math.pi / 16
places = [
    f"{n}, at={10.0 * c + at};"
    for c in range(8)
    for n, at in (("qf", 0.0), ("bd", 2.5), ("qd", 5.0), ("bd", 7.5))
]
ring = (
    f"qf: multipole, knl={{0, 0.2}};\nqd: multipole, knl={{0, -0.2}};\nbd: multipole, knl={{{th!r}}};\n"
    "ring: sequence, l=80.0, refer=entry;\n" + "\n".join(places) + "\nendsequence;\n"
)
x = track(read(ring, "ring"), 0.0, 0.0)
print(
    f"[thin-lens ring, 16 multipoles with knl[0] = 2*pi/16, design particle] x after one turn observed {x:+.6e} m, expected 0 (design orbit)"
)
bad &= abs(x) > 1e-6

# Control: a KICKER is an orbit kick in both codes; hkick = -K0L moves the particle to -K0L * 1 m.
x = track(read(f"el: kicker, hkick={-K0L};\n" + LINE, "seq"), 0.0, 0.0)
print(
    f"[control, kicker hkick=-0.001] x at 2 m observed {x:+.6e} m, expected {-K0L:+.6e} m (orbit kick, 1 m lever arm)"
)
controls_ok &= abs(x + K0L) < 1e-15
# Control: a thin quadrupole term gives dpx = -knl1*x0, so x = x0 - knl1*x0 * 1 m = 0.8 mm.
x = track(read("el: multipole, knl={0, 0.2};\n" + LINE, "seq"), 1e-3, 0.0)
print(
    f"[control, multipole knl={{0, 0.2}}, x0 = 1 mm] x at 2 m observed {x:+.6e} m, expected {8e-4:+.6e} m (x0 - knl1*x0*1 m)"
)
controls_ok &= abs(x - 8e-4) < 1e-15
print("DEFECT REPRODUCED" if bad and controls_ok else "NOT REPRODUCED")
