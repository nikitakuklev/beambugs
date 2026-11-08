"""TRK-06: A zero-strength TEAPOT quadrupole applies its kinetic correction on top of a full drift

For kq = 0 or a neutral bunch, the TEAPOT quadrupole's linear step `quad1` falls back to the full nonlinear drift, but `QuadTEAPOT`, and the linac `Quad` with its default TEAPOT tracker, still apply the kinetic step `quad2`, whose terms complete the linear transport and so duplicate part of the drift. A switched-off quadrupole of length L therefore differs from a drift: at dE = 0 its path-length term doubles, Δz = −L(px² + py²) instead of −L(px² + py²)/2, and with η = dE/(p₀β₀) it adds Δx = −Lη·px/(1 + η). A 1 m quadrupole with kq = 0 gives z = −5.000×10⁻⁶ m for px = 10⁻³ and py = 2×10⁻³, where a drift, and a quadrupole with kq = 10⁻¹² m⁻², give −2.500×10⁻⁶ m.

The script tracks one proton at 1 GeV with pₓ = 10⁻³ and p_y = 2·10⁻³ through a 1 m `QuadTEAPOT` with kq = 0, the same
quadrupole with a neutral bunch, a 1 m MAD-X quadrupole with K1 = 0 read by `readMADX`, and a 1 m linac `Quad` in a
`LinacAccLattice` at dB/dr = 0 and, with a neutral bunch, at dB/dr = 1 T/m, at dE = 0 and, for the zero-strength nodes,
dE = 10⁻³ GeV. It compares x and z with the TEAPOT drift map written out analytically. The controls are a 1 m
`DriftTEAPOT`, quadrupoles with kq = 10⁻¹² m⁻² and dB/dr = 10⁻⁹ T/m, which reproduce the drift map, and the linac `Quad`
at dB/dr = 0 with the linac tracker, which reproduces the linac `Drift`.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import tempfile

from orbit.core.bunch import Bunch
from orbit.py_linac.lattice import Drift, LinacAccLattice, Quad, Sequence
from orbit.teapot import DriftTEAPOT, QuadTEAPOT, TEAPOT_Lattice

M, K = 0.938272, 1.0  # GeV
PX, PY = 1e-3, 2e-3
E = K + M
P0 = math.sqrt(E * E - M * M)
BETA0, GAMMA = P0 / E, E / M


def drift_map(de, length):
    """TEAPOT drift in its own variables: eta = dE/(p0*beta0), a = 1/(1 + eta),
    x += L*a*px, z -= L*a*(a*(px^2 + py^2 + eta^2/gamma^2)/2 - eta/gamma^2)."""
    eta = de / (P0 * BETA0)
    a = 1.0 / (1.0 + eta)
    x = length * a * PX
    z = -length * a * (a * (PX**2 + PY**2 + eta**2 / GAMMA**2) / 2 - eta / GAMMA**2)
    return x, z


def track(lat, de, charge=1):
    b = Bunch()
    b.mass(M)
    b.charge(charge)
    b.getSyncParticle().kinEnergy(K)
    b.addParticle(0.0, PX, 0.0, PY, 0.0, de)
    lat.trackBunch(b)
    return b.x(0), b.z(0)


def one_node(node):
    lat = TEAPOT_Lattice("line")
    lat.addNode(node)
    lat.initialize()
    return lat


def linac_node(node, length=1.0, linac_tracker=False):
    lat = LinacAccLattice("line")
    seq = Sequence("seq")
    seq.setLinacAccLattice(lat)
    node.setLength(length)
    seq.addNode(node)
    lat.addNode(node)
    lat.initialize()
    if linac_tracker:
        lat.setLinacTracker(True)
    return lat


def linac_quad(gradient, linac_tracker=False):
    q = Quad("q")  # dB/dr defaults to 0
    if gradient:
        q.setParam("dB/dr", gradient)
    return linac_node(q, linac_tracker=linac_tracker)


def madx_quad():
    # 1 m quadrupole with K1 = 0 centred in a 2 m sequence: quad plus 1 m of drift, expected a 2 m drift
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "line.madx")
        with open(path, "w") as f:
            f.write("q: quadrupole, l=1.0, k1=0.0;\nline: sequence, l=2;\nq, at=1;\nendsequence;\n")
        lat = TEAPOT_Lattice()
        lat.readMADX(path, "line")
    return lat


cases = [
    ("QuadTEAPOT kq=0", lambda: one_node(QuadTEAPOT("q", 1.0, 2, 0.0)), 1, 1.0, (0.0, 1e-3), False),
    (
        "QuadTEAPOT kq=1, neutral bunch",
        lambda: one_node(QuadTEAPOT("q", 1.0, 2, 1.0)),
        0,
        1.0,
        (0.0,),
        False,
    ),
    ("readMADX K1=0 + 1 m drift", madx_quad, 1, 2.0, (0.0,), False),
    ("linac Quad dB/dr=0", lambda: linac_quad(0.0), 1, 1.0, (0.0, 1e-3), False),
    ("linac Quad dB/dr=1, neutral bunch", lambda: linac_quad(1.0), 0, 1.0, (0.0,), False),
    ("DriftTEAPOT (control)", lambda: one_node(DriftTEAPOT("d", 1.0)), 1, 1.0, (0.0, 1e-3), True),
    (
        "QuadTEAPOT kq=1e-12 (control)",
        lambda: one_node(QuadTEAPOT("q", 1.0, 2, 1e-12)),
        1,
        1.0,
        (0.0, 1e-3),
        True,
    ),
]
cases.append(
    ("linac Quad dB/dr=1e-9 (control)", lambda: linac_quad(1e-9), 1, 1.0, (0.0, 1e-3), True)
)
reproduced = True
for name, make, charge, length, des, is_control in cases:
    for de in des:
        x, z = track(make(), de, charge)
        xe, ze = drift_map(de, length)
        print(
            f"{name:34s} dE={de:.0e} GeV: x observed {x:.9e} m, expected {xe:.9e}; "
            f"z observed {z:.9e} m, expected {ze:.9e} (drift map, L = {length:.0f} m)"
        )
        off = abs(x - xe) > 1e-12 or abs(z - ze) > 1e-12
        reproduced &= off != is_control
# Control: with the linac tracker, the zero-gradient linac Quad falls back to the linac drift and matches it.
for de in (0.0, 1e-3):
    xq, zq = track(linac_quad(0.0, linac_tracker=True), de)
    xd, zd = track(linac_node(Drift("d"), linac_tracker=True), de)
    print(
        f"{'linac Quad dB/dr=0, linac tracker':34s} dE={de:.0e} GeV: x observed {xq:.9e} m, expected {xd:.9e}; "
        f"z observed {zq:.9e} m, expected {zd:.9e} (linac Drift, L = 1 m; control)"
    )
    reproduced &= abs(xq - xd) < 1e-15 and abs(zq - zd) < 1e-15
print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
