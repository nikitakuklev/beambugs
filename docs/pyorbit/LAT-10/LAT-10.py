"""LAT-10: `reverseOrder()` does not invert tilts, alignment errors, solenoid fields and hard-edge fringes

`reverseOrder()`, the lattice reversal used for bunch backtracking, only swaps each node's entrance and exit children. An element tilted by a or displaced by Δx then runs its exit transform first and acts as tilted by −a or displaced by −Δx; a solenoid keeps its field B, although inverting its map needs −B; and a quadrupole or bend hard-edge fringe acts twice instead of cancelling. Backtracking, x′, y′ and z negated around tracking through the reversed lattice, therefore misses the start. At 100 MeV, a 0.5 m, 3 T/m linac quadrupole tilted by 0.3 rad returns a ray started at x = −0.2 mm with y′ off by 0.2293 mrad. Untilted quadrupoles and bend bodies return to the start within 10⁻¹⁸.

The script builds one-element TEAPOT lattices at 1 GeV and linac lattices at 100 MeV, tracks two rays of amplitude about 0.1 mm forward, applies the backtracking transform of the upstream SNS backtracking example (x′, y′, z negated, dE kept, synchronous time reset), calls `reverseOrder()`, tracks again, applies the transform again and compares with the start. Untilted quadrupoles and bend bodies are the controls, and negating the tilt or the solenoid field after reversal returns the same cases to roundoff. For hard-edge fringes, with rays ten times larger, it compares the round-trip residual with the forward fringe effect, and exchanging the entrance and exit fringe functions after reversal is the control.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core.bunch import Bunch
from orbit.errors import AddErrorNode
from orbit.py_linac.errors import ErrorCntrlCoordDisplacement
from orbit.py_linac.lattice import Bend, LinacAccLattice, Quad, Sequence, Solenoid
from orbit.teapot import BendTEAPOT, DriftTEAPOT, QuadTEAPOT, SolenoidTEAPOT, TEAPOT_Lattice

MASS = 0.93827208816
NAMES = ("x", "x'", "y", "y'", "z", "dE")
# (x [m], x', y [m], y', z [m], dE [GeV]); small amplitudes keep TEAPOT's nonlinear terms near 1e-19
START = [(1e-4, 2e-5, -5e-5, 3e-5, 1e-4, 1e-6), (-2e-4, -1e-5, 1e-4, -2e-5, -2e-4, -2e-6)]


def flip(b):
    # BunchTransformerFunc of sns_linac_lattice_backtracking_test.py
    for i in range(b.getSize()):
        b.xp(i, -b.xp(i))
        b.yp(i, -b.yp(i))
        b.z(i, -b.z(i))
    b.getSyncParticle().time(0.0)


def coords(b):
    return [(b.x(i), b.xp(i), b.y(i), b.yp(i), b.z(i), b.dE(i)) for i in range(b.getSize())]


def round_trip(lat, ek, after_reverse=None, starts=START):
    """Return (max |back - start|, ray index, coordinate index, start value, back value)."""
    b = Bunch()
    b.mass(MASS)
    b.charge(1.0)
    b.getSyncParticle().kinEnergy(ek)
    for r in starts:
        b.addParticle(*r)
    lat.trackBunch(b)
    flip(b)
    lat.reverseOrder()
    if after_reverse is not None:
        after_reverse()
    lat.trackBunch(b)
    flip(b)
    back = coords(b)
    dev, i, k = max(
        (abs(back[i][k] - starts[i][k]), i, k) for i in range(len(starts)) for k in range(6)
    )
    return dev, i, k, starts[i][k], back[i][k]


def forward(lat, ek, starts=START):
    b = Bunch()
    b.mass(MASS)
    b.charge(1.0)
    b.getSyncParticle().kinEnergy(ek)
    for r in starts:
        b.addParticle(*r)
    lat.trackBunch(b)
    return coords(b)


def teapot(nodes, err=None):
    lat = TEAPOT_Lattice("t")
    for n in nodes:
        lat.addNode(n)
    lat.initialize()
    if err is not None:
        AddErrorNode(lat, 0.6, 0.7, {"errtype": "StraightError", "sample": "Fixed", **err})
    return lat


def linac(nodes):
    lat = LinacAccLattice("l")
    seq = Sequence("S")
    for n in nodes:
        seq.addNode(n)
        lat.addNode(n)
    lat.initialize()
    return lat


def t_quad(tilt=0.0, fringe=False):
    q = QuadTEAPOT("q", length=0.5, nparts=4, kq=1.5)
    q.setUsageFringeFieldIN(fringe)
    q.setUsageFringeFieldOUT(fringe)
    q.setTiltAngle(tilt)
    return q


def t_line(q):
    return [DriftTEAPOT("d1", 0.4), q, DriftTEAPOT("d2", 0.6)]


def t_bend(fringe=False):
    b = BendTEAPOT("b", length=1.0, nparts=8, theta=0.1)  # sector bend, no face angles
    b.setUsageFringeFieldIN(fringe)
    b.setUsageFringeFieldOUT(fringe)
    return b


def l_quad(tilt=0.0, fringe=False):
    q = Quad("q")
    q.setLength(0.5)
    q.setParam("dB/dr", 3.0)  # T/m
    q.setnParts(4)
    q.setUsageFringeFieldIN(fringe)
    q.setUsageFringeFieldOUT(fringe)
    q.setTiltAngle(tilt)
    return q


def l_offset_quad(dx):
    q = l_quad()
    err = ErrorCntrlCoordDisplacement("e")
    err.setDisplacementParameter("dx", dx)
    err.setOneNodeParent(q)
    return q


def l_bend():
    b = Bend("b")
    b.setLength(1.0)
    b.setParam("theta", 0.1)
    b.setnParts(8)
    b.setUsageFringeFieldIN(False)
    b.setUsageFringeFieldOUT(False)
    return b


def l_sol(b_field):
    s = Solenoid("s")
    s.setLength(1.0)
    s.setParam("B", b_field)  # normalized field B/(B rho) [1/m]
    s.setnParts(4)
    return s


def negate_tilt(node):
    # restore the forward geometry: after reversal the entrance child must rotate by +a again
    def fix():
        a = node.getTiltAngle()
        node.getNodeTiltIN().setTiltAngle(-a)
        node.getNodeTiltOUT().setTiltAngle(a)

    return fix


def negate_field(node):
    return lambda: node.setParam("B", -node.getParam("B"))


def swap_fringes(node):
    def fix():
        fi = node.getNodeFringeFieldIN().getFringeFieldFunction()
        fo = node.getNodeFringeFieldOUT().getFringeFieldFunction()
        node.getNodeFringeFieldIN().setFringeFieldFunction(fo)
        node.getNodeFringeFieldOUT().setFringeFieldFunction(fi)

    return fix


# Expected for every case: back = start. The transform R (x', y', z negated) satisfies R M R = M^-1 for
# every map of a static field whose Hamiltonian is even in the transverse momenta, so the reversed lattice
# tracked between two R must undo the forward pass; the controls show the roundoff level (1e-18).
ROUNDOFF = 1e-15
defects, controls = [], []


def case(label, lat, ek, is_control, fix=None):
    dev, i, k, s, b = round_trip(lat, ek, fix)
    print(
        f"[{label}] max |back - start| {dev:.4e}, expected 0 (roundoff); ray {i + 1} {NAMES[k]}: start {s:.4e}, back {b:.4e}"
    )
    (controls if is_control else defects).append(
        dev < ROUNDOFF if is_control else dev > 1e3 * ROUNDOFF
    )
    return dev


E_T, E_L = 1.0, 0.1  # kinetic energies [GeV]
case("TEAPOT control: quadrupole", teapot(t_line(t_quad())), E_T, True)
case("TEAPOT control: bend body", teapot([t_bend()]), E_T, True)
case("TEAPOT quadrupole tilted 0.3 rad", teapot(t_line(t_quad(0.3))), E_T, False)
q = t_quad(0.3)
case(
    "TEAPOT quadrupole tilted 0.3 rad, tilt children negated after reversal",
    teapot(t_line(q)),
    E_T,
    True,
    negate_tilt(q),
)
case(
    "TEAPOT AddErrorNode XYRot 0.3 rad",
    teapot(t_line(t_quad()), {"subtype": "XYRot", "angle": 0.3}),
    E_T,
    False,
)
case(
    "TEAPOT AddErrorNode TransDisp dx = 1 mm",
    teapot(t_line(t_quad()), {"subtype": "TransDisp", "dx": 1e-3, "dy": 0.0}),
    E_T,
    False,
)
s = SolenoidTEAPOT("s", B=0.5, length=1.0, nparts=4)
case("TEAPOT solenoid B = 0.5 1/m", teapot([s]), E_T, False)
s = SolenoidTEAPOT("s", B=0.5, length=1.0, nparts=4)
case(
    "TEAPOT solenoid B = 0.5 1/m, B negated after reversal", teapot([s]), E_T, True, negate_field(s)
)

case("linac control: quadrupole", linac([l_quad()]), E_L, True)
case("linac control: bend body", linac([l_bend()]), E_L, True)
case("linac quadrupole 3 T/m tilted 0.3 rad", linac([l_quad(0.3)]), E_L, False)
case(
    "linac quadrupole, ErrorCntrlCoordDisplacement dx = 1 mm",
    linac([l_offset_quad(1e-3)]),
    E_L,
    False,
)
case("linac solenoid B = 0.5 1/m", linac([l_sol(0.5)]), E_L, False)
s = l_sol(0.5)
case("linac solenoid B = 0.5 1/m, B negated after reversal", linac([s]), E_L, True, negate_field(s))

# Hard-edge fringes: compare the round-trip residual with the size of the forward fringe effect.
# Swapping the children puts the exit fringe map at the new entrance; it equals R F_in^-1 R, so the
# fringe acts twice and the residual is about twice the fringe effect. Larger rays make the third-order
# fringe terms stand well above roundoff.
MM = [tuple(10 * v for v in r) for r in START]


def fringe_case(label, make, ek, fix_factory=None):
    on, off = forward(make(True), ek, MM), forward(make(False), ek, MM)
    effect = max(abs(a - c) for ra, rc in zip(on, off) for a, c in zip(ra, rc))
    dev = round_trip(make(True), ek, starts=MM)[0]
    print(
        f"[{label} with hard-edge fringes, rays x 10] max |back - start| {dev:.4e}, expected 0 (roundoff); "
        f"forward fringe effect {effect:.4e}, ratio {dev / effect:.3f}"
    )
    defects.append(dev > 1.5 * effect)
    return effect


fringe_case("TEAPOT sector bend", lambda f: teapot([t_bend(f)]), E_T)
effect = fringe_case("linac quadrupole", lambda f: linac([l_quad(fringe=f)]), E_L)
q = l_quad(fringe=True)
dev_sw = round_trip(linac([q]), E_L, swap_fringes(q), starts=MM)[0]
print(
    f"[linac quadrupole with hard-edge fringes, rays x 10, fringe functions exchanged after reversal] "
    f"max |back - start| {dev_sw:.4e}, {dev_sw / effect:.1e} of the fringe effect"
)
controls.append(dev_sw < 1e-5 * effect)
print("DEFECT REPRODUCED" if all(defects) and all(controls) else "NOT REPRODUCED")
