"""PAR-11: MAD and MAD-X solenoid import applies the bunch charge to KS a second time

`readMADX` and `readMAD` copy a SOLENOID's strength KS into the parameter `B` of a TEAPOT solenoid, whose map multiplies `B` by the bunch charge number q. MAD defines KS = B_s/(Bρ), with B_s the longitudinal field and Bρ the beam particle's rigidity, so its map is the same for every charge. The imported solenoid applies q·KS instead: antiprotons, electrons and H⁻ ions (q = −1) rotate the opposite way, and a q = 2 ion by twice the angle. For KS = 0.5 m⁻¹ over 1 m and a particle starting at x = 1 mm, q = −1 leaves at y = +0.2397 mm and q = +2 at x = 0.7702 mm, where MAD-X gives y = −0.2397 mm and x = 0.9388 mm.

The script reads a 1 m MAD-X sequence and the equivalent MAD-8 line holding one solenoid with KS = 0.5 m⁻¹ and tracks a particle starting at x = 1 mm with zero energy deviation for q = +1, −1 (proton and electron mass), +2 and −2 at the same momentum. It compares the exit coordinates with the closed-form hard-edge solenoid map of normalized strength KS, which MAD-X applies for every charge. The q = +1 rows are the control, and each imported lattice is tracked again with its solenoid parameter set to KS/q, which gives the MAD map for every charge.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.core.bunch import Bunch
from orbit.teapot import TEAPOT_Lattice

KS, L = 0.5, 1.0
X0 = (1e-3, 0.0, 0.0, 0.0)  # x, px, y, py


def solenoid_map(ks, length, v):
    # Hard-edge solenoid of normalized strength ks = B_s/(B rho) at delta = 0, entrance and exit
    # fringes included (the MAD-X SOLENOID transfer matrix): K = ks/2, C = cos(K L), S = sin(K L).
    k = ks / 2.0
    c, s = math.cos(k * length), math.sin(k * length)
    x, px, y, py = v
    return (
        c * c * x + s * c / k * px + s * c * y + s * s / k * py,
        -k * s * c * x + c * c * px - k * s * s * y + s * c * py,
        -s * c * x - s * s / k * px + c * c * y + s * c / k * py,
        k * s * s * x - s * c * px - k * s * c * y + c * c * py,
    )


# MAD: KS already carries the sign and size of the beam particle's charge, so the map is the same
# for every q. For x0 = 1 mm: x = C^2 x0 = 0.9388 mm, y = -S C x0 = -0.2397 mm (K L = 0.25).
EXPECTED = solenoid_map(KS, L, X0)

SOURCES = {
    "readMADX": (
        "seq.madx",
        f"sol: solenoid, l={L}, ks={KS};\nring: sequence, l={L};\nsol, at={L / 2};\nendsequence;\n",
    ),
    "readMAD": ("seq.mad", f"sol: solenoid, l={L}, ks={KS}\nring: line=(sol)\n"),
}
SPECIES = [
    (+1.0, 0.93827208816, "proton, control"),
    (-1.0, 0.93827208816, "antiproton"),
    (-1.0, 0.00051099895, "electron"),
    (+2.0, 0.93827208816, "q = +2 ion"),
    (-2.0, 0.93827208816, "q = -2 ion"),
]


def track(lat, q, m):
    b = Bunch()
    b.mass(m)
    b.charge(q)
    b.getSyncParticle().momentum(2.0)
    b.addParticle(X0[0], X0[1], X0[2], X0[3], 0.0, 0.0)
    lat.trackBunch(b)
    return (b.x(0), b.xp(0), b.y(0), b.yp(0))


def fmt(v):
    return "(" + ", ".join(f"{u:+.6e}" for u in v) + ")"


print(
    f"expected exit (x, px, y, py) {fmt(EXPECTED)} for every charge (MAD solenoid map, KS = {KS} m^-1, L = {L} m)"
)
bad, controls_ok = False, True
for reader, (fname, text) in SOURCES.items():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, fname)
        with open(path, "w") as f:
            f.write(text)
        lat = TEAPOT_Lattice()
        getattr(lat, reader)(path, "ring")
    sol = [n for n in lat.getNodes() if n.getType() == "solenoid teapot"]
    for q, m, label in SPECIES:
        for node in sol:
            node.setParam("B", KS)
        obs = track(lat, q, m)
        dev = max(abs(o - e) for o, e in zip(obs, EXPECTED))
        # Control: the parameter B that soln multiplies by q must be KS/q to give the MAD map.
        for node in sol:
            node.setParam("B", KS / q)
        dev_fix = max(abs(o - e) for o, e in zip(track(lat, q, m), EXPECTED))
        print(
            f"[{reader}, {label}, q = {q:+.0f}] B = {KS} as imported: exit observed {fmt(obs)}, max |diff| {dev:.3e}; "
            f"control with B = KS/q: max |diff| {dev_fix:.3e}"
        )
        controls_ok &= dev_fix < 1e-15
        if q == 1.0:
            controls_ok &= dev < 1e-15
        else:
            bad |= dev > 1e-6
print("DEFECT REPRODUCED" if bad and controls_ok else "NOT REPRODUCED")
