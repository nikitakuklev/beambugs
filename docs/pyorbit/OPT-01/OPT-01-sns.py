"""OPT-01: the error on the SNS ring lattice shipped with the code

Setup: set `PYORBIT_SOURCE` to a PyORBIT3 source checkout; the script reads the SNS ring lattice
`tests/py/orbit/parsers/inputs/sns_ring.lat` from it.

The script imports the SNS ring at 1.3 GeV, calls `getChromaticitiesXY` and compares it with a
central difference in δ of the tune of the tracked one-turn Jacobian about the off-momentum closed
orbit, which it finds by Newton iteration. As a control it evaluates the method's formula with the
ring tune in the denominator.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

import numpy as np
from orbit.core.bunch import Bunch
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice

src = os.environ.get("PYORBIT_SOURCE")
if not src:
    raise SystemExit("set PYORBIT_SOURCE to a PyORBIT3 source checkout")
lat = TEAPOT_Lattice()
lat.readMADX(os.path.join(src, "tests/py/orbit/parsers/inputs/sns_ring.lat"), "rnginjsol")
lat.initialize()
ref = Bunch()
ref.mass(0.938272)
ref.charge(1)
ref.getSyncParticle().kinEnergy(1.3)  # GeV


def track(v, dE):
    b = Bunch()
    ref.copyEmptyBunchTo(b)
    b.addParticle(v[0], v[1], v[2], v[3], 0.0, dE)
    lat.trackBunch(b)
    return np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)])


def jacobian(v, dE, h=1e-7):
    J = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        J[:, j] = (track(v + e, dE) - track(v - e, dE)) / (2 * h)
    return J


def closed_orbit(dE):
    # Newton iteration for the transverse fixed point at fixed dE (no RF: dE is conserved)
    v = np.zeros(4)
    for _ in range(30):
        f = track(v, dE) - v
        if np.max(np.abs(f)) < 1e-15:
            break
        v = v - np.linalg.solve(jacobian(v, dE) - np.eye(4), f)
    return v


def fractional_tunes(J):
    # cos(mu) from the trace; the sign of M01 (= beta sin mu) selects mu in [0, 2 pi)
    out = []
    for i in (0, 2):
        c = (J[i, i] + J[i + 1, i + 1]) / 2
        s = np.sign(J[i, i + 1]) * math.sqrt(1 - c * c)
        out.append((math.atan2(s, c) / (2 * math.pi)) % 1.0)
    return out


ml = TEAPOT_MATRIX_Lattice(lat, ref)
xi = ml.getChromaticitiesXY()
nu_ring = ml.getTunes()
nu_1 = (ml.getRingTwissDataX()[0][1][1], ml.getRingTwissDataY()[0][1][1])
sp = ref.getSyncParticle()
pb = sp.momentum() ** 2 / (sp.kinEnergy() + sp.mass())  # dE = (p^2/E) * delta, as in the method
dd = 1e-5
nu = [fractional_tunes(jacobian(closed_orbit(s * dd * pb), s * dd * pb)) for s in (1.0, -1.0)]
fd = [(nu[0][k] - nu[1][k]) / (2 * dd) for k in (0, 1)]  # xi = d(nu)/d(delta) on the closed orbit
print(
    f"ring tunes (getTunes) {nu_ring[0]:.6f} {nu_ring[1]:.6f}; "
    f"first-slice phase advance {nu_1[0]:.6f} {nu_1[1]:.6f}"
)
bad = False
for k, plane in enumerate("xy"):
    print(
        f"xi_{plane}: observed {xi[k]:.4f}, expected {fd[k]:.4f} "
        "(central difference in delta of the closed-orbit one-turn tune)"
    )
    ctl = xi[k] * math.sin(2 * math.pi * nu_1[k]) / math.sin(2 * math.pi * nu_ring[k])
    print(f"control xi_{plane} with the ring tune in the denominator: {ctl:.4f}")
    if abs(xi[k] - fd[k]) > 0.01 * abs(fd[k]):
        bad = True
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
