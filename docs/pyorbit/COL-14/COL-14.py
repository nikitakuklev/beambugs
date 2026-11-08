"""COL-14: `SpaceChargeCalc3D.numExtBunches(0)` does not switch off neighbouring-bunch images

`SpaceChargeCalc3D.numExtBunches(n)` forwards n to the solver only when n > 0 and otherwise only returns the current count, so once images of neighbouring bunches are enabled, `numExtBunches(0)` leaves them on and the calculator keeps its periodic, wrapped analysis. A user who sets 2 images and then requests 0 gets 2 back, and an isolated bunch keeps receiving the field of its images; no other method resets the count. For a uniform sphere of radius 1 mm with images every 3.9 mm, a probe's energy kick stays at 0.9339 times the image-free value.

The script enables two images in one `SpaceChargeCalc3D(32, 32, 32)`, requests zero, and tracks a uniform sphere with an on-axis probe through it and through a fresh calculator without images. The sibling `PoissonSolverFFT3D.numExtBunches`, which accepts zero, is the control for the setter, and the fresh calculator for the kick.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import PoissonSolverFFT3D, SpaceChargeCalc3D
from orbit.utils.consts import mass_proton

rng = np.random.default_rng(5)
v = rng.normal(size=(20000, 3))
v /= np.linalg.norm(v, axis=1)[:, None]
v *= 1e-3 * rng.uniform(size=(20000, 1)) ** (1 / 3)  # uniform sphere, radius 1 mm


def kick(calc):
    b = Bunch()
    b.mass(mass_proton)
    b.macroSize(1e9 / 20000)
    b.getSyncParticle().kinEnergy(0.002)
    for p in v:
        b.addParticle(p[0], 0.0, p[1], 0.0, p[2], 0.0)
    ip = b.getSize()
    b.addParticle(0.5e-3, 0.0, 0.0, 0.0, 0.3e-3, 0.0)
    calc.trackBunch(b, 0.01)
    return b.dE(ip)


solver = PoissonSolverFFT3D(16, 16, 16, -1e-3, 1e-3, -1e-3, 1e-3, -1e-3, 1e-3)
s2, s0 = solver.numExtBunches(2), solver.numExtBunches(0)
print(
    f"control, PoissonSolverFFT3D.numExtBunches(2) returns {s2}, then numExtBunches(0) returns {s0}; expected 2, 0"
)
c = SpaceChargeCalc3D(32, 32, 32)
n2, n0 = c.numExtBunches(2), c.numExtBunches(0)
print(
    f"SpaceChargeCalc3D.numExtBunches(2) returns {n2}, then numExtBunches(0) returns {n0}; expected 2, 0"
)
c.freqOfBunches(5e9)  # bunch spacing beta c / f = 3.9 mm at 2 MeV, so images act strongly
fresh = SpaceChargeCalc3D(32, 32, 32)
fresh.freqOfBunches(5e9)
de_after, de_fresh = kick(c), kick(fresh)
# Estimate of the image effect: inside the sphere the self field is Q z / R^3; the two nearest images,
# as point charges at +-3.9 mm (rest frame, gamma ~ 1), give -Q (1/3.6^2 - 1/4.2^2) /mm^2 at z = 0.3 mm,
# so the kick with images is about 0.932 of the image-free kick.
print(
    f"probe dE after numExtBunches(0) observed {de_after:.6e} GeV ({de_after / de_fresh:.4f} of the image-free kick), "
    f"expected {de_fresh:.6e} GeV (fresh calculator without images, control)"
)
ctl_ok = (s2, s0) == (2, 0)
print(
    "DEFECT REPRODUCED"
    if ctl_ok and n0 != 0 and abs(de_after - de_fresh) > 1e-3 * abs(de_fresh)
    else "NOT REPRODUCED"
)
