"""TRK-03: Per-particle TTF gap kernels move a particle that sits on the reference

With `setLinacTracker(True)`, thin TTF gaps use `RfGapTTF_slow`, which evaluates each particle's transit-time factors
and phase coefficient at its entrance velocity, while the reference particle uses the velocity at the gap midpoint. A
particle with all six coordinates zero coincides with the reference and must leave the gap with zero z and zero energy
deviation. For a 2.5 MeV proton in a 402.5 MHz gap with field integral E₀L = 0.1 MeV at φ = −30°, it leaves with ΔE =
−0.1496 keV, 0.1962% of the 76.25 keV gain, and z = 5.731×10⁻⁷ m, at every gap. `RfGapThreePointTTF_slow`, used by
axis-field gaps, gives the same kind of z offset; `BaseRfGap_slow` keeps the particle on the reference.

The script tracks a bunch holding one particle at the origin of phase space through each per-particle gap kernel and its
ordinary counterpart, with TTF polynomials of a symmetric cosine-shaped gap field (for `RfGapTTF_slow`) and a single
three-point step of a parabolic field (for `RfGapThreePointTTF_slow`). The expected exit coordinates are zero, since the
particle is the reference; the ordinary kernels and `BaseRfGap_slow` are the controls.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

import numpy as np
from orbit.core.bunch import Bunch
from orbit.core.linac import (
    BaseRfGap,
    BaseRfGap_slow,
    RfGapThreePointTTF,
    RfGapThreePointTTF_slow,
    RfGapTTF,
    RfGapTTF_slow,
)
from orbit.core.orbit_utils import Polynomial

C = 2.99792458e8
M = 0.938272013
F = 402.5e6
K0 = 2.5e-3
G = 0.02
V = 1e-4  # E0L, GeV
PHI = math.radians(-30.0)
S_GRID = np.linspace(-G / 2, G / 2, 20001)
W = math.pi / (2 * G) * np.cos(math.pi * S_GRID / G)  # unit-integral field shape


def poly(coefs):
    p = Polynomial(len(coefs) - 1)
    for i, c in enumerate(coefs):
        p.coefficient(i, float(c))
    return p


beta0 = math.sqrt(1 - 1 / (1 + K0 / M) ** 2)
k_in = 2 * math.pi * F / (C * beta0)
ks = np.linspace(0.8 * k_in, 1.2 * k_in, 81)
# T(k) = int E cos(ks) ds / V, T'(k) = dT/dk; S = S' = 0 for a symmetric field
p_t = poly(
    np.polynomial.Polynomial.fit(ks, [np.trapezoid(W * np.cos(k * S_GRID), S_GRID) for k in ks], 10)
    .convert()
    .coef
)
p_tp = poly(
    np.polynomial.Polynomial.fit(
        ks, [-np.trapezoid(W * S_GRID * np.sin(k * S_GRID), S_GRID) for k in ks], 10
    )
    .convert()
    .coef
)
p_0 = poly([0.0])


def on_reference(track):
    b = Bunch()
    b.getSyncParticle().kinEnergy(K0)
    b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    track(b)
    return b.z(0), b.dE(0), b.getSyncParticle().kinEnergy() - K0


dz = 0.01
e0 = 1e9 * V / (4 * dz / 3)  # V/m; E0*(1 - s^2/dz^2) integrates to V over [-dz, dz]
cases = [
    ("RfGapTTF", lambda b: RfGapTTF().trackBunch(b, F, V, PHI, p_t, p_0, p_tp, p_0), False),
    (
        "RfGapTTF_slow",
        lambda b: RfGapTTF_slow().trackBunch(b, F, V, PHI, p_t, p_0, p_tp, p_0),
        True,
    ),
    (
        "RfGapThreePointTTF",
        lambda b: RfGapThreePointTTF().trackBunch(b, dz, 0.0, e0, 0.0, F, PHI),
        False,
    ),
    (
        "RfGapThreePointTTF_slow",
        lambda b: RfGapThreePointTTF_slow().trackBunch(b, dz, 0.0, e0, 0.0, F, PHI),
        True,
    ),
    ("BaseRfGap_slow", lambda b: BaseRfGap_slow().trackBunch(b, F, V, PHI), False),
    ("BaseRfGap", lambda b: BaseRfGap().trackBunch(b, F, V, PHI), False),
]
reproduced = True
for name, track, expect_defect in cases:
    z, de, gain = on_reference(track)
    print(
        f"{name:24s} reference-coincident particle: z_out {z:.4e} m, dE_out {de * 1e6:.4f} keV; "
        f"expected 0 and 0 (the particle is the reference); reference gain {gain * 1e6:.2f} keV"
    )
    moved = abs(z) > 1e-12 or abs(de) > 1e-12
    reproduced &= moved == expect_defect
print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
