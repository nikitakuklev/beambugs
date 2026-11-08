"""COL-09: 2.5-D space charge wraps the bunch head's line density into its tail

`SpaceChargeCalc2p5D` and `SpaceChargeCalc2p5Drb` fit the longitudinal grid exactly to the bunch extrema and deposit and interpolate the line density with the periodic linear stencil of `Grid1D`, so weight within half a bin of one end is shared with the bin at the other end. A particle near either end therefore receives a transverse kick scaled by a mixture of head and tail densities. For a uniform round beam whose line density rises linearly from zero at the tail, with 32 bins, a particle 0.4% of the bunch length from the tail gets 23.37 times the kick of the non-periodic binned density, one 0.4% from the head 0.6031 times, and a central one 0.9983 times. Flat-top profiles are unaffected.

The script builds a uniform round beam (sunflower pattern, 40000 macroparticles) with a linearly rising line density, adds probes near the tail, at the centre and near the head, and tracks it through both 2.5-D calculators with 32 longitudinal bins. It compares each probe's kick with the analytic uniform-beam kick for the line density of a non-periodic cloud-in-cell deposit on the same bins, which is the best the grid can resolve. The centre probe and a flat-top bunch are the controls.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import SpaceChargeCalc2p5D, SpaceChargeCalc2p5Drb
from orbit.utils.consts import mass_proton

A, LZ, NPHYS, N, LENGTH, NZ = 1e-3, 0.01, 1e9, 40000, 0.1, 32
PROBES = (
    ("tail, z = 0.004 LZ", 0.004),
    ("centre, z = 0.5 LZ (control)", 0.5),
    ("head, z = 0.996 LZ", 0.996),
)


def nonperiodic_density(zs, lo, hi):
    # Cloud-in-cell deposit on NZ cell-centred bins without wrap-around (end weights clamped to the
    # end bins), then linear interpolation between bin centres: the line density the grid can resolve.
    dz = (hi - lo) / NZ
    u = np.clip((zs - lo) / dz - 0.5, 0, NZ - 1 - 1e-12)
    i0 = u.astype(int)
    f = u - i0
    h = np.bincount(i0, 1 - f, NZ) + np.bincount(i0 + 1, f, NZ)
    centres = lo + (np.arange(NZ) + 0.5) * dz
    return lambda zp: np.interp(zp, centres, h * NPHYS / N / dz)


def run(profile, calc_name):
    k = np.arange(N) + 0.5
    r = A * np.sqrt(k / N)  # sunflower: uniform areal density in r < a
    t = k * np.pi * (3.0 - np.sqrt(5.0))
    # Quantiles of lambda(z) = 2 N z / LZ^2 (ramp) or N / LZ (flat top) on [0, LZ]
    z = LZ * np.sqrt(k / N) if profile == "ramp" else LZ * k / N
    z = z[np.random.default_rng(0).permutation(N)]  # decorrelate z from the transverse pattern
    b = Bunch()
    b.mass(mass_proton)
    b.macroSize(NPHYS / N)
    b.getSyncParticle().kinEnergy(0.1)
    for i in range(N):
        b.addParticle(r[i] * np.cos(t[i]), 0.0, r[i] * np.sin(t[i]), 0.0, z[i], 0.0)
    idx = []
    for _, p in PROBES:
        idx.append(b.getSize())
        b.addParticle(0.5 * A, 0.0, 0.0, 0.0, p * LZ, 0.0)
    if calc_name == "SpaceChargeCalc2p5D":
        SpaceChargeCalc2p5D(32, 32, NZ).trackBunch(b, LENGTH)
    else:
        SpaceChargeCalc2p5Drb(32, 32, NZ).trackBunch(b, LENGTH, 0.01)
    sp = b.getSyncParticle()
    zall = np.append(z, [p * LZ for _, p in PROBES])
    lam = nonperiodic_density(zall, zall.min(), zall.max())
    # Uniform round beam: dxp = 2 L r0 lambda x / (beta^2 gamma^3 a^2), probes at x = a/2
    unit = 2 * LENGTH * b.classicalRadius() / (sp.beta() ** 2 * sp.gamma() ** 3 * A**2) * 0.5 * A
    return [b.xp(i) / (unit * lam(p * LZ)) for i, (_, p) in zip(idx, PROBES)]


bad, ctl_ok = False, True
for calc_name in ("SpaceChargeCalc2p5D", "SpaceChargeCalc2p5Drb"):
    for profile in ("ramp", "flat"):
        for (name, _), ratio in zip(PROBES, run(profile, calc_name)):
            control = profile == "flat" or "control" in name
            where = name.replace(" (control)", "")
            label = f"{calc_name}, {'linear ramp' if profile == 'ramp' else 'flat top'}, {where}{' (control)' if control else ''}"
            print(
                f"{label}: xp/analytic(binned lambda) observed {ratio:.4g}, expected 1 (uniform beam, local line density)"
            )
            if control:
                ctl_ok &= abs(ratio - 1) < 0.01
            else:
                bad |= abs(ratio - 1) > 0.2
print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")
