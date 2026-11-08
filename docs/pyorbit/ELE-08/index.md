---
title: "ELE-08: Collimator material steps do not advance the longitudinal coordinate"
hide:
  - toc
---

# ELE-08 — Collimator material steps do not advance the longitudinal coordinate

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Elements and apertures |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`Collimator.cc#L714-L734`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/MaterialInteractions/Collimator.cc#L714-L734)<br>[`Collimator.cc#L344-L365`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/MaterialInteractions/Collimator.cc#L344-L365)<br>[`Collimator.cc#L218-L222`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/MaterialInteractions/Collimator.cc#L218-L222) |

A particle travelling through the material of a `TeapotCollimatorNode` keeps its z, although the synchronous time advances by L/(β₀c) over the collimator length L, with β₀c the reference speed. It arrives without the extra path of its slope and without the lower speed after its energy loss, both of which particles in the aperture get. A 1 GeV proton crossing 0.1 m of carbon at 1% density with x′ = 50 mrad leaves with z = 0 instead of −1.287×10⁻⁴ m. At full density with x′ = 0 the expected mean shift is −3.781×10⁻⁴ m.

Run `python ELE-08.py`.

??? example "Reproducer: ELE-08.py"

    [Download ELE-08.py](ELE-08.py)

    ```python
    """ELE-08: Collimator material steps do not advance the longitudinal coordinate

    A particle travelling through the material of a `TeapotCollimatorNode` keeps its z, although the synchronous time
    advances by L/(β₀c) over the collimator length L, with β₀c the reference speed. It arrives without the extra path of its
    slope and without the lower speed after its energy loss, both of which particles in the aperture get. A 1 GeV proton
    crossing 0.1 m of carbon at 1% density with x′ = 50 mrad leaves with z = 0 instead of −1.287×10⁻⁴ m. At full density
    with x′ = 0 the expected mean shift is −3.781×10⁻⁴ m.

    The script tracks 400 protons of 1 GeV kinetic energy through the jaw of a 0.1 m carbon collimator, at full density with
    x′ = 0 and at 1% density with x′ = 50 mrad, and computes each survivor's expected z from its own exit energy and slope:
    the flight time over a straight path at the mean of its entrance and exit speeds, minus the reference time. The control
    is one particle in the aperture, which the collimator drifts with the TEAPOT drift, z = −L·x′²/2.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.collimation import TeapotCollimatorNode
    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.core.orbit_utils import random as orbit_random  # ty: ignore[unresolved-import]
    from orbit.teapot import TEAPOT_Lattice

    MASS, EK = 0.93827208816, 1.0  # GeV
    E0 = EK + MASS
    BETA0 = math.sqrt(1 - (MASS / E0) ** 2)


    def beta_of(e):
        return math.sqrt(1 - (MASS / e) ** 2)


    def run(length, density, xp, n, seed):
        lat = TEAPOT_Lattice()
        # carbon (ma = 0), shape 1: circle of radius 10 mm, material outside it
        lat.addNode(TeapotCollimatorNode(length, 0, density, 1, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, "col"))
        lat.initialize()
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(EK)
        b.addParticle(1e-3, xp, 0.0, 0.0, 0.0, 0.0)  # aperture control; x stays below 10 mm
        for _ in range(n):
            b.addParticle(0.02, xp, 0.0, 0.0, 0.0, 0.0)  # inside the jaw
        lost = Bunch()
        b.copyEmptyBunchTo(lost)
        orbit_random.seed(seed)
        lat.trackBunch(b, {"lostbunch": lost})
        return b


    reproduced = True
    for length, density, xp in ((0.1, 1.0, 0.0), (0.1, 0.01, 0.05)):
        b = run(length, density, xp, 400, 7)
        label = f"L = {length} m carbon, density factor {density}, x' = {xp * 1e3:.0f} mrad"
        # Aperture: TEAPOT drift at dE = 0 gives z = -L x'^2 / 2 (z = -beta0 c dt, positive ahead of the reference).
        print(
            f"[{label}, aperture control] z observed {b.z(0):.6e} m, expected {-length * xp * xp / 2:.6e} m"
        )
        reproduced &= abs(b.z(0) + length * xp * xp / 2) < 1e-12
        z_obs, z_exp, de = [], [], []
        for i in range(1, b.getSize()):
            # mean of entrance and exit slope; scattering changes it little
            slope = 0.5 * (xp + b.xp(i))
            e_mean = E0 + b.dE(i) / 2  # linear energy loss along the jaw
            # z = -beta0 c (t - t_ref) with t = L sqrt(1 + slope^2) / (beta c) and t_ref = L / (beta0 c)
            z_exp.append(-length * (BETA0 * math.sqrt(1 + slope * slope) / beta_of(e_mean) - 1.0))
            z_obs.append(b.z(i))
            de.append(b.dE(i))
        n = len(z_obs)
        print(
            f"[{label}, jaw] {n} of 400 survive, mean dE {sum(de) / n:.4e} GeV; z observed: mean {sum(z_obs) / n:.4e} m, "
            f"max |z| {max(abs(z) for z in z_obs):.1e} m; expected mean {sum(z_exp) / n:.4e} m, "
            f"range [{min(z_exp):.4e}, {max(z_exp):.4e}] m"
        )
        reproduced &= max(abs(z) for z in z_obs) == 0.0 and sum(z_exp) / n < -1e-4
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

