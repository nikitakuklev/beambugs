---
title: "ELE-04: A collimator or foil node with several parts traverses the whole element in every part"
hide:
  - toc
---

# ELE-04 — A collimator or foil node with several parts traverses the whole element in every part

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Elements and apertures |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`TeapotCollimatorNode.py#L35-L42`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/collimation/TeapotCollimatorNode.py#L35-L42)<br>[`Collimator.cc#L68-L100`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/MaterialInteractions/Collimator.cc#L68-L100)<br>[`Collimator.cc#L218-L222`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/MaterialInteractions/Collimator.cc#L218-L222)<br>[`TeapotFoilNode.py#L37-L47`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/foils/TeapotFoilNode.py#L37-L47) |

`TeapotCollimatorNode.track` runs once per body part and calls the native collimator over the full collimator length each time. With N parts every particle drifts and scatters over N·L, and the synchronous clock advances N·L/(β₀c), with β₀c the reference speed, while the lattice length stays L. For a 0.5 m node in 4 parts, a particle in the aperture with x′ = 1 mrad starting at 2 mm leaves at 4.000 mm instead of 2.500 mm. A pencil beam in a 5 mm carbon jaw split into 4 parts has the median scattering angle of an unsplit 20 mm jaw, 2.0 to 2.1 times that of the 5 mm jaw. `TeapotFoilNode` crosses its foil N times in the same way.

Run `python ELE-04.py`.

??? example "Reproducer: ELE-04.py"

    [Download ELE-04.py](ELE-04.py)

    ```python
    """ELE-04: A collimator or foil node with several parts traverses the whole element in every part

    `TeapotCollimatorNode.track` runs once per body part and calls the native collimator over the full collimator length
    each time. With N parts every particle drifts and scatters over N·L, and the synchronous clock advances N·L/(β₀c), with
    β₀c the reference speed, while the lattice length stays L. For a 0.5 m node in 4 parts, a particle in the aperture with
    x′ = 1 mrad starting at 2 mm leaves at 4.000 mm instead of 2.500 mm. A pencil beam in a 5 mm carbon jaw split into 4
    parts has the median scattering angle of an unsplit 20 mm jaw, 2.0 to 2.1 times that of the 5 mm jaw.
    `TeapotFoilNode` crosses its foil N times in the same way.

    The script tracks one particle in the aperture of a 0.5 m circular carbon collimator with 1 to 4 parts and compares its
    exit x and the synchronous-time step with a 0.5 m drift. It then compares the median scattering angle of a pencil beam
    in a 5 mm jaw split into 4 parts with an unsplit 5 mm jaw and an unsplit 20 mm jaw, for two seeds of the native
    generator, and the mean energy loss in a carbon foil with 1, 2 and 4 parts. The one-part node is the control throughout.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.collimation import TeapotCollimatorNode
    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.core.orbit_utils import random as orbit_random  # ty: ignore[unresolved-import]
    from orbit.foils import TeapotFoilNode
    from orbit.teapot import TEAPOT_Lattice

    MASS, EK, C = 0.93827208816, 1.0, 2.99792458e8  # proton (GeV), kinetic energy (GeV), m/s


    def track(node, parts, rays, seed=None):
        node.setnParts(parts)
        lat = TEAPOT_Lattice()
        lat.addNode(node)
        lat.initialize()
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(EK)
        for r in rays:
            b.addParticle(*r)
        lost = Bunch()
        b.copyEmptyBunchTo(lost)
        if seed is not None:
            orbit_random.seed(seed)
        t0 = b.getSyncParticle().time()
        lat.trackBunch(b, {"lostbunch": lost})
        return lat.getLength(), b, b.getSyncParticle().time() - t0


    def collimator(length):
        # carbon (ma = 0), density factor 1, shape 1: circle of radius a = b = 10 mm, material outside it
        return TeapotCollimatorNode(length, 0, 1.0, 1, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, "col")


    reproduced = True
    L = 0.5
    for n in (1, 2, 3, 4):
        length, b, dt = track(collimator(L), n, [(2e-3, 1e-3, 0.0, 0.0, 0.0, 0.0)])
        t_ref = L / (b.getSyncParticle().beta() * C)
        # Inside the aperture the node is a drift of its length L: x = x0 + L x' = 2.5 mm, sync time step L/(beta0 c).
        print(
            f"[aperture ray, {n} part(s)] lattice length {length:.3f} m; x observed {b.x(0) * 1e3:.6f} mm, expected "
            f"{(2e-3 + L * 1e-3) * 1e3:.6f} mm; sync time step observed {dt:.6e} s = {dt / t_ref:.6f} L/(beta0 c), "
            f"expected {t_ref:.6e} s"
        )
        ok = abs(b.x(0) - 2.5e-3) < 1e-15 and abs(dt - t_ref) < 1e-20
        reproduced &= ok if n == 1 else not ok and abs(b.x(0) - (2e-3 + n * L * 1e-3)) < 1e-15


    def median_angle(length, parts, seed):
        # 6000 protons at x = 15 mm, inside the jaw, with zero angle; the median |x'| measures the multiple-scattering
        # core (the rms is dominated by single-scattering tails).
        _, b, _ = track(collimator(length), parts, [(0.015, 0.0, 0.0, 0.0, 0.0, 0.0)] * 6000, seed)
        angles = sorted(abs(b.xp(i)) for i in range(b.getSize()))
        return angles[len(angles) // 2]


    def highland(x, x0=0.1925, beta=0.8750, p=1.696):
        # PDG Highland width for thickness x of graphite (X0 = 0.1925 m) at 1 GeV kinetic energy
        return 13.6e-3 / (beta * p) * math.sqrt(x / x0) * (1 + 0.038 * math.log(x / x0))


    for seed in (101, 202):
        m5, m4p, m20 = (
            median_angle(0.005, 1, seed),
            median_angle(0.005, 4, seed),
            median_angle(0.020, 1, seed),
        )
        # Four parts of a 5 mm jaw must scatter like the 5 mm jaw (ratio 1), not like 20 mm of carbon.
        print(
            f"[5 mm carbon jaw, seed {seed}] median |x'| with 4 parts {m4p:.4e}, expected {m5:.4e} (1 part); ratio "
            f"{m4p / m5:.3f}, expected 1; unsplit 20 mm jaw {m20:.4e}, ratio 4 parts / 20 mm {m4p / m20:.3f}"
        )
        reproduced &= m4p / m5 > 1.5
    print(
        f"    Highland width ratio of 20 mm to 5 mm of carbon: {highland(0.020) / highland(0.005):.3f}"
    )

    mean_dE = {}
    for n in (1, 2, 4):
        # 200 ug/cm^2 carbon foil covering +-5 cm; full scattering model; every particle crosses it
        foil = TeapotFoilNode(-0.05, 0.05, -0.05, 0.05, 200.0, "foil")
        _, b, _ = track(foil, n, [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * 4000, seed=5)
        mean_dE[n] = sum(b.dE(i) for i in range(b.getSize())) / b.getSize()
        print(
            f"[carbon foil 200 ug/cm^2, {n} part(s)] mean dE observed {mean_dE[n]:.4e} GeV, expected {mean_dE[1]:.4e} GeV "
            f"(one crossing); ratio {mean_dE[n] / mean_dE[1]:.4f}"
        )
    reproduced &= abs(mean_dE[4] / mean_dE[1] - 4.0) < 0.05
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

