---
title: "COL-07: UniformEllipsoidFieldCalculator gives a wrong field outside an elongated ellipsoid"
hide:
  - toc
---

# COL-07 — `UniformEllipsoidFieldCalculator` gives a wrong field outside an elongated ellipsoid

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [UniformEllipsoidFieldCalculator.cc#L41](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/UniformEllipsoidFieldCalculator.cc#L41-L41)<br>[UniformEllipsoidFieldCalculator.cc#L74-L111](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/UniformEllipsoidFieldCalculator.cc#L74-L111)<br>[UniformEllipsoidFieldCalculator.cc#L177-L207](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/UniformEllipsoidFieldCalculator.cc#L177-L207)<br>[SpaceChargeCalcUnifEllipse.cc#L163-L168](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L163-L168)<br>[SpaceChargeCalcUnifEllipse.cc#L194-L201](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L194-L201) |

Outside a uniformly charged ellipsoid with semiaxes a, b, c, `UniformEllipsoidFieldCalculator` interpolates the field integral linearly in the confocal parameter λ from a 200-point table with spacing (2·max(a, b, c))²/199. For an elongated ellipsoid the transverse integral varies on the scale of the smallest semiaxis squared, far below that spacing, so the exterior field is wrong wherever λ is small against the step, near the surface and far away alike; the interior field is exact. The exact field is Fₓ = Q·x·R_D(c² + λ, b² + λ, a² + λ), with Q the charge and R_D Carlson's elliptic integral. For semiaxes (1, 1, 100) mm, Fₓ is 2.935 times exact at (1.5a, 0.5b, 0.4c) and 28.24 times at (4a, 3b, 0.5c).

Run `python COL-07.py`.

??? example "Reproducer: COL-07.py"

    [Download COL-07.py](COL-07.py)

    ```python
    """COL-07: `UniformEllipsoidFieldCalculator` gives a wrong field outside an elongated ellipsoid

    Outside a uniformly charged ellipsoid with semiaxes a, b, c, `UniformEllipsoidFieldCalculator` interpolates the field integral linearly in the confocal parameter λ from a 200-point table with spacing (2·max(a, b, c))²/199. For an elongated ellipsoid the transverse integral varies on the scale of the smallest semiaxis squared, far below that spacing, so the exterior field is wrong wherever λ is small against the step, near the surface and far away alike; the interior field is exact. The exact field is Fₓ = Q·x·R_D(c² + λ, b² + λ, a² + λ), with Q the charge and R_D Carlson's elliptic integral. For semiaxes (1, 1, 100) mm, Fₓ is 2.935 times exact at (1.5a, 0.5b, 0.4c) and 28.24 times at (4a, 3b, 0.5c).

    The script sets up the calculator for five semiaxis sets, from a sphere to a 1:1:100 ellipsoid, and compares Fₓ at an interior point, at an exterior point near the surface and at a transversely distant exterior point with the exact field, using λ from a bracketed root and SciPy's `elliprd`. The interior points and the sphere are the controls.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.spacecharge import UniformEllipsoidFieldCalculator
    from scipy.optimize import brentq
    from scipy.special import elliprd


    def exact_fx(a, b, c, x, y, z):
        # Uniform ellipsoid, total charge 1:
        # F_x = x (3/2) int_lam^inf ds / ((a^2 + s) sqrt((a^2 + s)(b^2 + s)(c^2 + s))) = x R_D(c^2 + lam, b^2 + lam, a^2 + lam),
        # lam = 0 inside, else the root of sum x_i^2/(a_i^2 + lam) = 1.
        s = x * x / a**2 + y * y / b**2 + z * z / c**2
        lam = 0.0
        if s > 1.0:
            f = lambda t: x * x / (a * a + t) + y * y / (b * b + t) + z * z / (c * c + t) - 1.0
            lam = brentq(f, 0.0, 1e3, xtol=1e-300, rtol=1e-15)
        return x * elliprd(c * c + lam, b * b + lam, a * a + lam)


    calc = UniformEllipsoidFieldCalculator()
    bad, ctl_ok = False, True
    for a, b, c in (
        (1e-3, 1e-3, 1e-3),
        (1e-3, 1e-3, 3e-3),
        (1e-3, 1e-3, 1e-2),
        (1e-3, 1e-3, 1e-1),
        (2e-3, 1e-3, 0.2),
    ):
        calc.setEllipsoid(a, b, c, 10 * max(a, b, c))
        sphere = a == b == c
        for where, (x, y, z) in (
            ("inside", (0.3 * a, 0.2 * b, 0.1 * c)),
            ("near outside", (1.5 * a, 0.5 * b, 0.4 * c)),
            ("far outside", (4 * a, 3 * b, 0.5 * c)),
        ):
            ratio = calc.calcField(x, y, z)[0] / exact_fx(a, b, c, x, y, z)
            tag = " (control)" if where == "inside" or sphere else ""
            print(
                f"semiaxes ({a * 1e3:g}, {b * 1e3:g}, {c * 1e3:g}) mm, {where} ({x * 1e3:g}, {y * 1e3:g}, {z * 1e3:g}) mm{tag}: "
                f"Fx/exact observed {ratio:.5f}, expected 1 (Carlson R_D field)"
            )
            if tag:
                ctl_ok &= abs(ratio - 1) < 1e-4
            else:
                bad |= abs(ratio - 1) > 0.01
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

