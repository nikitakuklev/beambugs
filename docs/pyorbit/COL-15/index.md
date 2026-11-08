---
title: "COL-15: SpaceChargeCalc2p5Drb.setLongAveragingPointsN(n) with n < 3 makes the energy kicks NaN"
hide:
  - toc
---

# COL-15 — `SpaceChargeCalc2p5Drb.setLongAveragingPointsN(n)` with n < 3 makes the energy kicks NaN

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [SpaceChargeCalc2p5Drb.cc#L257-L310](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L257-L310)<br>[SpaceChargeCalc2p5Drb.cc#L298-L304](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L298-L304)<br>[SpaceChargeCalc2p5Drb.cc#L313-L315](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L313-L315)<br>[wrap_spacechargecalc2p5d_rb.cc#L157-L166](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/wrap_spacechargecalc2p5d_rb.cc#L157-L166)<br>[SpaceChargeCalc2p5Drb.cc#L41](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L41-L41) |

`SpaceChargeCalc2p5Drb.setLongAveragingPointsN(n)` accepts any integer, but the longitudinal density derivative is a least-squares quadratic fit over n bins, whose normal equations are singular for n < 3; the determinant is exactly 0 for n = 1 and 2. With n = 2, 1, 0 or a negative value the energy kicks become NaN, or infinite near the bunch ends, without an error, while the transverse kicks stay finite; the default n = 3 and larger values give finite kicks. For a Gaussian bunch of 5000 macroparticles on 16 longitudinal bins, all 5000 energy kicks are NaN with n = 1; with n = 2, 4995 are NaN and 5 are −∞.

Run `python COL-15.py`.

??? example "Reproducer: COL-15.py"

    [Download COL-15.py](COL-15.py)

    ```python
    """COL-15: `SpaceChargeCalc2p5Drb.setLongAveragingPointsN(n)` with n < 3 makes the energy kicks NaN

    `SpaceChargeCalc2p5Drb.setLongAveragingPointsN(n)` accepts any integer, but the longitudinal density derivative is a least-squares quadratic fit over n bins, whose normal equations are singular for n < 3; the determinant is exactly 0 for n = 1 and 2. With n = 2, 1, 0 or a negative value the energy kicks become NaN, or infinite near the bunch ends, without an error, while the transverse kicks stay finite; the default n = 3 and larger values give finite kicks. For a Gaussian bunch of 5000 macroparticles on 16 longitudinal bins, all 5000 energy kicks are NaN with n = 1; with n = 2, 4995 are NaN and 5 are −∞.

    The script tracks one Gaussian bunch through `SpaceChargeCalc2p5Drb(32, 32, 16)` after setting the averaging count to −1, 0, 1, 2, 3, 5 and 40, and counts NaN and infinite energy kicks and non-finite horizontal kicks; counts of 3 and above are the controls. It also evaluates the determinant of the code's normal equations exactly for n = 1 to 4.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from fractions import Fraction

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.spacecharge import SpaceChargeCalc2p5Drb
    from orbit.utils.consts import mass_proton


    def det(n):
        # Determinant of the 3x3 normal matrix of y = a x^2 + b x + c over x = 1..n, written as in
        # calculateLongDerivative with S_j = sum x^j; rank <= n, so it vanishes for n < 3.
        s = [sum(Fraction(x) ** j for x in range(1, n + 1)) for j in range(5)]
        return (
            s[0] * s[2] * s[4]
            - s[1] ** 2 * s[4]
            - s[0] * s[3] ** 2
            + 2 * s[1] * s[2] * s[3]
            - s[2] ** 3
        )


    print(
        "normal-equation determinant for n = 1, 2, 3, 4: " + ", ".join(str(det(n)) for n in range(1, 5))
    )

    rng = np.random.default_rng(0)
    u = rng.normal(size=(5000, 3)) * np.array([1e-3, 1e-3, 1e-2])
    bad, ctl_ok = False, True
    for n in (-1, 0, 1, 2, 3, 5, 40):
        b = Bunch()
        b.mass(mass_proton)
        b.macroSize(1e9)
        b.getSyncParticle().kinEnergy(0.1)
        for x, y, z in u:
            b.addParticle(x, 0.0, y, 0.0, z, 0.0)
        calc = SpaceChargeCalc2p5Drb(32, 32, 16)
        calc.setLongAveragingPointsN(n)
        calc.trackBunch(b, 0.1, 0.05)
        de = np.array([b.dE(i) for i in range(b.getSize())])
        xp = np.array([b.xp(i) for i in range(b.getSize())])
        n_nan, n_inf = int(np.isnan(de).sum()), int(np.isinf(de).sum())
        n_bad_xp = int((~np.isfinite(xp)).sum())
        control = n >= 3
        # Expected 0: a fit with fewer than three points should be rejected or fall back, not yield
        # NaN (0/0 where the density is zero) or infinity (nonzero numerator over the zero determinant).
        print(
            f"setLongAveragingPointsN({n}){' (control)' if control else ''}: non-finite energy kicks observed "
            f"{n_nan + n_inf} of {b.getSize()} (NaN {n_nan}, infinite {n_inf}), non-finite xp {n_bad_xp}; expected 0 and 0"
        )
        ctl_ok &= n_bad_xp == 0
        if control:
            ctl_ok &= n_nan + n_inf == 0
        else:
            bad |= n_nan + n_inf > 0
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

