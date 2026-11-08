---
title: "COL-06: Grid1D smoothed gather centres its quadratic stencil on the lower node, so it jumps at every node"
hide:
  - toc
---

# COL-06 — `Grid1D` smoothed gather centres its quadratic stencil on the lower node, so it jumps at every node

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [Grid1D.cc#L640-L665](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L640-L665)<br>[Grid1D.cc#L705-L733](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L705-L733)<br>[Grid1D.cc#L213-L224](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L213-L224)<br>[Grid1D.cc#L538-L558](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L538-L558)<br>[LSpaceChargeCalc.cc#L214-L216](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/LSpaceChargeCalc.cc#L214-L216) |

`Grid1D.getValueSmoothed` and `calcGradientSmoothed` centre their three-point quadratic stencil on the grid node at or below z, so the offset f from the centre runs from 0 to 1 cell, but the weights ½(½ ∓ f)² and ¾ − f² form the quadratic B-spline only for |f| ≤ ½; the matching deposit `binValueSmoothed` uses the nearest node. The interpolant therefore jumps at node i by (gᵢ₊₁ − 3gᵢ + 3gᵢ₋₁ − gᵢ₋₂)/8, with gⱼ the grid values, 0.7500 for gⱼ = j³, and deposit and gather are not reciprocal. With `setUseGrad(1)` and `setSmoothGrad(1)`, `LSpaceChargeCalc` gives a symmetric Gaussian bunch (rms length 12.5 m, 128 bins over 250 m) a kick whose even part dE(z) + dE(−z) reaches 2.674% of the peak instead of zero.

Run `python COL-06.py`.

??? example "Reproducer: COL-06.py"

    [Download COL-06.py](COL-06.py)

    ```python
    """COL-06: `Grid1D` smoothed gather centres its quadratic stencil on the lower node, so it jumps at every node

    `Grid1D.getValueSmoothed` and `calcGradientSmoothed` centre their three-point quadratic stencil on the grid node at or below z, so the offset f from the centre runs from 0 to 1 cell, but the weights ½(½ ∓ f)² and ¾ − f² form the quadratic B-spline only for |f| ≤ ½; the matching deposit `binValueSmoothed` uses the nearest node. The interpolant therefore jumps at node i by (gᵢ₊₁ − 3gᵢ + 3gᵢ₋₁ − gᵢ₋₂)/8, with gⱼ the grid values, 0.7500 for gⱼ = j³, and deposit and gather are not reciprocal. With `setUseGrad(1)` and `setSmoothGrad(1)`, `LSpaceChargeCalc` gives a symmetric Gaussian bunch (rms length 12.5 m, 128 bins over 250 m) a kick whose even part dE(z) + dE(−z) reaches 2.674% of the peak instead of zero.

    The script fills a 16-node `Grid1D` with gⱼ = j³ and measures the jump of the smoothed value across two nodes, deposits a unit charge at one point and gathers it at another and vice versa, and tracks a mirror-symmetric Gaussian bunch through the smoothed gradient branch of `LSpaceChargeCalc`, whose kick must be odd in z. The control is the same bunch through the linear gradient branch, which uses the same deposit.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.spacecharge import Grid1D, LSpaceChargeCalc
    from orbit.utils.consts import mass_proton
    from scipy.special import ndtri

    bad = False

    # Case 1: continuity at a node. A quadratic-spline interpolant is continuous. With the lower-node
    # stencil, the left limit at node i has weights (1/8, -1/4, 9/8) on nodes (i-2, i-1, i) and the right
    # limit (1/8, 3/4, 1/8) on (i-1, i, i+1), so right minus left = (g[i+1] - 3g[i] + 3g[i-1] - g[i-2])/8,
    # which is 6/8 for g_j = j^3.
    n = 16
    g = Grid1D(n, 0.0, 1.0)
    for j in range(n):
        g.setValue(float(j**3), j)
    for i in (5, 8):
        zn, eps = g.getGridZ(i), 1e-9
        jump = g.getValueSmoothed(zn + eps) - g.getValueSmoothed(zn - eps)
        pred = ((i + 1) ** 3 - 3 * i**3 + 3 * (i - 1) ** 3 - (i - 2) ** 3) / 8
        print(
            f"value jump across node {i}: observed {jump:.4f}, expected 0 (continuous quadratic spline); third difference / 8 = {pred:.4f}"
        )
        bad |= abs(jump) > 1e-3


    # Case 2: reciprocity. Depositing at z1 and gathering at z2 must equal the reverse,
    # sum_k W_k(z1) W_k(z2), when deposit and gather use the same B-spline stencil.
    def kernel(z1, z2):
        h = Grid1D(n, 0.0, 1.0)
        h.binValueSmoothed(1.0, z1)
        return h.getValueSmoothed(z2)


    k12, k21 = kernel(0.40, 0.47), kernel(0.47, 0.40)
    print(
        f"deposit-gather kernel K(0.40, 0.47) observed {k12:.6f}, expected K(0.47, 0.40) = {k21:.6f} (reciprocal kernel)"
    )
    bad |= abs(k12 - k21) > 1e-9

    # Case 3: gradient-branch longitudinal space charge on a mirror-symmetric bunch centred in the ring.
    C, NB, N, SIG = 250.0, 128, 20000, 12.5
    half = SIG * ndtri((np.arange(N // 2) + 0.5) / N)  # negative half of a quantile sample
    zs = np.concatenate([half, -half[::-1]])  # exactly mirror-symmetric, grid centre at 0


    def even_part(smooth):
        bb = Bunch()
        bb.mass(mass_proton)
        bb.macroSize(2e14 / N)
        bb.getSyncParticle().kinEnergy(1.0)
        for zi in zs:
            bb.addParticle(0.0, 0.0, 0.0, 0.0, zi, 0.0)
        calc = LSpaceChargeCalc(3.0, C, 0, 1, NB)
        calc.setUseGrad(1)
        calc.setSmoothGrad(smooth)
        calc.trackBunch(bb)
        de = np.array([bb.dE(i) for i in range(N)])
        return np.max(np.abs(de + de[::-1])) / np.max(np.abs(de))  # particle i mirrors particle N-1-i


    anti = even_part(1)
    print(
        f"smoothed-gradient LSpaceChargeCalc: max|dE(z) + dE(-z)|/max|dE| observed {anti:.4g}, expected 0 (odd kick)"
    )
    anti0 = even_part(0)
    print(
        f"control, linear-gradient LSpaceChargeCalc: max|dE(z) + dE(-z)|/max|dE| observed {anti0:.2e}, expected 0"
    )
    ctl_ok = anti0 < 1e-6
    bad |= anti > 1e-3
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

