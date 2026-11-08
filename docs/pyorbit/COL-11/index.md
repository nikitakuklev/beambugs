---
title: "COL-11: A custom Boundary2D reports points outside its outline as inside"
hide:
  - toc
---

# COL-11 — A custom `Boundary2D` reports points outside its outline as inside

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [BaseBoundary2D.cc#L144-L152](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/BaseBoundary2D.cc#L144-L152)<br>[ShapedBoundary2D.cc#L284-L301](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/ShapedBoundary2D.cc#L284-L301)<br>[SpaceChargeCalc2p5D.cc#L110](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5D.cc#L110-L110)<br>[SpaceChargeCalcSliceBySlice2D.cc#L131](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcSliceBySlice2D.cc#L131-L131)<br>[wrap_boundary2d.cc#L215](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/wrap_boundary2d.cc#L215-L215)<br>[sc_2p5D_test.py#L58-L76](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/examples/SpaceCharge/sc2p5D/sc_2p5D_test.py#L58-L76) |

A `Boundary2D(nPoints, nModes)` built from user points answers `isInside(x, y)` with a test against the bounding rectangle of the points, not the closed outline through them. The 2.5-D and slice-by-slice calculators use this test to decide which particles receive a space-charge kick, so particles outside a custom non-rectangular boundary but inside its bounding box are kicked as if inside. For 128 points on a 10 mm circle, the point (9.5, 9.5) mm, 13.44 mm from the centre, is reported inside and receives a horizontal kick of 5.017 μrad in `SpaceChargeCalc2p5D`; the shaped `Boundary2D(..., "Circle", 0.02)` reports it outside and applies none. Rectangular outlines are exact.

Run `python COL-11.py`.

??? example "Reproducer: COL-11.py"

    [Download COL-11.py](COL-11.py)

    ```python
    """COL-11: A custom `Boundary2D` reports points outside its outline as inside

    A `Boundary2D(nPoints, nModes)` built from user points answers `isInside(x, y)` with a test against the bounding rectangle of the points, not the closed outline through them. The 2.5-D and slice-by-slice calculators use this test to decide which particles receive a space-charge kick, so particles outside a custom non-rectangular boundary but inside its bounding box are kicked as if inside. For 128 points on a 10 mm circle, the point (9.5, 9.5) mm, 13.44 mm from the centre, is reported inside and receives a horizontal kick of 5.017 μrad in `SpaceChargeCalc2p5D`; the shaped `Boundary2D(..., "Circle", 0.02)` reports it outside and applies none. Rectangular outlines are exact.

    The script builds a custom boundary from 128 points on a 10 mm circle and the shaped circle of the same radius, compares `isInside` at three points with the geometric answer, and tracks a Gaussian bunch with three probe particles through `SpaceChargeCalc2p5D(64, 64, 16)` with each boundary. The shaped circle and the probe inside the circle are the controls.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import math

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.spacecharge import Boundary2D, SpaceChargeCalc2p5D
    from orbit.utils.consts import mass_proton

    R, NP = 0.01, 128
    # The 128-gon through the points has inradius R cos(pi/128) = 9.997 mm, so points beyond that are outside.
    R_IN = R * math.cos(math.pi / NP)


    def boundaries():
        custom = Boundary2D(NP, 2)
        for i in range(NP):
            t = 2 * math.pi * i / NP
            custom.setBoundaryPoint(i, R * math.cos(t), R * math.sin(t))
        custom.initialize()
        shaped = Boundary2D(NP, 2, "Circle", 2 * R)  # shaped dimensions are diameters
        return custom, shaped


    custom, shaped = boundaries()
    bad, ctl_ok = False, True
    for x, y in ((0.0, 0.0), (0.0095, 0.0095), (0.011, 0.0)):
        expected = 1 if math.hypot(x, y) < R_IN else -1
        obs, ctl = custom.isInside(x, y), shaped.isInside(x, y)
        print(
            f"isInside({x * 1e3:g}, {y * 1e3:g}) mm, r = {math.hypot(x, y) * 1e3:.2f} mm: custom observed {obs:+d}, "
            f"shaped circle (control) {ctl:+d}, expected {expected:+d}"
        )
        bad |= obs != expected
        ctl_ok &= ctl == expected

    # Consequence in the 2.5-D kick: a probe outside the circle must receive no kick.
    rng = np.random.default_rng(5)
    u = rng.normal(size=(20000, 3)) * np.array([2e-3, 2e-3, 5e-3])
    probes = ((5e-3, 0.0), (9.5e-3, 9.5e-3), (8e-3, 8e-3))
    kicks = {}
    for tag in ("custom", "shaped"):
        bd = boundaries()[0 if tag == "custom" else 1]
        b = Bunch()
        b.mass(mass_proton)
        b.macroSize(1e9 / u.shape[0])
        b.getSyncParticle().kinEnergy(0.1)
        for x, y, z in u:
            b.addParticle(x, 0.0, y, 0.0, z, 0.0)
        idx = []
        for px, py in probes:
            idx.append(b.getSize())
            b.addParticle(px, 0.0, py, 0.0, 0.0, 0.0)
        SpaceChargeCalc2p5D(64, 64, 16).trackBunch(b, 0.1, bd)
        kicks[tag] = [(b.xp(i), b.yp(i)) for i in idx]
    for (px, py), kc, ks in zip(probes, kicks["custom"], kicks["shaped"]):
        r = math.hypot(px, py)
        inside = r < R_IN
        print(
            f"SpaceChargeCalc2p5D probe ({px * 1e3:g}, {py * 1e3:g}) mm, r = {r * 1e3:.2f} mm: custom boundary "
            f"(xp, yp) observed ({kc[0]:.4g}, {kc[1]:.4g}) rad, shaped circle ({ks[0]:.4g}, {ks[1]:.4g}) rad; "
            f"expected {'the shaped-circle kick' if inside else '(0, 0), outside the boundary'}"
        )
        if inside:
            ctl_ok &= abs(kc[0] - ks[0]) < 1e-3 * abs(ks[0])
        else:
            bad |= kc != (0.0, 0.0)
            ctl_ok &= ks == (0.0, 0.0)
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

