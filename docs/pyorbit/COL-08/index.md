---
title: "COL-08: SpaceChargeCalcUnifEllipse(nEllipses > 1) changes its field when weights are stored per particle or the bunch is displaced"
hide:
  - toc
---

# COL-08 — `SpaceChargeCalcUnifEllipse(nEllipses > 1)` changes its field when weights are stored per particle or the bunch is displaced

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [SpaceChargeCalcUnifEllipse.cc#L204-L278](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L204-L278)<br>[SpaceChargeCalcUnifEllipse.cc#L217-L218](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L217-L218)<br>[SpaceChargeCalcUnifEllipse.cc#L228-L229](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L228-L229)<br>[SpaceChargeCalcUnifEllipse.cc#L243](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L243-L243)<br>[SpaceChargeCalcUnifEllipse.cc#L257](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L257-L257)<br>[SpaceChargeCalcUnifEllipse.cc#L271-L272](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L271-L272)<br>[SpaceChargeCalcUnifEllipse.cc#L157-L159](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcUnifEllipse.cc#L157-L159) |

With more than one ellipsoid, `SpaceChargeCalcUnifEllipse` sorts particles into radial shells using raw, uncentred coordinates, with shell index int(nρ) − 1 for a scalar macro size but int(nρ) for a per-particle `macrosize` attribute, where n is the number of ellipsoids and ρ the normalized ellipsoidal radius. The same bunch therefore gets different fields when its identical weights are stored per particle or when it is displaced rigidly, and the scalar path itself misweights the shells. For a uniform ellipsoid, where a single ellipsoid gives the exact field, three ellipsoids give a core field of 1.038 times the exact value with a scalar macro size, 0.3135 times with the same weights stored per particle and 0.2138 times after a 3 mm horizontal displacement.

Run `python COL-08.py`.

??? example "Reproducer: COL-08.py"

    [Download COL-08.py](COL-08.py)

    ```python
    """COL-08: `SpaceChargeCalcUnifEllipse(nEllipses > 1)` changes its field when weights are stored per particle or the bunch is displaced

    With more than one ellipsoid, `SpaceChargeCalcUnifEllipse` sorts particles into radial shells using raw, uncentred coordinates, with shell index int(nρ) − 1 for a scalar macro size but int(nρ) for a per-particle `macrosize` attribute, where n is the number of ellipsoids and ρ the normalized ellipsoidal radius. The same bunch therefore gets different fields when its identical weights are stored per particle or when it is displaced rigidly, and the scalar path itself misweights the shells. For a uniform ellipsoid, where a single ellipsoid gives the exact field, three ellipsoids give a core field of 1.038 times the exact value with a scalar macro size, 0.3135 times with the same weights stored per particle and 0.2138 times after a 3 mm horizontal displacement.

    The script samples a uniform solid ellipsoid with semiaxes (2, 1, 3) mm and tracks it, with probes at 0.4 of the x and z semiaxes, through `SpaceChargeCalcUnifEllipse` with one, three and five ellipsoids, with scalar and per-particle weights and rigidly displaced. It prints the probe kicks relative to the single-ellipsoid kick of the undisplaced bunch, which is exact for a uniform ellipsoid; the single-ellipsoid runs with per-particle weights or a displacement are the controls.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.spacecharge import SpaceChargeCalcUnifEllipse
    from orbit.utils.consts import mass_proton

    N, AX, Q = 50000, np.array([2e-3, 1e-3, 3e-3]), 1e9
    rng = np.random.default_rng(11)
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v *= rng.uniform(size=(N, 1)) ** (1 / 3) * AX  # uniform solid ellipsoid
    PROBES = [(0.4 * AX[0], 0.0, 0.0), (0.0, 0.0, 0.4 * AX[2])]


    def run(n_ell, per_particle, shift_x):
        b = Bunch()
        b.mass(mass_proton)
        b.macroSize(Q / N)
        b.getSyncParticle().kinEnergy(0.002)
        for p in v:
            b.addParticle(p[0] + shift_x, 0.0, p[1], 0.0, p[2], 0.0)
        idx = []
        for x, y, z in PROBES:
            idx.append(b.getSize())
            b.addParticle(x + shift_x, 0.0, y, 0.0, z, 0.0)
        if per_particle:
            b.addPartAttr("macrosize")
            for i in range(b.getSize()):
                b.partAttrValue("macrosize", i, 0, Q / N)  # same weight as the scalar macro size
        SpaceChargeCalcUnifEllipse(n_ell).trackBunch(b, 0.01)
        return np.array([b.xp(idx[0]), b.dE(idx[1])])


    ref = run(1, False, 0.0)
    bad, ctl_ok = False, True
    for n_ell, pp, sx, label in (
        (1, True, 0.0, "1 ellipsoid, per-particle weights (control)"),
        (1, False, 3e-3, "1 ellipsoid, displaced 3 mm (control)"),
        (3, False, 0.0, "3 ellipsoids, scalar weights"),
        (3, True, 0.0, "3 ellipsoids, per-particle weights"),
        (3, False, 3e-3, "3 ellipsoids, displaced 3 mm"),
        (5, False, 0.0, "5 ellipsoids, scalar weights"),
        (5, True, 0.0, "5 ellipsoids, per-particle weights"),
    ):
        r = run(n_ell, pp, sx) / ref
        # Expected 1: for a uniform ellipsoid the shell fit should reproduce the single-ellipsoid
        # (exact) field, independent of the weight storage and of a rigid displacement.
        print(
            f"{label}: xp(0.4a, 0, 0) ratio observed {r[0]:.4f}, dE(0, 0, 0.4c) ratio {r[1]:.4f}; "
            f"expected 1 (uniform ellipsoid)"
        )
        if n_ell == 1:
            ctl_ok &= bool(np.all(np.abs(r - 1) < 1e-6))
        else:
            bad |= bool(np.any(np.abs(r - 1) > 0.02))
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

