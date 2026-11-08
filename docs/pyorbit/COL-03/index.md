---
title: "COL-03: SpaceChargeCalc2p5Drb takes the beam's vertical centre from the horizontal grid, so its energy kick depends on the transverse beam position"
hide:
  - toc
---

# COL-03 — `SpaceChargeCalc2p5Drb` takes the beam's vertical centre from the horizontal grid, so its energy kick depends on the transverse beam position

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [SpaceChargeCalc2p5Drb.cc#L229-L255](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L229-L255)<br>[SpaceChargeCalc2p5Drb.cc#L238](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L238-L238)<br>[SpaceChargeCalc2p5Drb.cc#L146-L167](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalc2p5Drb.cc#L146-L167) |

`SpaceChargeCalc2p5Drb` scales its energy kick by G(r) = 1 + 2ln(b/a) − r²/a², with a the fitted beam radius, b the pipe radius and r the distance from the fitted beam centre, but it reads the vertical coordinate of each grid row from the horizontal grid. On a square grid with the default aspect ratio, the fitted vertical centre is therefore off by x_min − y_min, the difference of the two grids' lower limits, for centred beams too; other grids also scale the fitted vertical spread. For a round Gaussian beam with σ = 1.5 mm in a 20 mm pipe, a rigid 3 mm horizontal displacement changes the energy kicks by up to 32.84% of their peak and a 3 mm vertical one by 32.98%, instead of not at all.

Run `python COL-03.py`.

??? example "Reproducer: COL-03.py"

    [Download COL-03.py](COL-03.py)

    ```python
    """COL-03: `SpaceChargeCalc2p5Drb` takes the beam's vertical centre from the horizontal grid, so its energy kick depends on the transverse beam position

    `SpaceChargeCalc2p5Drb` scales its energy kick by G(r) = 1 + 2ln(b/a) − r²/a², with a the fitted beam radius, b the pipe radius and r the distance from the fitted beam centre, but it reads the vertical coordinate of each grid row from the horizontal grid. On a square grid with the default aspect ratio, the fitted vertical centre is therefore off by x_min − y_min, the difference of the two grids' lower limits, for centred beams too; other grids also scale the fitted vertical spread. For a round Gaussian beam with σ = 1.5 mm in a 20 mm pipe, a rigid 3 mm horizontal displacement changes the energy kicks by up to 32.84% of their peak and a 3 mm vertical one by 32.98%, instead of not at all.

    The script tracks a round Gaussian beam, made symmetric under x ↔ y so that both grids coincide, through `SpaceChargeCalc2p5Drb(64, 64, 32)` centred and rigidly displaced, and compares the energy kicks with those of the centred beam. A diagonal displacement keeps the two grids' offsets equal and is the control. A last case tracks an undisplaced beam without the symmetrization and its x ↔ y mirror image, which must receive identical energy kicks because the model is symmetric in x and y.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.spacecharge import SpaceChargeCalc2p5Drb
    from orbit.utils.consts import mass_proton

    rng = np.random.default_rng(7)
    M = 10000
    xy = 1.5e-3 * rng.normal(size=(M, 2))
    z = 5e-3 * rng.normal(size=M)
    # Symmetric under x <-> y: include each point and its mirror, so the x and y extrema coincide.
    X = np.concatenate([xy[:, 0], xy[:, 1]])
    Y = np.concatenate([xy[:, 1], xy[:, 0]])
    Z = np.concatenate([z, z])


    def run(xs, ys, zs, dx=0.0, dy=0.0):
        b = Bunch()
        b.mass(mass_proton)
        b.macroSize(1e10 / xs.size)
        b.getSyncParticle().kinEnergy(0.05)
        for i in range(xs.size):
            b.addParticle(xs[i] + dx, 0.0, ys[i] + dy, 0.0, zs[i], 0.0)
        SpaceChargeCalc2p5Drb(64, 64, 32).trackBunch(b, 0.1, 0.02)  # length 0.1 m, pipe radius 20 mm
        return np.array([b.dE(i) for i in range(b.getSize())])


    ref = run(X, Y, Z)
    peak = np.max(np.abs(ref))
    bad, ctl_ok = False, True
    for dx, dy, label in (
        (3e-3, 0.0, "horizontal 3 mm"),
        (0.0, 3e-3, "vertical 3 mm"),
        (3e-3, 3e-3, "diagonal 3 mm (control)"),
    ):
        d = np.max(np.abs(run(X, Y, Z, dx, dy) - ref)) / peak
        # Expected 0: the model kick depends only on (x - x_c, y - y_c), a and dlambda/dz.
        print(
            f"displacement {label}: max|dE - dE(centred)|/peak observed {d:.4f}, expected 0 (rigid shift)"
        )
        if dx and dy:
            ctl_ok = d < 1e-6
        else:
            bad |= d > 0.01

    # Undisplaced beam without symmetrization: its x and y extrema differ. Swapping x and y swaps the
    # grids and leaves r^2 = (x - x_c)^2 + (y - y_c)^2 and a unchanged, so dE must not change.
    xa, ya = xy[:, 0], xy[:, 1]
    d1, d2 = run(xa, ya, z), run(ya, xa, z)
    print(
        f"undisplaced beam, x min - y min = {(float(np.min(xa)) - float(np.min(ya))) * 1e3:+.3f} mm: max|dE - dE(x <-> y)|/peak "
        f"observed {np.max(np.abs(d1 - d2)) / np.max(np.abs(d1)):.4f}, expected 0 (model symmetric in x and y)"
    )
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

