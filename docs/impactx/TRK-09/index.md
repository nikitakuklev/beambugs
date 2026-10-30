---
title: "TRK-09: ExactCFbend octupole and decapole kicks lose all precision at small curvature"
hide:
  - toc
---

# TRK-09 — `ExactCFbend` octupole and decapole kicks lose all precision at small curvature

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1050]] |
| Source | [`ExactCFbend.H#L375-L389`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactCFbend.H#L375-L389) |

`ExactCFbend` evaluates its octupole and decapole kicks through cancelling order-one terms in `1 + x/rc` and then multiplies by powers of the bend radius `rc`, so rounding dominates at small curvature: the error scales as `ε·rc³·k3` for the octupole and `ε·rc⁴·k4` for the decapole, with ε the double-precision epsilon. In the straight limit at `y = 0` the decapole kick rate is `−k4·x⁴/24`, which a 60-digit evaluation of the same expression reproduces. For `ds = 1e-4 m`, `rc = 1000 m`, `k4 = 1e4 m⁻⁵` and `x = 1 mm`, the decapole changes `px` by −6.889e-6 instead of −4.167e-14. Weakly curved combined-function magnets with these terms are affected.

## Reproducer

The curvilinear decapole kick at y = 0 is k4\*x^4/24 + O(k4 x^5/rc); the code evaluates it as a difference of O(1) polynomial and log terms in rho = 1 + x/rc, multiplied by rc^4\*k4. Expected: the decapole changes px by -ds\*k4\*x^4/24 relative to the same bend without it.

??? example "TRK-09.py"

    [Download TRK-09.py](TRK-09.py)

    ```python
    """TRK-09: ExactCFbend higher-order multipole kicks lose all precision at small curvature.

    The curvilinear decapole kick at y = 0 is k4*x^4/24 + O(k4 x^5/rc); the code evaluates it as a
    difference of O(1) polynomial and log terms in rho = 1 + x/rc, multiplied by rc^4*k4.
    Expected: the decapole changes px by -ds*k4*x^4/24 relative to the same bend without it.
    """
    import numpy as np
    from impactx import ImpactX, elements

    COLS = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.init_grids()


    def track(lattice, coords, kin_MeV=938.27208816):
        """Track proton(s) with (x, px, y, py, t, pt) through a lattice; return survivors."""
        beam = sim.beam
        beam.clear_particles()
        ref = beam.ref
        ref.reset()
        ref.set_species("proton").set_kin_energy_MeV(kin_MeV)
        u = np.atleast_2d(np.asarray(coords, dtype=float))
        beam.add_n_particles(
            x=u[:, 0].copy(), px=u[:, 1].copy(), y=u[:, 2].copy(),
            py=u[:, 3].copy(), t=u[:, 4].copy(), pt=u[:, 5].copy(),
            qm=ref.qm_ratio_SI, bunch_charge=1e-12,
        )
        sim.lattice.clear()
        sim.lattice.extend(lattice)
        sim.track_particles()
        df = beam.to_df(local=True)
        return np.zeros((0, 6)) if df is None else df[COLS].to_numpy(copy=True)


    ds, x, k4 = 1e-4, 1e-3, 1e4
    try:
        base = dict(ds=ds, k_skew=[0.0] * 5, int_order=2, mapsteps=1)
        with_k4 = track([elements.ExactCFbend(k_normal=[1e-3, 0, 0, 0, k4], **base)], [x, 0, 0, 0, 0, 0])[0, 1]
        without = track([elements.ExactCFbend(k_normal=[1e-3, 0, 0, 0, 0], **base)], [x, 0, 0, 0, 0, 0])[0, 1]
        expected = -ds * k4 * x**4 / 24
        print(f"rc = 1000 m: decapole px change observed {with_k4 - without:.4g}, expected {expected:.4g}")
        print("DEFECT REPRODUCED" if abs(with_k4 - without - expected) > 1e-3 * abs(expected) else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

