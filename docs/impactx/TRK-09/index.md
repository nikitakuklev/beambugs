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
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX commit [`fa0a43b`](https://github.com/BLAST-ImpactX/impactx/commit/fa0a43b9bb07), committed 2026-10-28 |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1050]] |
| Source | [`ExactCFbend.H#L375-L389`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactCFbend.H#L375-L389) |

`ExactCFbend` evaluates its octupole and decapole kicks as differences of order-one terms in 1 + x/`rc` and then multiplies them by powers of the bend radius `rc`, so rounding dominates at small curvature: the error scales as ε·`rc`³·k₃ for the octupole and ε·`rc`⁴·k₄ for the decapole, with k₃ and k₄ their coefficients and ε the double-precision machine epsilon. In the straight limit at y = 0 the decapole kick rate is −k₄x⁴/24, which a 60-digit evaluation of the code's own expression reproduces. For `rc` = 1000 m, `ds` = 1e-4 m, k₄ = 1e4 m⁻⁵ and a particle at x = 1 mm, the decapole changes `px` by −6.889e-6 instead of −4.167e-14.

Run `python TRK-09.py`.

??? example "Reproducer: TRK-09.py"

    [Download TRK-09.py](TRK-09.py)

    ```python
    """TRK-09: `ExactCFbend` octupole and decapole kicks lose all precision at small curvature

    `ExactCFbend` evaluates its octupole and decapole kicks as differences of order-one terms in 1 +
    x/`rc` and then multiplies them by powers of the bend radius `rc`, so rounding dominates at small
    curvature: the error scales as ε·`rc`³·k₃ for the octupole and ε·`rc`⁴·k₄ for the decapole, with k₃
    and k₄ their coefficients and ε the double-precision machine epsilon. In the straight limit at y = 0
    the decapole kick rate is −k₄x⁴/24, which a 60-digit evaluation of the code's own expression
    reproduces. For `rc` = 1000 m, `ds` = 1e-4 m, k₄ = 1e4 m⁻⁵ and a particle at x = 1 mm, the decapole
    changes `px` by −6.889e-6 instead of −4.167e-14.

    The script pushes a proton (938.3 MeV kinetic energy) at x = 1 mm with zero momenta through a 0.1 mm
    `ExactCFbend` with `k_normal[0]` = 1e-3 m⁻¹ (`rc` = 1000 m), once with k₄ = 1e4 m⁻⁵ and once
    without, using one second-order map step. It compares the difference in exit `px` with
    −`ds`·k₄x⁴/24.
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
        # The curvilinear decapole kick at y = 0 is k4*x^4/24 + O(k4 x^5/rc); the code evaluates it as a
        # difference of O(1) polynomial and log terms in rho = 1 + x/rc, multiplied by rc^4*k4.
        expected = -ds * k4 * x**4 / 24
        print(f"rc = 1000 m: decapole px change observed {with_k4 - without:.4g}, expected {expected:.4g}")
        print("DEFECT REPRODUCED" if abs(with_k4 - without - expected) > 1e-3 * abs(expected) else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-27</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-27</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

