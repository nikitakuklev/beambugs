---
title: "TRK-01: ExactCFbend ignores the skew dipole k_skew[0]"
hide:
  - toc
---

# TRK-01 — `ExactCFbend` ignores the skew dipole `k_skew[0]`

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1733]], filed from this evidence |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1734]], filed from this evidence |
| Source | [`ExactCFbend.H#L349-L366`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactCFbend.H#L349-L366) |

`ExactCFbend` omits `k_skew[0]` from its orbital kick loop, so the horizontal field of a skew dipole produces no vertical steering. The element's field model gives `Bx/(Bρ) = ks/(1+hx)`, with `h` the curvature, so an on-axis particle must gain `Δpy = ks·L`. For an on-axis proton in `ExactCFbend(ds=0.7, k_normal=[0.3], k_skew=[0.02])`, `py` is 0 instead of 0.01400 with one and with four slices. Users who model a skew dipole or a vertical corrector inside an `ExactCFbend` get no vertical kick from it.

## Reproducer

Curvilinear skew dipole: Bx/(B rho) = ks/(1+h x), so dpy/ds = (1+h x) Bx/(B rho) = ks and an on-axis particle leaves with py = ks\*L.

??? example "TRK-01.py"

    [Download TRK-01.py](TRK-01.py)

    ```python
    """TRK-01: ExactCFbend ignores the skew dipole k_skew[0] in orbital tracking.

    Curvilinear skew dipole: Bx/(B rho) = ks/(1+h x), so dpy/ds = (1+h x) Bx/(B rho) = ks and an
    on-axis particle leaves with py = ks*L."""
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


    L, ks = 0.7, 0.02
    try:
        out = track([elements.ExactCFbend(ds=L, k_normal=[0.3], k_skew=[ks], int_order=4, mapsteps=20)], np.zeros(6))
        print(f"py observed {out[0, 3]:.4g}, expected ks*L = {ks * L:.4g}")
        print("DEFECT REPRODUCED" if abs(out[0, 3] - ks * L) > 1e-6 else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

