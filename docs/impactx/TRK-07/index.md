---
title: "TRK-07: ExactCFbend with k_normal[0]=0 returns NaN"
hide:
  - toc
---

# TRK-07 — `ExactCFbend` with `k_normal[0]=0` returns NaN

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX commit [`fa0a43b`](https://github.com/BLAST-ImpactX/impactx/commit/fa0a43b9bb07), committed 2026-10-28 |
| Source | [`ExactCFbend.H#L340-L342`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactCFbend.H#L340-L342) |

An `ExactCFbend` with zero dipole strength, `k_normal[0]=0`, and any coefficient slot above the dipole, even one set to zero, returns NaN in `x`, `px`, `y`, `py` and `t`, because its multipole kick divides by a bend radius stored as zero. At zero curvature the element must reduce to the corresponding straight multipole; for a quadrupole that is `ExactQuad`, which agrees with an independent integration of the square-root (exact) Hamiltonian. For a proton with all six coordinates nonzero, `ExactCFbend(ds=0.7, k_normal=[0, 0.8])` gives NaN, while `ExactQuad(ds=0.7, k=0.8)` with the same integrator gives `x` = 0.02394 m, `px` = 0.01032 and `y` = −0.01456 m. An element with only the dipole coefficient, `k_normal=[0]`, stays finite.

Run `python TRK-07.py`.

??? example "Reproducer: TRK-07.py"

    [Download TRK-07.py](TRK-07.py)

    ```python
    """TRK-07: `ExactCFbend` with `k_normal[0]=0` returns NaN

    An `ExactCFbend` with zero dipole strength, `k_normal[0]=0`, and any coefficient slot above the
    dipole, even one set to zero, returns NaN in `x`, `px`, `y`, `py` and `t`, because its multipole
    kick divides by a bend radius stored as zero. At zero curvature the element must reduce to the
    corresponding straight multipole; for a quadrupole that is `ExactQuad`, which agrees with an
    independent integration of the square-root (exact) Hamiltonian. For a proton with all six
    coordinates nonzero, `ExactCFbend(ds=0.7, k_normal=[0, 0.8])` gives NaN, while `ExactQuad(ds=0.7,
    k=0.8)` with the same integrator gives `x` = 0.02394 m, `px` = 0.01032 and `y` = −0.01456 m. An
    element with only the dipole coefficient, `k_normal=[0]`, stays finite.

    The script tracks one proton (938.3 MeV kinetic energy) with all six coordinates nonzero through
    `ExactCFbend(ds=0.7, k_normal=[0, 0.8], k_skew=[0, 0])` and through `ExactQuad(ds=0.7, k=0.8)`, both
    with a sixth-order integrator and 50 map steps, and compares the two exit vectors.
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


    u = [0.013, 0.021, -0.017, 0.008, 0.015, -0.03]
    try:
        out = track([elements.ExactCFbend(ds=0.7, k_normal=[0.0, 0.8], k_skew=[0.0, 0.0], int_order=6, mapsteps=50)], u)[0]
        exp = track([elements.ExactQuad(ds=0.7, k=0.8, int_order=6, mapsteps=50)], u)[0]
        print(f"observed {np.round(out, 4).tolist()}\nexpected {np.round(exp, 4).tolist()}")
        print("DEFECT REPRODUCED" if not np.allclose(out, exp, atol=1e-8) else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

