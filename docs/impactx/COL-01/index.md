---
title: "COL-01: PIC 2p5D longitudinal kick uses a fixed 1 m reference length r0, which Gauss2p5D exposes"
hide:
  - toc
---

# COL-01 — PIC `2p5D` longitudinal kick uses a fixed 1 m reference length `r0`, which `Gauss2p5D` exposes

| | |
|---|---|
| Code | [ImpactX](../index.md) · Collective effects |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-28 on development (`b701a6e9d1f9`), source, CPU, double, serial |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1718]], filed from this evidence |
| Source | [`GatherAndPush.cpp#L222-L229`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/spacecharge/GatherAndPush.cpp#L222-L229) |

With `2p5D` particle-in-cell space charge, the longitudinal kick multiplies the absolute 2D potential, whose logarithmic kernel implicitly fixes a reference length `r0 = 1 m`. The `Gauss2p5D` model exposes this length as `gauss_long_scale` (default `1.103·γ·σz`), but `2p5D` has no such parameter and does not document it. Scaling all lengths by a common factor at fixed charge and energy must leave the length-scaled kick unchanged; the transverse kick obeys this, while the longitudinal kick changes by 7.879% when lengths double. Against the exact 3D field of a long round Gaussian bunch at `γσz/σ⊥ = 50`, the on-axis longitudinal field has L2 errors of 73.62% (`σ⊥ = 1 mm`) and 131.1% (`σ⊥ = 0.1 mm`), and 149.9% in a 25 mm round pipe.

## Reproducer

Scale every length (bunch, mesh, kick length) by a at fixed charge and energy. Coulomb fields scale as 1/a^2, so a\*(kick) must be unchanged for every component. The transverse kicks pass; the longitudinal kick multiplies the absolute 2D potential, whose log reference is 1 m.

??? example "COL-01.py"

    [Download COL-01.py](COL-01.py)

    ```python
    """COL-01: PIC 2.5D longitudinal space-charge kick depends on a hidden 1 m length scale.

    Scale every length (bunch, mesh, kick length) by a at fixed charge and energy. Coulomb fields
    scale as 1/a^2, so a*(kick) must be unchanged for every component. The transverse kicks pass;
    the longitudinal kick multiplies the absolute 2D potential, whose log reference is 1 m.
    """
    import numpy as np
    from impactx import ImpactX, elements

    sim = ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.diagnostics = False
    sim.space_charge = "2p5D"
    sim.poisson_solver = "fft"
    sim.particle_shape = 2
    sim.strang_split = False
    sim.n_cell = [64, 64, 8]
    sim.space_charge_num_longitudinal_bins = 65
    sim.init_grids()

    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(16384, 3)) * [1e-4, 1e-4, 1e-3]  # sigma x, y, t in m
    xyz -= xyz.mean(axis=0)


    def scaled_kicks(a):
        beam = sim.beam
        beam.clear_particles()
        ref = beam.ref
        ref.reset()
        ref.set_species("electron").set_kin_energy_MeV(4.59899055)  # gamma = 10
        z = np.zeros(len(xyz))
        beam.add_n_particles(x=a * xyz[:, 0], y=a * xyz[:, 1], t=a * xyz[:, 2], px=z, py=z, pt=z,
                             qm=ref.qm_ratio_SI, bunch_charge=1e-12)
        kick_only = elements.Programmable(ds=1e-3 * a)  # one collective kick, no transport
        kick_only.push = lambda pc, step, period: None
        sim.lattice.clear()
        sim.lattice.append(kick_only)
        sim.track_particles()
        df = beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
        return a * df[["momentum_x", "momentum_t"]].to_numpy()


    try:
        k1, k2 = scaled_kicks(1.0), scaled_kicks(2.0)
        rel = np.linalg.norm(k2 - k1, axis=0) / np.linalg.norm(k1, axis=0)
        print(f"relative change of a*kick under a=1 -> 2: px {rel[0]:.4g}, pt {rel[1]:.4g}; expected 0 for both")
        print("DEFECT REPRODUCED" if rel[1] > 1e-6 and rel[0] < 1e-9 else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed (physics review of the reference length) | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`) |
| Confirmed (validity review) | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed (validity review) | `claude-opus-5-5` |  | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed (filing-text review and development-build numbers) | `gpt-6-astra` | xhigh | 2026-10-28 | development (`b701a6e9d1f9`), source, CPU, double, serial |

