---
title: "TRK-08: Sol(ks=0) returns NaN positions"
hide:
  - toc
---

# TRK-08 — `Sol(ks=0)` returns NaN positions

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Source | [`Sol.H#L112`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/Sol.H#L112) |

A `Sol` with zero strength, `ks=0`, as when a solenoid is switched off in a scan or knob, returns NaN `x` and `y`, because the map divides by `ks/2`. The zero-field limit is a drift (`sin(αL)/α → L`), which `Sol(ks=1e-9)` and `Drift` both give. In the reproducer the particle exits with NaN `x` and `y` instead of `x = 0.0277`, `y = −0.0114`; the momenta and longitudinal coordinates are correct. Any lattice that contains a switched-off `Sol` is affected.

## Reproducer

sin(ks L/2)/(ks/2) -\> L as ks -\> 0, so a switched-off solenoid is the linear drift.

??? example "TRK-08.py"

    [Download TRK-08.py](TRK-08.py)

    ```python
    """TRK-08: Sol with ks = 0 returns NaN positions instead of a drift.

    sin(ks L/2)/(ks/2) -> L as ks -> 0, so a switched-off solenoid is the linear drift."""
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
        out = track([elements.Sol(ds=0.7, ks=0.0)], u)[0]
        exp = track([elements.Drift(ds=0.7)], u)[0]
        print(f"observed {np.round(out, 4).tolist()}\nexpected {np.round(exp, 4).tolist()}")
        print("DEFECT REPRODUCED" if not np.allclose(out, exp, atol=1e-12) else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

