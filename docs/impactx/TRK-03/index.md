---
title: "TRK-03: DipEdge(modify_ref_part=True) deflects the reference by the beam-normalized kick"
hide:
  - toc
---

# TRK-03 — `DipEdge(modify_ref_part=True)` deflects the reference by the beam-normalized kick

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-27 on ImpactX release [26.10](https://github.com/BLAST-ImpactX/impactx/tree/26.10) (`615921c`) |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1729]], filed from this evidence |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1730]], filed from this evidence |
| Source | [`DipEdge.H#L345-L353`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/DipEdge.H#L345-L353) |

With the nonlinear `DipEdge` model and `modify_ref_part=True`, ImpactX adds the fringe kick, normalized to the reference momentum P₀, directly to the reference particle's `px`, which is stored in units of mc. Since P₀/(mc) = βγ, the reference must instead be deflected by βγ times the kick the beam receives with `modify_ref_part=False`. For `psi` = 0.25 rad, `rc` = 1 m, `g` = 0.1 m and a reference entering along the axis, the reference `px` is −0.004474 at every energy instead of −0.007749 at γ = 2 and −0.04452 at γ = 10. The reference deflection is too small above βγ = 1 and too large below; the kick applied to the beam with `modify_ref_part=False` is not affected.

Run `python TRK-03.py`.

??? example "Reproducer: TRK-03.py"

    [Download TRK-03.py](TRK-03.py)

    ```python
    """TRK-03: `DipEdge(modify_ref_part=True)` deflects the reference by the beam-normalized kick

    With the nonlinear `DipEdge` model and `modify_ref_part=True`, ImpactX adds the fringe kick,
    normalized to the reference momentum P₀, directly to the reference particle's `px`, which is stored
    in units of mc. Since P₀/(mc) = βγ, the reference must instead be deflected by βγ times the kick the
    beam receives with `modify_ref_part=False`. For `psi` = 0.25 rad, `rc` = 1 m, `g` = 0.1 m and a
    reference entering along the axis, the reference `px` is −0.004474 at every energy instead of
    −0.007749 at γ = 2 and −0.04452 at γ = 10. The reference deflection is too small above βγ = 1 and
    too large below; the kick applied to the beam with `modify_ref_part=False` is not affected.

    The script pushes an on-axis proton through `DipEdge(psi=0.25, rc=1, g=0.1, model="nonlinear")` at
    γ = 2 and γ = 10, first with `modify_ref_part=False` to obtain the beam kick in `px`, then with
    `modify_ref_part=True`, and compares the reference `px` with βγ times that kick.
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


    edge = dict(psi=0.25, rc=1.0, g=0.1, model="nonlinear")
    bad = False
    try:
        for gamma in (2.0, 10.0):
            kin = (gamma - 1) * 938.27208816
            dpx = track([elements.DipEdge(**edge)], np.zeros(6), kin)[0, 1]
            track([elements.DipEdge(**edge, modify_ref_part=True)], np.zeros(6), kin)
            ref = sim.beam.ref
            # beam px = dPx/P0 and ref px = Px/(m c), so moving the same fringe deflection onto the
            # reference changes ref px by beta*gamma times the beam-side kick.
            print(f"gamma {gamma:g}: ref px observed {ref.px:.4g}, expected beta*gamma*dpx = {ref.beta_gamma * dpx:.4g}")
            bad |= abs(ref.px - ref.beta_gamma * dpx) > 1e-9
        print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-27</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-27</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

