---
title: "TRK-02: RFCavity spin transport rotates spin without a field and uses sqrt(1+pt²) for βγ"
hide:
  - toc
---

# TRK-02 — `RFCavity` spin transport rotates spin without a field and uses `sqrt(1+pt²)` for βγ

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX commit [`fa0a43b`](https://github.com/BLAST-ImpactX/impactx/commit/fa0a43b9bb07), committed 2026-10-28 |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`RFCavity.H#L383-L386`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/RFCavity.H#L383-L386) |

With spin tracking enabled (experimental in ImpactX), `RFCavity` rotates the spin incorrectly: one spin-coupling term omits the on-axis field profile, and two divide by √(1 + γ²) instead of the reference momentum βγ. Thomas–BMT spin precession vanishes where the field is zero and starts at the entrance with rate |dSₓ/ds| = E·`px`·(G + 1/(γ + 1)) for on-axis field E and magnetic anomaly G. For a proton at γ = 2, a cavity with `cos_coefficients=[0]` (no field) turns the spin from (0, 0, 1) to (−0.002864, −0.001091, 1.000); with E = 0.2 m⁻¹ and `px` = 0.001, the entrance rate is 3.294e-4 m⁻¹ instead of 4.252e-4 m⁻¹. Settings describing the same field give the same orbit but different spins; the orbit is unaffected.

Run `python TRK-02.py`.

??? example "Reproducer: TRK-02.py"

    [Download TRK-02.py](TRK-02.py)

    ```python
    """TRK-02: `RFCavity` spin transport rotates spin without a field and uses `sqrt(1+pt²)` for βγ

    With spin tracking enabled (experimental in ImpactX), `RFCavity` rotates the spin incorrectly: one
    spin-coupling term omits the on-axis field profile, and two divide by √(1 + γ²) instead of the
    reference momentum βγ. Thomas–BMT spin precession vanishes where the field is zero and starts at the
    entrance with rate |dSₓ/ds| = E·`px`·(G + 1/(γ + 1)) for on-axis field E and magnetic anomaly G. For
    a proton at γ = 2, a cavity with `cos_coefficients=[0]` (no field) turns the spin from (0, 0, 1) to
    (−0.002864, −0.001091, 1.000); with E = 0.2 m⁻¹ and `px` = 0.001, the entrance rate is 3.294e-4 m⁻¹
    instead of 4.252e-4 m⁻¹. Settings describing the same field give the same orbit but different spins;
    the orbit is unaffected.

    The script enables spin tracking and tracks a proton at 938.3 MeV kinetic energy (γ = 2) with
    initial spin (0, 0, 1). It first pushes an off-axis particle through a 0.7 m cavity with zero
    on-axis field, where the spin must stay (0, 0, 1); it then pushes a particle with `px` = 0.001
    through a 0.1 µm cavity with a constant profile (`cos_coefficients=[2]`, `escale` = 0.2) and
    compares |Sₓ|/L with the Thomas–BMT entrance rate. The printed ratio of the two equals βγ/√(1 + γ²).
    """
    import numpy as np
    from impactx import ImpactX, elements

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.spin = True
    sim.init_grids()


    def spin_after(cavity, coords):
        beam = sim.beam
        beam.clear_particles()
        ref = beam.ref
        ref.reset()
        ref.set_species("proton").set_kin_energy_MeV(938.27208816)
        x, px, y, py, t, pt = (np.array([c]) for c in coords)
        beam.add_n_particles(x=x, px=px, y=y, py=py, t=t, pt=pt, qm=ref.qm_ratio_SI,
                             bunch_charge=1e-12, sx=np.array([0.0]), sy=np.array([0.0]), sz=np.array([1.0]))
        sim.lattice.clear()
        sim.lattice.append(cavity)
        sim.track_particles()
        df = beam.to_df(local=True)
        return df[["spin_x", "spin_y", "spin_z"]].to_numpy()[0], ref


    try:
        u = [0.013, 0.021, -0.017, 0.008, 0.015, -0.03]
        s, _ = spin_after(elements.RFCavity(ds=0.7, escale=0.2, freq=1e8, phase=0.0, cos_coefficients=[0.0],
                                            sin_coefficients=[0.0], mapsteps=100), u)
        print(f"zero on-axis field: spin observed {np.round(s, 6).tolist()}, expected [0, 0, 1]")
        zero_field_bad = np.max(np.abs(s - [0, 0, 1])) > 1e-12

        L, px = 1e-7, 1e-3
        s, ref = spin_after(elements.RFCavity(ds=L, escale=0.2, freq=1e8, phase=0.0, cos_coefficients=[2.0],
                                              sin_coefficients=[0.0], mapsteps=10), [0, px, 0, 0, 0, 0])
        g = ref.gamma
        # On-axis field escale*ez with ez = cos_coefficients[0]/2 (the Fourier a0/2 term), phase 0. The
        # longitudinal E-field term of Thomas-BMT gives |dSx/ds| = escale*ez*px*(G + 1/(gamma+1)).
        expected = 0.2 * (2.0 / 2) * px * (ref.gyromagnetic_anomaly + 1 / (g + 1))
        ratio = abs(s[0] / L) / expected
        print(f"entrance |dSx/ds| observed {abs(s[0] / L):.4g}, expected {expected:.4g}; ratio {ratio:.4g} "
              f"(sqrt((g^2-1)/(g^2+1)) = {np.sqrt((g**2 - 1) / (g**2 + 1)):.4g})")
        print("DEFECT REPRODUCED" if zero_field_bad or abs(ratio - 1) > 1e-4 else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

