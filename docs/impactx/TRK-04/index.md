---
title: "TRK-04: Soft-edge elements and their reverse() do not cancel in a lattice"
hide:
  - toc
---

# TRK-04 — Soft-edge elements and their `reverse()` do not cancel in a lattice

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1731]], filed from this evidence |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1732]], filed from this evidence |
| Source | [`SoftQuad.H#L548-L555`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/SoftQuad.H#L548-L555)<br>[`SoftSol.H#L648-L655`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/SoftSol.H#L648-L655)<br>[`RFCavity.H#L604-L611`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/RFCavity.H#L604-L611)<br>[`tracking/common.H#L168-L170`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/tracking/common.H#L168-L170) |

A `SoftQuadrupole`, `SoftSolenoid` or `RFCavity` followed by its `reverse()` copy in a lattice does not return particles to their initial coordinates, although the manual documents `reverse()` as undoing the original element. Lattice tracking resets the field-profile origin at each element, so the reversed element (negative `ds`) integrates outside its field profile; with a flat profile it reduces to a drift of length −`ds`. Pushing the reversed element directly, which keeps the forward element's profile origin, closes the round trip to 2.1e-16. With a constant profile (`cos_coefficients=[2]`, `ds` = 0.7 m), the largest coordinate error after the pair, in metres or normalized momentum, is 0.01080 for `SoftQuadrupole`, 0.004498 for `SoftSolenoid` and 0.002723 for `RFCavity`. Any lattice containing a reversed soft-edge element is affected.

??? example "Reproducer: TRK-04.py"

    [Download TRK-04.py](TRK-04.py)

    ```python
    """TRK-04: Soft-edge elements and their `reverse()` do not cancel in a lattice

    A `SoftQuadrupole`, `SoftSolenoid` or `RFCavity` followed by its `reverse()` copy in a lattice does
    not return particles to their initial coordinates, although the manual documents `reverse()` as
    undoing the original element. Lattice tracking resets the field-profile origin at each element, so
    the reversed element (negative `ds`) integrates outside its field profile; with a flat profile it
    reduces to a drift of length −`ds`. Pushing the reversed element directly, which keeps the forward
    element's profile origin, closes the round trip to 2.1e-16. With a constant profile
    (`cos_coefficients=[2]`, `ds` = 0.7 m), the largest coordinate error after the pair, in metres or
    normalized momentum, is 0.01080 for `SoftQuadrupole`, 0.004498 for `SoftSolenoid` and 0.002723 for
    `RFCavity`. Any lattice containing a reversed soft-edge element is affected.

    For each of the three classes, the script tracks one proton (938.3 MeV kinetic energy) with all six
    coordinates nonzero through a two-element lattice, the element and its `reverse()` copy, with a
    constant profile, `ds` = 0.7 m and 100 map steps. It reports the largest difference between final
    and initial coordinates, which must be 0. The constant profile rules out profile asymmetry as an
    explanation.
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
    profile = dict(ds=0.7, cos_coefficients=[2.0], sin_coefficients=[0.0], mapsteps=100)
    cases = [
        ("SoftQuadrupole", dict(gscale=0.8)),
        ("SoftSolenoid", dict(bscale=0.8)),
        ("RFCavity", dict(escale=0.2, freq=1e8, phase=0.0)),
    ]
    bad = False
    try:
        for name, kw in cases:
            el = getattr(elements, name)(**profile, **kw)
            inv = el.copy()
            inv.reverse()
            out = track([el, inv], u)
            err = np.max(np.abs(out[0] - u))
            print(f"{name}: max |final - initial| = {err:.4g}, expected 0")
            bad |= err > 1e-12
        print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

