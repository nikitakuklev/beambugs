---
title: "TRK-06: Linear CFbend returns NaN at a zero focusing frequency"
hide:
  - toc
---

# TRK-06 — Linear `CFbend` returns NaN at a zero focusing frequency

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P2 |
| Severity | Silent wrong result; crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Discussed in | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1664#issuecomment-5706911843]] |
| Source | [`CFbend.H#L133-L140`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/CFbend.H#L133-L140) |

The linear `CFbend` returns NaN coordinates at valid zero focusing frequencies: `k = 0` gives NaN `y` and `py`, and `k = −1/rc²` (horizontal focusing cancelled, `k1 = −h²`) gives NaN `x`, `px` and `t`. The transfer functions have finite zero-frequency limits, and at `k = 0` the element must agree with `Sbend`. Near cancellation, division by the small horizontal focusing coefficient gives finite but wrong results: for `rc = 1/0.3 m`, `k = −0.09 m⁻²` and `ds = 0.7 m`, `x` is 0.0277 m instead of 0.0302 m and `t` is 0.0112 m instead of 0.0131 m. Zero-gradient bends and combined-function bends tuned to cancel horizontal focusing are affected.

## Reproducer

k = 0 is a sector bend (equal to Sbend); k = -1/rc^2 cancels horizontal focusing. Both maps are finite: sin(w L)/w -\> L, (1 - cos)/w^2 -\> L^2/2. A neighbouring k gives the limit. Exactly zero gives NaN; a rounding residue (h = 0.3: k = -h^2 leaves gx = -1.4e-17) gives finite wrong x and t.

??? example "TRK-06.py"

    [Download TRK-06.py](TRK-06.py)

    ```python
    """TRK-06: linear CFbend returns NaN when a focusing frequency is zero.

    k = 0 is a sector bend (equal to Sbend); k = -1/rc^2 cancels horizontal focusing. Both maps are
    finite: sin(w L)/w -> L, (1 - cos)/w^2 -> L^2/2. A neighbouring k gives the limit. Exactly zero
    gives NaN; a rounding residue (h = 0.3: k = -h^2 leaves gx = -1.4e-17) gives finite wrong x and t."""
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
    bad = False
    try:
        for rc, k, ref_el in [
            (3.0, 0.0, elements.Sbend(ds=0.7, rc=3.0)),
            (2.0, -0.25, elements.CFbend(ds=0.7, rc=2.0, k=-0.25 + 1e-9)),
            (1 / 0.3, -(0.3**2), elements.CFbend(ds=0.7, rc=1 / 0.3, k=-(0.3**2) + 1e-9)),
        ]:
            out = track([elements.CFbend(ds=0.7, rc=rc, k=k)], u)[0]
            exp = track([ref_el], u)[0]
            print(f"rc={rc:.4g} k={k:.4g}: observed {np.round(out, 4).tolist()}\n          expected {np.round(exp, 4).tolist()}")
            bad |= not np.allclose(out, exp, atol=1e-8)
        print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Found (silently wrong near-zero case) | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

