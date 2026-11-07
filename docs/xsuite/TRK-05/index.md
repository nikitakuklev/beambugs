---
title: "TRK-05: Sliced thick Multipole repeats the full reference bend angle"
hide:
  - toc
---

# TRK-05 — Sliced thick Multipole repeats the full reference bend angle

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Source | [`thin_slice_multipole.h`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_multipole.h#L22-L48)<br>[`track_magnet_particles`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L338-L369)<br>[reference-frame kick](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_kick.h#L151-L160) |

Thin slicing of a `Multipole(isthick=True)` applies the element's full reference bend angle `hxl` in every slice, so each extra slice adds spurious bending. At the reference momentum, a dipole kick `knl[0]` equal to `hxl` cancels the reference-frame kick exactly, so a particle starting on the reference orbit must stay there for any number of slices. For a 1 m element with `knl[0]` = `hxl` = 0.01 rad and a 1 GeV/c proton, four slices give x = 15.00 mm and normalized horizontal momentum `px` = 0.03000 instead of zero for both. The unsliced element and a single slice keep the particle on axis.

Run `python TRK-05.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-05.py"

    [Download TRK-05.py](TRK-05.py)

    ```python
    """TRK-05: Sliced thick Multipole repeats the full reference bend angle

    Thin slicing of a `Multipole(isthick=True)` applies the element's full reference bend angle `hxl` in
    every slice, so each extra slice adds spurious bending. At the reference momentum, a dipole kick
    `knl[0]` equal to `hxl` cancels the reference-frame kick exactly, so a particle starting on the
    reference orbit must stay there for any number of slices. For a 1 m element with `knl[0]` = `hxl` =
    0.01 rad and a 1 GeV/c proton, four slices give x = 15.00 mm and normalized horizontal momentum `px`
    = 0.03000 instead of zero for both. The unsliced element and a single slice keep the particle on
    axis.

    The script tracks the reference proton from the origin through the element unsliced and with
    `Teapot(1)`, `Teapot(2)` and `Teapot(4)` slicing, and expects x = `px` = 0 each time to an absolute
    tolerance of 2 × 10⁻¹⁵. The unsliced and single-slice cases must pass; two and four slices show the
    bug.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 0.0, 2e-15


    def evaluate():
        cases = {}
        for count in (None, 1, 2, 4):
            line = xt.Line(
                elements={
                    "m": xt.Multipole(
                        length=1,
                        isthick=True,
                        hxl=0.01,
                        knl=[0.01],
                    )
                }
            )
            line.particle_ref = xt.Particles(p0c=1e9)
            if count is not None:
                line.slice_thick_elements(slicing_strategies=[xt.Strategy(xt.Teapot(count))])
            particle = xt.Particles(p0c=1e9)
            line.track(particle)
            assert particle.state[0] == 1
            # On the reference orbit each fractional dipole kick cancels its
            # fractional reference-frame kick; all curvature feed-down terms vanish.
            # Source tracing predicts a leading excess angle (N-1)*hxl for N slices.
            observed = np.array([particle.x[0], particle.px[0]])
            expected = np.zeros(2)
            if count in (None, 1):
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            cases[f"Teapot count={count}, x,px"] = observed, expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: knl[0]=hxl leaves the on-momentum reference orbit at x=px=0 for every slice count."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | low | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |

