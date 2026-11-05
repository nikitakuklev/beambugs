---
title: "OPT-06: A subrange's second-order chromaticity depends on its absolute s position"
hide:
  - toc
---

# OPT-06 — A subrange's second-order chromaticity depends on its absolute s position

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`_get_chromatic_functions`, chromatic_functions.py:110–134](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/chromatic_functions.py#L110-L134) |

A periodic Twiss over a named subrange of a line reports a second-order chromaticity that depends on the range's absolute longitudinal position s. Differentiating the prescribed tune polynomial twice gives the correct value, and moving an otherwise unchanged range along s cannot change it. For a 2 m `LineSegmentMap` with horizontal tune Qₓ(δ) = 0.2 + 1.2δ + 1.2δ², δ being the relative momentum deviation, a 10 m drift placed before it, outside the selected range and not tracked, changes the reported `ddqx` (d²Qₓ/dδ²) from 2.400 to 86.40, a factor of 36.00 for both finite-difference steps tried. The error arises for constant-energy periodic ranges that start after s = 0; the tune, the first-order chromaticity and the drift-free control are correct.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: OPT-06.py"

    [Download OPT-06.py](OPT-06.py)

    ```python
    """OPT-06: A subrange's second-order chromaticity depends on its absolute s position

    A periodic Twiss over a named subrange of a line reports a second-order chromaticity that depends on
    the range's absolute longitudinal position s. Differentiating the prescribed tune polynomial twice
    gives the correct value, and moving an otherwise unchanged range along s cannot change it. For a 2 m
    `LineSegmentMap` with horizontal tune Qₓ(δ) = 0.2 + 1.2δ + 1.2δ², δ being the relative momentum
    deviation, a 10 m drift placed before it, outside the selected range and not tracked, changes the
    reported `ddqx` (d²Qₓ/dδ²) from 2.400 to 86.40, a factor of 36.00 for both finite-difference steps
    tried. The error arises for constant-energy periodic ranges that start after s = 0; the tune, the
    first-order chromaticity and the drift-free control are correct.

    The script places that `LineSegmentMap` after a drift of length 0 or 10 m, ending at a marker, and
    runs a periodic `twiss4d` over the segment alone with finite-difference steps `delta_chrom` = 10⁻³
    and 5 × 10⁻⁴. It checks the tune and `dqx` first, then compares `ddqx` with 2.4; the drift-free case
    is the control.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 0.0, 5e-8


    def evaluate():
        cases = {}
        for offset in (0, 10):
            line = xt.Line(
                elements={
                    "lead": xt.Drift(length=offset),
                    # ddqx is d²Qx/dδ², so Qx(δ) = 0.2 + 1.2δ + 1.2δ².
                    "arc": xt.LineSegmentMap(
                        length=2, qx=0.2, qy=0.3, dqx=1.2, ddqx=2.4, longitudinal_mode="frozen"
                    ),
                    "end": xt.Marker(),
                }
            )
            line.particle_ref = xt.Particles(p0c=1e9)
            for step in (1e-3, 5e-4):
                tw = line.twiss4d(start="arc", end="end", init="periodic", delta_chrom=step)
                # The selected range contains only the prescribed tune polynomial.
                np.testing.assert_allclose([tw.qx, tw.dqx], [0.2, 1.2], rtol=0, atol=1e-10)
                # xtrack/twiss/chromatic_functions.py (_get_chromatic_functions) averages δ over
                # the range as trapz(delta, s)/s[-1], dividing by the
                # absolute end position (12 m) instead of the range length (2 m). Both momentum
                # separations in the second difference shrink by 2/12, so ddqx grows by 36.
                if offset == 0:
                    np.testing.assert_allclose(tw.ddqx, 2.4, rtol=RTOL, atol=ATOL)
                cases[f"start s={offset}, delta_chrom={step}, ddqx"] = tw.ddqx, 2.4
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: d²Qx/ddelta²=2.4 from Qx=0.2+1.2*delta+1.2*delta², independent of preceding drift."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

