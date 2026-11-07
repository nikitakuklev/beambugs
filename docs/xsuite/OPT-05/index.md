---
title: "OPT-05: 4D Twiss retains RF energy kicks from LineSegmentMap"
hide:
  - toc
---

# OPT-05 — 4D Twiss retains RF energy kicks from LineSegmentMap

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`, head build) |
| Source | [`longitudinal_motion`, linesegmentmap.h:240–305](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/linesegmentmap.h#L240-L305)<br>[4D Twiss sets explicitly](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_defaults_and_input_preparation.py#L412-L413) |

`twiss4d()` keeps the RF energy kicks of a `LineSegmentMap` in its longitudinal modes `linear_fixed_rf` and `nonlinear`, although 4D (transverse-only) optics assumes fixed energy. With RF off, 4D optics requires constant momentum, and a frozen segment followed by an ordinary cavity satisfies it. For a 1 GeV/c proton through a 1 m segment with tunes 0.2 and 0.3, 1 MV at 100 MHz, momentum compaction 0.01 and initial longitudinal offset 1 mm, the exit relative momentum deviation δ is −3.940 × 10⁻⁶ in `linear_fixed_rf` and −3.941 × 10⁻⁶ in `nonlinear` mode instead of zero. Periodic and open Twiss are both affected; the `linear_fixed_qs` mode and the ordinary cavity keep the momentum fixed.

Run `python OPT-05.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-05.py"

    [Download OPT-05.py](OPT-05.py)

    ```python
    """OPT-05: 4D Twiss retains RF energy kicks from LineSegmentMap

    `twiss4d()` keeps the RF energy kicks of a `LineSegmentMap` in its longitudinal modes
    `linear_fixed_rf` and `nonlinear`, although 4D (transverse-only) optics assumes fixed energy. With
    RF off, 4D optics requires constant momentum, and a frozen segment followed by an ordinary cavity
    satisfies it. For a 1 GeV/c proton through a 1 m segment with tunes 0.2 and 0.3, 1 MV at 100 MHz,
    momentum compaction 0.01 and initial longitudinal offset 1 mm, the exit relative momentum deviation
    δ is −3.940 × 10⁻⁶ in `linear_fixed_rf` and −3.941 × 10⁻⁶ in `nonlinear` mode instead of zero.
    Periodic and open Twiss are both affected; the `linear_fixed_qs` mode and the ordinary cavity keep
    the momentum fixed.

    The script runs `twiss4d(zeta0=0.001, chrom=False)` on a 1 m `LineSegmentMap` with tunes 0.2 and 0.3
    and 1 GeV/c protons in each longitudinal mode, with 1 MV at 100 MHz and momentum compaction 0.01 for
    `linear_fixed_rf` and `nonlinear`, checks the tunes, and compares δ at entrance and exit with zero.
    The `linear_fixed_qs` mode and a frozen segment followed by an ordinary `Cavity` are the controls.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 0.0, 2e-15


    def evaluate():
        # A frozen segment plus an ordinary cavity supplies a control with the
        # same transverse tunes and RF; 4D optics must keep energy fixed.
        control = xt.Line(
            elements=[
                xt.LineSegmentMap(length=1, qx=0.2, qy=0.3, longitudinal_mode="frozen"),
                xt.Cavity(voltage=1e6, frequency=1e8),
            ]
        )
        control.particle_ref = xt.Particles(p0c=1e9)
        tw = control.twiss4d(zeta0=0.001, chrom=False)
        np.testing.assert_allclose(tw.delta, 0, rtol=RTOL, atol=ATOL)
        cases = {}
        for mode in ("linear_fixed_qs", "linear_fixed_rf", "nonlinear"):
            kwargs = (
                {"qs": 0.01, "bets": -10}
                if mode == "linear_fixed_qs"
                else {"voltage_rf": 1e6, "frequency_rf": 1e8, "momentum_compaction_factor": 0.01}
            )
            line = xt.Line(
                elements={
                    "arc": xt.LineSegmentMap(
                        length=1,
                        qx=0.2,
                        qy=0.3,
                        longitudinal_mode=mode,
                        **kwargs,
                    )
                }
            )
            line.particle_ref = xt.Particles(p0c=1e9)
            tw = line.twiss4d(zeta0=0.001, chrom=False)
            np.testing.assert_allclose([tw.qx, tw.qy], [0.2, 0.3], rtol=0, atol=1e-12)
            if mode == "linear_fixed_qs":
                np.testing.assert_allclose(tw.delta, 0, rtol=RTOL, atol=ATOL)
            cases[f"4D Twiss {mode}, delta at entrance/exit"] = tw.delta, np.zeros(2)
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: 4D Twiss suppresses RF energy kicks, so delta stays zero.")
        print("Ordinary Cavity and linear_fixed_qs controls must obey the same invariant.")
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

