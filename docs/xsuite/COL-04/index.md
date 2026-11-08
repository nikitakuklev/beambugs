---
title: "COL-04: CollectiveMonitor slice emittance omits the covariance square"
hide:
  - toc
---

# COL-04 — CollectiveMonitor slice emittance omits the covariance square

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xfields release [v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5) (`223b4fa`) |
| Source | [`CollectiveMonitor._update_slice_buffer`, collective_monitor.py:348–364](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/collective_monitor.py#L348-L364)<br>[`UniformBinSlicer.cov`](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/slicers/uniform.py#L319-L333) |

`CollectiveMonitor` computes each slice's emittance as √(var(x)·var(px) − cov(x, px)), subtracting the covariance instead of its square. The RMS emittance is the square root of the covariance-matrix determinant, which an area-preserving shear leaves unchanged, and the slicer's own variances and covariance give that value directly. For four 1 GeV/c protons with x = ±1 mm, `px` = ±2 × 10⁻⁴ and a shear `px` → `px` + a·x, the normalized emittance is 2.132 × 10⁻⁷ m for every a, but the monitor reports 4.766 × 10⁻⁴ m, about 2236 times too large, at a = −0.2 and NaN with a warning at a = +0.2. The error appears in x, y and ζ; uncorrelated slices and the slice moments are correct.

Run `python COL-04.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-04.py"

    [Download COL-04.py](COL-04.py)

    ```python
    """COL-04: CollectiveMonitor slice emittance omits the covariance square

    `CollectiveMonitor` computes each slice's emittance as √(var(x)·var(px) − cov(x, px)), subtracting
    the covariance instead of its square. The RMS emittance is the square root of the covariance-matrix
    determinant, which an area-preserving shear leaves unchanged, and the slicer's own variances and
    covariance give that value directly. For four 1 GeV/c protons with x = ±1 mm, `px` = ±2 × 10⁻⁴ and a
    shear `px` → `px` + a·x, the normalized emittance is 2.132 × 10⁻⁷ m for every a, but the monitor
    reports 4.766 × 10⁻⁴ m, about 2236 times too large, at a = −0.2 and NaN with a warning at a = +0.2.
    The error appears in x, y and ζ; uncorrelated slices and the slice moments are correct.

    The script tracks four particles per bunch, x = 10⁻³·(−1, −1, 1, 1) and `px` = 2 × 10⁻⁴·(−1, 1, −1,
    1) + a·x for a = 0, 0.2 and −0.2, through a one-slice monitor with one and with two bunches. It
    checks the slicer's variances and covariance against a direct calculation and compares the monitor's
    `epsn_x` with β₀γ₀·√(det C), C being the 2 × 2 covariance matrix of (x, `px`). The unsheared case is
    the control.
    """

    import numpy as np
    import xfields as xf
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-16


    def evaluate():
        cases = {}
        x = 0.001 * np.array([-1, -1, 1, 1])
        independent = 0.0002 * np.array([-1, 1, -1, 1])
        for bunches in (1, 2):
            for shear in (0.0, 0.2, -0.2):
                px = independent + shear * x
                p = xt.Particles(
                    p0c=1e9,
                    x=np.tile(x, bunches),
                    px=np.tile(px, bunches),
                    zeta=np.repeat(-np.arange(bunches, dtype=float), 4),
                )
                monitor = xf.CollectiveMonitor(
                    monitor_bunches=False,
                    monitor_slices=True,
                    monitor_particles=False,
                    zeta_range=(-0.1, 0.1),
                    num_slices=1,
                    flush_data_every=2,
                    bunch_spacing_zeta=1,
                    filling_scheme=np.ones(bunches, dtype=int),
                    stats_to_store=["epsn_x"],
                )
                monitor.track(p)
                assert monitor.slicer is not None
                assert monitor.slice_buffer is not None
                # The transformation px -> px+a*x is an area-preserving shear.
                # Its covariance determinant equals the unsheared one, for either sign.
                covariance = np.cov(x, px, ddof=0)
                beta_gamma = float(p.beta0[0] * p.gamma0[0])
                expected = np.sqrt(np.linalg.det(covariance)) * beta_gamma
                np.testing.assert_allclose(expected, 0.001 * 0.0002 * beta_gamma, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(
                    monitor.slicer.var("x"), covariance[0, 0], rtol=RTOL, atol=ATOL
                )
                np.testing.assert_allclose(
                    monitor.slicer.var("px"), covariance[1, 1], rtol=RTOL, atol=ATOL
                )
                np.testing.assert_allclose(
                    monitor.slicer.cov("x", "px"), covariance[0, 1], rtol=RTOL, atol=ATOL
                )
                for bunch in range(bunches):
                    observed = monitor.slice_buffer[bunch]["epsn_x"][
                        monitor.i_turn % monitor.flush_data_every, 0
                    ]
                    if shear == 0:
                        np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
                    else:
                        cases[f"{bunches} bunches, bunch {bunch}, shear={shear}, epsn_x"] = (
                            observed,
                            expected,
                        )
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: epsn_x=beta0*gamma0*sqrt(var(x)*var(px)-cov(x,px)**2), invariant under px += shear*x."
        )
        print("Both signs of the shear and both single/multiple-bunch monitor branches are checked.")
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-26</span> | xfields 0.27.5 (`223b4fa134a1`), installed wheel and main-branch source, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-30</span> | xfields 0.27.5 (`223b4fa134a1`), installed packages, kernels compiled from installed source, CPU, double, serial |

