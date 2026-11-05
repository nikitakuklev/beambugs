---
title: "TRK-06: Zero linear chromaticity shifts the quadratic coefficient into the linear slot"
hide:
  - toc
---

# TRK-06 — Zero linear chromaticity shifts the quadratic coefficient into the linear slot

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`LineSegmentMap.__init__`, line_segment_map.py:270–291](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/line_segment_map.py#L270-L291) |

`LineSegmentMap` treats a scalar second tune derivative `ddqx` or `ddqy` as a first derivative when the matching linear chromaticity `dqx` or `dqy` is exactly zero. A specified second derivative must make the tune depend quadratically, and therefore evenly, on the relative momentum deviation δ. With `qx` = 0.2, `dqx` = 0, `ddqx` = 2.4 and initial x = 1 mm, the exit position is 0.5776 mm at δ = −2% and 0.01257 mm at δ = +2%, instead of 0.3061 mm at both. The analytic rotation and the equivalent derivative array `dnqx=[0.2, 0, 2.4]` agree on the correct value, so specifying arrays avoids the bug; the vertical keywords have the same defect.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: TRK-06.py"

    [Download TRK-06.py](TRK-06.py)

    ```python
    """TRK-06: Zero linear chromaticity shifts the quadratic coefficient into the linear slot

    `LineSegmentMap` treats a scalar second tune derivative `ddqx` or `ddqy` as a first derivative when
    the matching linear chromaticity `dqx` or `dqy` is exactly zero. A specified second derivative must
    make the tune depend quadratically, and therefore evenly, on the relative momentum deviation δ. With
    `qx` = 0.2, `dqx` = 0, `ddqx` = 2.4 and initial x = 1 mm, the exit position is 0.5776 mm at δ = −2%
    and 0.01257 mm at δ = +2%, instead of 0.3061 mm at both. The analytic rotation and the equivalent
    derivative array `dnqx=[0.2, 0, 2.4]` agree on the correct value, so specifying arrays avoids the
    bug; the vertical keywords have the same defect.

    The script tracks particles at δ = −2%, 0 and +2% from x = 1 mm, y = 2 mm through a `LineSegmentMap`
    with frozen longitudinal motion, specified once by the scalar keywords (`ddqx` = 2.4, `ddqy` = −1.6,
    zero first derivatives) and once by the equivalent derivative arrays. Both are compared with a
    unit-beta rotation by the phase 2πQ(δ), where Q(δ) = Q₀ + Q″δ²/2. The array form must agree; the
    scalar form shows the bug.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-16


    def evaluate():
        delta = np.array([-0.02, 0, 0.02])
        x0, y0 = 0.001, 0.002
        # Q(delta) = Q0 + Q' delta + Q'' delta^2/2 with Q' = 0 is even in delta; unit-beta
        # rotation x = x0 cos(2 pi Q), px = -x0 sin(2 pi Q), likewise for y.
        phix = 2 * np.pi * (0.2 + 2.4 * delta**2 / 2)
        phiy = 2 * np.pi * (0.3 - 1.6 * delta**2 / 2)
        expected = np.array(
            [x0 * np.cos(phix), -x0 * np.sin(phix), y0 * np.cos(phiy), -y0 * np.sin(phiy)]
        ).T
        cases = {}
        for label, kwargs in (
            ("explicit derivative arrays", {"dnqx": [0.2, 0, 2.4], "dnqy": [0.3, 0, -1.6]}),
            (
                "scalar ddqx/ddqy with zero first derivatives",
                {"qx": 0.2, "qy": 0.3, "ddqx": 2.4, "ddqy": -1.6},
            ),
        ):
            element = xt.LineSegmentMap(longitudinal_mode="frozen", **kwargs)
            particle = xt.Particles(p0c=1e9, delta=delta, x=x0, y=y0)
            element.track(particle)
            assert np.all(particle.state == 1)
            observed = np.array([particle.x, particle.px, particle.y, particle.py]).T
            if label == "explicit derivative arrays":
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            cases[label + "; rows delta=-0.02/0/0.02, columns x,px,y,py"] = observed, expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: Q(delta)=Q0+Q''*delta**2/2 when Q'=0, an even function of momentum offset.")
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

