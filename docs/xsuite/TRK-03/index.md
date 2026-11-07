---
title: "TRK-03: RF-derived linear segment map produces NaNs for electrons"
hide:
  - toc
---

# TRK-03 — RF-derived linear segment map produces NaNs for electrons

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`, head build) |
| Source | [`longitudinal_motion`, linesegmentmap.h:282–304](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/linesegmentmap.h#L282-L304) |

`LineSegmentMap` with `longitudinal_mode='linear_fixed_rf'` gives an electron NaN longitudinal coordinates at a valid, stable RF phase. Xsuite's RF convention uses the magnitude of the reference charge, so the analytic linear synchrotron rotation is the same for electrons and positrons of equal momentum. For a 1 GeV/c electron starting 1 mm ahead of the reference in a 1 m segment with 1 MV at 100 MHz, momentum compaction 0.01 and RF phase π, the expected exit position `zeta` is 1.000 mm and its normalized conjugate momentum `pzeta` is 2.096 × 10⁻⁶; both come out NaN. Tracking the element on its own leaves the particle alive, while the default `Line.track()` marks it lost; positrons are tracked correctly.

Run `python TRK-03.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-03.py"

    [Download TRK-03.py](TRK-03.py)

    ```python
    """TRK-03: RF-derived linear segment map produces NaNs for electrons

    `LineSegmentMap` with `longitudinal_mode='linear_fixed_rf'` gives an electron NaN longitudinal
    coordinates at a valid, stable RF phase. Xsuite's RF convention uses the magnitude of the reference
    charge, so the analytic linear synchrotron rotation is the same for electrons and positrons of equal
    momentum. For a 1 GeV/c electron starting 1 mm ahead of the reference in a 1 m segment with 1 MV at
    100 MHz, momentum compaction 0.01 and RF phase π, the expected exit position `zeta` is 1.000 mm and
    its normalized conjugate momentum `pzeta` is 2.096 × 10⁻⁶; both come out NaN. Tracking the element
    on its own leaves the particle alive, while the default `Line.track()` marks it lost; positrons are
    tracked correctly.

    The script tracks an electron and a positron (reference charge −1 and +1, electron mass) through the
    segment and compares (`zeta`, `pzeta`) with the linear synchrotron rotation whose tune follows from
    the charge magnitude, the slip factor and the RF parameters. The positron must match and checks the
    convention; the electron shows the bug.
    """

    import numpy as np
    import xtrack as xt
    from scipy.constants import c

    RTOL, ATOL = 2e-13, 2e-16


    def evaluate():
        length, voltage, frequency, zeta = 1.0, 1e6, 1e8, 0.001
        element = xt.LineSegmentMap(
            length=length,
            qx=0.2,
            qy=0.3,
            longitudinal_mode="linear_fixed_rf",
            voltage_rf=voltage,
            frequency_rf=frequency,
            phase_rf=np.pi,
            momentum_compaction_factor=0.01,
        )
        cases = {}
        for charge in (1, -1):
            particle = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=1e9, q0=charge, zeta=zeta)
            beta0, gamma0 = float(particle.beta0[0]), float(particle.gamma0[0])
            eta = 0.01 - 1 / gamma0**2
            energy0 = xt.ELECTRON_MASS_EV * gamma0
            # Xsuite's RF phase convention gives the same focusing for +/-q0:
            # qs^2 = |q0| |eta| L f V / (2 pi beta0^3 E0 c), eta = alpha_c - 1/gamma0^2,
            # bets = eta L / (2 pi qs), (zeta, pzeta) -> (zeta cos 2 pi qs, zeta sin(2 pi qs)/bets).
            qs = np.sqrt(
                abs(charge)
                * abs(eta)
                * length
                * frequency
                * voltage
                / (2 * np.pi * beta0**3 * energy0 * c)
            )
            bets = eta * length / (2 * np.pi * qs)
            expected = np.array([zeta * np.cos(2 * np.pi * qs), zeta * np.sin(2 * np.pi * qs) / bets])
            element.track(particle)
            assert particle.state[0] == 1
            observed = np.array([np.asarray(particle.zeta)[0], particle.pzeta[0]])
            if charge == 1:
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            cases[f"q0={charge}, zeta,pzeta"] = observed, expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: stable linear RF rotation with qs proportional to sqrt(abs(q0)); same electron/positron map."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

