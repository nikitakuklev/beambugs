---
title: "COL-06: Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy"
hide:
  - toc
---

# COL-06 — Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS master (`a3f730a`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [reciprocals, wofz.c:191,197](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/wofz.c#L191-L197)<br>[accuracy contract, wofz.c:65](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/wofz.c#L65)<br>[double-precision statement, wofz.c:113](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/wofz.c#L113)<br>[types, f2c.h:21,26](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/include/f2c.h#L21-L26)<br>[complexErf, complex.cc:27–33](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/complex.cc#L27-L33)<br>[wa and wb, insertSCeffects.cc:581–584](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L581-L584)<br>[kicks, insertSCeffects.cc:592–602](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L592-L602)<br>[nonlinear force, scmult.tex:24–43](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L24-L43) |

Nonlinear SCMULT kicks for elliptical Gaussian beams lose accuracy when the SDDS complex error function evaluates its power-series reciprocals in single precision despite a 14-digit accuracy contract. For 1 nC of electrons at 10 mₑc with rms sizes 2 mm horizontally and 1 mm vertically, a sampled particle's horizontal kick is −3.149×10⁻⁵ rad, differing from the independent Poisson-integral value by −7.904×10⁻¹³ rad (2.510×10⁻⁸ relative). A scalar model reproduces the native kicks with single-precision reciprocals and agrees with the integral to about 10⁻¹⁵ relative with double-precision reciprocals.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _scring.py](../helpers/_scring.py) · [View source](../helpers/index.md#_scring)

??? example "Reproducer: COL-06.py"

    [Download COL-06.py](COL-06.py)

    ```python
    """COL-06: Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy

    Nonlinear SCMULT kicks for elliptical Gaussian beams lose accuracy when the SDDS complex error function evaluates its power-series reciprocals in single precision despite a 14-digit accuracy contract. For 1 nC of electrons at 10 mₑc with rms sizes 2 mm horizontally and 1 mm vertically, a sampled particle's horizontal kick is −3.149×10⁻⁵ rad, differing from the independent Poisson-integral value by −7.904×10⁻¹³ rad (2.510×10⁻⁸ relative). A scalar model reproduces the native kicks with single-precision reciprocals and agrees with the integral to about 10⁻¹⁵ relative with double-precision reciprocals.

    The script tracks 16 electrons through one drift of a closed four-bend ring, with an SCMULT after
    the drift, and compares each slope kick with the Gaussian field from the Poisson integral. The
    cause is isolated by a scalar model of the power series of `wofz` that divides in binary32 or in
    binary64: the binary32 model must reproduce the native kicks, and the binary64 model the Poisson
    integral.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scring import beam, compare, kick, levels, track
    from scipy.special import wofz

    SX, SY = 2e-3, 1e-3  # m, rms sizes; sx > sy, so the kick is evaluated without swapping planes
    LENGTH = 0.5  # m, drift with the SCMULT
    # Four transverse positions (x = ±σₓ, y = ±σᵧ) and two arrival times (±0.1 ns), each combination
    # twice.
    PARTICLES = beam(SX, SY, 1e-10 * levels(2), n=2, copies=2)
    RTOL_FAULT = 1e-9  # a correct kick agrees with the Poisson integral to 1e-12
    RTOL_MODEL = 1e-11  # the models agree with the native kicks, or the integral, to 1e-14


    def f32(value: float) -> float:
        """A C `(float)` cast of a decimal constant, promoted back to double."""
        return float(np.float32(value))


    def faddeeva(z: complex, single: bool) -> complex:
        """w(z) for Im z > 0, with the power-series branch of mdbmth/wofz.c reproduced operation by
        operation. Its reciprocals are `(float)1./j`: a binary32 division of the integer j, promoted to
        double afterwards. single=False divides in binary64 instead. The model covers only the
        power-series region; the native continued-fraction branch is not modelled.
        """
        xa, ya = abs(z.real), abs(z.imag)
        qrho = (xa / f32(6.3)) ** 2 + (ya / f32(4.4)) ** 2
        if qrho >= 0.085264:
            raise ValueError("argument outside the audited power-series region")
        xquad, yquad = xa * xa - ya * ya, xa * 2 * ya

        def reciprocal(j: int) -> float:
            return float(np.float32(1) / np.float32(j)) if single else 1.0 / j

        qrho = (1 - ya / f32(4.4) * f32(0.85)) * np.sqrt(qrho)
        n = int(np.floor(qrho * 72 + 6 + 0.5))  # i_dnnt: nearest integer
        j = 2 * n + 1
        xsum, ysum = reciprocal(j), 0.0
        for i in range(n, 0, -1):
            j -= 2
            xaux = (xsum * xquad - ysum * yquad) / i
            ysum = (xsum * yquad + ysum * xquad) / i
            xsum = xaux + reciprocal(j)
        u1 = (xsum * ya + ysum * xa) * -1.12837916709551257388 + 1.0
        v1 = (xsum * xa - ysum * ya) * 1.12837916709551257388
        damp = np.exp(-xquad)
        u2, v2 = damp * np.cos(yquad), -damp * np.sin(yquad)
        u, v = u1 * u2 - v1 * v2, u1 * v2 + v1 * u2
        return complex(u, -v if z.real < 0 else v)


    def model_kick(physical: np.ndarray, single: bool) -> np.ndarray:
        """Kick of the nonlinear branch of SCMULT with `faddeeva` in place of the exact function.

        The kick is proportional to Im w for x and Re w for y, with w = w(a) - C3 w(b), where
        a = (x + i|y|)/sd, b = (x sy/sx + i|y| sx/sy)/sd, sd = sqrt(2 (sx^2 - sy^2)) and
        C3 = exp(-x^2/(2 sx^2) - y^2/(2 sy^2)). The physical kick has the same form with the exact
        function, so the model kick is the physical kick times the ratio of the two w components.
        """
        x, y = PARTICLES["x"], PARTICLES["y"]
        sd = np.sqrt(2 * (SX**2 - SY**2))
        c3 = np.exp(-(x**2) / (2 * SX**2) - y**2 / (2 * SY**2))
        ratio = np.empty_like(physical)
        for i in range(len(x)):
            a = complex(x[i] / sd, abs(y[i]) / sd)
            b = complex(x[i] / sd * SY / SX, abs(y[i]) / sd * SX / SY)
            w = faddeeva(a, single) - c3[i] * faddeeva(b, single)
            exact = complex(wofz(a)) - c3[i] * complex(wofz(b))
            ratio[i] = [w.imag / exact.imag, w.real / exact.real]
        return physical * ratio


    def deviation(observed: np.ndarray, expected: np.ndarray) -> str:
        relative = np.abs(observed / expected - 1)
        return f"max relative deviation x={relative[:, 0].max():.4g}, y={relative[:, 1].max():.4g}"


    defect = False
    for mode in modes():
        observed, leff = track(mode, LENGTH, PARTICLES)
        physical = kick(PARTICLES, leff, nonlinear=True)
        # Faulty arithmetic: native kicks differ from the Poisson integral at the 1e-8 level.
        wrong = compare(f"{mode} native vs Poisson integral", observed, physical, RTOL_FAULT)
        print(f"{mode} native vs Poisson integral: {deviation(observed, physical)}")
        single = model_kick(physical, single=True)
        double = model_kick(physical, single=False)
        # Control: the binary32 model reproduces the native kicks, the binary64 model the integral.
        if compare(f"{mode} native vs binary32 model", observed, single, RTOL_MODEL):
            sys.exit("The binary32 model does not reproduce the native kicks.")
        print(f"{mode} native vs binary32 model: {deviation(observed, single)}")
        if compare(f"{mode} Poisson integral vs binary64 model", physical, double, RTOL_MODEL):
            sys.exit("The binary64 model does not reproduce the Poisson integral.")
        print(f"{mode} Poisson integral vs binary64 model: {deviation(physical, double)}")
        defect |= wrong
    print(f"comparison: rtol={RTOL_FAULT:g} (native vs integral), {RTOL_MODEL:g} (models)")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (force discrepancy and the single-precision reciprocal candidate) | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Found (force discrepancy) | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | SDDS master (`a3f730a14c49`), Elegant 0355dd6aca12 source, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | SDDS master (`a60b9bbed7cc`), Elegant 75a2e1a1a58b source, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | SDDS master (`a60b9bbed7cc`), Elegant 75a2e1a1a58b source, CPU, serial and 3-rank Pelegant |

