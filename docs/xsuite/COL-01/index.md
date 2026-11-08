---
title: "COL-01: q-Gaussian normalization overflows near the Gaussian limit"
hide:
  - toc
---

# COL-01 — q-Gaussian normalization overflows near the Gaussian limit

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xfields release [v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5) (`223b4fa`) |
| Source | [`LongitudinalProfileQGaussian.cq_from_q`, qgaussian.py:45–58](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/longitudinal_profiles/qgaussian.py#L45-L58)<br>[manual's q-Gaussian formula](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L411-L439) |

`LongitudinalProfileQGaussian` normalizes the q-Gaussian line density with gamma functions whose numerator and denominator overflow separately, so for an index `q` near, but not equal to, 1 the density and the space-charge kicks it drives are NaN. The normalization is finite, as a log-beta-function evaluation and adaptive quadrature confirm. For a unit-population, unit-RMS profile at `q` = 0.999 and 1.001, the centre density is NaN instead of 0.3988 and 0.3991 m⁻¹, and a 1 GeV/c proton at x = 1 mm in a Gaussian space-charge lens gets `px` NaN instead of 2.561 × 10⁻⁵ and 2.562 × 10⁻⁵. Values from 0.995 to 1.005 fail except within 10⁻⁶ of 1, where an intentional Gaussian approximation applies; 0.99 and 1.01 are correct.

Run `python COL-01.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-01.py"

    [Download COL-01.py](COL-01.py)

    ```python
    """COL-01: q-Gaussian normalization overflows near the Gaussian limit

    `LongitudinalProfileQGaussian` normalizes the q-Gaussian line density with gamma functions whose
    numerator and denominator overflow separately, so for an index `q` near, but not equal to, 1 the
    density and the space-charge kicks it drives are NaN. The normalization is finite, as a
    log-beta-function evaluation and adaptive quadrature confirm. For a unit-population, unit-RMS
    profile at `q` = 0.999 and 1.001, the centre density is NaN instead of 0.3988 and 0.3991 m⁻¹, and a
    1 GeV/c proton at x = 1 mm in a Gaussian space-charge lens gets `px` NaN instead of 2.561 × 10⁻⁵ and
    2.562 × 10⁻⁵. Values from 0.995 to 1.005 fail except within 10⁻⁶ of 1, where an intentional Gaussian
    approximation applies; 0.99 and 1.01 are correct.

    The script builds unit-population, unit-RMS profiles for `q` = 0.9, 0.999, 1, 1.001 and 1.1 and
    compares the line density and its derivative at z = 0 and ±0.5 with a log-beta normalization that
    adaptive quadrature checks to unit integral. It then tracks a 1 GeV/c proton (x = 1 mm, ζ = 0.05 m)
    through a 1 m `SpaceChargeBiGaussian` lens (population 10¹⁰, longitudinal RMS 0.1 m, transverse RMS
    1 mm) and compares `px` with the round-Gaussian Lorentz kick. The controls `q` = 0.9, 1 and 1.1
    agree; 0.999 and 1.001 return NaN.
    """

    import numpy as np
    import xfields as xf
    import xtrack as xt
    from scipy.integrate import quad
    from scipy.special import betaln

    RTOL, ATOL = 5e-12, 2e-15
    # Match the C kernel constants so the check isolates normalization arithmetic.
    QELEM, EPSILON_0 = 1.60217662e-19, 8.854187817620e-12


    def density(z, q):
        """Unit population and unit RMS length, evaluated with a stable log beta."""
        beta = 1 / (5 - 3 * q)
        if q == 1:
            return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        if q < 1:
            log_c = betaln(0.5, 1 / (1 - q) + 1) - 0.5 * np.log(1 - q)
        else:
            log_c = betaln(0.5, 1 / (q - 1) - 0.5) - 0.5 * np.log(q - 1)
        argument = -(1 - q) * beta * z * z
        if argument <= -1:
            return 0.0
        return np.exp(0.5 * np.log(beta) - log_c + np.log1p(argument) / (1 - q))


    def evaluate():
        cases = {}
        points = np.array([0.0, 0.5, -0.5])
        for q in (0.9, 0.999, 1.0, 1.001, 1.1):
            # An independent numerical normalization checks the beta-function oracle.
            support = np.sqrt((5 - 3 * q) / (1 - q)) if q < 1 else np.inf
            integral, error = quad(density, -support, support, args=(q,), epsabs=2e-13, epsrel=2e-13)
            np.testing.assert_allclose(integral, 1, rtol=0, atol=2e-12)
            assert error < 2e-12
            profile = xf.LongitudinalProfileQGaussian(number_of_particles=1, sigma_z=1, q_parameter=q)
            expected = np.array([density(z, q) for z in points])
            beta = 1 / (5 - 3 * q)
            expected_derivative = -2 * beta * points * expected / (1 - (1 - q) * beta * points**2)
            observed = profile.line_density(points)
            derivative = profile.line_derivative(points)
            if q in (0.9, 1.0, 1.1):
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(derivative, expected_derivative, rtol=RTOL, atol=ATOL)
            else:
                cases[f"q={q}, density"] = observed, expected
                cases[f"q={q}, derivative"] = derivative, expected_derivative

            sc = xf.SpaceChargeBiGaussian(
                length=1,
                longitudinal_profile=xf.LongitudinalProfileQGaussian(
                    number_of_particles=1e10, sigma_z=0.1, q_parameter=q
                ),
                sigma_x=0.001,
                sigma_y=0.001,
            )
            p = xt.Particles(p0c=1e9, x=0.001, zeta=0.05)
            beta0, gamma0 = float(p.beta0[0]), float(p.gamma0[0])
            factor = -QELEM / p.mass0 * (1 - beta0**2) / (gamma0 * beta0**2)
            dphi_dx = np.expm1(-0.5) / (2 * np.pi * EPSILON_0 * 0.001)
            expected_px = factor * 1e11 * density(0.5, q) * dphi_dx
            sc.track(p)
            assert p.state[0] == 1
            if q in (0.9, 1.0, 1.1):
                np.testing.assert_allclose(p.px[0], expected_px, rtol=RTOL, atol=ATOL)
            else:
                cases[f"q={q}, space-charge px of alive particle"] = p.px[0], expected_px
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: finite normalized q-Gaussian density, its analytic derivative, and the round-beam Lorentz kick."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-26</span> | xfields 0.27.5 (`223b4fa134a1`), installed wheel and main-branch source, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-30</span> | xfields 0.27.5 (`223b4fa134a1`), installed packages, kernels compiled from installed source, CPU, double, serial |

