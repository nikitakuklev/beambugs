---
title: "COL-02: Displaced Gaussian space charge integrates the wrong longitudinal-kick ray"
hide:
  - toc
---

# COL-02 — Displaced Gaussian space charge integrates the wrong longitudinal-kick ray

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xfields main (`223b4fa`) |
| Source | [`SpaceChargeBiGaussian_track_local_particle`, spacechargebigaussian.h:53–93](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/spacecharge_src/spacechargebigaussian.h#L53-L93)<br>[manual prescribes the signed ray](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L302-L323) |

`SpaceChargeBiGaussian` computes the longitudinal kick by integrating the transverse field along a ray from the origin to (|x|, |y|), so a particle at a negative coordinate receives the kick of its mirror point. For a beam displaced from the axis (`mean_x` or `mean_y` nonzero), the manual's signed ray, an analytic exponential-integral potential and adaptive quadrature give different kicks on the two sides. For a 1 GeV/c proton beam (population 10¹⁰, `sigma_z` = 0.1 m, transverse RMS 1 mm) displaced by 2 mm, particles at x = ±1 mm in a 1 m lens get identical kicks in the relative momentum deviation δ: the difference is 0 instead of 2.668 × 10⁻⁷. Transverse kicks and centred beams are correct.

??? example "Reproducer: COL-02.py"

    [Download COL-02.py](COL-02.py)

    ```python
    """COL-02: Displaced Gaussian space charge integrates the wrong longitudinal-kick ray

    `SpaceChargeBiGaussian` computes the longitudinal kick by integrating the transverse field along a
    ray from the origin to (|x|, |y|), so a particle at a negative coordinate receives the kick of its
    mirror point. For a beam displaced from the axis (`mean_x` or `mean_y` nonzero), the manual's signed
    ray, an analytic exponential-integral potential and adaptive quadrature give different kicks on the
    two sides. For a 1 GeV/c proton beam (population 10¹⁰, `sigma_z` = 0.1 m, transverse RMS 1 mm)
    displaced by 2 mm, particles at x = ±1 mm in a 1 m lens get identical kicks in the relative momentum
    deviation δ: the difference is 0 instead of 2.668 × 10⁻⁷. Transverse kicks and centred beams are
    correct.

    The script tracks particles at x = ±1 mm, y = 0, ζ = 0.05 m through a frozen round Gaussian
    displaced by `mean_x` = 2 mm and compares the difference of their δ kicks with A·λ′(ζ)·[Φ(+x) −
    Φ(−x)], where A is the kick multiplier, λ the line density and Φ the transverse potential, so the
    potential's arbitrary constant cancels. The analytic Φ difference is checked against adaptive
    quadrature, and the transverse kicks and the centred beam, where reflection is exact, serve as
    controls. Results at integration counts 257, 513 and 1025 are extrapolated to remove the quadrature
    error.
    """

    import numpy as np
    import xfields as xf
    import xtrack as xt
    from scipy.integrate import quad
    from scipy.special import exp1

    RTOL, ATOL = 2e-12, 2e-15
    # The installed and main C kernels use these constants.
    QELEM, EPSILON_0 = 1.60217662e-19, 8.854187817620e-12


    def evaluate():
        sigma, center, offset, sigma_z, zeta = 0.001, 0.002, 0.001, 0.1, 0.05
        profile = xf.LongitudinalProfileQGaussian(number_of_particles=1e10, sigma_z=sigma_z)
        seed = xt.Particles(p0c=1e9, x=[offset, -offset], zeta=zeta)
        beta0, gamma0 = float(seed.beta0[0]), float(seed.gamma0[0])
        factor = -QELEM / seed.mass0 * (1 - beta0**2) / (gamma0 * beta0**2)
        line_density = 1e10 / (np.sqrt(2 * np.pi) * sigma_z) * np.exp(-(zeta**2) / (2 * sigma_z**2))
        line_derivative = -zeta / sigma_z**2 * line_density

        def gradient(x):
            u = x - center
            return np.expm1(-(u**2) / (2 * sigma**2)) / (2 * np.pi * EPSILON_0 * u)

        # Phi(+offset)-Phi(-offset) is gauge independent. The analytic round-beam
        # potential difference agrees with an independent signed-ray quadrature.
        u_plus = (offset - center) ** 2 / (2 * sigma**2)
        u_minus = (-offset - center) ** 2 / (2 * sigma**2)
        phi_difference = -(np.log(u_plus / u_minus) + exp1(u_plus) - exp1(u_minus)) / (
            4 * np.pi * EPSILON_0
        )
        integral, error = quad(gradient, -offset, offset, epsabs=0.001, epsrel=2e-13)
        np.testing.assert_allclose(integral, phi_difference, rtol=2e-14, atol=0)
        assert error < abs(integral) * 1e-12
        expected_difference = factor * line_derivative * phi_difference
        cases = {}
        differences = []
        analytic_quadratures = []
        for steps in (257, 513, 1025):
            sc = xf.SpaceChargeBiGaussian(
                length=1,
                longitudinal_profile=profile,
                mean_x=center,
                sigma_x=sigma,
                sigma_y=sigma,
                z_kick_num_integ_per_sigma=steps,
            )
            p = seed.copy()
            sc.track(p)
            assert np.all(p.state == 1)
            np.testing.assert_allclose(
                p.px, factor * line_density * gradient(seed.x), rtol=RTOL, atol=ATOL
            )
            differences.append(float(p.delta[0] - p.delta[1]))
            ray = np.linspace(-offset, offset, 2 * steps - 1)
            analytic_quadratures.append(factor * line_derivative * np.trapezoid(gradient(ray), ray))
            # The round centered case has the reflection symmetry used by the code.
            sc.mean_x = 0
            centered = seed.copy()
            sc.track(centered)
            np.testing.assert_allclose(centered.delta[0], centered.delta[1], rtol=0, atol=ATOL)
        # Extrapolate in h=1/(n-1), removing the first-endpoint O(h) term
        # and trapezoidal O(h**2) error. This pins the physical integral rather
        # than requiring finite quadrature to equal its continuum limit exactly.
        coefficients = np.array([1 / 3, -2, 8 / 3])
        np.testing.assert_allclose(
            coefficients @ analytic_quadratures, expected_difference, rtol=RTOL, atol=ATOL
        )
        cases["extrapolated delta(+x)-delta(-x)"] = coefficients @ differences, expected_difference
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: delta(+x)-delta(-x) = A lambda'(zeta) [Phi(+x)-Phi(-x)], independent of potential gauge."
        )
        print(
            "Analytic round-beam potential, independent quadrature and transverse kicks are checked first."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

??? example "Helper: _common.py"

    [Download _common.py](_common.py)

    ```python
    """Shared setup and output for the defect reproducers.

    `prepare()` moves to a fresh temporary working directory, where CFFI compiles the kernels, and
    prints the package versions and context; `report()` prints the observed and expected rows and the
    verdict line.
    """

    import atexit
    import importlib
    import importlib.metadata
    import os
    import platform
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import xobjects as xo


    def prepare(*, radiation="off", random_seed=None):
        # CFFI compiles into cwd; keep generated files outside the source trees.
        scratch = Path(tempfile.mkdtemp(prefix="xsuite-repro-"))
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        os.chdir(scratch)
        xo.settings.allow_kernel_compilation = True
        print(f"Python {platform.python_version()}; {platform.platform()}")
        for name in ("xsuite", "xtrack", "xobjects", "xdeps", "xpart", "xfields", "xcoll"):
            module = importlib.import_module(name)
            print(f"{name} {module.__version__}: {module.__file__}")
        for name in ("numpy", "scipy", "madng-tpsa", "cffi"):
            print(f"{name} {importlib.metadata.version(name)}")
        randomness = (
            "deterministic, no random seed"
            if random_seed is None
            else f"fixed RNG seed(s)={random_seed}"
        )
        print(f"ContextCpu, serial; radiation={radiation}; {randomness}")
        print(
            f"allow_kernel_compilation={xo.settings.allow_kernel_compilation}; "
            f"force_kernel_compilation={xo.settings.force_kernel_compilation}"
        )


    def report(cases, *, rtol, atol):
        np.set_printoptions(precision=15, linewidth=140)
        mismatch = False
        for label, (observed, expected) in cases.items():
            print(f"{label}: observed={np.asarray(observed)} expected={np.asarray(expected)}")
            mismatch |= not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"comparison: rtol={rtol:g}, atol={atol:g}")
        print("DEFECT REPRODUCED" if mismatch else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xfields 0.27.5 (`223b4fa134a1`), installed wheel and main worktree, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xfields 0.27.5 (`223b4fa134a1`), installed packages, kernels compiled from installed source, CPU, double, serial |

