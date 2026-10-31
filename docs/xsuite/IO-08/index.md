---
title: "IO-08: Accelerating RF contour emittance uses symmetric turning points"
hide:
  - toc
---

# IO-08 — Accelerating RF contour emittance uses symmetric turning points

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xpart main (`2633e52`) |
| Source | [RFBucket.emittance_single_particle, rf_bucket.py:539–557](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/longitudinal/rf_bucket.py#L539-L557) |

With an explicit `z`, `RFBucket.emittance_single_particle` computes a contour's longitudinal emittance assuming turning points symmetric about the stable point, although acceleration makes them asymmetric. Solving the tilted single-harmonic potential for both turning points and integrating with two independent quadratures gives the closed-contour area. For a 100 m proton bucket with Lorentz factor γ = 10, harmonic 10, voltage 1 MV and momentum compaction 0.001, requesting `z` = 0.6 m with the default `sigma` = 2 gives 1.238 eV·s instead of 1.288 eV·s (3.882% low) at acceleration fraction a = +0.5, and 0.9808 eV·s instead of 0.9465 eV·s (3.620% high) at a = −0.5. Here a is the energy gain per turn divided by the peak RF energy gain. The stationary single-harmonic control passes.

??? example "Reproducer: IO-08.py"

    [Download IO-08.py](IO-08.py)

    ```python
    """IO-08: Accelerating RF contour emittance uses symmetric turning points

    With an explicit `z`, `RFBucket.emittance_single_particle` computes a contour's longitudinal
    emittance assuming turning points symmetric about the stable point, although acceleration makes them
    asymmetric. Solving the tilted single-harmonic potential for both turning points and integrating
    with two independent quadratures gives the closed-contour area. For a 100 m proton bucket with
    Lorentz factor γ = 10, harmonic 10, voltage 1 MV and momentum compaction 0.001, requesting `z` = 0.6
    m with the default `sigma` = 2 gives 1.238 eV·s instead of 1.288 eV·s (3.882% low) at acceleration
    fraction a = +0.5, and 0.9808 eV·s instead of 0.9465 eV·s (3.620% high) at a = −0.5. Here a is the
    energy gain per turn divided by the peak RF energy gain. The stationary single-harmonic control
    passes.

    The script builds a 100 m proton bucket with γ = 10, harmonic 10, voltage 1 MV and momentum
    compaction 0.001 at a = 0 and ±0.5, where a = Δp·βc/(eV) with Δp the momentum gain per turn
    (`p_increment`), and requests the emittance of the contour through `z` = 0.6 m. It finds both true
    turning points in the analytic tilted potential and integrates the closed contour with adaptive
    quadrature and an independent 160-node Gauss rule, without calling the bucket's own Hamiltonian or
    potential. The stationary case a = 0 is the control.
    """

    from _prebuilt import identity, np, xt
    from numpy.polynomial.legendre import leggauss
    from scipy.constants import c, e
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from xpart.longitudinal.rf_bucket import RFBucket

    identity()
    print("EXECUTION Python only; no tracking kernels invoked")
    C, gamma, h, V = 100.0, 10.0, 10.0, 1e6
    mass = xt.PROTON_MASS_EV * e / c**2
    beta = np.sqrt(1 - gamma**-2)
    p0 = np.sqrt(gamma**2 - 1) * mass * c
    eta = 0.001 - gamma**-2
    k = 2 * np.pi * h / C


    def check(a):
        b = RFBucket(C, gamma, mass, e, [0.001], a * V * e / (beta * c), [h], [V], [0.0], zeta0=None)
        # Integrate the closed orbit in the analytic single-harmonic tilted potential.
        # This oracle never calls RFBucket.hamiltonian/equihamiltonian/potential.
        phase_s = -np.arcsin(a)
        z = 0.6
        phase_r = phase_s + 2 * z * k
        g = lambda phase: -np.cos(phase) + a * phase
        level = g(phase_r)
        phase_l = brentq(lambda phase: g(phase) - level, -np.pi + np.arcsin(a), phase_s, xtol=5e-15)
        midpoint, radius = (phase_l + phase_r) / 2, (phase_r - phase_l) / 2
        coefficient = 2 * e * V / (C * k) / (eta * beta * c * p0)

        def integrand(t):
            phase = midpoint + radius * np.sin(t)
            radicand = coefficient * (g(phase) - level)
            assert np.min(radicand) > -1e-18
            return np.sqrt(np.maximum(0.0, radicand)) * radius * np.cos(t)

        integral, _err = quad(integrand, -np.pi / 2, np.pi / 2, epsabs=1e-16, epsrel=2e-13)
        expected = 2 * p0 / (e * k) * integral
        nodes, weights = leggauss(160)
        gauss = 2 * p0 / (e * k) * np.pi / 2 * np.dot(weights, integrand(nodes * np.pi / 2))
        np.testing.assert_allclose(gauss, expected, rtol=2e-13, atol=0.0)
        got = b.emittance_single_particle(z=z)
        print(
            "a",
            a,
            "true_left/right",
            phase_l / k,
            phase_r / k,
            "assumed_left",
            (phase_s - 2 * z * k) / k,
        )
        print(
            "a",
            a,
            "reported",
            f"{got:.14g}",
            "closed_area",
            f"{expected:.14g}",
            "relative_error",
            f"{got / expected - 1:.12g}",
            "oracle_relative_difference",
            gauss / expected - 1,
        )
        np.testing.assert_allclose(float(b.z_sfp_extr), phase_s / k, rtol=0.0, atol=3e-13)
        if a == 0:
            np.testing.assert_allclose(got, expected, rtol=2e-12, atol=0.0)
        else:
            assert abs(got / expected - 1) > 0.03


    for a in (0.0, 0.5, -0.5):
        check(a)
    print("DEFECT REPRODUCED")

    ```

??? example "Helper: _prebuilt.py"

    [Download _prebuilt.py](_prebuilt.py)

    ```python
    """Serial CPU setup that runs Xsuite's prebuilt kernels with kernel compilation forbidden.

    `identity()` prints the package versions and context; `kernels()` asserts that a prebuilt kernel
    library was loaded and prints it.
    """

    import importlib
    import importlib.metadata
    import os
    import platform
    from pathlib import Path

    # Configure thread pools before importing NumPy or Xsuite.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"
    os.environ["XSUITE_ALLOW_KERNEL_COMPILATION"] = "0"
    os.environ["XSUITE_FORCE_KERNEL_COMPILATION"] = "0"
    os.environ["XSUITE_CFFI_FORBID_COMPILE"] = "1"

    import numpy as np
    import xcoll as xc
    import xobjects as xo
    import xtrack as xt
    from _common import report

    __all__ = ["CTX", "identity", "kernels", "np", "particles", "report", "xc", "xt"]

    np.set_printoptions(precision=17)
    xo.settings.allow_kernel_compilation = False
    xo.settings.force_kernel_compilation = False
    CTX = xo.ContextCpu(omp_num_threads=0)


    def identity(*, radiation="off", random_seed=None):
        print(f"Python {platform.python_version()}; {platform.platform()}")
        for name in ("numpy", "scipy", "cffi"):
            print(name, importlib.metadata.version(name))
        for name in ("xsuite", "xtrack", "xobjects", "xpart", "xfields", "xdeps", "xcoll"):
            mod = importlib.import_module(name)
            print("PACKAGE", name, mod.__version__, mod.__file__)
        randomness = (
            "deterministic, no random seed"
            if random_seed is None
            else f"fixed RNG seed(s)={random_seed}"
        )
        print(
            f"CONTEXT CPU serial; radiation {radiation}; {randomness}; "
            "kernel compilation forbidden; prebuilt kernels"
        )


    def kernels():
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/xsuite/lib/" in line and ".so" in line
            }
        )
        assert libraries, "No prebuilt kernel library loaded"
        print("KERNEL_LIBRARIES", libraries)


    def particles(**kwargs):
        return xt.Particles(_context=CTX, p0c=1e9, **kwargs)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xpart 0.23.23 (`2633e528018a`), installed packages, Python only, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xpart 0.23.23 (`2633e528018a`), installed packages, Python only, serial |

