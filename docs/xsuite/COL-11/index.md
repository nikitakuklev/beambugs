---
title: "COL-11: Interpolated electron lens omits the transit-speed factor"
hide:
  - toc
---

# COL-11 — Interpolated electron lens omits the transit-speed factor

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`) |
| Source | [`ElectronLensInterpolated_track_local_particle`, electronlens_interpolated.h:46–57](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/electronlens_src/electronlens_interpolated.h#L46-L57) |

`ElectronLensInterpolated` omits the inverse particle-speed factor from the transit time and uses the reference speed throughout, so kicks are too small by the factor β₀, the reference speed over c, already at zero relative momentum deviation δ. Integrating the Lorentz force of the counter-propagating electrons over the transit time L/(βc), with the same stored field derivative, gives the correct impulse. For a 0.6 A, 15 kV, 0.8 m lens and a 600 MeV/c proton (β₀ = 0.5387), `px` is −3.870 × 10⁻⁵ instead of −7.184 × 10⁻⁵, and at δ = 0.25 the kick stays unchanged instead of becoming 0.8784 times the reference kick. Lower-speed beams and off-momentum particles are affected.

??? example "Reproducer: COL-11.py"

    [Download COL-11.py](COL-11.py)

    ```python
    """COL-11: Interpolated electron lens omits the transit-speed factor

    `ElectronLensInterpolated` omits the inverse particle-speed factor from the transit time and uses
    the reference speed throughout, so kicks are too small by the factor β₀, the reference speed over c,
    already at zero relative momentum deviation δ. Integrating the Lorentz force of the
    counter-propagating electrons over the transit time L/(βc), with the same stored field derivative,
    gives the correct impulse. For a 0.6 A, 15 kV, 0.8 m lens and a 600 MeV/c proton (β₀ = 0.5387), `px`
    is −3.870 × 10⁻⁵ instead of −7.184 × 10⁻⁵, and at δ = 0.25 the kick stays unchanged instead of
    becoming 0.8784 times the reference kick. Lower-speed beams and off-momentum particles are affected.

    The script builds a lens from a normalized Gaussian density on a 49 × 49 grid and tracks positive
    unit-charge particles with electron and proton masses at 0.6 and 1.7 GeV/c from a grid vertex, where
    the interpolated field derivative equals the stored one. It compares each kick with the Lorentz
    impulse computed from that derivative, and tracks protons at δ = 0 and 0.25 to compare the ratio of
    their kicks with the change in transit speed. It requires the installed prebuilt kernels
    (compilation is forbidden).
    """

    import xfields as xf
    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import c

    identity()

    axis = np.linspace(-0.012, 0.012, 49)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    dx = axis[1] - axis[0]
    rho = np.exp(-(x * x + y * y) / (2 * 0.0025**2))
    rho /= rho.sum() * dx * dx
    current, length, voltage = 0.6, 0.8, 15000.0
    lens = xf.ElectronLensInterpolated(
        _context=CTX, x_grid=axis, y_grid=axis, rho=rho, current=current, length=length, voltage=voltage
    )
    # At a grid vertex the tricubic derivative equals its stored Taylor derivative.
    ix, iy, iz = 30, 24, 5
    # Xobjects generates this field from the class field specification.
    fm = lens.fieldmap  # ty: ignore[unresolved-attribute]
    idx = 8 * (ix + int(fm.nx) * iy + int(fm.nx) * int(fm.ny) * iz)
    grad = float(fm._phi_taylor[idx + 1]) / float(fm.dx)
    be = np.sqrt(1.0 - (1.0 + voltage / 510998.928) ** -2)
    assert grad < 0

    for p0c in (6e8, 1.7e9):
        kicks, beta = [], []
        for mass in (xt.ELECTRON_MASS_EV, xt.PROTON_MASS_EV):
            particle = xt.Particles(
                _context=CTX, p0c=p0c, mass0=mass, q0=1, x=axis[ix], y=axis[iy], delta=0
            )
            bb = p0c / np.hypot(p0c, mass)
            # E = -I/(be*c) grad(phi), |F|=q*E*(1+be*bb), dt=L/(bb*c).
            # Divide impulse by reference momentum; p0c in eV cancels q's elementary charge.
            physical = current * length * grad * (1 + be * bb) / (p0c * c * be * bb)
            lens.track(particle)
            actual = float(particle.px[0])
            assert particle.state[0] > 0
            np.testing.assert_allclose(actual / physical, bb, rtol=2e-13, atol=0)
            assert not np.isclose(actual, physical, rtol=1e-9, atol=0)
            kicks.append(actual)
            beta.append(bb)
            print(
                f"p0c={p0c:.9g} mass0={mass:.12g} beta0={bb:.15g} observed={actual:.16g} Lorentz_impulse={physical:.16g} observed/expected={actual / physical:.15g}"
            )
        observed_ratio = kicks[1] / kicks[0]
        expected_ratio = ((1 + be * beta[1]) / beta[1]) / ((1 + be * beta[0]) / beta[0])
        print(
            f"p0c={p0c:.9g} proton/electron observed={observed_ratio:.15g} expected={expected_ratio:.15g}"
        )

    reference_momentum = 6e8
    momenta = reference_momentum * np.array([1.0, 1.25])
    particles = xt.Particles(
        _context=CTX, p0c=reference_momentum, x=axis[ix], y=axis[iy], delta=[0.0, 0.25]
    )
    lens.track(particles)
    velocities = momenta / np.hypot(momenta, xt.PROTON_MASS_EV)
    forces = (1 + be * velocities) / velocities
    observed_ratio = float(particles.px[1] / particles.px[0])
    expected_ratio = forces[1] / forces[0]
    np.testing.assert_allclose(observed_ratio, 1.0, rtol=0, atol=2e-15)
    print(
        f"delta=0.25/reference kick ratio={observed_ratio:.15g} Lorentz_expected={expected_ratio:.15g}"
    )
    kernels()
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

