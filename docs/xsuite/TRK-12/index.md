---
title: "TRK-12: Wire kicks omit particle charge-to-mass scaling"
hide:
  - toc
---

# TRK-12 — Wire kicks omit particle charge-to-mass scaling

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`Wire_track_local_particle`, wire.h:33–50](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/wire.h#L33-L50)<br>[species-coordinate Hamiltonian](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L321-L466) |

`Wire` gives every particle the reference-species kick regardless of its charge or mass, so even opposite charges bend the same way. Integrating the Biot–Savart field of the finite wire shows that the normalized kick must scale with the particle's charge-to-mass ratio relative to the reference species, χ. For a 7 A wire of physical length 0.7 m and interaction length 1.1 m at (x, y) = (−1, 0.2) mm, a 1 GeV/c particle at (2, 0.5) mm with twice the reference charge and the same mass gets normalized horizontal momentum −9.696 × 10⁻⁵ instead of −1.939 × 10⁻⁴. Reference-species and off-momentum particles are tracked correctly; any species set through `charge_ratio` or `mass_ratio` is affected.

??? example "Reproducer: TRK-12.py"

    [Download TRK-12.py](TRK-12.py)

    ```python
    """TRK-12: Wire kicks omit particle charge-to-mass scaling

    `Wire` gives every particle the reference-species kick regardless of its charge or mass, so even
    opposite charges bend the same way. Integrating the Biot–Savart field of the finite wire shows that
    the normalized kick must scale with the particle's charge-to-mass ratio relative to the reference
    species, χ. For a 7 A wire of physical length 0.7 m and interaction length 1.1 m at (x, y) = (−1,
    0.2) mm, a 1 GeV/c particle at (2, 0.5) mm with twice the reference charge and the same mass gets
    normalized horizontal momentum −9.696 × 10⁻⁵ instead of −1.939 × 10⁻⁴. Reference-species and
    off-momentum particles are tracked correctly; any species set through `charge_ratio` or `mass_ratio`
    is affected.

    The script tracks particles with χ = 1, 2, −1 and 0.5 past the wire and compares their horizontal
    kicks with the Biot–Savart result scaled by χ. Three controls must agree: the reference particle in
    both planes, a particle whose doubled charge is set through `q0` instead of `charge_ratio`, and a
    reference particle at relative momentum deviation 0.35, whose kick must not change. It runs serially
    on the installed prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, particles, report, xt

    identity()
    cases = {}
    c, mu0 = 299792458.0, 4 * np.pi * 1e-7
    I, physical_length, interaction_length = 7.0, 0.7, 1.1
    x, y, xwire, ywire = 0.002, 0.0005, -0.001, 0.0002
    dx, dy = x - xwire, y - ywire
    radius_squared = dx * dx + dy * dy
    # Integral of the finite-wire field over the interaction interval. The beta
    # in q*v*B cancels dt=ds/(beta*c), so no off-momentum factor remains in dp/P0.
    field_integral = np.hypot(
        physical_length + interaction_length, 2 * np.sqrt(radius_squared)
    ) - np.hypot(interaction_length - physical_length, 2 * np.sqrt(radius_squared))
    base = -mu0 * I * c / (4 * np.pi * 1e9) * field_integral / radius_squared
    wire = xt.Wire(
        _context=CTX,
        current=I,
        L_phy=physical_length,
        L_int=interaction_length,
        xma=xwire,
        yma=ywire,
    )
    charge = np.array([1.0, 2.0, -1.0, 1.0])
    mass = np.array([1.0, 1.0, 1.0, 2.0])
    p = particles(
        x=np.full(4, x),
        y=np.full(4, y),
        delta=np.zeros(4),
        charge_ratio=charge,
        mass_ratio=mass,
    )
    expected = base * dx * charge / mass
    wire.track(p)
    print("PX observed", np.asarray(p.px), "expected", expected)
    print("RATIOS observed", np.asarray(p.px) / p.px[0], "expected", charge / mass)
    np.testing.assert_allclose(p.px[0], expected[0], rtol=2e-14, atol=0)
    np.testing.assert_allclose(p.py[0], base * dy, rtol=2e-14, atol=0)
    cases["Wire species kick ratios"] = (np.asarray(p.px) / p.px[0], charge / mass)
    q0_two = particles(q0=2, x=x, y=y, delta=0.0)
    wire.track(q0_two)
    print(
        "SAME_PHYSICAL_CHARGE q0=2:",
        q0_two.px[0],
        "q0=1 charge_ratio=2:",
        p.px[1],
        "ratio=",
        q0_two.px[0] / p.px[1],
    )
    np.testing.assert_allclose(q0_two.px[0], expected[1], rtol=2e-14, atol=0)
    momentum = particles(x=[x, x], y=[y, y], delta=[0.0, 0.35])
    wire.track(momentum)
    np.testing.assert_array_equal(momentum.px, [p.px[0], p.px[0]])
    print("OFF_MOMENTUM control ratio=", momentum.px[1] / momentum.px[0])
    cases["equivalent physical charge px"] = (p.px[1], q0_two.px[0])
    kernels()
    report(cases, rtol=4e-15, atol=0)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |

