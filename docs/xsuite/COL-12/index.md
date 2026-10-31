---
title: "COL-12: ElectronCloud applies inconsistent reference-charge factors"
hide:
  - toc
---

# COL-12 — ElectronCloud applies inconsistent reference-charge factors

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`) |
| Source | [`ElectronCloud_track_local_particle`, electroncloud.h:45–57](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/electroncloud_src/electroncloud.h#L45-L57) |

`ElectronCloud` multiplies its longitudinal kick by the reference charge `q0` but not its transverse kicks, so for `q0` ≠ 1 the kicks no longer derive from one scalar potential and the map is not symplectic. Kicks from a single potential have equal mixed derivatives, ∂(Δpx)/∂ζ = ∂(Δpζ)/∂x, and the canonical energy coordinate `pzeta` (pζ) carries no charge factor. For a 3 GeV/c reference in a 0.7 m element with potential 0.031·xζ − 0.017·yζ, ∂(Δpx)/∂ζ is −0.02170, while ∂(Δpζ)/∂x is −0.04340 for `q0` = 2 and +0.04340 for `q0` = −2 instead of −0.02170. A negative reference charge reverses only the longitudinal kick; references with `q0` = 1 are unaffected.

??? example "Reproducer: COL-12.py"

    [Download COL-12.py](COL-12.py)

    ```python
    """COL-12: ElectronCloud applies inconsistent reference-charge factors

    `ElectronCloud` multiplies its longitudinal kick by the reference charge `q0` but not its transverse
    kicks, so for `q0` ≠ 1 the kicks no longer derive from one scalar potential and the map is not
    symplectic. Kicks from a single potential have equal mixed derivatives, ∂(Δpx)/∂ζ = ∂(Δpζ)/∂x, and
    the canonical energy coordinate `pzeta` (pζ) carries no charge factor. For a 3 GeV/c reference in a
    0.7 m element with potential 0.031·xζ − 0.017·yζ, ∂(Δpx)/∂ζ is −0.02170, while ∂(Δpζ)/∂x is −0.04340
    for `q0` = 2 and +0.04340 for `q0` = −2 instead of −0.02170. A negative reference charge reverses
    only the longitudinal kick; references with `q0` = 1 are unaffected.

    The script loads exact tricubic Taylor data for that potential and tracks a 3 GeV/c reference
    particle with `charge_ratio` = 1 from (x, y, ζ) = (0.12, −0.08, 0.18) m for `q0` = 1, 2 and −2. It
    compares the kicks with −L·∇φ and measures the mixed derivatives by central differences with step
    10⁻⁴ m; `q0` = 1 is the control. It requires the installed prebuilt kernels (compilation is
    forbidden).
    """

    import xfields as xf
    from _prebuilt import CTX, identity, kernels, np, xt

    identity()

    # Exact tricubic data for phi(x,y,zeta) = A*x*zeta + B*y*zeta.
    a, b, length = 0.031, -0.017, 0.7
    grid = np.linspace(-0.6, 0.6, 7)
    fmap = xf.TriCubicInterpolatedFieldMap(_context=CTX, x_grid=grid, y_grid=grid, z_grid=grid)
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    h = grid[1] - grid[0]
    taylor = np.stack(
        (
            a * x * z + b * y * z,
            a * z * h,
            b * z * h,
            (a * x + b * y) * h,
            np.zeros_like(x),
            np.full_like(x, a * h * h),
            np.full_like(x, b * h * h),
            np.zeros_like(x),
        ),
        axis=-1,
    )
    # Xobjects generates this array field from the class field specification.
    fmap._phi_taylor[:] = taylor.ravel()  # ty: ignore[unresolved-attribute]
    cloud = xf.ElectronCloud(_buffer=fmap._buffer, fieldmap=fmap, length=length)


    def kick(q, xx=0.12, zz=0.18):
        p = xt.Particles(_context=CTX, p0c=3e9, q0=q, x=xx, y=-0.08, zeta=zz, pzeta=0.015)
        # These particle properties return linked arrays despite scalar annotations.
        assert p.chi[0] == 1 and p.charge_ratio[0] == 1  # ty: ignore[not-subscriptable]
        before = float(p.pzeta[0])
        cloud.track(p)
        assert p.state[0] > 0
        return np.array([p.px[0], p.py[0], p.pzeta[0] - before])


    expected = -length * np.array([a * 0.18, b * 0.18, a * 0.12 + b * (-0.08)])
    for q in [1, 2, -2]:
        observed = kick(q)
        implemented = expected.copy()
        implemented[2] *= q
        np.testing.assert_allclose(observed, implemented, rtol=0, atol=5e-16)
        step = 1e-4
        dpx_dz = (kick(q, zz=0.18 + step)[0] - kick(q, zz=0.18 - step)[0]) / (2 * step)
        dpz_dx = (kick(q, xx=0.12 + step)[2] - kick(q, xx=0.12 - step)[2]) / (2 * step)
        # Scalar-potential kicks give dpx/dzeta = dpzeta/dx = -L*a; the extra longitudinal q0
        # leaves the reciprocity error |(q0 - 1)*L*a| instead of zero.
        expected_residual = abs((q - 1) * length * a)
        np.testing.assert_allclose(abs(dpx_dz - dpz_dx), expected_residual, rtol=0, atol=2e-12)
        print(f"q0={q} kicks={observed.tolist()} scalar_gradient_expected={expected.tolist()}")
        print(
            f"q0={q} dpx/dzeta={dpx_dz:.14g} dpzeta/dx={dpz_dx:.14g} reciprocity_error={abs(dpx_dz - dpz_dx):.14g}"
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

