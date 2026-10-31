---
title: "TRK-14: Elens residual kicks fail forward/backward tracking"
hide:
  - toc
---

# TRK-14 — Elens residual kicks fail forward/backward tracking

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`Elens_track_local_particle`, elens.h:137–147](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/elens.h#L137-L147)<br>[`Elens` class advertises `has_backtrack=True`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elens.py#L47) |

Backtracking through an `Elens` repeats its hollow-core residual kicks instead of undoing them, so a forward-then-backward round trip doubles the residual momentum. The inverse of a constant momentum kick has the opposite sign, and an explicit inverse element with both residual kicks negated restores the initial state exactly. For a 1.4 m, 3 A, 12 kV lens with radii 1 and 4 mm and residual normalized kicks (2.3, −4.1) × 10⁻⁷, a 1 GeV/c proton in the core returns with normalized transverse momenta (`px`, `py`) = (4.600, −8.200) × 10⁻⁷ instead of zero. Backtracking a lens with nonzero residual kicks is affected; the annular electron-lens force round-trips correctly.

??? example "Reproducer: TRK-14.py"

    [Download TRK-14.py](TRK-14.py)

    ```python
    """TRK-14: Elens residual kicks fail forward/backward tracking

    Backtracking through an `Elens` repeats its hollow-core residual kicks instead of undoing them, so a
    forward-then-backward round trip doubles the residual momentum. The inverse of a constant momentum
    kick has the opposite sign, and an explicit inverse element with both residual kicks negated
    restores the initial state exactly. For a 1.4 m, 3 A, 12 kV lens with radii 1 and 4 mm and residual
    normalized kicks (2.3, −4.1) × 10⁻⁷, a 1 GeV/c proton in the core returns with normalized transverse
    momenta (`px`, `py`) = (4.600, −8.200) × 10⁻⁷ instead of zero. Backtracking a lens with nonzero
    residual kicks is affected; the annular electron-lens force round-trips correctly.

    The script tracks two 1 GeV/c protons, one inside the core and one in the annulus, forward through
    the lens and back with `Line.track(backtrack=True)`, and expects their transverse momenta to return
    to exactly zero with positions and momentum deviation unchanged. As controls, the annular proton
    must round-trip, and an explicit inverse with negated length and residual kicks must restore both
    protons. It runs serially on the installed prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, particles, report, xt

    identity()
    cases = {}
    kx, ky = 2.3e-7, -4.1e-7
    lens_length = 1.4
    lens = xt.Elens(
        _context=CTX,
        inner_radius=0.001,
        outer_radius=0.004,
        current=3.0,
        voltage=12000.0,
        elens_length=lens_length,
        residual_kick_x=kx,
        residual_kick_y=ky,
    )
    assert lens.has_backtrack is True
    for mode in ("line_backtrack", "explicit_inverse_control"):
        p = particles(x=[0.0002, 0.002], y=[-0.0001, 0.0003], delta=[0.0, 0.0])
        initial = np.stack([np.array(getattr(p, f)) for f in ("x", "px", "y", "py", "zeta", "delta")])
        if mode == "explicit_inverse_control":
            inverse = lens.copy()
            inverse.elens_length = -lens_length
            inverse.residual_kick_x = -kx
            inverse.residual_kick_y = -ky
            lens.track(p)
            after_forward = (p.px.copy(), p.py.copy())
            inverse.track(p)
        else:
            line = xt.Line(elements=[lens])
            line.build_tracker(_context=CTX)
            line.track(p)
            after_forward = (p.px.copy(), p.py.copy())
            line.track(p, backtrack=True)
        final = np.stack([np.array(getattr(p, f)) for f in ("x", "px", "y", "py", "zeta", "delta")])
        print(mode, "FORWARD px=", after_forward[0], "py=", after_forward[1])
        print(mode, "ROUNDTRIP px=", p.px, "py=", p.py, "expected=[0,0]")
        if mode == "line_backtrack":
            np.testing.assert_array_equal([p.px[1], p.py[1]], [0.0, 0.0])
            cases["core round trip px,py"] = ([p.px[0], p.py[0]], [0.0, 0.0])
        else:
            np.testing.assert_array_equal(p.px, [0.0, 0.0])
            np.testing.assert_array_equal(p.py, [0.0, 0.0])
        np.testing.assert_array_equal(final[[0, 2, 4, 5]], initial[[0, 2, 4, 5]])
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

