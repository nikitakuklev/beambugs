---
title: "CTX-02: Copied nested fields disconnect from parent storage"
hide:
  - toc
---

# CTX-02 — Copied nested fields disconnect from parent storage

| | |
|---|---|
| Code | [Xsuite](../index.md) · Contexts and kernels |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xobjects main (`dfe34ae`) |
| Source | [HybridClass.copy, hybrid_class.py:344–360](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/xobjects/hybrid_class.py#L344-L360)<br>[nested-object tests](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/tests/test_hybrid_class.py#L95-L138) |

Copying a `SpaceChargeBiGaussian` and changing its public longitudinal-profile population leaves the tracking kernel using the old intensity, because the copied child fields are disconnected from the parent's storage. At fixed geometry the kick must scale linearly with population, as a freshly constructed control confirms. For a 0.7 m element with longitudinal RMS size 0.13 m and transverse RMS size 1.2 mm, increasing the population from 7.000×10⁹ to 1.750×10¹⁰ leaves a 1.3 GeV/c proton at the bunch center and x = 0.7 mm with normalized horizontal momentum `px` = 3.145×10⁻⁶ instead of 7.862×10⁻⁶. Both default-buffer and same-buffer copies fail. The original object remains unchanged.

??? example "Reproducer: CTX-02.py"

    [Download CTX-02.py](CTX-02.py)

    ```python
    """CTX-02: Copied nested fields disconnect from parent storage

    Copying a `SpaceChargeBiGaussian` and changing its public longitudinal-profile population leaves the
    tracking kernel using the old intensity, because the copied child fields are disconnected from the
    parent's storage. At fixed geometry the kick must scale linearly with population, as a freshly
    constructed control confirms. For a 0.7 m element with longitudinal RMS size 0.13 m and transverse
    RMS size 1.2 mm, increasing the population from 7.000×10⁹ to 1.750×10¹⁰ leaves a 1.3 GeV/c proton at
    the bunch center and x = 0.7 mm with normalized horizontal momentum `px` = 3.145×10⁻⁶ instead of
    7.862×10⁻⁶. Both default-buffer and same-buffer copies fail. The original object remains unchanged.

    The script first shows on a minimal nested struct that setting a copied child field (gain 11) leaves
    the copy's raw storage at the old value (3). It then copies a frozen round `SpaceChargeBiGaussian`
    (population 7×10⁹, longitudinal RMS 0.13 m, transverse RMS 1.2 mm, length 0.7 m) into a new and into
    the same buffer, sets the copy's population to 1.75×10¹⁰ and tracks two 1.3 GeV/c protons at x = 0.7
    and 1.9 mm, `zeta` = 0 and 0.08 m. The kicks must be 2.5 times the original, as a freshly built
    element with the higher population gives; the copies give exactly the original kick.
    """

    from typing import ClassVar

    import xfields as xf
    import xobjects as xo
    from _prebuilt import CTX, identity, kernels, np, xt

    identity()
    print("ORACLE population scaling")


    class Cell(xo.HybridClass):
        gain: float
        _xofields: ClassVar[dict] = {"gain": xo.Float64}


    class Pair(xo.HybridClass):
        cell: Cell
        _xofields: ClassVar[dict] = {"cell": Cell}


    p = Pair(cell=Cell(gain=3), _context=CTX)
    q = p.copy()
    q.cell.gain = 11
    print("STRUCT copied Python/raw/original", q.cell.gain, q._xobject.cell.gain, p.cell.gain)
    assert q.cell.gain == 11 and q._xobject.cell.gain == 3 and p.cell.gain == 3


    def make(pop):
        return xf.SpaceChargeBiGaussian(
            longitudinal_profile=xf.LongitudinalProfileQGaussian(
                number_of_particles=pop, sigma_z=0.13, _context=CTX
            ),
            length=0.7,
            sigma_x=0.0012,
            sigma_y=0.0012,
            _context=CTX,
        )


    def kick(element):
        particles = xt.Particles(p0c=1.3e9, x=[0.0007, 0.0019], zeta=[0, 0.08], _context=CTX)
        element.track(particles)
        return np.array(particles.px)


    original = make(7e9)
    base = kick(original)
    control = kick(make(17.5e9))
    np.testing.assert_allclose(control, 2.5 * base, rtol=3e-15, atol=0)
    for mode in ("new_buffer", "same_buffer"):
        copied = original.copy(**({"_buffer": original._buffer} if mode == "same_buffer" else {}))
        copied.longitudinal_profile.number_of_particles = 17.5e9
        got = kick(copied)
        print(
            mode,
            "population Python/raw",
            copied.longitudinal_profile.number_of_particles,
            copied._xobject.longitudinal_profile.number_of_particles,
        )
        print(mode, "kick observed", got, "expected", control, "ratio_to_original", got / base)
        np.testing.assert_array_equal(got, base)
        assert (
            copied.longitudinal_profile.number_of_particles
            != copied._xobject.longitudinal_profile.number_of_particles
        )
        assert original.longitudinal_profile.number_of_particles == 7e9
        copied._xobject.longitudinal_profile.number_of_particles = 17.5e9
        np.testing.assert_allclose(kick(copied), control, rtol=3e-15, atol=0)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xobjects 0.7.2 (`dfe34ae824bd`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xobjects 0.7.2 (`dfe34ae824bd`), installed packages, prebuilt kernels, CPU, serial |

