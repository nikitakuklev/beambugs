---
title: "IO-09: RF phase setter leaves cached bucket geometry stale"
hide:
  - toc
---

# IO-09 — RF phase setter leaves cached bucket geometry stale

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xpart main (`2633e52`) |
| Source | [RFBucket.phi_offset_list setter and cached fixed points, rf_bucket.py:164–190](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/longitudinal/rf_bucket.py#L164-L190) |

After an `RFBucket`'s geometry has been read, setting the RF phase `phi_offset_list` leaves the cached stable points and bucket boundaries unchanged, and the synchrotron tune Qₛ is recalculated from the new phase and the stale stable point, giving an inconsistent value. For a 100 m accelerating proton bucket with Lorentz factor γ = 10, harmonic 10, voltage 1 MV and acceleration fraction 0.5, changing the phase from 0 to 0.4 rad leaves the stable point at −0.8333 m instead of −0.1967 m. Fresh construction and the analytic translation give the new geometry and an unchanged Qₛ = 1.156×10⁻³, while the stale-root calculation gives 9.643×10⁻⁴. Setting the phase before reading geometry, or calling `add_fields([], [])`, gives correct results.

??? example "Reproducer: IO-09.py"

    [Download IO-09.py](IO-09.py)

    ```python
    """IO-09: RF phase setter leaves cached bucket geometry stale

    After an `RFBucket`'s geometry has been read, setting the RF phase `phi_offset_list` leaves the
    cached stable points and bucket boundaries unchanged, and the synchrotron tune Qₛ is recalculated
    from the new phase and the stale stable point, giving an inconsistent value. For a 100 m
    accelerating proton bucket with Lorentz factor γ = 10, harmonic 10, voltage 1 MV and acceleration
    fraction 0.5, changing the phase from 0 to 0.4 rad leaves the stable point at −0.8333 m instead of
    −0.1967 m. Fresh construction and the analytic translation give the new geometry and an unchanged Qₛ
    = 1.156×10⁻³, while the stale-root calculation gives 9.643×10⁻⁴. Setting the phase before reading
    geometry, or calling `add_fields([], [])`, gives correct results.

    The script builds a 100 m proton bucket with γ = 10, harmonic 10, voltage 1 MV, momentum compaction
    0.001 and acceleration fraction 0.5, reads its stable point, boundaries and Qₛ, then changes the
    phase from 0 to 0.4 rad. It compares the cached values with a freshly built bucket and with the
    analytic translation of the stable point, boundaries and tune, and shows that the stale stable point
    has nonzero total force. Setting the phase before any geometry is read, and calling `add_fields([],
    [])` after the change, are controls that must give the fresh result.
    """

    from _prebuilt import identity, np, xt
    from scipy.constants import c, e
    from xpart.longitudinal.rf_bucket import RFBucket

    identity()
    print("EXECUTION Python only; no tracking kernels invoked")
    C, gamma, h, V = 100.0, 10.0, 10.0, 1e6
    mass = xt.PROTON_MASS_EV * e / c**2
    beta = np.sqrt(1 - gamma**-2)
    a = 0.5
    phase_change = 0.4


    def bucket(phase):
        return RFBucket(
            C,
            gamma,
            mass,
            e,
            [0.001],
            a * V * e / (beta * c),
            [h],
            [V],
            [phase],
            zeta0=None,
        )


    cached = bucket(0.0)
    old_s = cached.z_sfp.copy()
    old_bounds = np.array([cached.z_left, cached.z_right])
    old_tune = cached.Q_s.copy()
    cached.phi_offset_list = np.array([phase_change])
    fresh = bucket(phase_change)
    expected_s = (phase_change - np.arcsin(a)) * C / (2 * np.pi * h)
    shift = phase_change * C / (2 * np.pi * h)
    expected_tune = np.sqrt(
        -e * cached.eta0 * h * V * np.sqrt(1 - a * a) / (2 * np.pi * cached.p0 * beta * c)
    )
    print("stable cached/fresh/analytic", cached.z_sfp, fresh.z_sfp, expected_s)
    print(
        "bounds cached",
        [cached.z_left, cached.z_right],
        "fresh",
        [fresh.z_left, fresh.z_right],
        "expected shift",
        shift,
    )
    print(
        "force_at_cached_stable / RF_amplitude",
        cached.total_force(cached.z_sfp) / (e * V / C),
    )
    print("Qs cached/fresh/analytic", cached.Q_s, fresh.Q_s, expected_tune)
    np.testing.assert_allclose(fresh.z_sfp, expected_s, rtol=0, atol=3e-13)
    np.testing.assert_allclose([fresh.z_left, fresh.z_right], old_bounds + shift, rtol=0, atol=3e-12)
    np.testing.assert_allclose(fresh.Q_s, expected_tune, rtol=2e-14, atol=0)
    np.testing.assert_array_equal(cached.z_sfp, old_s)
    np.testing.assert_array_equal([cached.z_left, cached.z_right], old_bounds)
    assert abs(float(cached.total_force(cached.z_sfp)[0]) / (e * V / C)) > 0.2
    # Falsification control: same setter before the cache is populated is correct.
    uncached = bucket(0.0)
    uncached.phi_offset_list = np.array([phase_change])
    np.testing.assert_allclose(uncached.z_sfp, fresh.z_sfp, rtol=0, atol=3e-13)
    # add_fields explicitly invalidates the geometry caches (an available workaround).
    cached.add_fields([], [])
    np.testing.assert_allclose(cached.z_sfp, fresh.z_sfp, rtol=0, atol=3e-13)
    np.testing.assert_allclose(
        [cached.z_left, cached.z_right], [fresh.z_left, fresh.z_right], rtol=0, atol=3e-12
    )
    np.testing.assert_allclose(cached.Q_s, expected_tune, rtol=2e-14, atol=0)
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

