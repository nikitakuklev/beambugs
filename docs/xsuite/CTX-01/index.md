---
title: "CTX-01: CPU linked-array views bypass dependent-state updates"
hide:
  - toc
---

# CTX-01 — CPU linked-array views bypass dependent-state updates

| | |
|---|---|
| Code | [Xsuite](../index.md) · Contexts and kernels |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xobjects main (`dfe34ae`) |
| Source | [`LinkedArrayCpu`, context_cpu.py:172–187](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/xobjects/context_cpu.py#L172-L187)<br>[`BaseLinkedArray.__setitem__`, linkedarray.py:27–36](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/xobjects/linkedarray.py#L27-L36)<br>[`Particles.delta` setter](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1213-L1225) |

Writing through a shared CPU slice of `Particles.delta` changes the relative momentum deviation δ without updating the dependent energy and momentum fields, so subsequent tracking uses inconsistent coordinates. Relativistic kinematics and the expanded-drift map require those fields to change together, as direct indexed assignment does. Setting a 1 GeV/c proton to δ = 0.3 through a view, then tracking 1.7 m with normalized horizontal momentum `px` = 0.0008, gives x = 1.360 mm instead of 1.046 mm. Strided views, the output of NumPy universal functions (`out=`) and aliased in-place writes also bypass updates, and slices of read-only arrays accept prohibited writes. Direct indexed assignment preserves the required consistency.

??? example "Reproducer: CTX-01.py"

    [Download CTX-01.py](CTX-01.py)

    ```python
    """CTX-01: CPU linked-array views bypass dependent-state updates

    Writing through a shared CPU slice of `Particles.delta` changes the relative momentum deviation δ
    without updating the dependent energy and momentum fields, so subsequent tracking uses inconsistent
    coordinates. Relativistic kinematics and the expanded-drift map require those fields to change
    together, as direct indexed assignment does. Setting a 1 GeV/c proton to δ = 0.3 through a view,
    then tracking 1.7 m with normalized horizontal momentum `px` = 0.0008, gives x = 1.360 mm instead of
    1.046 mm. Strided views, the output of NumPy universal functions (`out=`) and aliased in-place
    writes also bypass updates, and slices of read-only arrays accept prohibited writes. Direct indexed
    assignment preserves the required consistency.

    The script sets δ = 0.3 on the second of four 1 GeV/c protons with `px` = 0.0008 through five paths
    (direct indexed assignment, a slice view, a strided view, a universal function with `out=`, and an
    aliased in-place add) and tracks each through a 1.7 m expanded drift. It compares the resulting
    `rpp`, `ptau` and x with 1/(1 + δ), the relativistic mass-shell value of p_τ and L·`px`/(1 + δ);
    direct assignment is the control. It also writes through a slice of the read-only `rpp` array, which
    must be rejected as direct indexing is.
    """

    from _prebuilt import CTX, identity, kernels, np, particles, report, xt

    identity()
    cases = {}
    drift = xt.Drift(_context=CTX, length=1.7, model="expanded")
    for mode in ("direct", "view", "strided_view", "ufunc_out", "aliased_inplace"):
        p = particles(delta=[0.0, 0.0, 0.0, 0.0], px=[0.0008] * 4)
        if mode == "direct":
            p.delta[1] = 0.3
        elif mode == "view":
            view = p.delta[1:]
            print(
                "VIEW",
                type(view).__name__,
                "mode=",
                view.mode,
                "shares_memory=",
                np.shares_memory(view, p.delta),
            )
            view[0] = 0.3
        elif mode == "strided_view":
            p.delta[1::2][0] = 0.3
        elif mode == "ufunc_out":
            increments = np.array([0.0, 0.3, 0.0, 0.0])
            np.add(p.delta, increments, out=p.delta)
        else:
            alias = p.delta
            alias += [0.0, 0.3, 0.0, 0.0]
        np.testing.assert_array_equal(p.delta, [0.0, 0.3, 0.0, 0.0])
        expected_rpp = 1 / 1.3
        # Rationalized relativistic energy conversion, independent of Xtrack setter.
        beta0 = 1e9 / np.hypot(1e9, p.mass0)
        u = beta0**2 * 0.3 * 2.3
        expected_ptau = beta0 * 0.3 * 2.3 / (np.sqrt(1 + u) + 1)
        expected_x = 1.7 * 0.0008 / 1.3
        rpp, ptau = float(p.rpp[1]), float(p.ptau[1])
        drift.track(p)
        print(
            mode,
            f"rpp={rpp:.17g} expected={expected_rpp:.17g}",
            f"ptau={ptau:.17g} expected={expected_ptau:.17g}",
            f"x={p.x[1]:.17g} expected={expected_x:.17g}",
        )
        if mode == "direct":
            np.testing.assert_allclose(rpp, expected_rpp, rtol=2e-15, atol=0)
            np.testing.assert_allclose(ptau, expected_ptau, rtol=3e-15, atol=0)
            np.testing.assert_allclose(p.x[1], expected_x, rtol=2e-15, atol=0)
        else:
            cases[mode + " rpp,ptau,x"] = (
                [rpp, ptau, p.x[1]],
                [expected_rpp, expected_ptau, expected_x],
            )

    # No subtle energy-coordinate contract is needed to demonstrate that the same
    # ndarray subclass silently loses its explicit readonly setting through a view.
    p = particles(delta=[0.0, 0.0])
    try:
        p.rpp[1] = 0.5
    except ValueError as exc:
        print("READONLY direct:", str(exc))
    else:
        raise AssertionError("Readonly direct control did not reject mutation")
    try:
        p.rpp[1:][0] = 0.5
    except ValueError:
        pass  # A view that enforces the read-only contract leaves rpp unchanged.
    print("READONLY sliced:", p.rpp, "delta=", p.delta)
    cases["read-only view rpp,delta"] = ([p.rpp[1], p.delta[1]], [1.0, 0.0])
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xobjects 0.7.2 (`dfe34ae824bd`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xobjects 0.7.2 (`dfe34ae824bd`), installed packages, prebuilt kernels, CPU, serial |

