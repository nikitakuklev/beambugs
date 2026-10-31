---
title: "OPT-11: Variable weights distort physical maximum steps"
hide:
  - toc
---

# OPT-11 — Variable weights distort physical maximum steps

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xdeps main (`6484795`) |
| Source | [MeritFunctionForMatch._clip_to_max_steps, optimize.py:417–425](https://github.com/xsuite/xdeps/blob/73147cf7d9f1fc7b0f49ddb55c0ce037b70b7353/xdeps/optimize/optimize.py#L417-L425)<br>[_x_to_knobs, optimize.py:220](https://github.com/xsuite/xdeps/blob/73147cf7d9f1fc7b0f49ddb55c0ce037b70b7353/xdeps/optimize/optimize.py#L220)<br>[Vary.max_step contract](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/match.py#L461-L462) |

Giving a matching variable a `weight` other than one changes its effective `max_step` in the variable's own units. The solver works in weighted coordinates, so converting back to the variable must convert the maximum step too, leaving the accepted physical bound independent of weight. For a dimensionless variable starting at 3, targeting 20 and limited to `max_step` = 0.2, one accepted `Line.match()` iteration changes it by 2.000 with weight 10 and by 0.02000 with weight 0.1, instead of 0.2000. Weighted variables can therefore exceed the requested bound or be needlessly restricted; unit weight is correct, and a single variable suffices to trigger the error.

??? example "Reproducer: OPT-11.py"

    [Download OPT-11.py](OPT-11.py)

    ```python
    """OPT-11: Variable weights distort physical maximum steps

    Giving a matching variable a `weight` other than one changes its effective `max_step` in the
    variable's own units. The solver works in weighted coordinates, so converting back to the variable
    must convert the maximum step too, leaving the accepted physical bound independent of weight. For a
    dimensionless variable starting at 3, targeting 20 and limited to `max_step` = 0.2, one accepted
    `Line.match()` iteration changes it by 2.000 with weight 10 and by 0.02000 with weight 0.1, instead
    of 0.2000. Weighted variables can therefore exceed the requested bound or be needlessly restricted;
    unit weight is correct, and a single variable suffices to trigger the error.

    The script takes one accepted `Line.match()` iteration on an empty line with a single variable, read
    back as the target by a tracking-free action, for weights w = 0.1, 1 and 10. Solver coordinates x
    relate to the variable k by k = w·x, so the bound in solver units must be `max_step`/w; the script
    checks that conversion and shows that the physical change is w × 0.2 instead. Unit weight is the
    control.
    """

    from _matching import optimizer
    from _prebuilt import identity, np

    identity()
    print("EXECUTION Python only; no tracking kernels invoked")
    for weight in (0.1, 1.0, 10.0):
        line, opt = optimizer({"k": 3.0}, {"k": 20.0}, {"k": 0.2}, weights={"k": weight})
        opt.step(1)
        physical_change = line["k"] - 3.0
        print(
            "weight",
            weight,
            "physical_change",
            f"{physical_change:.17g}",
            "max_step",
            0.2,
            "solver_change",
            opt.solver._last_xstep,
        )
        np.testing.assert_allclose(physical_change, weight * 0.2, rtol=0, atol=1e-14)
        if weight > 1:
            assert physical_change > 0.2 + 1e-12
        # Independent unit conversion: k=w*x implies |dx| <= max_step/w.
        np.testing.assert_allclose(opt._err._x_to_knobs([0.2 / weight]), [0.2], rtol=1e-15, atol=0)
    print("DEFECT REPRODUCED")

    ```

??? example "Helper: _matching.py"

    [Download _matching.py](_matching.py)

    ```python
    """Tracking-free matching setup shared by the step-limit and knob reproducers.

    `ReadKnobs` is an action that returns the current values of named line variables, so targets can be
    set on the variables themselves. `optimizer` builds a public `Line.match()` (not yet solved) on an
    empty line without a tracking kernel, with given starting values, goals, maximum steps, optional
    weights and variable order.
    """

    import xdeps as xd
    from _prebuilt import xt


    class ReadKnobs(xd.Action):
        def __init__(self, line, names):
            self.line = line
            self.names = names

        def run(self):
            return {name: self.line[name] for name in self.names}


    def optimizer(initial, goals, maxima, weights=None, order=None):
        line = xt.Line(elements=[])
        line.build_tracker(compile=False)  # Actions read variables; no tracking kernel is needed.
        for name, value in initial.items():
            line[name] = value
        names = tuple(initial)
        action = ReadKnobs(line, names)
        opt = line.match(
            solve=False,
            verbose=False,
            show_call_counter=False,
            vary=[
                xt.Vary(
                    name,
                    step=1e-4,
                    max_step=maxima[name],
                    weight=1.0 if weights is None else weights[name],
                )
                for name in (names if order is None else order)
            ],
            targets=[xt.Target(name, goals[name], tol=1e-11, action=action) for name in names],
        )
        return line, opt

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xdeps 0.10.21 (`73147cf7d9f1`), installed wheels, Python only (no tracking kernels), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xdeps 0.10.21 (`73147cf7d9f1`), installed wheels, Python only (no tracking kernels), CPU, serial |

