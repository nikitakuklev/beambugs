---
title: "OPT-12: Knob construction applies absolute limits to additive corrections"
hide:
  - toc
---

# OPT-12 — Knob construction applies absolute limits to additive corrections

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [KnobOptimizer.__init__, match.py:1334–1351](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/match.py#L1334-L1351)<br>[Vary.limits contract](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/match.py#L454-L456) |

`match_knob()` applies a variable's absolute `Vary.limits` to the additive correction it solves for, without subtracting the variable's starting value. The correction's allowed interval must be the limits shifted by that starting value, and ordinary matching respects the absolute bounds. For a dimensionless variable starting at 8 with limits [−10, 10] and target 12, the generated knob reaches 12.00 instead of stopping at the allowed maximum 10.00. Conversely, starting at 8 with valid limits [7, 10] and target 9 fails during knob setup. Knob construction from a nonzero starting value is affected at the matched endpoint itself, without extrapolation or later combination of knobs.

??? example "Reproducer: OPT-12.py"

    [Download OPT-12.py](OPT-12.py)

    ```python
    """OPT-12: Knob construction applies absolute limits to additive corrections

    `match_knob()` applies a variable's absolute `Vary.limits` to the additive correction it solves for,
    without subtracting the variable's starting value. The correction's allowed interval must be the
    limits shifted by that starting value, and ordinary matching respects the absolute bounds. For a
    dimensionless variable starting at 8 with limits [−10, 10] and target 12, the generated knob reaches
    12.00 instead of stopping at the allowed maximum 10.00. Conversely, starting at 8 with valid limits
    [7, 10] and target 9 fails during knob setup. Knob construction from a nonzero starting value is
    affected at the matched endpoint itself, without extrapolation or later combination of knobs.

    The script uses one variable k on an empty line, starting at 8 and read back as the target by a
    tracking-free action. With limits [−10, 10] and target 12, ordinary `Line.match()` stops at 10,
    while `match_knob()` keeps the unshifted limits on the correction a in k = 8 + a instead of [−18,
    2]; the generated knob is checked at its start, midpoint and endpoint. With limits [7, 10] and
    target 9, ordinary matching succeeds and knob setup raises an error.
    """

    from _matching import ReadKnobs
    from _prebuilt import identity, np, xt

    identity()
    print("EXECUTION Python only; no tracking kernels invoked")


    def construct(limits, goal, method):
        line = xt.Line(elements=[])
        line.build_tracker(compile=False)  # This check evaluates variables without tracking.
        line["k"] = 8.0
        action = ReadKnobs(line, ("k",))
        params = {
            "vary": [xt.Vary("k", limits=limits, step=1e-4)],
            "targets": [xt.Target("k", goal, tol=1e-11, action=action)],
            "verbose": False,
            "show_call_counter": False,
        }
        if method == "knob":
            opt = line.match_knob(
                knob_name="scan",
                knob_value_start=0.0,
                knob_value_end=1.0,
                run=False,
                **params,
            )
        else:
            opt = line.match(solve=False, **params)
        return line, opt


    for mode in ("direct", "knob"):
        line, opt = construct((-10.0, 10.0), 12.0, mode)
        opt.step(2)
        print(
            mode,
            "after_two_steps k",
            line["k"],
            "varied",
            [(v.name, v.get_value(), v.limits) for v in opt.vary],
        )
        if mode == "direct":
            np.testing.assert_allclose(line["k"], 10.0, rtol=0, atol=1e-12)
        else:
            opt.solve()
            np.testing.assert_allclose(line["k"], 12.0, rtol=0, atol=1e-11)
            opt.generate_knob()
            np.testing.assert_allclose(line["k"], 8.0, rtol=0, atol=1e-12)
            line["scan"] = 0.5
            np.testing.assert_allclose(line["k"], 10.0, rtol=0, atol=1e-11)
            line["scan"] = 1.0
            print("generated endpoint k", line["k"], "specified interval", (-10.0, 10.0))
            np.testing.assert_allclose(line["k"], 12.0, rtol=0, atol=1e-11)

    # Complementary falsification: an initial value and target both within limits.
    line, opt = construct((7.0, 10.0), 9.0, "direct")
    opt.solve()
    np.testing.assert_allclose(line["k"], 9.0, rtol=0, atol=1e-11)
    try:
        construct((7.0, 10.0), 9.0, "knob")
    except ValueError as exc:
        print("legal k=8, target=9, limits=[7,10] rejected during knob setup:", str(exc))
        assert "below lower limit" in str(exc)
    else:
        raise AssertionError("Expected untranslated auxiliary bound to reject zero correction")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, Python only (no tracking kernels), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, Python only (no tracking kernels), CPU, serial |

