---
title: "COL-17: All-lost tracking calls freeze wake history"
hide:
  - toc
---

# COL-17 — All-lost tracking calls freeze wake history

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`) |
| Source | [`WakeTracker.track`, waketracker.py:125–150](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/waketracker/waketracker.py#L125-L150)<br>[advancing its moments history](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/element_with_slicer.py#L213) |

`WakeTracker.track` returns early when no particle is alive, before advancing the stored wake-source history, so a turn in which the whole bunch is lost does not age the history. A turn without sources must still age it, as calls with live particles outside the slicing window or with zero weight do, and direct convolution gives the expected delays and expiry. With a three-turn history in a 20 m ring and a 1.4 GeV/c proton source of weight 5, a witness one empty turn later receives `px` = 4.024 × 10⁻⁹ instead of 2.441 × 10⁻⁹, and two empty turns later 4.024 × 10⁻⁹ instead of 0, after the source should have expired.

??? example "Reproducer: COL-17.py"

    [Download COL-17.py](COL-17.py)

    ```python
    """COL-17: All-lost tracking calls freeze wake history

    `WakeTracker.track` returns early when no particle is alive, before advancing the stored wake-source
    history, so a turn in which the whole bunch is lost does not age the history. A turn without sources
    must still age it, as calls with live particles outside the slicing window or with zero weight do,
    and direct convolution gives the expected delays and expiry. With a three-turn history in a 20 m
    ring and a 1.4 GeV/c proton source of weight 5, a witness one empty turn later receives `px` = 4.024
    × 10⁻⁹ instead of 2.441 × 10⁻⁹, and two empty turns later 4.024 × 10⁻⁹ instead of 0, after the
    source should have expired.

    The script uses four slices over [−1, 1] m and a causal transverse wake W(d) = 10¹⁹·exp(d/40) for d
    < −5 m and zero otherwise, so no kick arises within the same turn. A source at ζ = 0.75 m is
    followed by zero, one or two explicitly numbered empty turns and then a witness at ζ = −0.75 m,
    whose kick is compared with direct convolution. The empty turns consist of lost particles, live
    particles outside the slicing window, or live particles of zero weight; the last two are controls.
    It requires the installed prebuilt kernels (compilation is forbidden).
    """

    from typing import ClassVar

    import xfields as xf
    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import e

    identity()


    # Delayed causal monopole transverse wake: no self-wake within this bunch.
    class DelayedWake:
        source_exponents = (0, 0)
        test_exponents = (0, 0)
        source_moments: ClassVar[list] = ["num_particles"]
        _source_moment_exponents: ClassVar[list] = [1]
        plane = "x"
        kick = "px"

        def function_vs_zeta(self, zeta, beta0, dzeta):
            return np.where(zeta < -5, 1e19 * np.exp(zeta / 40), 0)


    def make():
        return xf.WakeTracker(
            components=[DelayedWake()],
            num_turns=3,
            circumference=20,
            zeta_range=(-1, 1),
            num_slices=4,
            _context=CTX,
        )


    def bunch(zeta, turn, state=1, weight=1):
        return xt.Particles(
            p0c=1.4e9, zeta=zeta, at_turn=turn, state=state, weight=weight, _context=CTX
        )


    def populations(w):
        return [
            float(np.sum(w.moments_data.get_source_moment_profile("num_particles", i, 0)[1]))
            for i in range(3)
        ]


    for gaps in (0, 1, 2):
        observations = {}
        for empty in ("lost", "outside", "zero_weight"):
            wake = make()
            wake.track(bunch(0.75, 0, weight=5))
            for t in range(1, gaps + 1):
                wake.track(
                    bunch(
                        4 if empty == "outside" else 0.75,
                        t,
                        state=0 if empty == "lost" else 1,
                        weight=0 if empty == "zero_weight" else 1,
                    )
                )
            before = populations(wake)
            test = bunch(-0.75, gaps + 1)
            wake.track(test)
            got = float(test.px[0])
            expected = (
                (5e19 * np.exp((-1.5 - (gaps + 1) * 20) / 40) * e / (1.4e9 * float(test.beta0[0])))
                if gaps + 1 < 3
                else 0.0
            )
            observations[empty] = got
            print(
                "gap",
                gaps,
                empty,
                "history_before_witness",
                before,
                "kick",
                f"{got:.17g}",
                "expected",
                f"{expected:.17g}",
            )
            if empty != "lost" or gaps == 0:
                np.testing.assert_allclose(got, expected, rtol=6e-15, atol=1e-27)
            else:
                assert abs(got - expected) > 1e-11
        np.testing.assert_allclose(
            observations["outside"], observations["zero_weight"], rtol=0, atol=1e-27
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

