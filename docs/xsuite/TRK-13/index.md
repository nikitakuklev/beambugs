---
title: "TRK-13: Exciter activates before its stated starting time"
hide:
  - toc
---

# TRK-13 — Exciter activates before its stated starting time

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`Exciter_track_local_particle`, exciter.h:43–47](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/exciter.h#L43-L47)<br>[`start_turn` contract](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/exciter.py#L33-L36) |

`Exciter` applies its first waveform sample to particles arriving less than one sample period before `start_turn`. The documented start time, and an independent calculation of which sample covers each arrival time, require no excitation before then. With `start_turn` = 5, revolution frequency 1 MHz, sampling frequency 0.25 MHz, `knl=[2.5e-6]` and first sample 0.75, the reference particle receives normalized horizontal momentum −1.875 × 10⁻⁶ at turn 4 instead of zero. Early arrivals are affected, and when sampling is slower than revolution the kick can start a whole turn early; samples within the waveform, arrivals more than one sample period early and times after the waveform ends are handled correctly.

??? example "Reproducer: TRK-13.py"

    [Download TRK-13.py](TRK-13.py)

    ```python
    """TRK-13: Exciter activates before its stated starting time

    `Exciter` applies its first waveform sample to particles arriving less than one sample period before
    `start_turn`. The documented start time, and an independent calculation of which sample covers each
    arrival time, require no excitation before then. With `start_turn` = 5, revolution frequency 1 MHz,
    sampling frequency 0.25 MHz, `knl=[2.5e-6]` and first sample 0.75, the reference particle receives
    normalized horizontal momentum −1.875 × 10⁻⁶ at turn 4 instead of zero. Early arrivals are affected,
    and when sampling is slower than revolution the kick can start a whole turn early; samples within
    the waveform, arrivals more than one sample period early and times after the waveform ends are
    handled correctly.

    The script scans arrival times from 1.25 sample periods before the start to 3.25 after it, through
    an exciter with three single-precision samples at 400 MHz, and compares each horizontal kick with
    zero-order hold applied only where the real-valued sample index lies inside the waveform. A second
    exciter, sampled at a quarter of the revolution frequency, kicks the reference particle one turn
    before `start_turn`. Arrivals inside the waveform, after it and more than one period early must
    match; the three arrivals just before the start and the early reference particle show the bug. It
    runs serially on the installed prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, particles, report, xt

    identity()
    cases = {}
    fs, frev, start = 4e8, 1e6, 5
    k = 2.5e-6
    samples = np.array([0.75, -0.5, 0.25], dtype=np.float32)
    element = xt.Exciter(
        _context=CTX,
        knl=[k],
        samples=samples,
        sampling_frequency=fs,
        frev=frev,
        start_turn=start,
    )
    # Real-valued sample coordinate u = fs ((at_turn - start_turn)/frev - zeta/(beta0 c)), support
    # [0, nsamples); inside it the kick is -knl*samples[floor(u)], outside it zero. The kernel
    # converts u to int64 first, truncating negative values toward zero.
    sample_times = np.array([-1.25, -0.99, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25, 2.25, 3.25])
    p = particles(zeta=np.zeros(len(sample_times)), at_turn=start)
    p.zeta = -sample_times * p.beta0[0] * 299792458.0 / fs
    # Derive expected waveform after checking the real-valued half-open support;
    # use floor for in-domain zero-order hold, without converting negatives to int.
    expected = np.zeros(len(sample_times))
    active = (sample_times >= 0) & (sample_times < len(samples))
    # The kernel promotes stored Float32 samples to double for multiplication by
    # its double-strength coefficient. NumPy otherwise keeps this product Float32.
    expected[active] = -k * samples.astype(np.float64)[np.floor(sample_times[active]).astype(int)]
    element.track(p)
    print("TIME_FROM_START in sample periods=", sample_times)
    print("PX observed=", np.asarray(p.px))
    print("PX expected=", expected)
    np.testing.assert_array_equal(p.px[[0, 4, 5, 6, 7, 8, 9]], expected[[0, 4, 5, 6, 7, 8, 9]])
    cases["before-start px"] = (np.asarray(p.px)[1:4], expected[1:4])
    np.testing.assert_array_equal(expected[1:4], [0.0, 0.0, 0.0])

    # The same early kick occurs with zeta=0 one full revolution before start
    # when sampling is slower than the revolution frequency. No oversized zeta.
    slow = xt.Exciter(
        _context=CTX,
        knl=[k],
        samples=[0.75],
        sampling_frequency=frev / 4,
        frev=frev,
        start_turn=start,
    )
    early_reference = particles(at_turn=start - 1, zeta=0.0)
    slow.track(early_reference)
    print(
        "EARLY_REFERENCE turn=",
        early_reference.at_turn[0],
        "start=",
        start,
        "sample_time=-0.25 px=",
        early_reference.px[0],
        "expected=0",
    )
    cases["early reference px"] = (early_reference.px[0], 0.0)
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

