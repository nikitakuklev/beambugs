---
title: "OPT-02: Native Taylor maps ignore an absolute-time cavity's RF clock"
hide:
  - toc
---

# OPT-02 — Native Taylor maps ignore an absolute-time cavity's RF clock

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Tags | <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> [?](../../methodology.md#tags) |
| Source | [`track_rf.h:51–58`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L51-L58) |

Tracking a `Cavity` with `absolute_time=True` through the experimental `ParticlesTpsa` API, which propagates native truncated-power-series (Taylor) maps, silently ignores the absolute RF clock and so uses the wrong RF phase. The absolute-arrival-time definition and the relativistic energy–momentum relation give the correct kick, and ordinary particle tracking reproduces it. For a 1 GeV/c proton at zero position and RF phase, a 1 MV, 100 MHz cavity with `time_s` = 2.5 ns gives a relative momentum change δ of zero instead of 1.371 × 10⁻³, a 1.000 MeV energy gain; relative-time controls are correct. A source comment acknowledges the restriction, but the public interface accepts the configuration without warning or rejection.

??? example "Reproducer: OPT-02.py"

    [Download OPT-02.py](OPT-02.py)

    ```python
    """OPT-02: Native Taylor maps ignore an absolute-time cavity's RF clock

    Tracking a `Cavity` with `absolute_time=True` through the experimental `ParticlesTpsa` API, which
    propagates native truncated-power-series (Taylor) maps, silently ignores the absolute RF clock and
    so uses the wrong RF phase. The absolute-arrival-time definition and the relativistic
    energy–momentum relation give the correct kick, and ordinary particle tracking reproduces it. For a
    1 GeV/c proton at zero position and RF phase, a 1 MV, 100 MHz cavity with `time_s` = 2.5 ns gives a
    relative momentum change δ of zero instead of 1.371 × 10⁻³, a 1.000 MeV energy gain; relative-time
    controls are correct. A source comment acknowledges the restriction, but the public interface
    accepts the configuration without warning or rejection.

    The script tracks a 1 GeV/c proton at a quarter RF period (`time_s` = 2.5 ns) through a 1 MV, 100
    MHz `Cavity` at zero phase, as an ordinary particle and as a first-order `ParticlesTpsa` map. With
    `absolute_time=True` the RF phase is π/2, so the proton must gain exactly 1 MeV; ordinary tracking
    is checked against that before the map's constant part is compared. With `absolute_time=False` both
    give zero, which isolates the clock term.
    """

    import numpy as np
    import xtrack as xt
    from xtrack.tpsa import ParticlesTpsa

    RTOL, ATOL = 2e-13, 2e-16


    def evaluate():
        frequency, voltage, p0c = 1e8, 1e6, 1e9
        cases = {}
        for absolute_time in (False, True):
            element = xt.Cavity(
                voltage=voltage, frequency=frequency, phase=0, absolute_time=absolute_time
            )
            kwargs = {"p0c": p0c, "time_s": 0.25 / frequency, "s": 0, "zeta": 0}
            scalar = xt.Particles(**kwargs)
            series = ParticlesTpsa(order=1, p0c=p0c, time_s=0.25 / frequency, s=0, zeta=0)
            # At t=T/4 the absolute RF phase is pi/2; at relative phase zero it is zero.
            ptau = voltage / p0c if absolute_time else 0.0
            beta0 = float(scalar.beta0[0])
            u = ptau * (ptau + 2 / beta0)
            expected = u / (np.sqrt(1 + u) + 1)
            element.track(scalar)
            np.testing.assert_allclose(scalar.delta[0], expected, rtol=RTOL, atol=ATOL)
            element.track(series)
            assert series._xobject.state == 1
            if not absolute_time:
                np.testing.assert_allclose(series.const_part[5], expected, rtol=RTOL, atol=ATOL)
            cases[f"Cavity absolute_time={absolute_time}, TPSA delta"] = series.const_part[5], expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: at absolute time T_RF/4, Delta E=V=1e6 eV; relative-time control has zero kick."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

