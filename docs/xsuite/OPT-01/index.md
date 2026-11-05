---
title: "OPT-01: Native Taylor maps silently omit RFMultipole and CrabCavity kicks"
hide:
  - toc
---

# OPT-01 — Native Taylor maps silently omit RFMultipole and CrabCavity kicks

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`track_rf_kick_single_particle`, track_rf.h:68–169](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L68-L169)<br>[`TPSA_SUPPORTED_ELEMENTS`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/prebuilt_kernel_definitions/element_types.py#L127-L148)<br>[public API](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/apireference.rst#L1090-L1126) |

Tracking an `RFMultipole` or `CrabCavity` with the experimental `ParticlesTpsa` API, which propagates native truncated-power-series (Taylor) maps, silently drops the element's transverse kick and the energy change that goes with it. The RF potential predicts both kicks, and ordinary particle tracking reproduces them. For a 1 GeV/c proton at x = 3 mm crossing a 1 MV, 100 MHz crab cavity at phase π/4 rad, the map gives zero normalized horizontal momentum pₓ and relative momentum change δ instead of 7.071 × 10⁻⁴ and −6.097 × 10⁻⁶; an RF dipole with `knl[0]` = 0.001 likewise loses its opposite kick. Ordinary finite-difference `Line.twiss()` is not shown to fail.

??? example "Reproducer: OPT-01.py"

    [Download OPT-01.py](OPT-01.py)

    ```python
    """OPT-01: Native Taylor maps silently omit RFMultipole and CrabCavity kicks

    Tracking an `RFMultipole` or `CrabCavity` with the experimental `ParticlesTpsa` API, which
    propagates native truncated-power-series (Taylor) maps, silently drops the element's transverse kick
    and the energy change that goes with it. The RF potential predicts both kicks, and ordinary particle
    tracking reproduces them. For a 1 GeV/c proton at x = 3 mm crossing a 1 MV, 100 MHz crab cavity at
    phase π/4 rad, the map gives zero normalized horizontal momentum pₓ and relative momentum change δ
    instead of 7.071 × 10⁻⁴ and −6.097 × 10⁻⁶; an RF dipole with `knl[0]` = 0.001 likewise loses its
    opposite kick. Ordinary finite-difference `Line.twiss()` is not shown to fail.

    The script tracks a 1 GeV/c proton at x = 3 mm through an `RFMultipole` (`knl[0]` = 0.001) and a 1
    MV `CrabCavity`, both at 100 MHz and phase π/4, once as an ordinary particle and once as a
    first-order `ParticlesTpsa` map. The ordinary particle must first match the analytic kick of the
    order-zero RF potential, with δ from the exact relativistic conversion of the energy change; the
    constant part of the map is then compared with the same values.
    """

    import numpy as np
    import xtrack as xt
    from scipy.constants import c
    from xtrack.tpsa import ParticlesTpsa

    RTOL, ATOL = 2e-13, 2e-16


    def evaluate():
        p0c, x, frequency = 1e9, 0.003, 1e8
        phase = np.pi / 4
        # Both chosen order-zero potentials give the same longitudinal kick.
        ptau = -(2 * np.pi * frequency / c) * 0.001 * x * np.sin(phase)
        beta0 = float(xt.Particles(p0c=p0c).beta0[0])
        radicand_change = ptau * (ptau + 2 / beta0)
        delta = radicand_change / (np.sqrt(1 + radicand_change) + 1)
        cases = {}
        # Order-zero transverse kicks: px = -knl[0]*cos(phi) for the RF dipole and
        # (V/p0c)*sin(phi) for the horizontal crab cavity, both 0.001 in magnitude here.
        for element, px in (
            (xt.RFMultipole(knl=[0.001], phase_n=[phase], frequency=frequency), -0.001 * np.cos(phase)),
            (xt.CrabCavity(crab_voltage=1e6, phase=phase, frequency=frequency), 0.001 * np.sin(phase)),
        ):
            expected = np.array([px, delta])
            scalar = xt.Particles(p0c=p0c, x=x)
            series = ParticlesTpsa(order=1, p0c=p0c, x=x)
            element.track(scalar)
            np.testing.assert_allclose([scalar.px[0], scalar.delta[0]], expected, rtol=RTOL, atol=ATOL)
            element.track(series)
            assert series._xobject.state == 1
            cases[f"{type(element).__name__} TPSA px,delta"] = series.const_part[[1, 5]], expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: RF dipole potential, including its longitudinal energy derivative.")
        print("Scalar controls must match the analytic kick before testing the TPSA constant part.")
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

