---
title: "TRK-04: Optimizing RF slices drops roll and phase taper"
hide:
  - toc
---

# TRK-04 — Optimizing RF slices drops roll and phase taper

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`ThinSliceCrabCavity.get_equivalent_element`, slice_elements_thin.py:162–173](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L162-L173)<br>[`ThinSliceCavity` converter](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L139-L149) |

`Line.optimize_for_tracking()` drops the roll of sliced `CrabCavity` elements and the `phase_taper` of sliced `Cavity` elements. Rotating the transverse kick by the roll angle and applying the RF sine law to the tapered phase give the correct kicks, and the unoptimized slices reproduce them. A 1 GeV/c proton through a 1 m, 1 MV crab cavity at zero frequency, phase π/2 and roll π/2 exits displaced 0.5000 mm horizontally instead of vertically. Through a 1 MV accelerating cavity at zero frequency with phase 0.2 rad and taper 0.3 rad, it gains 198.7 keV instead of 479.4 keV. Rolled crab cavities are affected, and so are phase-tapered cavities when the mean-radiation configuration enables the taper.

??? example "Reproducer: TRK-04.py"

    [Download TRK-04.py](TRK-04.py)

    ```python
    """TRK-04: Optimizing RF slices drops roll and phase taper

    `Line.optimize_for_tracking()` drops the roll of sliced `CrabCavity` elements and the `phase_taper`
    of sliced `Cavity` elements. Rotating the transverse kick by the roll angle and applying the RF sine
    law to the tapered phase give the correct kicks, and the unoptimized slices reproduce them. A 1
    GeV/c proton through a 1 m, 1 MV crab cavity at zero frequency, phase π/2 and roll π/2 exits
    displaced 0.5000 mm horizontally instead of vertically. Through a 1 MV accelerating cavity at zero
    frequency with phase 0.2 rad and taper 0.3 rad, it gains 198.7 keV instead of 479.4 keV. Rolled crab
    cavities are affected, and so are phase-tapered cavities when the mean-radiation configuration
    enables the taper.

    The script slices each cavity with `Teapot(1)` and tracks a proton through it before and after
    optimization. For the crab cavity it expects the kick V/p0c·(cos roll, sin roll) at mid-element
    followed by a half-length drift; for the tapered cavity it expects the energy gain V·sin(phase +
    taper), with the kernel configured for mean radiation so that the taper term is active (the line
    contains no radiating element). The unoptimized lines and an unrolled crab cavity after optimization
    must agree.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-16


    def coordinates(particle):
        return np.array([getattr(particle, key)[0] for key in ("x", "px", "y", "py")])


    def evaluate():
        cases = {}
        for roll in (0, np.pi / 2):
            line = xt.Line(
                elements={
                    "crab": xt.CrabCavity(
                        length=1,
                        crab_voltage=1e6,
                        frequency=0,
                        phase=np.pi / 2,
                        rot_s_rad=roll,
                    )
                }
            )
            line.particle_ref = xt.Particles(p0c=1e9)
            line.slice_thick_elements(slicing_strategies=[xt.Strategy(xt.Teapot(1))])
            line.build_tracker()
            px, py = 0.001 * np.cos(roll), 0.001 * np.sin(roll)
            expected = np.array([0.5 * px, px, 0.5 * py, py])
            before = xt.Particles(p0c=1e9)
            line.track(before)
            np.testing.assert_allclose(coordinates(before), expected, rtol=RTOL, atol=ATOL)
            line.optimize_for_tracking(verbose=False)
            after = xt.Particles(p0c=1e9)
            line.track(after)
            assert after.state[0] == 1
            if roll == 0:
                np.testing.assert_allclose(coordinates(after), expected, rtol=RTOL, atol=ATOL)
            cases[f"optimized roll={roll}, x,px,y,py"] = coordinates(after), expected
        line = xt.Line(
            elements={"c": xt.Cavity(length=1, voltage=1e6, frequency=0, phase=0.2, phase_taper=0.3)}
        )
        line.particle_ref = xt.Particles(p0c=1e9)
        line.slice_thick_elements(slicing_strategies=[xt.Strategy(xt.Teapot(1))])
        # Phase taper is enabled only in the radiation-capable kernel configuration.
        line.configure_radiation(model="mean")
        before = xt.Particles(p0c=1e9)
        line.track(before)
        expected_ptau = 0.001 * np.sin(0.5)
        np.testing.assert_allclose(before.ptau[0], expected_ptau, rtol=RTOL, atol=ATOL)
        line.optimize_for_tracking(verbose=False)
        after = xt.Particles(p0c=1e9)
        line.track(after)
        assert after.state[0] == 1
        cases["optimized Cavity with phase_taper=0.3, ptau"] = after.ptau[0], expected_ptau
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare(radiation="off for crab; mean configuration for cavity phase taper")
        print(
            "Expected: rotated kick (px,py)=V/p0c*(cos(roll),sin(roll)), followed by a half-length drift."
        )
        print("Expected tapered cavity energy: ptau=V/p0c*sin(phase+phase_taper).")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | low | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |

