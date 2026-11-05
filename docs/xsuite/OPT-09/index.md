---
title: "OPT-09: Sliced decelerating cavities are counted as synchrotron-radiation loss"
hide:
  - toc
---

# OPT-09 — Sliced decelerating cavities are counted as synchrotron-radiation loss

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`_get_eneloss_and_damping_rates`, radiation.py:21–32](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L21-L32)<br>[`radiation_analysis=True` output](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L354-L362) |

With `radiation_analysis=True`, Twiss counts the kicks of decelerating `ThinSliceCavity` elements as synchrotron-radiation energy loss, although it excludes the equivalent unsliced `Cavity`. Monitoring the energy change element by element isolates the bend's radiation and shows that slicing leaves it unchanged. In a 3 GeV/c electron ring with a 1 m, 0.01 rad bend and a 10 kV decelerating cavity, slicing each cavity into one Teapot slice changes the reported loss from 114.0 eV to 10.11 keV instead of keeping 114.0 eV; the excess is the full 10.00 keV RF deceleration. Accelerating kicks and zero-deceleration controls show no such excess.

??? example "Reproducer: OPT-09.py"

    [Download OPT-09.py](OPT-09.py)

    ```python
    """OPT-09: Sliced decelerating cavities are counted as synchrotron-radiation loss

    With `radiation_analysis=True`, Twiss counts the kicks of decelerating `ThinSliceCavity` elements as
    synchrotron-radiation energy loss, although it excludes the equivalent unsliced `Cavity`. Monitoring
    the energy change element by element isolates the bend's radiation and shows that slicing leaves it
    unchanged. In a 3 GeV/c electron ring with a 1 m, 0.01 rad bend and a 10 kV decelerating cavity,
    slicing each cavity into one Teapot slice changes the reported loss from 114.0 eV to 10.11 keV
    instead of keeping 114.0 eV; the excess is the full 10.00 keV RF deceleration. Accelerating kicks
    and zero-deceleration controls show no such excess.

    The script builds that ring from a `LineSegmentMap` for focusing and damping, the bend, the
    decelerating cavity and an accelerating cavity that compensates the losses (both cavities 0.4 m
    long, zero frequency, phases at the energy extrema), with mean radiation. It compares the Twiss
    `energy_loss` with the bend's loss from element-by-element tracking, before and after slicing only
    the cavities, and checks that slicing leaves the one-turn matrix unchanged. The unsliced line is the
    control.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-12, 2e-6


    def evaluate():
        p0c = 3e9
        line = xt.Line(
            elements={
                "arc": xt.LineSegmentMap(
                    length=5,
                    qx=0.2,
                    qy=0.3,
                    betx=10,
                    bety=10,
                    qs=0.01,
                    bets=-10,
                    damping_rate_x=0.001,
                    damping_rate_px=0.001,
                    damping_rate_y=0.001,
                    damping_rate_py=0.001,
                    damping_rate_zeta=0.001,
                    damping_rate_pzeta=0.001,
                ),
                "bend": xt.Bend(length=1, angle=0.01),
                "decel": xt.Cavity(length=0.4, voltage=1e4, frequency=0, phase=-np.pi / 2),
                "accel": xt.Cavity(
                    length=0.4, voltage=1e4 + 114.04218301938643, frequency=0, phase=np.pi / 2
                ),
            }
        )
        line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=-1, p0c=p0c)
        line.configure_radiation(model="mean")
        cases = {}
        baseline_matrix = None
        for sliced in (False, True):
            if sliced:
                line.discard_tracker()
                line.slice_thick_elements(
                    slicing_strategies=[
                        xt.Strategy(None),
                        xt.Strategy(xt.Teapot(1), element_type=xt.Cavity),
                    ]
                )
            tw = line.twiss(chrom=False, radiation_analysis=True, radiation_method="full")
            if baseline_matrix is None:
                baseline_matrix = tw.R_matrix.copy()
            else:
                np.testing.assert_allclose(tw.R_matrix, baseline_matrix, rtol=0, atol=1e-11)
            # Independently account for physical energy changes using scalar tracking.
            particle = tw.particle_on_co.copy()
            line.track(particle, turn_by_turn_monitor="ONE_TURN_EBE")
            assert particle.state[0] == 1
            ptau = line.record_last_track.ptau[0]
            bend_index = list(line.element_names).index("bend")
            expected_loss = -p0c * (ptau[bend_index + 1] - ptau[bend_index])
            assert expected_loss > 100
            # The segment is at its zero orbit, so its artificial damping removes no energy.
            np.testing.assert_allclose(ptau[1] - ptau[0], 0, rtol=0, atol=2e-15)
            if not sliced:
                np.testing.assert_allclose(tw.energy_loss, expected_loss, rtol=RTOL, atol=ATOL)
            cases[f"cavities sliced={sliced}, radiation energy_loss [eV]"] = (
                tw.energy_loss,
                expected_loss,
            )
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare(radiation="mean")
        print(
            "Expected: radiation loss equals the independently tracked bend loss; the -10 keV RF kick is not radiation."
        )
        print("The sliced and unsliced one-turn matrices must agree before comparing loss accounting.")
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

